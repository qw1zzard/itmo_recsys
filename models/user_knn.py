from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import ItemItemRecommender
from rectools import Columns
from scipy.sparse import coo_matrix, spmatrix


class UserKnn:
    """
    A user-based KNN model wrapper around `implicit.nearest_neighbours.ItemItemRecommender`
    """

    SIMILAR_USER_COLUMN = 'similar_user_id'
    SIMILARITY_COLUMN = 'similarity'
    IDF_COLUMN = 'idf'

    def __init__(self, model: ItemItemRecommender, N_similar_users: int):
        self.model = model
        self.N_similar_users = N_similar_users

        self.users_inv_mapping = None
        self.users_mapping = None
        self.items_inv_mapping = None
        self.items_mapping = None

        self.item_idf = None
        self.popular_items = None
        self.cold_model_fitted = False
        self.interacted_items_dataframe = None

    def _set_mappings(self, interactions: pd.DataFrame) -> None:
        """
        Create dictionaries to map external IDs (users, items) to internal IDs and vice versa.
        """
        unique_users = interactions[Columns.User].unique()
        self.users_inv_mapping = dict(enumerate(unique_users))
        self.users_mapping = {v: k for k, v in self.users_inv_mapping.items()}

        unique_items = interactions[Columns.Item].unique()
        self.items_inv_mapping = dict(enumerate(unique_items))
        self.items_mapping = {v: k for k, v in self.items_inv_mapping.items()}

    def _get_user_item_matrix(self, interactions: pd.DataFrame) -> spmatrix:
        """
        Construct a sparse user-item matrix in CSR format.
        Rows represent users, and columns represent items.
        """
        user_idx = interactions[Columns.User].map(self.users_mapping.get)
        item_idx = interactions[Columns.Item].map(self.items_mapping.get)
        data = interactions[Columns.Weight].astype(np.float32)

        user_item_coo = coo_matrix((data, (user_idx, item_idx)))
        return user_item_coo.tocsr()

    def _set_interacted_items_dataframe(self, interactions: pd.DataFrame) -> None:
        """
        Groups interactions by user to get item_id list for each user
        """
        self.interacted_items_dataframe = (
            interactions.groupby(Columns.User, as_index=False)
            .agg({Columns.Item: list})
            .rename(columns={Columns.User: self.SIMILAR_USER_COLUMN})
        )

    @staticmethod
    def idf(n: int, x: float):
        """
        Calculates IDF for one item
        """
        return np.log((1 + n) / (1 + x) + 1)

    def _count_item_idf(self, interactions: pd.DataFrame) -> None:
        """
        Calculate IDF values for all items present in the interactions dataset
         and store the result in self.item_idf.
        """
        item_freqs = Counter(interactions[Columns.Item].values)
        item_idf_df = pd.DataFrame.from_dict(
            item_freqs, orient='index', columns=['doc_freq']
        ).reset_index()
        total_interactions = len(interactions)
        item_idf_df[self.IDF_COLUMN] = item_idf_df['doc_freq'].apply(
            lambda x: self.idf(total_interactions, x)
        )
        self.item_idf = item_idf_df

    def _prepare_for_model(self, train_interactions: pd.DataFrame) -> None:
        """
        Sets mappings, grouped interactions, calculates idf
        """
        self._set_mappings(train_interactions)
        self._set_interacted_items_dataframe(train_interactions)
        self._count_item_idf(train_interactions)

    def fit_cold_model(self, train_interactions: pd.DataFrame) -> None:
        """
        Fit a model for cold recommendations.

        Parameters:
        train_interactions (pd.DataFrame): interaction data used to train the model.
        """

        self.popular_items = (
            train_interactions.groupby('item_id')
            .size()
            .sort_values(ascending=False)
            .index.tolist()
        )
        self.cold_model_fitted = True

    def recommend_cold(
        self, users: Union[list, np.array], k: int = 100
    ) -> pd.DataFrame:
        """
        Return recommendations for the given cold users.
        Can be called separately or within the class. Supports both list and numpy array as input.

        Parameters:
        users (list | np.array): List or array of users for whom recommendations will be generated.
        k (int, optional): Number of recommendations to generate per user. Default is 100.

        Returns:
        pd.DataFrame: A dataframe containing user-item recommendations.
        """

        if not self.cold_model_fitted:
            raise ValueError('Cold model is not fitted yet.')

        top_items = self.popular_items[:k]
        cold_recs = pd.DataFrame(
            [
                (user, item, rank + 1)
                for user in users
                for rank, item in enumerate(top_items)
            ],
            columns=[Columns.User, Columns.Item, Columns.Rank],
        )
        return cold_recs

    def fit(self, train_interactions: pd.DataFrame) -> None:
        """
        Fit the model on the provided training data.

        Internally:
        1) Prepare mappings, watchlist DataFrame, and item IDF.
        2) Create a user-item matrix and fit the underlying Implicit model.
        """
        self._prepare_for_model(train_interactions)
        user_item_matrix = self._get_user_item_matrix(train_interactions)
        self.model.fit(user_item_matrix.T)

        if not self.cold_model_fitted:
            self.fit_cold_model(train_interactions)

    def _get_similar_users(
        self, external_user_id: int
    ) -> tuple[list[int], list[float]]:
        """
        Retrieve a list of similar users and corresponding similarities
        from the underlying Implicit model.
        """
        if external_user_id not in self.users_mapping:
            # if user doesn't exist in mapping, return sentinel (-1).
            return [-1], [-1]

        internal_user_id = self.users_mapping[external_user_id]
        user_ids, similarities = self.model.similar_items(
            internal_user_id, N=self.N_similar_users
        )
        # convert back to external IDs
        external_user_ids = [self.users_inv_mapping[u_id] for u_id in user_ids]
        return external_user_ids, similarities

    @staticmethod
    def get_rank(recs: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Sort recommendations by score in descending order,
        assign ranks within each user group, and then truncate by top-k.
        """
        recs = recs.sort_values([Columns.User, Columns.Score], ascending=False)
        recs = recs.drop_duplicates([Columns.User, Columns.Item])
        recs[Columns.Rank] = recs.groupby(Columns.User).cumcount() + 1
        recs = recs[recs[Columns.Rank] <= k][
            [Columns.User, Columns.Item, Columns.Score, Columns.Rank]
        ]

        return recs

    def recommend(self, users: np.ndarray, k: int) -> pd.DataFrame:
        """
        Generate top-k recommendations for the specified list of users.

        Steps:
        1) Find similar users for each target user.
        2) Join watched items from these similar users.
        3) Compute a final score as similarity * IDF.
        4) Return top-k items per user.
        """

        recs = pd.DataFrame({Columns.User: users})
        recs[self.SIMILAR_USER_COLUMN], recs[self.SIMILARITY_COLUMN] = zip(
            *recs[Columns.User].map(lambda user_id: self._get_similar_users(user_id))
        )
        recs = recs.set_index(Columns.User).apply(pd.Series.explode).reset_index()

        recs = (
            recs[~(recs[Columns.User] == recs[self.SIMILAR_USER_COLUMN])]
            .merge(
                self.interacted_items_dataframe,
                on=[self.SIMILAR_USER_COLUMN],
                how='left',
            )
            .explode(Columns.Item)
            .sort_values([Columns.User, self.SIMILARITY_COLUMN], ascending=False)
            .drop_duplicates([Columns.User, Columns.Item], keep='first')
            .merge(self.item_idf, left_on=Columns.Item, right_on='index', how='left')
        )

        recs[Columns.Score] = recs[self.SIMILARITY_COLUMN] * recs[self.IDF_COLUMN]
        recs = recs[[Columns.User, Columns.Item, Columns.Score]]

        cold_recs = self.recommend_cold(users=users, k=k)
        cold_recs[Columns.Score] = 0.0

        recs = pd.concat([recs, cold_recs])
        recs = self.get_rank(recs, k=k)
        return recs


def get_knn_recomendations(user_id: int, knn_recomendations: pd.DataFrame):
    if user_id in knn_recomendations.index:
        return knn_recomendations.loc[user_id, 'item_id']

    return [10440, 15297, 9728, 13865, 4151, 3734, 2657, 4880, 142, 6809]
