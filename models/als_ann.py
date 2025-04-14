from rectools.models import ImplicitALSWrapperModel
from rectools.tools import UserToItemAnnRecommender

from models.utils import ALS_MODEL, POPULAR_RECOMMENDATIONS, load_pickle

model: ImplicitALSWrapperModel = load_pickle(ALS_MODEL)

if model:
    user_id_map = load_pickle("models/als_ann_dataset_user_id_map.pickle")
    item_id_map = load_pickle("models/als_ann_dataset_item_id_map.pickle")

    user_vectors, item_vectors = model.get_vectors()

    model_ann = UserToItemAnnRecommender(
        user_vectors=user_vectors,
        item_vectors=item_vectors,
        user_id_map=user_id_map,
        item_id_map=item_id_map,
        index_init_params={"method": "hnsw", "space": "cosinesimil"},
    )
    model_ann.index.loadIndex("models/als_ann_with_features_index.pickle")


def get_als_recomendations(user_id: int) -> list[int]:
    if model_ann and user_id in user_id_map.external_ids:
        model_ann.get_item_list_for_user(user_id, top_n=10).tolist()

    return POPULAR_RECOMMENDATIONS
