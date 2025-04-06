import pickle

POPULAR_RECOMMENDATIONS: list[int] = [
    10440,
    15297,
    9728,
    13865,
    4151,
    3734,
    2657,
    4880,
    142,
    6809,
]

KNN_RECOMMENDATIONS: str = "models/knn_tfidf_model_with_popular_df.pickle"
ALS_MODEL: str = "models/als_with_features_model.pickle"


def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    except pickle.UnpicklingError:
        raise pickle.UnpicklingError()
