import os

import pandas as pd

from models.utils import DSSM_RECOMMENDATIONS, POPULAR_RECOMMENDATIONS

if os.path.exists(DSSM_RECOMMENDATIONS):
    dssm_recomendations = pd.read_pickle(DSSM_RECOMMENDATIONS)
    dssm_recomendations.set_index('user_id', inplace=True)


def get_dssm_recomendations(user_id: int) -> list[int]:
    if dssm_recomendations is not None and user_id in dssm_recomendations.index:
        return dssm_recomendations.loc[user_id, 'item_id']

    return POPULAR_RECOMMENDATIONS
