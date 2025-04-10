import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from starlette import status

from models.user_knn import get_knn_recomendations
from service.api.exceptions import UserNotFoundError
from service.log import app_logger

if not os.getenv("API_KEY"):
    load_dotenv()

auth_scheme = HTTPBearer()


KNN_MODEL_PREDICTIONS_PATH = "models/knn_tfidf_model_predictions.parquet"
knn_tfidf_model_with_popular_df = None

if os.path.exists(KNN_MODEL_PREDICTIONS_PATH):
    knn_tfidf_model_with_popular_df = pd.read_parquet(KNN_MODEL_PREDICTIONS_PATH)
    knn_tfidf_model_with_popular_df.set_index("user_id", inplace=True)


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


class UnauthorizedMessage(BaseModel):
    detail: str = "Bearer token missing or unknown"
    description: str = "Неверно указанный Bearer token"


class ModelNotFoundMessage(BaseModel):
    detail: str = "Model is not found"
    description: str = "Введено неправильное имя модели"


class SuccessMessage(BaseModel):
    detail: str = "Successful response from /health"


router = APIRouter()


@router.get(
    path="/health",
    tags=["Health"],
    responses={status.HTTP_200_OK: {"model": SuccessMessage}},
)
async def health() -> str:
    return "I am alive"


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    API_KEY = os.getenv("API_KEY")
    if credentials.scheme != "Bearer" or credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    dependencies=[Depends(verify_token)],
    responses={
        status.HTTP_200_OK: {"model": RecoResponse},
        status.HTTP_401_UNAUTHORIZED: {"model": UnauthorizedMessage},
        status.HTTP_404_NOT_FOUND: {"model": ModelNotFoundMessage},
    },
)
async def get_reco(request: Request, model_name: str, user_id: int) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    if model_name == "knn_tfidf_model_with_popular":
        if knn_tfidf_model_with_popular_df is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="knn_tfidf_model_with_popular is not loaded",
            )
        reco = get_knn_recomendations(user_id, knn_tfidf_model_with_popular_df)

    elif model_name == "baseline_first_10_items":
        reco = list(range(k_recs))

    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ModelNotFoundMessage().detail,
        )

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
