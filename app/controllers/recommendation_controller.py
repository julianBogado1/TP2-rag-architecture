from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.core.exceptions import (
    UserNotFoundError, LLMProviderError, VectorStoreError, RecommendationError,
)
from app.services.retrieval.recommendation_orchestrator import RecommendationOrchestrator
from app.models.recommendation_response import RecommendationResponse

RECOMMEND_MAX_RETRIES = 3       # retries after the first attempt


class RecommendRequestBody(BaseModel):
    user_id: str
    raw_prompt: str = Field(min_length=1, max_length=2000)


def build_recommendation_router(orchestrator: RecommendationOrchestrator) -> APIRouter:
    router = APIRouter()

    @router.post("/recommend", response_model=RecommendationResponse)
    def recommend(body: RecommendRequestBody) -> RecommendationResponse:
        try:
            response = orchestrator.recommend(body.user_id, body.raw_prompt)
            for _ in range(RECOMMEND_MAX_RETRIES):
                if response.recommendations:
                    break
                response = orchestrator.recommend(body.user_id, body.raw_prompt)
            return response
        except UserNotFoundError:
            raise HTTPException(404, "User not found")
        except LLMProviderError:
            raise HTTPException(502, "Upstream model failure")
        except VectorStoreError:
            raise HTTPException(502, "Vector store failure")
        except RecommendationError:
            raise HTTPException(502, "Recommendation pipeline failure")

    return router
