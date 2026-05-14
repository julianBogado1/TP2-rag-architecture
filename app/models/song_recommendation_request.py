from datetime import date
from pydantic import BaseModel, Field
from app.models.prompt_score import PromptScore, PromptAudioFeatures
from app.models.user_profile import UserProfileData


class MetadataFilters(BaseModel):
    genres_in: list[str] | None = None
    genres_not_in: list[str] | None = None
    artist_not_in: list[str] | None = None
    preferred_language: str | None = None
    min_popularity: int | None = None
    release_date_min: date | None = None
    release_date_max: date | None = None


class RankingWeights(BaseModel):
    w_lyrics: float = 0.55
    w_audio: float = 0.30
    w_profile: float = 0.10
    w_popularity: float = 0.03
    w_recency: float = 0.02


class SongRecommendationRequest(BaseModel):
    user_id: str
    raw_prompt: str
    prompt_score: PromptScore
    user_profile: UserProfileData
    semantic_query: str
    target_audio_features: PromptAudioFeatures
    metadata_filters: MetadataFilters
    ranking_weights: RankingWeights
    top_k_retrieval: int = Field(default=50, gt=0)
    top_n_output: int = Field(default=10, gt=0)
