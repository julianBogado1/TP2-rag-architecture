from pydantic import BaseModel
from app.models.prompt_score import PromptScore
from app.models.user_profile import UserProfileData
from app.models.ranked_song import TopRecommendedSong


class RecommendationContext(BaseModel):
    raw_prompt: str
    prompt_score: PromptScore
    user_profile: UserProfileData
    top_songs: list[TopRecommendedSong]
    explanation_rules: list[str]


class SongRecommendation(BaseModel):
    rank: int
    track_name: str
    artist_name: str
    explanation: str
    matched_mood: list[str]
    matched_audio_features: list[str]
    score_total: float | None = None


class RecommendationResponse(BaseModel):
    message: str
    recommendations: list[SongRecommendation]
