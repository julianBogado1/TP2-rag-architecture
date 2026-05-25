from app.core.llm_client import LLMClient
from app.models.song_recommendation_request import SongRecommendationRequest
from app.models.ranked_song import TopRecommendedSong
from app.models.recommendation_response import RecommendationContext, RecommendationResponse


_RESPONSE_SYSTEM_PROMPT = """You are a music recommendation assistant. You receive
a RecommendationContext with the user's raw prompt, their parsed PromptScore,
their UserProfile, and the top N candidate songs already ranked. Produce a
RecommendationResponse: a short opener message in the user's original language,
plus one SongRecommendation per top song. For each, write a short explanation
(<=40 words) citing at least one phrase from evidence_chunks, naming matched_mood
fields from prompt_score (e.g., 'happy', 'energetic'), and matched_audio_features.
Never invent songs not present in top_songs."""

_EXPLANATION_RULES = [
    "Reply in the same language as raw_prompt.",
    "Cite at least one phrase from evidence_chunks per song.",
    "Reference matched_mood explicitly.",
    "Do not invent songs not present in top_songs.",
    "Keep each explanation under 40 words.",
]


class ResponseGeneratorService:
    """Final LLM call: builds RecommendationContext and produces a typed response."""

    def __init__(self, llm_client: LLMClient, model: str) -> None:
        self._llm = llm_client
        self._model = model

    def generate(
        self,
        top_songs: list[TopRecommendedSong],
        req: SongRecommendationRequest,
    ) -> RecommendationResponse:
        ctx = RecommendationContext(
            raw_prompt=req.raw_prompt,
            prompt_score=req.prompt_score,
            user_profile=req.user_profile,
            top_songs=top_songs,
            explanation_rules=_EXPLANATION_RULES,
        )
        return self._llm.generate_structured(
            model=self._model,
            system=_RESPONSE_SYSTEM_PROMPT,
            context=ctx,
            schema=RecommendationResponse,
        )
