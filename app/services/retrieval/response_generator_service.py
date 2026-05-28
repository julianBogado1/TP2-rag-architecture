from app.core.llm_client import LLMClient
from app.models.prompt_score import PromptScore
from app.models.song_recommendation_request import SongRecommendationRequest
from app.models.ranked_song import TopRecommendedSong
from app.models.recommendation_response import (
    RecommendationContext, RecommendationResponse, SongRecommendation,
)


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

SKIP_LLM_MESSAGE = "Top picks (response LLM disabled)."

# Thresholds for synthesising matched_* fields when LLM is skipped
MOOD_MATCH_THRESHOLD  = 0.5
AUDIO_MATCH_THRESHOLD = 0.5

_MOOD_FIELDS = ("happy", "sad", "energetic", "calm", "nostalgic",
                "romantic", "assertive", "deep", "playful")
_AUDIO_FIELDS = ("valence", "energy", "danceability",
                 "acousticness", "instrumentalness", "tempo_norm")


class ResponseGeneratorService:
    """Final LLM call (or synthesised fallback when skip_llm=True)."""

    def __init__(self, llm_client: LLMClient, model: str, skip_llm: bool = False) -> None:
        self._llm = llm_client
        self._model = model
        self._skip_llm = skip_llm

    def generate(
        self,
        top_songs: list[TopRecommendedSong],
        req: SongRecommendationRequest,
    ) -> RecommendationResponse:
        if self._skip_llm:
            return self._synthesize(top_songs, req)
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

    @staticmethod
    def _synthesize(
        top_songs: list[TopRecommendedSong],
        req: SongRecommendationRequest,
    ) -> RecommendationResponse:
        matched_mood  = _fields_above(req.prompt_score, _MOOD_FIELDS, MOOD_MATCH_THRESHOLD)
        matched_audio = _fields_above(req.target_audio_features, _AUDIO_FIELDS, AUDIO_MATCH_THRESHOLD)
        recs = [
            SongRecommendation(
                rank=i + 1,
                track_name=s.track_name,
                artist_name=s.artist_name,
                explanation="",
                matched_mood=matched_mood,
                matched_audio_features=matched_audio,
                score_total=s.score_total,
            )
            for i, s in enumerate(top_songs)
        ]
        return RecommendationResponse(message=SKIP_LLM_MESSAGE, recommendations=recs)


def _fields_above(obj, names: tuple[str, ...], threshold: float) -> list[str]:
    # PromptScore moods and audio axes may be None when the prompt is silent on them;
    # treat None as "no signal" rather than letting the comparison raise.
    return [n for n in names
            if (v := getattr(obj, n, None)) is not None and v > threshold]
