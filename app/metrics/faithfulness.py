import logging

from pydantic import BaseModel
from app.core.llm_client import OpenAILLMClient
from app.core.exceptions import LLMProviderError
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.ranked_song import TopRecommendedSong

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SYSTEM = (
    "You are an evaluation judge. Given a lyric context and an explanation, "
    "decide if every factual claim in the explanation can be inferred from the context. "
    "Return supported=true only if ALL claims are grounded in the context."
)


class _Judgment(BaseModel):
    supported: bool


def faithfulness_score(
    response: RecommendationResponse,
    top_songs: list[TopRecommendedSong],
    llm: OpenAILLMClient,
) -> float:
    """
    Faithfulness: fraction of LLM explanations fully supported by lyric evidence.

    Faithfulness = |explanations supported by evidence_chunks| / |total recommendations|
    """
    if not response.recommendations:
        return 0.0

    chunks_by_track = {
        (s.track_name.strip().lower(), s.artist_name.strip().lower()): s.evidence_chunks
        for s in top_songs
    }

    supported = 0
    judged = 0
    for rec in response.recommendations:
        if not rec.explanation:
            continue
        key = (rec.track_name.strip().lower(), rec.artist_name.strip().lower())
        chunks = chunks_by_track.get(key)
        if not chunks:
            continue
        context = " | ".join(chunks)
        user_msg = f"Context: {context}\n\nExplanation: {rec.explanation}"
        try:
            judgment = llm.parse_structured(_MODEL, _SYSTEM, user_msg, _Judgment)
        except LLMProviderError as e:
            # A judge failure is an infra error, not a hallucination — exclude it
            # from the denominator so it doesn't silently lower the score.
            logger.warning("faithfulness judge failed for %s: %s", key, e)
            continue
        if judgment is None:
            logger.warning("faithfulness judge returned None for %s", key)
            continue
        judged += 1
        if judgment.supported:
            supported += 1

    return supported / judged if judged > 0 else 0.0
