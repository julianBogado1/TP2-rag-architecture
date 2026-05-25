from pydantic import BaseModel
from app.core.llm_client import OpenAILLMClient
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.ranked_song import TopRecommendedSong

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
    total = 0
    for rec in response.recommendations:
        total += 1
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
            if judgment and judgment.supported:
                supported += 1
        except Exception:
            pass

    return supported / total if total > 0 else 0.0
