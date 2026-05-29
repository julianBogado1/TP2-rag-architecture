import logging
import re

from pydantic import BaseModel
from app.core.llm_client import OpenAILLMClient
from app.core.exceptions import LLMProviderError
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.ranked_song import TopRecommendedSong

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SYSTEM = (
    "You are an evaluation judge. You are given the CONTEXT the recommender actually "
    "had for a song — lyric excerpts plus structured facts (genres, audio features, and "
    "the moods/audio traits the system matched) — and an EXPLANATION it produced. "
    "Decompose the explanation into its atomic factual claims (one verifiable statement "
    "each). For every claim, set supported=true if it can be inferred from the context, "
    "false otherwise. A claim about mood, energy, danceability, tempo or genre is "
    "supported when the corresponding fact appears in the context (e.g. 'high energy' is "
    "supported by energy=0.8; 'happy' by a matched mood or high valence). Return one "
    "entry per claim."
)

# audio axes carried on SongMetadata.audio_features
_AUDIO_AXES = ("valence", "energy", "danceability",
               "acousticness", "instrumentalness", "tempo_norm")


class _Claim(BaseModel):
    text: str
    supported: bool


class _Judgment(BaseModel):
    claims: list[_Claim]


def _normalize(name: str) -> str:
    """Loose normalisation so 'Song (feat. X)' and 'song' match the same key."""
    name = name.lower().strip()
    name = re.sub(r"\(feat\.?.*?\)|\bfeat\.?\b.*$", "", name)  # drop featured-artist noise
    name = re.sub(r"[^a-z0-9 ]+", "", name)                     # drop punctuation
    return re.sub(r"\s+", " ", name).strip()


def _match_song(rec: SongRecommendation,
                by_key: dict[tuple[str, str], TopRecommendedSong],
                ordered: list[TopRecommendedSong]) -> TopRecommendedSong | None:
    """Map a recommendation back to its source song: try a normalised (track, artist)
    key first, then fall back to rank position (one rec per top song, in order)."""
    hit = by_key.get((_normalize(rec.track_name), _normalize(rec.artist_name)))
    if hit is not None:
        return hit
    idx = rec.rank - 1
    if 0 <= idx < len(ordered):
        return ordered[idx]
    return None


def _build_context(song: TopRecommendedSong, rec: SongRecommendation) -> str:
    """The full grounded context the recommender had: lyrics + structured facts."""
    parts: list[str] = []
    if song.evidence_chunks:
        parts.append("Lyrics: " + " | ".join(song.evidence_chunks))
    if song.metadata.genres:
        parts.append("Genres: " + ", ".join(song.metadata.genres))
    af = song.metadata.audio_features
    if af is not None:
        parts.append("Audio features: " + ", ".join(
            f"{axis}={getattr(af, axis):.2f}" for axis in _AUDIO_AXES))
    if rec.matched_mood:
        parts.append("Matched moods: " + ", ".join(rec.matched_mood))
    if rec.matched_audio_features:
        parts.append("Matched audio traits: " + ", ".join(rec.matched_audio_features))
    return "\n".join(parts)


def faithfulness_score(
    response: RecommendationResponse,
    top_songs: list[TopRecommendedSong],
    llm: OpenAILLMClient,
) -> float:
    """
    Faithfulness (RAGAS, claim-level): fraction of the individual factual claims across
    all explanations that are grounded in the context the recommender actually had —
    lyric evidence plus the structured facts (genres, audio features, matched moods)
    the explanation is allowed to cite.

    The judge decomposes each explanation into atomic claims and marks each one
    supported/unsupported; we pool all claims across recommendations (micro-average):

        Faithfulness = |claims inferable from context| / |total claims judged|
    """
    if not response.recommendations:
        return 0.0

    by_key = {
        (_normalize(s.track_name), _normalize(s.artist_name)): s
        for s in top_songs
    }

    supported = 0
    total = 0
    for rec in response.recommendations:
        if not rec.explanation:
            continue
        song = _match_song(rec, by_key, top_songs)
        if song is None:
            logger.warning("faithfulness: no source song for rec %r", rec.track_name)
            continue
        context = _build_context(song, rec)
        if not context:
            continue
        user_msg = f"Context:\n{context}\n\nExplanation: {rec.explanation}"
        try:
            judgment = llm.parse_structured(_MODEL, _SYSTEM, user_msg, _Judgment)
        except LLMProviderError as e:
            # A judge failure is an infra error, not a hallucination — exclude it
            # from the denominator so it doesn't silently lower the score.
            logger.warning("faithfulness judge failed for %s: %s", rec.track_name, e)
            continue
        if judgment is None:
            logger.warning("faithfulness judge returned None for %s", rec.track_name)
            continue
        if not judgment.claims:
            # No claims extracted — nothing to attribute, contributes nothing.
            continue
        total += len(judgment.claims)
        supported += sum(1 for c in judgment.claims if c.supported)

    return supported / total if total > 0 else 0.0
