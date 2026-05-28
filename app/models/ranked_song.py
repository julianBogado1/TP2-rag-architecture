from pydantic import BaseModel
from app.models.song_candidate import SongMetadata


class ScoreBreakdown(BaseModel):
    score_lyrics: float
    score_audio: float
    score_profile: float
    score_popularity: float
    score_recency: float


class RankedSongCandidate(BaseModel):
    """A candidate after reranking (all candidates, scored)."""
    song_id: str
    track_name: str
    artist_name: str
    score_total: float
    score_breakdown: ScoreBreakdown
    evidence_chunks: list[str]
    metadata: SongMetadata


class TopRecommendedSong(BaseModel):
    """A candidate that survived top-N selection (dedup + per-artist cap)."""
    song_id: str
    track_name: str
    artist_name: str
    score_total: float
    score_breakdown: ScoreBreakdown
    evidence_chunks: list[str]
    metadata: SongMetadata
