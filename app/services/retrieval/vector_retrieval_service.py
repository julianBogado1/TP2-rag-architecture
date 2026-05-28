import json
from datetime import date
from app.models.song_recommendation_request import SongRecommendationRequest, MetadataFilters
from app.models.song_candidate import CandidateChunk, SongMetadata, AudioFeatures
from app.models.parsed_audio_features import ParsedAudioFeatures
from app.models.raw_pinecone_match import RawPineconeMatch
from app.persistence.vector.pinecone_repository import PineconeRepository

# Tunables for vector retrieval
OVERFETCH_MULTIPLIER = 10                       # Alternative: fixed 500 — simpler but ignores per-request top_k.
TEMPO_NORMALIZATION_DIVISOR = 250.0             # BPM ceiling used to map tempo into [0, 1]
DEFAULT_RELEASE_YEAR_FALLBACK = 1900             # used only when release_date is missing/malformed AND no filter excludes it


class VectorRetrievalService:
    """Pinecone query + JSON metadata parsing + Python post-retrieval filtering."""

    def __init__(self, repo: PineconeRepository) -> None:
        self._repo = repo

    def retrieve(
        self,
        req: SongRecommendationRequest,
        query_vec: list[float],
    ) -> list[CandidateChunk]:
        raw_matches = self._repo.query(query_vec, top_k=req.top_k_retrieval * OVERFETCH_MULTIPLIER)
        candidates: list[CandidateChunk] = []
        for m in raw_matches:
            # audio is None for lyrics-only songs (no Spotify match): keep them and
            # let the reranker score lyrics-only. chars is always a dict (never None).
            audio = self._parse_audio(m.metadata.get("audio_features_chunk"))
            chars = self._parse_chars(m.metadata.get("song_characteristics_chunk"))
            if not self._passes_filters(m.metadata, chars, req.metadata_filters):
                continue
            candidates.append(self._build_candidate_chunk(m, audio, chars))
            if len(candidates) >= req.top_k_retrieval:
                break
        return candidates

    @staticmethod
    def _parse_audio(blob: str | None) -> ParsedAudioFeatures | None:
        """Return parsed audio features, or None meaning 'no audio' (the song is
        still kept and scored lyrics-only). Missing/empty/malformed/incomplete all
        map to None."""
        if not blob:
            return None
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            return None
        required = ("valence", "energy", "danceability", "acousticness", "instrumentalness")
        if not isinstance(data, dict) or not all(k in data for k in required):
            return None
        try:
            return ParsedAudioFeatures(
                valence=data["valence"],
                energy=data["energy"],
                danceability=data["danceability"],
                acousticness=data["acousticness"],
                instrumentalness=data["instrumentalness"],
                tempo_norm=_normalize_tempo(data.get("tempo")),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_chars(blob: str | None) -> dict:
        """Return characteristics dict; empty dict on missing/malformed (never drops a song)."""
        if not blob:
            return {}
        try:
            parsed = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _passes_filters(meta: dict, chars: dict, filters: MetadataFilters) -> bool:
        genres = meta.get("genres") or []
        artist = (meta.get("artist_name") or "").lower()
        track  = (meta.get("track_name")  or "").lower()
        if filters.genres_in and not (set(genres) & set(filters.genres_in)):
            return False
        if filters.genres_not_in and (set(genres) & set(filters.genres_not_in)):
            return False
        if filters.artist_in and artist not in {a.lower() for a in filters.artist_in}:
            return False
        if filters.artist_not_in and artist in {a.lower() for a in filters.artist_not_in}:
            return False
        if filters.songs_not_in and track in {s.lower() for s in filters.songs_not_in}:
            return False
        if filters.preferred_language and chars.get("language") != filters.preferred_language:
            return False
        if filters.min_popularity is not None and chars.get("popularity", 0) < filters.min_popularity:
            return False
        if filters.release_date_min is not None:
            year = _safe_year(meta.get("release_date"))
            if year is None or year < filters.release_date_min.year:
                return False
        if filters.release_date_max is not None:
            year = _safe_year(meta.get("release_date"))
            if year is None or year > filters.release_date_max.year:
                return False
        return True

    @staticmethod
    def _build_candidate_chunk(m: RawPineconeMatch, audio: ParsedAudioFeatures | None, chars: dict) -> CandidateChunk:
        year = _safe_year(m.metadata.get("release_date")) or DEFAULT_RELEASE_YEAR_FALLBACK
        af = None if audio is None else AudioFeatures(
            valence=audio.valence,
            energy=audio.energy,
            danceability=audio.danceability,
            acousticness=audio.acousticness,
            instrumentalness=audio.instrumentalness,
            tempo_norm=audio.tempo_norm,
        )
        song_id = str(m.metadata.get("song_id", ""))
        return CandidateChunk(
            chunk_id=m.chunk_id,
            song_id=song_id,
            lyrics_chunk="",
            lyrics_similarity=m.score,
            metadata=SongMetadata(
                track_name=m.metadata.get("track_name", ""),
                artist_name=m.metadata.get("artist_name", ""),
                genres=m.metadata.get("genres") or [],
                popularity=chars.get("popularity", 0),
                release_date=date(year, 1, 1),
                audio_features=af,
            ),
        )


def _safe_year(value) -> int | None:
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _normalize_tempo(tempo: float | int | None) -> float:
    # Alternative: step buckets — slow/<90, mid/90-130, fast/>130 — simpler but loses granularity.
    if tempo is None:
        return 0.5
    return max(0.0, min(1.0, float(tempo) / TEMPO_NORMALIZATION_DIVISOR))
