from app.models.song_candidate import CandidateChunk, CandidateSong


class CandidateAggregatorService:
    """Groups CandidateChunk[] by song_id and looks up evidence lyrics from Mongo."""

    def __init__(self, song_repo, max_evidence_chunks: int) -> None:
        self._song_repo = song_repo
        self._max_evidence_chunks = max_evidence_chunks

    def aggregate(self, chunks: list[CandidateChunk]) -> list[CandidateSong]:
        if not chunks:
            return []
        groups: dict[str, list[CandidateChunk]] = {}
        for c in chunks:
            groups.setdefault(c.song_id, []).append(c)

        song_ids = [int(sid) for sid in groups.keys() if sid.isdigit()]
        lyrics_by_id = {str(s.song_id): s.lyrics for s in self._song_repo.get_by_ids(song_ids)}

        return [
            self._build(song_id, chunks_for_song, lyrics_by_id.get(song_id, ""))
            for song_id, chunks_for_song in groups.items()
        ]

    def _build(self, song_id: str, chunks: list[CandidateChunk], lyrics: str) -> CandidateSong:
        chunks_sorted = sorted(chunks, key=lambda c: c.lyrics_similarity, reverse=True)
        best = chunks_sorted[0]
        return CandidateSong(
            song_id=song_id,
            track_name=best.metadata.track_name,
            artist_name=best.metadata.artist_name,
            genres=best.metadata.genres,
            popularity=best.metadata.popularity,
            release_date=best.metadata.release_date,
            best_lyrics_chunks=self._slice_evidence(lyrics),
            best_lyrics_similarity=best.lyrics_similarity,
            audio_features=best.metadata.audio_features,
        )

    def _slice_evidence(self, lyrics: str) -> list[str]:
        # Alternative: just truncate to 400 chars — simpler but loses verse structure.
        if not lyrics:
            return []
        lines = [ln.strip() for ln in lyrics.split("\n") if ln.strip()]
        return lines[: self._max_evidence_chunks]
