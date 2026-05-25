from app.models.ranked_song import RankedSongCandidate, TopRecommendedSong


class TopNSelectorService:
    """Sort by score, dedup by normalized title, cap per artist, return top N."""

    def __init__(self, max_per_artist: int) -> None:
        self._max_per_artist = max_per_artist

    def select(self, ranked: list[RankedSongCandidate], top_n: int) -> list[TopRecommendedSong]:
        sorted_ranked = sorted(ranked, key=lambda r: r.score_total, reverse=True)
        seen_titles: set[str] = set()
        per_artist: dict[str, int] = {}
        out: list[TopRecommendedSong] = []
        for r in sorted_ranked:
            key = r.track_name.strip().lower()
            if key in seen_titles:
                continue
            if per_artist.get(r.artist_name, 0) >= self._max_per_artist:
                continue
            seen_titles.add(key)
            per_artist[r.artist_name] = per_artist.get(r.artist_name, 0) + 1
            out.append(TopRecommendedSong(
                song_id=r.song_id, track_name=r.track_name, artist_name=r.artist_name,
                score_total=r.score_total, score_breakdown=r.score_breakdown,
                evidence_chunks=r.evidence_chunks, metadata=r.metadata,
            ))
            if len(out) >= top_n:
                break
        return out
