from app.models.ranked_song import TopRecommendedSong


def context_precision_at_k(
    ranked_songs: list[TopRecommendedSong],
    gt_song_ids: set[str],
    k: int,
) -> float:
    """
    Context Precision@K: are relevant songs ranked higher than irrelevant ones?

    CP@K = Σ(Precision@k × v_k) / total_relevant_in_top_k
    where v_k = 1 if song at rank k is in gt_song_ids, else 0
    and Precision@k = TP@k / k
    """
    top = ranked_songs[:k]
    total_relevant = sum(1 for s in top if s.song_id in gt_song_ids)
    if total_relevant == 0:
        return 0.0

    numerator = 0.0
    tp = 0
    for i, song in enumerate(top, start=1):
        if song.song_id in gt_song_ids:
            tp += 1
            numerator += tp / i

    return numerator / total_relevant
