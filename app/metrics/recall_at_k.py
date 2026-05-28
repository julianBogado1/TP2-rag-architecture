from app.models.ranked_song import TopRecommendedSong


def recall_at_k(
    ranked_songs: list[TopRecommendedSong],
    gt_song_ids: set[str],
    k: int,
) -> float:
    """
    Recall@K (adapted): fraction of relevant songs surfaced in top K.

    Recall@K = |relevant ∩ top_k| / min(K, |gt_song_ids|)

    The denominator is capped at K because in a recommendation system the GT
    set can contain thousands of songs — raw Context Recall would always be ~0.
    """
    if not gt_song_ids or k <= 0:
        return 0.0
    top_ids = {s.song_id for s in ranked_songs[:k]}
    relevant_retrieved = len(top_ids & gt_song_ids)
    denom = min(k, len(gt_song_ids))
    return relevant_retrieved / denom
