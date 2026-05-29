from app.models.ranked_song import TopRecommendedSong


def recall_at_k(
    ranked_songs: list[TopRecommendedSong],
    gt_song_ids: set[str],
    k: int,
    pool_size: int | None = None,
) -> float:
    """
    Recall@K (adapted): fraction of relevant songs surfaced in top K.

    Recall@K = |relevant ∩ top_k| / min(K, |gt_song_ids|, pool_size)

    The denominator is capped at K because in a recommendation system the GT
    set can contain thousands of songs — raw Context Recall would always be ~0.

    `pool_size` caps it further at the number of *eligible* candidates evaluated.
    For audio cases the harness scores only audio-bearing songs, so a top-K with
    few audio songs would otherwise pin recall at |pool|/K no matter how good the
    matches are. Defaults to K (no extra cap) for existing callers.
    """
    if not gt_song_ids or k <= 0:
        return 0.0
    top_ids = {s.song_id for s in ranked_songs[:k]}
    relevant_retrieved = len(top_ids & gt_song_ids)
    denom = min(k, len(gt_song_ids))
    if pool_size is not None:
        denom = min(denom, pool_size)
    return relevant_retrieved / denom if denom > 0 else 0.0
