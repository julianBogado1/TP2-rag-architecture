from math import log2
from app.models.ranked_song import TopRecommendedSong


def ndcg_at_k(
    ranked_songs: list[TopRecommendedSong],
    gt_song_ids: set[str],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain @K — a rank-aware quality metric.

    Unlike recall@k it does not divide by |GT|, so it stays meaningful when the
    ground-truth set is huge: it normalizes by the best achievable ordering within K
    and rewards placing relevant songs near the top.

        DCG@k  = Σ_{i=1..k} rel_i / log2(i + 1)         (rel_i binary: 1 if in GT)
        IDCG@k = Σ_{i=1..m} 1     / log2(i + 1)         (m = min(k, |GT|))
        NDCG@k = DCG@k / IDCG@k                          (0.0 when IDCG@k == 0)
    """
    if not gt_song_ids or k <= 0:
        return 0.0

    dcg = sum(
        1.0 / log2(i + 1)
        for i, song in enumerate(ranked_songs[:k], start=1)
        if song.song_id in gt_song_ids
    )
    m = min(k, len(gt_song_ids))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0
