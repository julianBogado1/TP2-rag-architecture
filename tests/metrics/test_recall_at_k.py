from types import SimpleNamespace
from app.metrics.recall_at_k import recall_at_k


def _s(sid):
    return SimpleNamespace(song_id=sid)


def test_all_relevant_retrieved():
    # gt={1,2}, both in top, denom=min(10,2)=2 -> 2/2
    assert recall_at_k([_s("1"), _s("2"), _s("3")], {"1", "2"}, k=10) == 1.0


def test_partial_recall_capped_denominator():
    # retrieved relevant={1}, gt={1,2,3}, denom=min(2,3)=2 -> 0.5
    assert recall_at_k([_s("1"), _s("9")], {"1", "2", "3"}, k=2) == 0.5


def test_k_zero_returns_zero_not_crash():
    assert recall_at_k([_s("1")], {"1"}, k=0) == 0.0


def test_empty_gt_returns_zero():
    assert recall_at_k([_s("1")], set(), k=10) == 0.0


def test_pool_size_caps_denominator():
    # 1 relevant of a pool of 2 eligible songs, huge GT, k=10.
    # Without pool_size: denom=10 -> 0.1. With pool_size=2: denom=2 -> 0.5.
    gt = {str(i) for i in range(1000)}
    assert recall_at_k([_s("1"), _s("999999")], gt, k=10) == 0.1
    assert recall_at_k([_s("1"), _s("999999")], gt, k=10, pool_size=2) == 0.5


def test_pool_size_does_not_inflate_above_gt_or_k():
    # pool_size only ever tightens the denominator, never loosens it.
    assert recall_at_k([_s("1"), _s("2")], {"1", "2"}, k=10, pool_size=100) == 1.0
