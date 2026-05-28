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
