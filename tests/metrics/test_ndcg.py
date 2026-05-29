from math import isclose
from types import SimpleNamespace
from app.metrics.ndcg import ndcg_at_k


def _s(sid):
    return SimpleNamespace(song_id=sid)


def test_perfect_ordering_is_one():
    # relevant songs front-loaded -> DCG == IDCG
    ranked = [_s("1"), _s("2"), _s("9")]
    assert ndcg_at_k(ranked, {"1", "2"}, k=3) == 1.0


def test_front_loaded_beats_back_loaded():
    gt = {"1"}
    front = ndcg_at_k([_s("1"), _s("8"), _s("9")], gt, k=3)
    back  = ndcg_at_k([_s("8"), _s("9"), _s("1")], gt, k=3)
    assert front > back
    assert front == 1.0  # single relevant at rank 1 -> DCG == IDCG


def test_all_miss_is_zero():
    assert ndcg_at_k([_s("8"), _s("9")], {"1", "2"}, k=2) == 0.0


def test_empty_gt_or_k_zero():
    assert ndcg_at_k([_s("1")], set(), k=10) == 0.0
    assert ndcg_at_k([_s("1")], {"1"}, k=0) == 0.0


def test_known_value():
    # ranked [hit, miss, hit], gt has >=3 songs, k=3
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5 = 2.1309
    val = ndcg_at_k([_s("1"), _s("8"), _s("3")], {"1", "3", "5"}, k=3)
    assert isclose(val, 1.5 / 2.130929753, rel_tol=1e-6)
