from types import SimpleNamespace
from app.metrics.context_precision import context_precision_at_k


def _s(sid):
    return SimpleNamespace(song_id=sid)


def test_relevant_at_ranks_1_and_3():
    # (1/1 + 2/3) / 2 = 0.83333
    val = context_precision_at_k([_s("1"), _s("x"), _s("3")], {"1", "3"}, k=3)
    assert abs(val - 0.833333) < 1e-4


def test_docs_example_ranks_2_and_3():
    # (1/2 + 2/3) / 2 = 0.58333  (matches docs/metrics.md)
    val = context_precision_at_k([_s("x"), _s("2"), _s("3")], {"2", "3"}, k=3)
    assert abs(val - 0.583333) < 1e-4


def test_no_relevant_in_top_k_returns_zero():
    assert context_precision_at_k([_s("x"), _s("y")], {"1"}, k=3) == 0.0
