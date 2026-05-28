import importlib.util
from pathlib import Path

# sample_songs.py is a script (not a package module); load it directly.
_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "sample_songs.py"
_spec = importlib.util.spec_from_file_location("sample_songs", _PATH)
sample_songs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample_songs)


def test_balance_round_robins_across_genres():
    ids_by_genre = {
        "pop": list(range(0, 10)),
        "rock": list(range(10, 20)),
        "rap": list(range(20, 30)),
    }
    out = sample_songs.balance(ids_by_genre, n=5)

    assert len(out) == 5
    assert len(set(out)) == 5  # no duplicates
    # each genre contributes 1-2 (round-robin), none dominates
    per_genre = {g: sum(1 for x in out if x in ids) for g, ids in ids_by_genre.items()}
    assert all(1 <= c <= 2 for c in per_genre.values())


def test_balance_handles_small_genre():
    ids_by_genre = {"pop": [1, 2, 3], "rock": [10]}  # rock smaller than its share
    out = sample_songs.balance(ids_by_genre, n=4)
    assert len(out) == 4
    assert 10 in out  # the single rock id is included
