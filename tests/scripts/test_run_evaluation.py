import importlib.util
from pathlib import Path
from types import SimpleNamespace

# run_evaluation.py is a script (not a package module); load it directly.
_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "run_evaluation.py"
_spec = importlib.util.spec_from_file_location("run_evaluation", _PATH)
run_evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_evaluation)


def _song(song_id: str, has_audio: bool):
    af = object() if has_audio else None
    return SimpleNamespace(song_id=song_id, metadata=SimpleNamespace(audio_features=af))


def test_audio_bearing_drops_no_audio_songs():
    songs = [_song("1", True), _song("2", False), _song("3", True)]
    kept = run_evaluation.audio_bearing(songs)
    assert [s.song_id for s in kept] == ["1", "3"]


def test_score_case_audio_ignores_no_audio_songs():
    # "2" is in GT but has no audio -> excluded from the audio-case pool entirely.
    # Pool = [s1]; recall denom = min(10, |GT|, pool=1) = 1 -> the one audio match scores 1.0.
    songs = [_song("1", True), _song("2", False)]
    cp, rec, ndcg = run_evaluation.score_case("audio", songs, {"1", "2"}, k=10)
    assert cp == 1.0           # the single audio-bearing song is relevant
    assert rec == 1.0          # 1 relevant of a pool of 1 audio-bearing song
    # ndcg's IDCG still targets min(k,|GT|)=2 ideal hits but only 1 audio song exists,
    # so it's < 1.0 — ndcg keeps a coverage signal that recall (pool-capped) drops.
    assert 0.0 < ndcg < 1.0


def test_score_case_genre_counts_all_returned_songs():
    songs = [_song("1", True), _song("2", False)]
    cp, rec, ndcg = run_evaluation.score_case("genre", songs, {"1", "2"}, k=10)
    assert cp == 1.0
    assert rec == 1.0          # both GT songs surfaced; no-audio song still counts
    assert ndcg == 1.0
