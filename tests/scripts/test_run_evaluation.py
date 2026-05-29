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
    # "2" is in GT but has no audio -> excluded from the audio-case denominator/numerator.
    songs = [_song("1", True), _song("2", False)]
    cp, rec = run_evaluation.score_case("audio", songs, {"1", "2"}, k=10)
    assert cp == 1.0           # the single audio-bearing song is relevant
    assert rec == 0.5          # 1 of min(10, 2) GT songs surfaced among audio-bearing


def test_score_case_genre_counts_all_returned_songs():
    songs = [_song("1", True), _song("2", False)]
    cp, rec = run_evaluation.score_case("genre", songs, {"1", "2"}, k=10)
    assert cp == 1.0
    assert rec == 1.0          # both GT songs surfaced; no-audio song still counts
