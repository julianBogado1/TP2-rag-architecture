import json
import importlib.util
from pathlib import Path

from app.models.genre import Genre

# build_gt.py is a script (not a package module); load it directly.
_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "build_gt.py"
_spec = importlib.util.spec_from_file_location("build_gt", _PATH)
build_gt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_gt)


def test_genre_case_shape():
    case = build_gt.genre_case("pop")
    assert case["kind"] == "genre"
    assert case["tag"] == "pop"
    assert "pop" in case["prompt"].lower()
    assert case["user_id"] == build_gt.EVAL_USER["user_id"]


def test_audio_case_shape():
    case = build_gt.audio_case(build_gt.AUDIO_CASES[0])
    assert case["kind"] == "audio"
    assert case["target_feature"] == "valence"
    assert "threshold" in case and "operator" in case


def test_genre_cases_cover_all_canonical_genres():
    tags = {build_gt.genre_case(g.value)["tag"] for g in Genre}
    assert tags == {g.value for g in Genre}
    assert len(tags) == 6


def test_audio_cases_use_soft_thresholds():
    by_label = {c["label"]: c for c in build_gt.AUDIO_CASES}
    assert by_label["Happy"]["threshold"] == 0.6 and by_label["Happy"]["operator"] == "gt"
    assert by_label["Sad"]["threshold"] == 0.4 and by_label["Sad"]["operator"] == "lt"


def test_restrict_to_indexed_filters():
    assert build_gt.restrict_to_indexed(["1", "2", "3"], {"2", "3"}) == ["2", "3"]


def test_restrict_to_indexed_noop_when_none():
    ids = ["1", "2", "3"]
    assert build_gt.restrict_to_indexed(ids, None) == ids


def test_load_indexed_ids_missing_returns_none():
    assert build_gt.load_indexed_ids(Path("/no/such/file.json")) is None


def test_load_indexed_ids_coerces_to_str_set(tmp_path):
    p = tmp_path / "ids.json"
    p.write_text(json.dumps([1, 2, 3]))  # ids stored as ints
    assert build_gt.load_indexed_ids(p) == {"1", "2", "3"}
