import json
from app.models.song_document import SongDocument
from app.services.loader_service import LoaderService


class _FakeRepo:
    def __init__(self, songs):
        self._songs = songs

    def get_all(self):
        yield from self._songs

    def get_by_ids_stream(self, song_ids):
        yield from (s for s in self._songs if s.song_id in song_ids)


def _song(**overrides):
    base = dict(
        song_id=1, title="T", tag="pop", artist="A", year=2024, views=10,
        lyrics="la la la",
    )
    base.update(overrides)
    return SongDocument(**base)


def _meta_for(song):
    repo = _FakeRepo([song])
    return next(iter(LoaderService(repo).load())).metadata


def test_missing_tag_yields_empty_genres():
    meta = _meta_for(_song(tag=""))
    assert meta["genres"] == []


def test_tempo_lives_in_audio_blob_not_characteristics():
    meta = _meta_for(_song(tempo=120.0))
    audio = json.loads(meta["audio_features_chunk"])
    chars = json.loads(meta["song_characteristics_chunk"])
    assert audio["tempo"] == 120.0
    assert "tempo" not in chars


def test_no_spotify_match_yields_empty_audio_blob():
    meta = _meta_for(_song())  # all spotify fields default None
    assert json.loads(meta["audio_features_chunk"]) == {}


def test_missing_year_yields_empty_release_date():
    meta = _meta_for(_song(year=0))
    assert meta["release_date"] == ""
