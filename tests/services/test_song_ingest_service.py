from unittest.mock import MagicMock, patch
from app.services.song_ingest_service import SongIngestService


def _spotify_row(track_name: str, artists: str) -> dict:
    return {
        "track_id": "sp1",
        "artists": artists,
        "album_name": "Album",
        "track_name": track_name,
        "popularity": 80,
        "duration_ms": 200000,
        "explicit": False,
        "danceability": 0.7,
        "energy": 0.8,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.1,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.6,
        "tempo": 120.0,
        "time_signature": 4,
        "track_genre": "pop",
    }


def _genius_row(i: int, title: str | None = None, artist: str | None = None) -> dict:
    return {
        "id": i,
        "title": title or f"Song {i}",
        "tag": "rap",
        "artist": artist or "Artist",
        "year": 2020,
        "views": 100,
        "features": None,
        "lyrics": "some lyrics",
        "language_cld3": "en",
        "language_ft": "en",
        "language": "en",
    }


def _mock_db(inserted_per_call: int) -> MagicMock:
    db = MagicMock()
    db["songs"].insert_many.return_value.inserted_ids = list(range(inserted_per_call))
    return db


def test_run_counts_spotify_match():
    db = _mock_db(1)
    spotify = [_spotify_row("my song", "my artist")]
    genius = [_genius_row(1, title="My Song", artist="My Artist")]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        result = SongIngestService(db).run(max_songs=1, batch_size=10)

    assert result.spotify_matches == 1
    assert result.total_inserted == 1


def test_run_no_match_gives_zero_spotify_matches():
    db = _mock_db(1)
    spotify = [_spotify_row("other song", "other artist")]
    genius = [_genius_row(1, title="My Song", artist="My Artist")]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        result = SongIngestService(db).run(max_songs=1, batch_size=10)

    assert result.spotify_matches == 0
    assert result.total_inserted == 1


def test_run_miss_inserts_null_spotify_fields():
    db = _mock_db(1)
    spotify = []
    genius = [_genius_row(1)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        SongIngestService(db).run(max_songs=1, batch_size=10)

    doc = db["songs"].insert_many.call_args[0][0][0]
    assert doc["track_id"] is None
    assert doc["popularity"] is None
    assert doc["danceability"] is None
    assert doc["track_genre"] is None


def test_run_hit_populates_spotify_fields():
    db = _mock_db(1)
    spotify = [_spotify_row("my song", "my artist")]
    genius = [_genius_row(1, title="My Song", artist="My Artist")]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        SongIngestService(db).run(max_songs=1, batch_size=10)

    doc = db["songs"].insert_many.call_args[0][0][0]
    assert doc["track_id"] == "sp1"
    assert doc["popularity"] == 80
    assert doc["danceability"] == 0.7
    assert doc["track_genre"] == "pop"


def test_run_limits_to_max_songs():
    db = _mock_db(3)
    spotify = []
    genius = [_genius_row(i) for i in range(10)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        result = SongIngestService(db).run(max_songs=3, batch_size=10)

    assert result.total_inserted == 3


def test_run_splits_into_batches():
    db = _mock_db(2)
    spotify = []
    genius = [_genius_row(i) for i in range(4)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        result = SongIngestService(db).run(max_songs=4, batch_size=2)

    assert result.batches == 2
    assert result.total_inserted == 4


def test_run_flushes_remainder_batch():
    db = _mock_db(3)
    spotify = []
    genius = [_genius_row(i) for i in range(3)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter(spotify), iter(genius)]):
        result = SongIngestService(db).run(max_songs=3, batch_size=10)

    assert result.batches == 1
    assert result.total_inserted == 3


def test_run_drops_collection_before_inserting():
    db = _mock_db(0)

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter([]), iter([])]):
        SongIngestService(db).run(max_songs=5)

    db["songs"].drop.assert_called_once()


def test_run_empty_genius_stream_returns_zero():
    db = MagicMock()

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter([]), iter([])]):
        result = SongIngestService(db).run(max_songs=100)

    assert result.total_inserted == 0
    assert result.batches == 0
    assert result.spotify_matches == 0


def test_run_continues_on_batch_insert_error():
    db = MagicMock()
    db["songs"].insert_many.side_effect = Exception("DB error")
    genius = [_genius_row(i) for i in range(4)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter([]), iter(genius)]):
        result = SongIngestService(db).run(max_songs=4, batch_size=2)

    assert result.batches == 2
    assert result.total_inserted == 0


def test_run_remainder_batch_handles_insert_error():
    db = MagicMock()
    db["songs"].insert_many.side_effect = Exception("DB error")
    genius = [_genius_row(i) for i in range(3)]

    with patch("app.services.song_ingest_service.load_dataset", side_effect=[iter([]), iter(genius)]):
        result = SongIngestService(db).run(max_songs=3, batch_size=10)

    assert result.batches == 1
    assert result.total_inserted == 0
