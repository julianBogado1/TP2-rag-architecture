from unittest.mock import MagicMock, patch
from app.services.ingest_service import IngestService


def _row(i: int) -> dict:
    return {
        "id": i,
        "title": f"Song {i}",
        "tag": "rap",
        "artist": "Artist",
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
    db["dataset"].insert_many.return_value.inserted_ids = list(range(inserted_per_call))
    return db


def test_run_inserts_up_to_max_songs():
    db = _mock_db(3)
    rows = [_row(i) for i in range(10)]

    with patch("app.services.ingest_service.load_dataset", return_value=iter(rows)):
        result = IngestService(db).run(max_songs=3, batch_size=10)

    assert result.total_inserted == 3


def test_run_splits_into_batches():
    db = _mock_db(2)
    rows = [_row(i) for i in range(4)]

    with patch("app.services.ingest_service.load_dataset", return_value=iter(rows)):
        result = IngestService(db).run(max_songs=4, batch_size=2)

    assert result.batches == 2
    assert result.total_inserted == 4


def test_run_flushes_remainder_batch():
    db = _mock_db(3)
    rows = [_row(i) for i in range(3)]

    with patch("app.services.ingest_service.load_dataset", return_value=iter(rows)):
        result = IngestService(db).run(max_songs=3, batch_size=10)

    assert result.batches == 1
    assert result.total_inserted == 3


def test_run_drops_collection_before_inserting():
    db = _mock_db(0)

    with patch("app.services.ingest_service.load_dataset", return_value=iter([])):
        IngestService(db).run(max_songs=5)

    db["dataset"].drop.assert_called_once()


def test_run_empty_stream_returns_zero():
    db = MagicMock()

    with patch("app.services.ingest_service.load_dataset", return_value=iter([])):
        result = IngestService(db).run(max_songs=100)

    assert result.total_inserted == 0
    assert result.batches == 0
