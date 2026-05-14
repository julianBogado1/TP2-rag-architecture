from unittest.mock import MagicMock
from app.models.dataset_song_dto import DatasetSongDTO
from app.persistence.mongo import dataset_repository


def _song(**kwargs) -> DatasetSongDTO:
    defaults = dict(
        song_id=1, title="T", tag="rap", artist="A",
        year=2020, views=0, lyrics="lyrics",
    )
    return DatasetSongDTO(**(defaults | kwargs))


def test_insert_many_songs_returns_inserted_count():
    db = MagicMock()
    db["dataset"].insert_many.return_value.inserted_ids = [1, 2, 3]
    songs = [_song(song_id=i) for i in range(3)]

    result = dataset_repository.insert_many_songs(db, songs)

    assert result == 3
    db["dataset"].insert_many.assert_called_once()


def test_insert_many_songs_empty_list_returns_zero():
    db = MagicMock()

    result = dataset_repository.insert_many_songs(db, [])

    assert result == 0
    db["dataset"].insert_many.assert_not_called()


def test_insert_many_songs_serialises_to_dicts():
    db = MagicMock()
    db["dataset"].insert_many.return_value.inserted_ids = [1]
    song = _song(song_id=99, title="My Song")

    dataset_repository.insert_many_songs(db, [song])

    call_args = db["dataset"].insert_many.call_args[0][0]
    assert call_args[0]["song_id"] == 99
    assert call_args[0]["title"] == "My Song"


def test_drop_collection_calls_drop():
    db = MagicMock()

    dataset_repository.drop_collection(db)

    db["dataset"].drop.assert_called_once()


def test_count_returns_document_count():
    db = MagicMock()
    db["dataset"].count_documents.return_value = 42

    result = dataset_repository.count(db)

    assert result == 42
    db["dataset"].count_documents.assert_called_once_with({})
