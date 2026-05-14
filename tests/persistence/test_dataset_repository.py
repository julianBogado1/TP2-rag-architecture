from unittest.mock import MagicMock
from app.models.dataset_song_dto import DatasetSongDTO
from app.persistence.mongo.dataset_repository import DatasetRepository


def _song(**kwargs) -> DatasetSongDTO:
    defaults = dict(
        song_id=1, title="T", tag="rap", artist="A",
        year=2020, views=0, lyrics="lyrics",
    )
    return DatasetSongDTO(**(defaults | kwargs))


def _repo(inserted_ids=None) -> tuple[DatasetRepository, MagicMock]:
    db = MagicMock()
    if inserted_ids is not None:
        db["dataset"].insert_many.return_value.inserted_ids = inserted_ids
    return DatasetRepository(db), db


def test_insert_many_songs_returns_inserted_count():
    repo, db = _repo(inserted_ids=[1, 2, 3])
    songs = [_song(song_id=i) for i in range(3)]

    result = repo.insert_many_songs(songs)

    assert result == 3
    db["dataset"].insert_many.assert_called_once()


def test_insert_many_songs_empty_list_returns_zero():
    repo, db = _repo()

    result = repo.insert_many_songs([])

    assert result == 0
    db["dataset"].insert_many.assert_not_called()


def test_insert_many_songs_serialises_to_dicts():
    repo, db = _repo(inserted_ids=[1])
    song = _song(song_id=99, title="My Song")

    repo.insert_many_songs([song])

    call_args = db["dataset"].insert_many.call_args[0][0]
    assert call_args[0]["song_id"] == 99
    assert call_args[0]["title"] == "My Song"


def test_drop_collection_calls_drop():
    repo, db = _repo()

    repo.drop_collection()

    db["dataset"].drop.assert_called_once()


def test_count_returns_document_count():
    repo, db = _repo()
    db["dataset"].count_documents.return_value = 42

    result = repo.count()

    assert result == 42
    db["dataset"].count_documents.assert_called_once_with({})
