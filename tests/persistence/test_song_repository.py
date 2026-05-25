from unittest.mock import MagicMock
from app.models.song_document import SongDocument
from app.persistence.mongo.song_repository import SongRepository


def _song(**kwargs) -> SongDocument:
    defaults = dict(
        song_id=1, title="T", tag="rap", artist="A",
        year=2020, views=0, lyrics="lyrics",
    )
    return SongDocument(**(defaults | kwargs))


def _repo() -> tuple[SongRepository, MagicMock]:
    collection = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    return SongRepository(db), collection


def test_insert_many_returns_inserted_count():
    repo, col = _repo()
    col.insert_many.return_value.inserted_ids = [1, 2, 3]
    songs = [_song(song_id=i) for i in range(3)]

    result = repo.insert_many(songs)

    assert result == 3
    col.insert_many.assert_called_once()


def test_insert_many_empty_list_returns_zero():
    repo, col = _repo()

    result = repo.insert_many([])

    assert result == 0
    col.insert_many.assert_not_called()


def test_insert_many_serialises_to_dicts():
    repo, col = _repo()
    col.insert_many.return_value.inserted_ids = [1]
    song = _song(song_id=99, title="My Song")

    repo.insert_many([song])

    call_args = col.insert_many.call_args[0][0]
    assert call_args[0]["song_id"] == 99
    assert call_args[0]["title"] == "My Song"
    assert call_args[0]["track_id"] is None


def test_drop_collection_calls_drop():
    repo, col = _repo()

    repo.drop_collection()

    col.drop.assert_called_once()


def test_count_returns_document_count():
    repo, col = _repo()
    col.count_documents.return_value = 7

    result = repo.count()

    assert result == 7
    col.count_documents.assert_called_once_with({})


def test_get_all_returns_all_songs():
    repo, col = _repo()
    col.find.return_value = [
        _song(song_id=1, title="Song A").model_dump(),
        _song(song_id=2, title="Song B").model_dump(),
    ]

    results = repo.get_all()

    assert len(results) == 2
    col.find.assert_called_once_with({}, {"_id": 0})


def test_get_by_id_returns_song():
    repo, col = _repo()
    col.find_one.return_value = _song(song_id=38, title="Stronger").model_dump()

    result = repo.get_by_id(38)

    assert result is not None
    assert result.song_id == 38
    assert result.title == "Stronger"
    col.find_one.assert_called_once_with({"song_id": 38}, {"_id": 0})


def test_get_by_id_returns_none_when_not_found():
    repo, col = _repo()
    col.find_one.return_value = None

    result = repo.get_by_id(999)

    assert result is None


def test_get_by_artist_returns_list():
    repo, col = _repo()
    col.find.return_value = [
        _song(artist="Kanye West", title="Stronger").model_dump(),
        _song(artist="Kanye West", title="Gold Digger").model_dump(),
    ]

    results = repo.get_by_artist("Kanye West")

    assert len(results) == 2
    assert all(s.artist == "Kanye West" for s in results)
    col.find.assert_called_once_with({"artist": "Kanye West"}, {"_id": 0})


def test_get_by_genre_queries_tag_field():
    repo, col = _repo()
    col.find.return_value = [_song(tag="pop").model_dump()]

    results = repo.get_by_genre("pop")

    assert len(results) == 1
    col.find.assert_called_once_with({"tag": "pop"}, {"_id": 0})


def test_get_by_year_returns_list():
    repo, col = _repo()
    col.find.return_value = [_song(year=2007).model_dump()]

    results = repo.get_by_year(2007)

    assert len(results) == 1
    assert results[0].year == 2007
    col.find.assert_called_once_with({"year": 2007}, {"_id": 0})


def test_get_by_ids_returns_matching_songs():
    repo, col = _repo()
    col.find.return_value = [
        _song(song_id=1).model_dump(),
        _song(song_id=3).model_dump(),
    ]

    results = repo.get_by_ids([1, 3, 999])

    assert sorted(s.song_id for s in results) == [1, 3]
    col.find.assert_called_once_with({"song_id": {"$in": [1, 3, 999]}}, {"_id": 0})


def test_get_by_ids_empty_list_returns_empty():
    repo, col = _repo()

    results = repo.get_by_ids([])

    assert results == []
    col.find.assert_not_called()
