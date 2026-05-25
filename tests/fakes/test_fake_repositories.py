from app.models.song_document import SongDocument
from app.models.user_profile import UserProfileData
from tests.fakes.fake_song_repository import FakeSongRepository
from tests.fakes.fake_user_profile_repository import FakeUserProfileRepository


def _song(song_id: int) -> SongDocument:
    return SongDocument(
        song_id=song_id, title=f"t{song_id}", tag="pop", artist="a",
        year=2024, views=0, lyrics="hello world",
    )


def test_fake_song_repo_get_by_id_and_ids():
    repo = FakeSongRepository(songs=[_song(1), _song(2), _song(3)])
    assert repo.get_by_id(2).song_id == 2
    assert repo.get_by_id(999) is None
    ids = sorted(s.song_id for s in repo.get_by_ids([1, 3, 999]))
    assert ids == [1, 3]


def test_fake_user_profile_repo():
    p = UserProfileData(user_id="u1", favourite_genres=[], favourite_artists=[],
                        favourite_songs=[], preferred_language="es")
    repo = FakeUserProfileRepository(profiles={"u1": p})
    assert repo.get_by_user_id("u1") is p
    assert repo.get_by_user_id("nope") is None
