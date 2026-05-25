from unittest.mock import MagicMock
from app.models.user_profile import UserProfileData
from app.persistence.mongo.user_profile_repository import UserProfileRepository


def _repo() -> tuple[UserProfileRepository, MagicMock]:
    collection = MagicMock()
    db = MagicMock()
    db.__getitem__.return_value = collection
    return UserProfileRepository(db), collection


def test_get_by_user_id_returns_user():
    repo, col = _repo()
    col.find_one.return_value = {
        "user_id": "user_001",
        "favourite_genres": ["pop"],
        "favourite_artists": ["X"],
        "favourite_songs": [],
        "preferred_language": "es",
    }

    user = repo.get_by_user_id("user_001")

    assert isinstance(user, UserProfileData)
    assert user.user_id == "user_001"
    col.find_one.assert_called_once_with({"user_id": "user_001"}, {"_id": 0})


def test_get_by_user_id_missing_returns_none():
    repo, col = _repo()
    col.find_one.return_value = None

    assert repo.get_by_user_id("ghost") is None
