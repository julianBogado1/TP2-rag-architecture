from app.models.user_profile import UserProfileData


class FakeUserProfileRepository:
    def __init__(self, profiles: dict[str, UserProfileData] | None = None) -> None:
        self._profiles = profiles or {}

    def get_by_user_id(self, user_id: str) -> UserProfileData | None:
        return self._profiles.get(user_id)
