from pymongo.database import Database
from pymongo.collection import Collection
from app.models.user_profile import UserProfileData


class UserProfileRepository:
    COLLECTION = "user_profiles"

    def __init__(self, db: Database) -> None:
        self._collection: Collection = db[self.COLLECTION]

    def get_by_user_id(self, user_id: str) -> UserProfileData | None:
        doc = self._collection.find_one({"user_id": user_id}, {"_id": 0})
        return UserProfileData(**doc) if doc else None
