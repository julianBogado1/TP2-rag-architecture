from motor.motor_asyncio import AsyncIOMotorDatabase
from app.models.user_profile import UserProfileData


class UserProfileRepository:
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    async def get_by_user_id(self, user_id: str) -> UserProfileData | None:
        doc = await self._db["user_profiles"].find_one(
            {"user_id": user_id}, {"_id": 0}
        )
        if doc is None:
            return None
        return UserProfileData(**doc)
