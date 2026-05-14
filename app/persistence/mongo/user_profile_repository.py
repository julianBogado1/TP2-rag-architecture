from app.models.user_profile import UserProfileData
from app.persistence.mongo.database import get_db


async def get_by_user_id(user_id: str) -> UserProfileData | None:
    db = get_db()
    doc = await db["user_profiles"].find_one(
        {"user_id": user_id}, {"_id": 0}
    )
    if doc is None:
        return None
    return UserProfileData(**doc)
