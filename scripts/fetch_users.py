#!/usr/bin/env python3
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.persistence.mongo.database import MongoDatabase
from app.persistence.mongo.user_profile_repository import UserProfileRepository


async def main():
    mongo = MongoDatabase()
    await mongo.connect()
    repo = UserProfileRepository(mongo.get_db())
    for i in range(1, 11):
        user_id = f"user_{i:03d}"
        user = await repo.get_by_user_id(user_id)
        if user:
            print(user)
        else:
            print(f"{user_id}: not found")
    await mongo.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
