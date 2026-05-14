#!/usr/bin/env python3
import asyncio
from app.persistence.mongo.database import connect, disconnect
from app.persistence.mongo.user_profile_repository import get_by_user_id


async def main():
    await connect()
    for i in range(1, 11):
        user_id = f"user_{i:03d}"
        user = await get_by_user_id(user_id)
        if user:
            print(user)
        else:
            print(f"{user_id}: not found")
    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
