from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

_client: AsyncIOMotorClient | None = None


async def connect():
    global _client
    _client = AsyncIOMotorClient(settings.mongo_uri)


async def disconnect():
    if _client:
        _client.close()


def get_db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise RuntimeError("MongoDB client is not connected. Call connect() first.")
    return _client[settings.mongo_db_name]

