from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings


class MongoDatabase:
    def __init__(self) -> None:
        self._client: AsyncIOMotorClient | None = None

    async def connect(self) -> None:
        self._client = AsyncIOMotorClient(settings.mongo_uri)

    async def disconnect(self) -> None:
        if self._client:
            self._client.close()

    def get_db(self) -> AsyncIOMotorDatabase:
        if self._client is None:
            raise RuntimeError("MongoDB client is not connected. Call connect() first.")
        return self._client[settings.mongo_db_name]
