from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.persistence.mongo.database import MongoDatabase
from app.persistence.mongo.user_profile_repository import UserProfileRepository
from app.controllers.indexing_controller import router as indexing_router


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield
    await mongo.disconnect()


app = FastAPI(lifespan=lifespan)
app.include_router(indexing_router)
