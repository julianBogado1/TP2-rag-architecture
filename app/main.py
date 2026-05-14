from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.persistence.mongo.database import connect, disconnect
from app.persistence.mongo.user_profile_repository import get_by_user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect()
    for i in range(1, 11):
        user_id = f"user_{i:03d}"
        user = await get_by_user_id(user_id)
        if user:
            print(user)
        else:
            print(f"{user_id}: not found")
    yield
    await disconnect()


app = FastAPI(lifespan=lifespan)


