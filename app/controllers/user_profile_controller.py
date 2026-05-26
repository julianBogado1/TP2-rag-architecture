from fastapi import APIRouter, HTTPException
from app.models.user_profile import UserProfileData
from app.persistence.mongo.user_profile_repository import UserProfileRepository


def build_user_profile_router(user_repo: UserProfileRepository) -> APIRouter:
    router = APIRouter()

    @router.get("/users/{user_id}", response_model=UserProfileData)
    def get_profile(user_id: str) -> UserProfileData:
        profile = user_repo.get_by_user_id(user_id)
        if profile is None:
            raise HTTPException(404, "User not found")
        return profile

    @router.put("/users/{user_id}", response_model=UserProfileData)
    def upsert_profile(user_id: str, body: UserProfileData) -> UserProfileData:
        if body.user_id != user_id:
            raise HTTPException(422, "user_id en URL y body deben coincidir")
        user_repo.upsert(body)
        return body

    return router