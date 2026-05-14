from datetime import datetime
from pydantic import BaseModel, Field


class ListeningEvent(BaseModel):
    song: str
    artist: str
    listened_at: datetime


class UserProfileData(BaseModel):
    user_id: str
    favourite_genres: list[str]
    favourite_artists: list[str]
    favourite_songs: list[str]
    preferred_language: str
    disliked_genres: list[str] = Field(default_factory=list)
    disliked_artists: list[str] = Field(default_factory=list)
    listening_history: list[ListeningEvent] = Field(default_factory=list)
