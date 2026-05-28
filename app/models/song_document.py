import math

from pydantic import BaseModel, field_validator


class SongDocument(BaseModel):
    # Genius fields — always present
    song_id: int
    title: str
    tag: str
    artist: str
    year: int | None = None
    views: int | None = None
    features: str | None = None
    lyrics: str
    language_cld3: str | None = None
    language_ft: str | None = None
    language: str | None = None
    # Spotify fields — None when no match found
    track_id: str | None = None
    album_name: str | None = None
    popularity: int | None = None
    duration_ms: int | None = None
    explicit: bool | None = None
    danceability: float | None = None
    energy: float | None = None
    key: int | None = None
    loudness: float | None = None
    mode: int | None = None
    speechiness: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    liveness: float | None = None
    valence: float | None = None
    tempo: float | None = None
    time_signature: int | None = None
    track_genre: str | None = None

    @field_validator(
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo",
        mode="before",
    )
    @classmethod
    def _nan_to_none(cls, v):
        # Spotify source rows can carry NaN floats; persist them as None so they
        # don't leak NaN into Mongo or the JSON-serialized Pinecone metadata.
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
