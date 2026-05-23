from pydantic import BaseModel


class SongDocument(BaseModel):
    # Genius fields — always present
    song_id: int
    title: str
    tag: str
    artist: str
    year: int
    views: int
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
