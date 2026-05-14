from pydantic import BaseModel


class DatasetSongDTO(BaseModel):
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
