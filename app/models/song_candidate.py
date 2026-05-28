from datetime import date
from pydantic import BaseModel, Field
from app.models.song_recommendation_request import MetadataFilters


class AudioFeatures(BaseModel):
    valence: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    instrumentalness: float = Field(ge=0.0, le=1.0)
    tempo_norm: float = Field(ge=0.0, le=1.0)


class SongMetadata(BaseModel):
    track_name: str
    artist_name: str
    genres: list[str]
    popularity: int
    release_date: date
    audio_features: AudioFeatures | None = None


class SongVectorDocument(BaseModel):
    id: str
    song_id: str
    lyrics_chunk: str
    lyrics_embedding: list[float]
    metadata: SongMetadata


class CandidateChunk(BaseModel):
    chunk_id: str
    song_id: str
    # Empty at retrieval time; evidence text is filled later by the aggregator
    # from Mongo (kept here for shape parity with the vector document).
    lyrics_chunk: str
    lyrics_similarity: float
    metadata: SongMetadata


class CandidateSong(BaseModel):
    song_id: str
    track_name: str
    artist_name: str
    genres: list[str]
    popularity: int
    release_date: date
    best_lyrics_chunks: list[str]
    best_lyrics_similarity: float
    audio_features: AudioFeatures | None = None
