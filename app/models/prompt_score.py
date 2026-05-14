from pydantic import BaseModel, Field


class PromptAudioFeatures(BaseModel):
    valence: float = Field(ge=0.0, le=1.0)
    energy: float = Field(ge=0.0, le=1.0)
    danceability: float = Field(ge=0.0, le=1.0)
    acousticness: float = Field(ge=0.0, le=1.0)
    instrumentalness: float = Field(ge=0.0, le=1.0)
    tempo_norm: float = Field(ge=0.0, le=1.0)


class PromptScore(BaseModel):
    happy: float = Field(ge=0.0, le=1.0)
    sad: float = Field(ge=0.0, le=1.0)
    energetic: float = Field(ge=0.0, le=1.0)
    calm: float = Field(ge=0.0, le=1.0)
    nostalgic: float = Field(ge=0.0, le=1.0)
    romantic: float = Field(ge=0.0, le=1.0)
    assertive: float = Field(ge=0.0, le=1.0)
    deep: float = Field(ge=0.0, le=1.0)
    playful: float = Field(ge=0.0, le=1.0)
    wants_recent_songs: float = Field(ge=0.0, le=1.0)
    wants_popular_songs: float = Field(ge=0.0, le=1.0)
    wants_obscure_songs: float = Field(ge=0.0, le=1.0)
    wants_lyrics_focus: float = Field(ge=0.0, le=1.0)
    wants_mood_focus: float = Field(ge=0.0, le=1.0)
    semantic_query: str
    extracted_genres: list[str] | None = None
    extracted_artists: list[str] | None = None
    preferred_language: str | None = None
    audio_features: PromptAudioFeatures
