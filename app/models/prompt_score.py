import re
from pydantic import BaseModel, Field, field_validator

# Pinecone stores `language` as a lowercase ISO 639-1 2-letter code ("es", "en", ...).
# This regex defines the format PromptScore.preferred_language is normalised to.
ISO_639_1_PATTERN = re.compile(r"^[a-z]{2}$")


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

    @field_validator("preferred_language", mode="before")
    @classmethod
    def _normalize_language(cls, v):
        """Coerce to lowercase ISO 639-1 2-letter code or None. Anything else (full
        names like 'Spanish', empty strings, malformed input) becomes None so the
        retrieval-time language filter is simply skipped instead of excluding
        every song due to a format mismatch with what Pinecone stores."""
        if v is None:
            return None
        if not isinstance(v, str):
            return None
        v = v.strip().lower()
        if ISO_639_1_PATTERN.match(v):
            return v
        return None
