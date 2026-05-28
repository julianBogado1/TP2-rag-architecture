import re
from pydantic import BaseModel, Field, field_validator
from app.models.genre import Genre

# Pinecone stores `language` as a lowercase ISO 639-1 2-letter code ("es", "en", ...).
# This regex defines the format PromptScore.preferred_language is normalised to.
ISO_639_1_PATTERN = re.compile(r"^[a-z]{2}$")
_VALID_GENRES = {g.value for g in Genre}


class PromptAudioFeatures(BaseModel):
    # Each axis is None when the prompt doesn't imply it; ranker scores only present axes.
    valence: float | None = Field(default=None, ge=0.0, le=1.0)
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    danceability: float | None = Field(default=None, ge=0.0, le=1.0)
    acousticness: float | None = Field(default=None, ge=0.0, le=1.0)
    instrumentalness: float | None = Field(default=None, ge=0.0, le=1.0)
    tempo_norm: float | None = Field(default=None, ge=0.0, le=1.0)


class PromptScore(BaseModel):
    # Moods + wants_* are None when the prompt is silent; 0.0 means explicitly negated.
    happy: float | None = Field(default=None, ge=0.0, le=1.0)
    sad: float | None = Field(default=None, ge=0.0, le=1.0)
    energetic: float | None = Field(default=None, ge=0.0, le=1.0)
    calm: float | None = Field(default=None, ge=0.0, le=1.0)
    nostalgic: float | None = Field(default=None, ge=0.0, le=1.0)
    romantic: float | None = Field(default=None, ge=0.0, le=1.0)
    assertive: float | None = Field(default=None, ge=0.0, le=1.0)
    deep: float | None = Field(default=None, ge=0.0, le=1.0)
    playful: float | None = Field(default=None, ge=0.0, le=1.0)
    wants_recent_songs: float | None = Field(default=None, ge=0.0, le=1.0)
    wants_popular_songs: float | None = Field(default=None, ge=0.0, le=1.0)
    wants_obscure_songs: float | None = Field(default=None, ge=0.0, le=1.0)
    wants_lyrics_focus: float | None = Field(default=None, ge=0.0, le=1.0)
    wants_mood_focus: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_query: str
    wanted_genres: list[Genre] | None = None
    wanted_artists: list[str] | None = None
    unwanted_genres: list[Genre] | None = None
    unwanted_artists: list[str] | None = None
    unwanted_songs: list[str] | None = None
    preferred_language: str | None = None
    audio_features: PromptAudioFeatures | None = None

    @field_validator("wanted_genres", "unwanted_genres", mode="before")
    @classmethod
    def _filter_known_genres(cls, v):
        """Silently drop genres outside the canonical vocabulary; empty list -> None."""
        if v is None:
            return None
        if not isinstance(v, list):
            return None
        keep = [s.strip().lower() for s in v if isinstance(s, str)]
        keep = [s for s in keep if s in _VALID_GENRES]
        return keep or None

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
