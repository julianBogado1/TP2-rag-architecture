from datetime import date, datetime
import pytest
from app.models.prompt_score import PromptScore, PromptAudioFeatures
from app.models.user_profile import UserProfileData
from app.models.song_document import SongDocument
from app.models.song_candidate import CandidateSong, AudioFeatures
from app.models.request_context import RequestContext


@pytest.fixture
def sample_prompt_audio_features() -> PromptAudioFeatures:
    return PromptAudioFeatures(
        valence=0.85, energy=0.70, danceability=0.70,
        acousticness=0.15, instrumentalness=0.05, tempo_norm=0.55,
    )


@pytest.fixture
def sample_prompt_score(sample_prompt_audio_features) -> PromptScore:
    return PromptScore(
        happy=0.95, sad=0.0, energetic=0.65, calm=0.10, nostalgic=0.0,
        romantic=0.0, assertive=0.20, deep=0.0, playful=0.70,
        wants_recent_songs=0.30, wants_popular_songs=0.40, wants_obscure_songs=0.0,
        wants_lyrics_focus=0.30, wants_mood_focus=0.75,
        semantic_query="happy upbeat songs with positive mood",
        wanted_genres=None, wanted_artists=None,
        unwanted_genres=None, unwanted_artists=None, unwanted_songs=None,
        preferred_language="es",
        audio_features=sample_prompt_audio_features,
    )


@pytest.fixture
def sample_user_profile() -> UserProfileData:
    return UserProfileData(
        user_id="user_001",
        favourite_genres=["pop", "indie"],
        favourite_artists=["Taylor Swift"],
        favourite_songs=["Cruel Summer"],
        preferred_language="es",
        disliked_genres=["metal"],
        disliked_artists=[],
        listening_history=[],
    )


@pytest.fixture
def sample_request_context() -> RequestContext:
    return RequestContext(
        user_id="user_001",
        raw_prompt="quiero canciones felices :)",
        timestamp=datetime(2026, 5, 24, 14, 0, 0),
        session_id="test-session",
    )


@pytest.fixture
def sample_song_document() -> SongDocument:
    return SongDocument(
        song_id=42, title="Friday Lights", tag="pop", artist="Luna Reyes",
        year=2024, views=1_000_000, lyrics="Friday lights, everybody's smiling",
        language="es",
        track_id="tr_42", album_name="Album", popularity=78,
        duration_ms=210000, explicit=False, danceability=0.79, energy=0.82,
        key=5, loudness=-5.0, mode=1, speechiness=0.05, acousticness=0.08,
        instrumentalness=0.0, liveness=0.1, valence=0.88, tempo=122.0,
        time_signature=4, track_genre="pop",
    )


@pytest.fixture
def sample_candidate_song() -> CandidateSong:
    return CandidateSong(
        song_id="42", track_name="Friday Lights", artist_name="Luna Reyes",
        genres=["pop"], popularity=78, release_date=date(2024, 1, 1),
        best_lyrics_chunks=["Friday lights, the city's coming alive"],
        best_lyrics_similarity=0.81,
        audio_features=AudioFeatures(
            valence=0.88, energy=0.82, danceability=0.79,
            acousticness=0.08, instrumentalness=0.0, tempo_norm=122/250.0,
        ),
    )
