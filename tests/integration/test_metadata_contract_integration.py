"""Proves the indexing writer (W1 loader) and the retrieval reader (W2) agree on
the Pinecone metadata contract: tempo lives in the audio blob, and a lyrics-only
song (no Spotify match) round-trips as a kept candidate with audio_features=None."""
from unittest.mock import MagicMock

from app.lambdas.indexing_pipeline import IndexingPipeline
from app.services.loader_service import LoaderService
from app.services.splitter_service import SplitterService
from app.services.embedder_service import EmbedderService
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from app.models.song_document import SongDocument
from app.models.song_recommendation_request import (
    SongRecommendationRequest, MetadataFilters, RankingWeights,
)
from app.models.prompt_score import PromptScore, PromptAudioFeatures
from app.models.user_profile import UserProfileData
from tests.fakes.fake_song_repository import FakeSongRepository
from tests.fakes.fake_pinecone_repository import FakePineconeRepository


def _request(top_k=10):
    audio = PromptAudioFeatures(valence=0.5, energy=0.5, danceability=0.5,
                                acousticness=0.5, instrumentalness=0.5, tempo_norm=0.5)
    score = PromptScore(
        happy=0.5, sad=0.0, energetic=0.5, calm=0.0, nostalgic=0.0,
        romantic=0.0, assertive=0.0, deep=0.0, playful=0.0,
        wants_recent_songs=0.0, wants_popular_songs=0.0, wants_obscure_songs=0.0,
        wants_lyrics_focus=0.0, wants_mood_focus=0.0,
        semantic_query="q", audio_features=audio,
    )
    return SongRecommendationRequest(
        user_id="u", raw_prompt="r", prompt_score=score,
        user_profile=UserProfileData(user_id="u", favourite_genres=[], favourite_artists=[],
                                     favourite_songs=[], preferred_language="es"),
        semantic_query="q", target_audio_features=audio,
        metadata_filters=MetadataFilters(), ranking_weights=RankingWeights(),
        top_k_retrieval=top_k, top_n_output=10,
    )


def test_audio_and_lyrics_only_songs_round_trip():
    audio_song = SongDocument(
        song_id=1, title="Audio", tag="pop", artist="A", year=2024, views=0,
        lyrics="line one\nline two\n", language="es",
        popularity=70, danceability=0.7, energy=0.7, valence=0.8,
        acousticness=0.1, instrumentalness=0.0, tempo=120.0,
    )
    lyrics_only_song = SongDocument(
        song_id=2, title="LyricsOnly", tag="rap", artist="B", year=2024, views=0,
        lyrics="just words\nno spotify\n",  # all Spotify fields None
    )
    song_repo = FakeSongRepository(songs=[audio_song, lyrics_only_song])
    vec_repo = FakePineconeRepository()

    embedder = EmbedderService.__new__(EmbedderService)
    embedder._vector_repo = vec_repo
    embedder._batch_size = 10
    fake_embed = MagicMock()
    fake_embed.embed_documents.return_value = [[0.1] * 384] * 100
    embedder._embeddings = fake_embed

    IndexingPipeline(
        loader=LoaderService(song_repo),
        splitter=SplitterService(chunk_size=400, chunk_overlap=50),
        embedder=embedder,
    ).run()

    candidates = VectorRetrievalService(vec_repo).retrieve(_request(), [0.1] * 384)
    by_id = {c.song_id: c for c in candidates}

    # both songs are retrievable (lyrics-only is not silently dropped)
    assert {"1", "2"} <= set(by_id)
    # audio song: full features, tempo read from the audio blob (not the dead 0.5 default)
    assert by_id["1"].metadata.audio_features is not None
    assert by_id["1"].metadata.audio_features.tempo_norm == 120 / 250.0
    # lyrics-only song: kept, scored lyrics-only
    assert by_id["2"].metadata.audio_features is None
