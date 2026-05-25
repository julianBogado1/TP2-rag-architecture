import json
from unittest.mock import MagicMock
from app.core.config import Settings
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.song_document import SongDocument
from app.services.retrieval.prompt_parser_service import PromptParserService
from app.services.retrieval.recommendation_request_builder import RecommendationRequestBuilder
from app.services.retrieval.query_embedder_service import QueryEmbedderService
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from app.services.retrieval.candidate_aggregator_service import CandidateAggregatorService
from app.services.retrieval.hybrid_reranker_service import HybridRerankerService
from app.services.retrieval.top_n_selector_service import TopNSelectorService
from app.services.retrieval.response_generator_service import ResponseGeneratorService
from app.services.retrieval.recommendation_orchestrator import RecommendationOrchestrator
from tests.fakes.fake_llm_client import FakeLLMClient
from tests.fakes.fake_pinecone_repository import FakePineconeRepository
from tests.fakes.fake_song_repository import FakeSongRepository
from tests.fakes.fake_user_profile_repository import FakeUserProfileRepository


def _vec_for_song(i: int) -> list[float]:
    v = [0.0] * 384
    v[i % 384] = 1.0
    return v


def _meta_for(song_id: int, popularity=70):
    return {
        "song_id": song_id, "track_name": f"track_{song_id}", "artist_name": "Luna Reyes",
        "genres": ["pop"], "release_date": "2024",
        "song_characteristics_chunk": json.dumps({"popularity": popularity, "tempo": 120, "language": "es"}),
        "audio_features_chunk": json.dumps({"valence": 0.85, "energy": 0.7, "danceability": 0.7,
                                             "acousticness": 0.1, "instrumentalness": 0.0}),
    }


def test_e2e_with_fakes(sample_prompt_score, sample_user_profile):
    vec_repo = FakePineconeRepository()
    vec_repo.upsert_batch([
        (f"{i}_0", _vec_for_song(i), _meta_for(i, popularity=50 + i))
        for i in (1, 2, 3)
    ])

    song_repo = FakeSongRepository(songs=[
        SongDocument(song_id=i, title=f"track_{i}", tag="pop", artist="Luna Reyes",
                     year=2024, views=1000, lyrics=f"some lyrics for {i}", language="es")
        for i in (1, 2, 3)
    ])

    user_repo = FakeUserProfileRepository(profiles={"user_001": sample_user_profile})

    expected_resp = RecommendationResponse(
        message="¡Tu playlist feliz!",
        recommendations=[
            SongRecommendation(rank=1, track_name="track_1", artist_name="Luna Reyes",
                               explanation="x", matched_mood=["happy"],
                               matched_audio_features=["valence"]),
        ],
    )
    llm = FakeLLMClient(returns={
        "PromptScore": sample_prompt_score,
        "RecommendationResponse": expected_resp,
    })

    embedder = MagicMock()
    embedder.embed_query.return_value = _vec_for_song(1)

    s = Settings(mongo_uri="x", mongo_db_name="x", pinecone_api_key="x", openai_api_key="x", _env_file=None)
    orch = RecommendationOrchestrator(
        prompt_parser      = PromptParserService(llm, "m"),
        user_repo          = user_repo,
        request_builder    = RecommendationRequestBuilder(s),
        query_embedder     = QueryEmbedderService(embedder),
        vector_retrieval   = VectorRetrievalService(vec_repo),
        aggregator         = CandidateAggregatorService(song_repo, s.aggregator_max_evidence_chunks),
        reranker           = HybridRerankerService(),
        selector           = TopNSelectorService(s.selector_max_per_artist),
        response_generator = ResponseGeneratorService(llm, "m"),
    )

    resp = orch.recommend("user_001", "happy please")

    assert isinstance(resp, RecommendationResponse)
    assert resp.message
    assert len(resp.recommendations) >= 1
