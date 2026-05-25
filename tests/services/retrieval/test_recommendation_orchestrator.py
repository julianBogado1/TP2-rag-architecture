import pytest
from unittest.mock import MagicMock
from app.core.exceptions import UserNotFoundError
from app.core.config import Settings
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


def _settings():
    return Settings(mongo_uri="x", mongo_db_name="x",
                    pinecone_api_key="x", openai_api_key="x", _env_file=None)


def _build_orch(llm, vec_repo, song_repo, user_repo, embedder):
    s = _settings()
    return RecommendationOrchestrator(
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


def test_user_not_found_raises(sample_prompt_score):
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    llm = FakeLLMClient(returns={"PromptScore": sample_prompt_score})
    orch = _build_orch(llm, FakePineconeRepository(), FakeSongRepository(),
                       FakeUserProfileRepository(profiles={}), embedder)
    with pytest.raises(UserNotFoundError):
        orch.recommend("ghost", "happy")


def test_empty_chunks_returns_empty_message(sample_prompt_score, sample_user_profile):
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    llm = FakeLLMClient(returns={"PromptScore": sample_prompt_score})
    orch = _build_orch(llm, FakePineconeRepository(), FakeSongRepository(),
                       FakeUserProfileRepository(profiles={"user_001": sample_user_profile}),
                       embedder)
    resp = orch.recommend("user_001", "happy")
    assert resp.recommendations == []
    assert resp.message != ""
