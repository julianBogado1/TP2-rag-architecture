"""Smoke test: real Mongo + Pinecone + OpenAI.

Usage:
    .venv/bin/python scripts/smoke_recommend.py "<prompt>" [user_id]
"""
import sys
from pymongo import MongoClient
from app.core.config import settings
from app.core.llm_client import OpenAILLMClient
from app.persistence.mongo.song_repository import SongRepository
from app.persistence.mongo.user_profile_repository import UserProfileRepository
from app.persistence.vector.pinecone_repository import PineconeRepository
from app.services.embedder_service import EmbedderService
from app.services.retrieval.prompt_parser_service import PromptParserService
from app.services.retrieval.recommendation_request_builder import RecommendationRequestBuilder
from app.services.retrieval.query_embedder_service import QueryEmbedderService
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from app.services.retrieval.candidate_aggregator_service import CandidateAggregatorService
from app.services.retrieval.hybrid_reranker_service import HybridRerankerService
from app.services.retrieval.top_n_selector_service import TopNSelectorService
from app.services.retrieval.response_generator_service import ResponseGeneratorService
from app.services.retrieval.recommendation_orchestrator import RecommendationOrchestrator


def main(prompt: str, user_id: str = "user_001") -> None:
    mongo = MongoClient(settings.mongo_uri)
    db = mongo[settings.mongo_db_name]
    pinecone = PineconeRepository(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    llm = OpenAILLMClient(api_key=settings.openai_api_key)
    embedder = EmbedderService(
        vector_repo=pinecone,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    orch = RecommendationOrchestrator(
        prompt_parser      = PromptParserService(llm, settings.openai_model_parser),
        user_repo          = UserProfileRepository(db),
        request_builder    = RecommendationRequestBuilder(settings),
        query_embedder     = QueryEmbedderService(embedder),
        vector_retrieval   = VectorRetrievalService(pinecone),
        aggregator         = CandidateAggregatorService(
            SongRepository(db), settings.aggregator_max_evidence_chunks
        ),
        reranker           = HybridRerankerService(),
        selector           = TopNSelectorService(settings.selector_max_per_artist),
        response_generator = ResponseGeneratorService(llm, settings.openai_model_response),
    )
    resp = orch.recommend(user_id, prompt)
    print(resp.model_dump_json(indent=2))
    mongo.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: smoke_recommend.py '<prompt>' [user_id]")
        sys.exit(1)
    user = sys.argv[2] if len(sys.argv) >= 3 else "user_001"
    main(sys.argv[1], user)
