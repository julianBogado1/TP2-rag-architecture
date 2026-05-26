from contextlib import asynccontextmanager
from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.staticfiles import StaticFiles
import webbrowser
import threading

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
from app.controllers.recommendation_controller import build_recommendation_router
from app.controllers.indexing_controller import router as indexing_router
from app.controllers.user_profile_controller import build_user_profile_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    mongo = MongoClient(settings.mongo_uri)
    db = mongo[settings.mongo_db_name]

    pinecone_repo = PineconeRepository(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    llm_client = OpenAILLMClient(api_key=settings.openai_api_key)

    embedder = EmbedderService(
        vector_repo=pinecone_repo,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )

    song_repo = SongRepository(db)
    user_repo = UserProfileRepository(db)

    orchestrator = RecommendationOrchestrator(
        prompt_parser      = PromptParserService(llm_client, settings.openai_model_parser),
        user_repo          = user_repo,
        request_builder    = RecommendationRequestBuilder(settings),
        query_embedder     = QueryEmbedderService(embedder),
        vector_retrieval   = VectorRetrievalService(pinecone_repo),
        aggregator         = CandidateAggregatorService(song_repo, settings.aggregator_max_evidence_chunks),
        reranker           = HybridRerankerService(),
        selector           = TopNSelectorService(settings.selector_max_per_artist),
        response_generator = ResponseGeneratorService(
            llm_client, settings.openai_model_response, skip_llm=True,
        ),
    )

    app.include_router(indexing_router)
    app.include_router(build_recommendation_router(orchestrator))
    app.include_router(build_user_profile_router(user_repo))
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    print("Frontend disponible en http://localhost:8000")
    yield
    mongo.close()


app = FastAPI(lifespan=lifespan, title="Song Recommendation RAG API")
