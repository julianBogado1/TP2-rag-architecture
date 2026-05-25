from datetime import datetime, timezone
from uuid import uuid4
from app.core.exceptions import UserNotFoundError
from app.models.request_context import RequestContext
from app.models.recommendation_response import RecommendationResponse
from app.persistence.mongo.user_profile_repository import UserProfileRepository
from app.services.retrieval.prompt_parser_service import PromptParserService
from app.services.retrieval.recommendation_request_builder import RecommendationRequestBuilder
from app.services.retrieval.query_embedder_service import QueryEmbedderService
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from app.services.retrieval.candidate_aggregator_service import CandidateAggregatorService
from app.services.retrieval.hybrid_reranker_service import HybridRerankerService
from app.services.retrieval.top_n_selector_service import TopNSelectorService
from app.services.retrieval.response_generator_service import ResponseGeneratorService

EMPTY_RESULT_MESSAGE = "No encontré canciones que coincidan con tu pedido. Probá con otros términos."


class RecommendationOrchestrator:
    """Wires the 8 retrieval services into a single end-to-end recommend() call."""

    def __init__(
        self,
        prompt_parser:      PromptParserService,
        user_repo:          UserProfileRepository,
        request_builder:    RecommendationRequestBuilder,
        query_embedder:     QueryEmbedderService,
        vector_retrieval:   VectorRetrievalService,
        aggregator:         CandidateAggregatorService,
        reranker:           HybridRerankerService,
        selector:           TopNSelectorService,
        response_generator: ResponseGeneratorService,
    ) -> None:
        self._prompt_parser      = prompt_parser
        self._user_repo          = user_repo
        self._request_builder    = request_builder
        self._query_embedder     = query_embedder
        self._vector_retrieval   = vector_retrieval
        self._aggregator         = aggregator
        self._reranker           = reranker
        self._selector           = selector
        self._response_generator = response_generator

    def recommend(self, user_id: str, raw_prompt: str) -> RecommendationResponse:
        ctx = RequestContext(
            user_id=user_id, 
            raw_prompt=raw_prompt,
            timestamp=datetime.now(timezone.utc), 
            session_id=str(uuid4()),
        )
        score = self._prompt_parser.parse(raw_prompt)
        profile = self._user_repo.get_by_user_id(user_id)

        if profile is None:
            raise UserNotFoundError(user_id)

        req = self._request_builder.build(ctx, score, profile)
        qvec = self._query_embedder.embed(req.semantic_query)
        chunks = self._vector_retrieval.retrieve(req, qvec)
        if not chunks:
            return RecommendationResponse(message=EMPTY_RESULT_MESSAGE, recommendations=[])

        cands  = self._aggregator.aggregate(chunks)
        ranked = self._reranker.rerank(cands, req)
        top    = self._selector.select(ranked, req.top_n_output)
        return self._response_generator.generate(top, req)

