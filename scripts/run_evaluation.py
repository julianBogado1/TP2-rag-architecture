#!/usr/bin/env python3
"""Run the full metrics evaluation against the ground truth test cases.

Requires:
  - data/gt_test_cases.json (run scripts/build_gt.py first)
  - MongoDB and Pinecone running
  - OPENAI_API_KEY set in .env
  - skip_llm=False in ResponseGeneratorService

Usage:
    python scripts/run_evaluation.py
"""
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pymongo import MongoClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from app.core.config import settings
from app.core.llm_client import OpenAILLMClient
from app.models.request_context import RequestContext
from app.persistence.mongo.song_repository import SongRepository
from app.persistence.mongo.user_profile_repository import UserProfileRepository
from app.persistence.vector.pinecone_repository import PineconeRepository
from app.services.embedder_service import EmbedderService
from app.services.retrieval.candidate_aggregator_service import CandidateAggregatorService
from app.services.retrieval.hybrid_reranker_service import HybridRerankerService
from app.services.retrieval.prompt_parser_service import PromptParserService
from app.services.retrieval.query_embedder_service import QueryEmbedderService
from app.services.retrieval.recommendation_request_builder import RecommendationRequestBuilder
from app.services.retrieval.response_generator_service import ResponseGeneratorService
from app.services.retrieval.top_n_selector_service import TopNSelectorService
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from app.metrics import (
    context_precision_at_k,
    recall_at_k,
    faithfulness_score,
    answer_relevance_score,
)

GT_PATH = Path(__file__).parent.parent / "data" / "gt_test_cases.json"
RESULTS_PATH = Path(__file__).parent.parent / "data" / "evaluation_results.json"
K = 10


def audio_bearing(top_songs: list) -> list:
    """Keep only songs that carry Spotify audio features.

    Audio test cases are scored against this subset so that no-audio songs
    retrieved on lyric similarity don't count as misses against an
    audio-feature ground truth.
    """
    return [s for s in top_songs if s.metadata.audio_features is not None]


def score_case(kind: str, top_songs: list, gt_ids: set, k: int) -> tuple[float, float]:
    """Compute (context_precision@k, recall@k) for one case.

    audio cases score over the audio-bearing subset; genre cases score over all
    returned songs (relevance is decided by the GT id set).
    """
    scored = audio_bearing(top_songs) if kind == "audio" else top_songs
    return context_precision_at_k(scored, gt_ids, k), recall_at_k(scored, gt_ids, k)


def build_services():
    mongo = MongoClient(settings.mongo_uri)
    db = mongo[settings.mongo_db_name]
    pinecone_repo = PineconeRepository(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    llm = OpenAILLMClient(api_key=settings.openai_api_key)
    embedder = EmbedderService(
        vector_repo=pinecone_repo,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
    )
    return dict(
        mongo=mongo,
        user_repo=UserProfileRepository(db),
        llm=llm,
        embedder=embedder,
        pinecone_repo=pinecone_repo,
        prompt_parser=PromptParserService(llm, settings.openai_model_parser),
        request_builder=RecommendationRequestBuilder(settings),
        query_embedder=QueryEmbedderService(embedder),
        vector_retrieval=VectorRetrievalService(pinecone_repo),
        aggregator=CandidateAggregatorService(SongRepository(db), settings.aggregator_max_evidence_chunks),
        reranker=HybridRerankerService(),
        selector=TopNSelectorService(settings.selector_max_per_artist),
        response_generator=ResponseGeneratorService(llm, settings.openai_model_response),
    )


def run_pipeline(svc: dict, user_id: str, raw_prompt: str):
    """Run pipeline step-by-step to capture both top_songs and response."""
    ctx = RequestContext(
        user_id=user_id,
        raw_prompt=raw_prompt,
        timestamp=datetime.now(timezone.utc),
        session_id=str(uuid4()),
    )
    score = svc["prompt_parser"].parse(raw_prompt)
    profile = svc["user_repo"].get_by_user_id(user_id)
    req = svc["request_builder"].build(ctx, score, profile)
    qvec = svc["query_embedder"].embed(req.semantic_query)
    chunks = svc["vector_retrieval"].retrieve(req, qvec)
    if not chunks:
        return None, None
    cands = svc["aggregator"].aggregate(chunks)
    ranked = svc["reranker"].rerank(cands, req)
    top_songs = svc["selector"].select(ranked, req.top_n_output)
    response = svc["response_generator"].generate(top_songs, req)
    return top_songs, response


def main() -> None:
    if not GT_PATH.exists():
        print(f"ERROR: {GT_PATH} not found. Run scripts/build_gt.py first.")
        return

    test_cases = json.loads(GT_PATH.read_text())
    print(f"Loaded {len(test_cases)} test cases\n")

    svc = build_services()

    results = []
    header = f"{'Label':<16} {'Kind':<6} {'CP@K':>6} {'Rec@K':>6} {'Faith':>6} {'AnsRel':>7}"
    print(header)
    print("-" * len(header))

    try:
        for tc in test_cases:
            label = tc["label"]
            prompt = tc["prompt"]
            user_id = tc["user_id"]
            kind = tc.get("kind", "audio")  # default keeps old GT files working
            gt_ids = set(tc["gt_song_ids"])

            try:
                top_songs, response = run_pipeline(svc, user_id, prompt)
                if top_songs is None:
                    print(f"{label:<16} {kind:<6} — no results")
                    continue

                cp, rec = score_case(kind, top_songs, gt_ids, K)
                faith = faithfulness_score(response, top_songs, svc["llm"])
                ar = answer_relevance_score(prompt, response, svc["embedder"], svc["llm"])

                print(f"{label:<16} {kind:<6} {cp:>6.3f} {rec:>6.3f} {faith:>6.3f} {ar:>7.3f}")

                results.append({
                    "label": label,
                    "kind": kind,
                    "prompt": prompt,
                    "context_precision_at_k": round(cp, 4),
                    "recall_at_k": round(rec, 4),
                    "faithfulness": round(faith, 4),
                    "answer_relevance": round(ar, 4),
                })
            except Exception as e:  # isolate one bad case from the whole run
                print(f"{label:<16} {kind:<6} — ERROR: {e}")
                results.append({"label": label, "kind": kind, "prompt": prompt, "error": str(e)})

            # write incrementally so a later crash doesn't discard prior results
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

        scored = [r for r in results if "error" not in r]
        if scored:
            print("-" * len(header))
            _print_avg("AVERAGE (all)", scored, header)
            for k in ("audio", "genre"):
                subset = [r for r in scored if r.get("kind") == k]
                if subset:
                    _print_avg(f"avg [{k}]", subset, header)
            print(f"\nSaved to {RESULTS_PATH}")
    finally:
        svc["mongo"].close()


def _print_avg(name: str, rows: list[dict], header: str) -> None:
    n = len(rows)
    avg_cp = sum(r["context_precision_at_k"] for r in rows) / n
    avg_rec = sum(r["recall_at_k"] for r in rows) / n
    avg_faith = sum(r["faithfulness"] for r in rows) / n
    avg_ar = sum(r["answer_relevance"] for r in rows) / n
    print(f"{name:<16} {'':<6} {avg_cp:>6.3f} {avg_rec:>6.3f} {avg_faith:>6.3f} {avg_ar:>7.3f}")


if __name__ == "__main__":
    main()
