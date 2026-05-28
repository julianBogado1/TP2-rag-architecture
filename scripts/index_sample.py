#!/usr/bin/env python3
"""Index a pre-sampled set of songs into Pinecone.

Usage:
    .venv/bin/python scripts/index_sample.py [--ids-file scripts/sample_200k_ids.json]

Reads a JSON array of song_ids, runs the full indexing pipeline for only those songs.
"""
import sys
import json
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

from pymongo import MongoClient

from app.core.config import settings
from app.persistence.mongo.song_repository import SongRepository
from app.persistence.vector.pinecone_repository import PineconeRepository
from app.services.loader_service import LoaderService
from app.services.splitter_service import SplitterService
from app.services.embedder_service import EmbedderService
from app.lambdas.indexing_pipeline import IndexingPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ids-file",
        default=str(ROOT / "scripts" / "sample_200k_ids.json"),
        help="Path to JSON file containing list of song_ids",
    )
    return p.parse_args()


def main():
    args = parse_args()

    ids_path = Path(args.ids_file)
    if not ids_path.exists():
        logger.error(f"IDs file not found: {ids_path}")
        sys.exit(1)

    with open(ids_path) as f:
        song_ids: list[int] = json.load(f)

    logger.info(f"Loaded {len(song_ids):,} song_ids from {ids_path}")

    # Wire up services
    mongo_client = MongoClient(settings.mongo_uri)
    db = mongo_client[settings.mongo_db_name]
    song_repo = SongRepository(db)

    pinecone_repo = PineconeRepository(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=settings.pinecone_dimension,
    )

    embedder = EmbedderService(
        vector_repo=pinecone_repo,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    loader = LoaderService(repo=song_repo)
    splitter = SplitterService()
    pipeline = IndexingPipeline(loader=loader, splitter=splitter, embedder=embedder)

    logger.info("Starting indexing pipeline for sampled songs...")
    result = pipeline.run(song_ids=song_ids)
    logger.info(
        f"Done. docs={result.total_docs:,}  chunks={result.total_chunks:,}  indexed={result.total_indexed:,}"
    )

    mongo_client.close()


if __name__ == "__main__":
    main()
