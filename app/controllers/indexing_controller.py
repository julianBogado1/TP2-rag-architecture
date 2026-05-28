from fastapi import APIRouter
from pymongo import MongoClient

from app.core.config import settings
from app.lambdas.indexing_pipeline import IndexingPipeline, IndexingResult
from app.persistence.mongo.song_repository import SongRepository
from app.persistence.vector.pinecone_repository import PineconeRepository
from app.services.embedder_service import EmbedderService
from app.services.loader_service import LoaderService
from app.services.splitter_service import SplitterService

router = APIRouter()


def _build_pipeline() -> tuple[IndexingPipeline, MongoClient]:
    client = MongoClient(settings.mongo_uri)
    pipeline = IndexingPipeline(
        loader=LoaderService(
            repo=SongRepository(client[settings.mongo_db_name])
        ),
        splitter=SplitterService(chunk_size=1000, chunk_overlap=200),
        embedder=EmbedderService(
            vector_repo=PineconeRepository(
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index_name,
            ),
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        ),
    )
    return pipeline, client


@router.post("/index")
def run_indexing() -> IndexingResult:
    pipeline, client = _build_pipeline()
    try:
        return pipeline.run()
    finally:
        client.close()


def handler(event: dict, context: object) -> dict:
    pipeline, client = _build_pipeline()
    try:
        result = pipeline.run()
        return {"statusCode": 200, "body": result.__dict__}
    finally:
        client.close()
