from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from app.core.exceptions import VectorStoreError
from app.services.embedder_service import EmbedderService
from tests.fakes.fake_pinecone_repository import FakePineconeRepository


def test_embed_query_delegates_and_returns_vector():
    service = EmbedderService.__new__(EmbedderService)
    fake_embeddings = MagicMock()
    fake_embeddings.embed_query.return_value = [0.1] * 384
    service._embeddings = fake_embeddings
    service._vector_repo = FakePineconeRepository()
    service._batch_size = 100

    result = service.embed_query("happy songs")

    assert len(result) == 384
    fake_embeddings.embed_query.assert_called_once_with("happy songs")


def test_embed_and_index_reraises_vector_store_error():
    service = EmbedderService.__new__(EmbedderService)
    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.return_value = [[0.1] * 384]
    service._embeddings = fake_embeddings
    service._batch_size = 100

    failing_repo = MagicMock()
    failing_repo.upsert_batch.side_effect = VectorStoreError("boom")
    service._vector_repo = failing_repo

    chunks = [Document(page_content="x", metadata={"song_id": 1, "start_index": 0})]

    with pytest.raises(VectorStoreError):
        service.embed_and_index(chunks)
