from unittest.mock import MagicMock
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
