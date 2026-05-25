from unittest.mock import MagicMock
from app.services.retrieval.query_embedder_service import QueryEmbedderService


def test_delegates_to_embedder_embed_query():
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.5] * 384
    svc = QueryEmbedderService(embedder=embedder)

    result = svc.embed("happy songs")

    assert result == [0.5] * 384
    embedder.embed_query.assert_called_once_with("happy songs")
