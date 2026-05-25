from app.services.embedder_service import EmbedderService


class QueryEmbedderService:
    """Adapter on EmbedderService.embed_query that names the retrieval-side intent."""

    def __init__(self, embedder: EmbedderService) -> None:
        self._embedder = embedder

    def embed(self, semantic_query: str) -> list[float]:
        return self._embedder.embed_query(semantic_query)
