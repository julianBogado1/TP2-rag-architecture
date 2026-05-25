class RecommendationError(Exception):
    """Base for all domain errors raised by the recommendation pipeline."""


class UserNotFoundError(RecommendationError):
    def __init__(self, user_id: str) -> None:
        super().__init__(f"User '{user_id}' not found in profile store.")
        self.user_id = user_id


class LLMProviderError(RecommendationError):
    """Raised when the LLM call fails after retry (timeout, rate limit, refusal)."""


class VectorStoreError(RecommendationError):
    """Raised on Pinecone failures (network, auth, index-not-found)."""


class EmbeddingError(RecommendationError):
    """Raised when the embedding model fails (loading, OOM, encode error)."""
