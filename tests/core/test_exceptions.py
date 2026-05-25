import pytest
from app.core.exceptions import (
    RecommendationError, UserNotFoundError,
    LLMProviderError, VectorStoreError, EmbeddingError,
)


def test_user_not_found_carries_user_id():
    err = UserNotFoundError("user_001")
    assert err.user_id == "user_001"
    assert "user_001" in str(err)
    assert isinstance(err, RecommendationError)


@pytest.mark.parametrize("cls", [LLMProviderError, VectorStoreError, EmbeddingError])
def test_other_errors_subclass_base(cls):
    assert issubclass(cls, RecommendationError)
