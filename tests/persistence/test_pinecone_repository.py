import pytest
from unittest.mock import MagicMock
from app.core.exceptions import VectorStoreError
from app.models.raw_pinecone_match import RawPineconeMatch
from app.persistence.vector import pinecone_repository as pinecone_module
from app.persistence.vector.pinecone_repository import PineconeRepository


def _repo_with_fake_index() -> tuple[PineconeRepository, MagicMock]:
    repo = PineconeRepository.__new__(PineconeRepository)
    fake_index = MagicMock()
    repo._index = fake_index
    return repo, fake_index


def test_query_returns_typed_matches():
    repo, fake_index = _repo_with_fake_index()
    fake_index.query.return_value = {
        "matches": [
            {"id": "42_3", "score": 0.81, "metadata": {"song_id": 42}},
            {"id": "13_0", "score": 0.55, "metadata": {"song_id": 13}},
        ],
    }

    result = repo.query([0.1] * 384, top_k=10)

    assert all(isinstance(m, RawPineconeMatch) for m in result)
    assert result[0].chunk_id == "42_3"
    assert result[0].score == 0.81
    assert result[0].metadata["song_id"] == 42


def test_query_handles_missing_metadata():
    repo, fake_index = _repo_with_fake_index()
    fake_index.query.return_value = {"matches": [{"id": "x", "score": 0.1}]}

    result = repo.query([0.1] * 384, top_k=1)

    assert result[0].metadata == {}


def test_query_empty_matches():
    repo, fake_index = _repo_with_fake_index()
    fake_index.query.return_value = {"matches": []}

    assert repo.query([0.1] * 384, top_k=10) == []


class _ScoredVector:
    """Minimal stand-in for pinecone-client v3+ ScoredVector (attribute access)."""
    def __init__(self, id, score, metadata):
        self.id = id
        self.score = score
        self.metadata = metadata


class _QueryResponse:
    def __init__(self, matches):
        self.matches = matches


def test_query_handles_pinecone_v3_object_response():
    repo, fake_index = _repo_with_fake_index()
    fake_index.query.return_value = _QueryResponse(matches=[
        _ScoredVector("42_3", 0.81, {"song_id": 42}),
        _ScoredVector("13_0", 0.55, None),
    ])

    result = repo.query([0.1] * 384, top_k=10)

    assert [m.chunk_id for m in result] == ["42_3", "13_0"]
    assert result[0].metadata == {"song_id": 42}
    assert result[1].metadata == {}


def test_upsert_batch_succeeds_after_transient_failure(monkeypatch):
    monkeypatch.setattr(pinecone_module.time, "sleep", lambda *_: None)
    repo, fake_index = _repo_with_fake_index()
    fake_index.upsert.side_effect = [RuntimeError("transient"), None]

    records = [("a", [0.1] * 384, {"song_id": 1})]
    repo.upsert_batch(records)

    assert fake_index.upsert.call_count == 2
    fake_index.upsert.assert_called_with(vectors=records)


def test_upsert_batch_raises_vector_store_error_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(pinecone_module.time, "sleep", lambda *_: None)
    repo, fake_index = _repo_with_fake_index()
    fake_index.upsert.side_effect = RuntimeError("down")

    with pytest.raises(VectorStoreError):
        repo.upsert_batch([("a", [0.1] * 384, {"song_id": 1})])

    assert fake_index.upsert.call_count == pinecone_module._MAX_RETRIES


def test_query_wraps_sdk_error_as_vector_store_error():
    repo, fake_index = _repo_with_fake_index()
    fake_index.query.side_effect = RuntimeError("boom")

    with pytest.raises(VectorStoreError):
        repo.query([0.1] * 384, top_k=10)
