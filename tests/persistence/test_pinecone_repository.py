from unittest.mock import MagicMock
from app.models.raw_pinecone_match import RawPineconeMatch
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
