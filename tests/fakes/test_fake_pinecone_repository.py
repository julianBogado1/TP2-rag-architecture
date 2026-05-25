from tests.fakes.fake_pinecone_repository import FakePineconeRepository


def test_upsert_and_query_returns_cosine_ordered():
    repo = FakePineconeRepository()
    repo.upsert_batch([
        ("a", [1.0, 0.0, 0.0], {"song_id": 1}),
        ("b", [0.9, 0.1, 0.0], {"song_id": 2}),
        ("c", [0.0, 1.0, 0.0], {"song_id": 3}),
    ])

    result = repo.query([1.0, 0.0, 0.0], top_k=2)

    assert [m.chunk_id for m in result] == ["a", "b"]
    assert result[0].score > result[1].score


def test_clear_empties_store():
    repo = FakePineconeRepository()
    repo.upsert("a", [1.0, 0.0], {})
    repo.clear()
    assert repo.query([1.0, 0.0], top_k=5) == []
