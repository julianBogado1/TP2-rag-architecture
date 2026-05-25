from app.models.raw_pinecone_match import RawPineconeMatch


def test_raw_pinecone_match_round_trip():
    m = RawPineconeMatch(chunk_id="42_3", score=0.81, metadata={"song_id": 42})
    assert m.chunk_id == "42_3"
    assert m.score == 0.81
    assert m.metadata["song_id"] == 42
