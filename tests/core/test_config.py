from app.core.config import Settings


def test_settings_loads_with_defaults(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://test")
    monkeypatch.setenv("MONGO_DB_NAME", "test")
    monkeypatch.setenv("PINECONE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "y")
    s = Settings(_env_file=None)
    assert s.pinecone_index_name == "song-lyrics-chunks"
    assert s.pinecone_dimension == 384
    assert s.openai_model_parser == "gpt-4o-mini"
    assert s.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.splitter_chunk_size == 1000
    assert s.retrieval_top_k == 150
    assert s.output_top_n == 10
    assert s.default_w_lyrics == 0.55
