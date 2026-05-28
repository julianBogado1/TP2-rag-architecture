from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Mongo
    mongo_uri: str
    mongo_db_name: str

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "song-lyrics-chunks"
    pinecone_dimension: int = 384
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # OpenAI
    openai_api_key: str
    openai_model_parser: str = "gpt-4o-mini"
    openai_model_response: str = "gpt-4o-mini"

    # Embedding
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 512

    # Splitter
    # 400/50 keeps chunks inside all-MiniLM-L6-v2's 256-token limit (~750-900 chars),
    # so the embedder never silently truncates a chunk's tail.
    splitter_chunk_size: int = 400
    splitter_chunk_overlap: int = 50

    # Retrieval defaults
    retrieval_top_k: int = 150
    output_top_n: int = 10
    aggregator_max_evidence_chunks: int = 3
    selector_max_per_artist: int = 2

    # Default ranking weights
    default_w_lyrics: float = 0.55
    default_w_audio: float = 0.30
    default_w_profile: float = 0.10
    default_w_popularity: float = 0.03
    default_w_recency: float = 0.02

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


class _LazySettings:
    """Lazy singleton: constructs on first attribute access. Avoids import-time
    validation failures when env vars are missing (e.g., in unit tests that
    monkeypatch the environment)."""
    _instance: Settings | None = None

    def __getattr__(self, name: str):
        if _LazySettings._instance is None:
            _LazySettings._instance = Settings()
        return getattr(_LazySettings._instance, name)


settings = _LazySettings()
