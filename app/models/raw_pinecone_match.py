from pydantic import BaseModel


class RawPineconeMatch(BaseModel):
    """Raw, untyped-metadata match returned by PineconeRepository.query().
    The VectorRetrievalService parses metadata downstream into typed models."""
    chunk_id: str
    score: float
    metadata: dict
