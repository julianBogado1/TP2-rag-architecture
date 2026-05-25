import math
from app.models.raw_pinecone_match import RawPineconeMatch


class FakePineconeRepository:
    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, doc_id: str, embedding: list[float], metadata: dict) -> None:
        self._store[doc_id] = (embedding, metadata)

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> None:
        for doc_id, embedding, metadata in records:
            self._store[doc_id] = (embedding, metadata)

    def clear(self) -> None:
        self._store.clear()

    def query(self, vector: list[float], top_k: int) -> list[RawPineconeMatch]:
        scored = [
            (doc_id, self._cosine(vector, vec), meta)
            for doc_id, (vec, meta) in self._store.items()
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            RawPineconeMatch(chunk_id=doc_id, score=score, metadata=meta)
            for doc_id, score, meta in scored[:top_k]
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
