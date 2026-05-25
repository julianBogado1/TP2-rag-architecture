from pinecone import Pinecone, ServerlessSpec
from app.models.raw_pinecone_match import RawPineconeMatch

EMBEDDING_DIMENSION = 384


class PineconeRepository:
    def __init__(self, api_key: str, index_name: str, dimension: int = EMBEDDING_DIMENSION) -> None:
        self._pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self._dimension = dimension
        self._initialize_index()
        self._index = self._pc.Index(index_name)

    def _initialize_index(self) -> None:
        if self.index_name not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=self.index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert(self, doc_id: str, embedding: list[float], metadata: dict) -> None:
        self._index.upsert(vectors=[(doc_id, embedding, metadata)])

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> None:
        self._index.upsert(vectors=records)

    def clear(self) -> None:
        self._index.delete(delete_all=True)

    def query(self, vector: list[float], top_k: int) -> list[RawPineconeMatch]:
        result = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            RawPineconeMatch(
                chunk_id=m["id"],
                score=m["score"],
                metadata=m.get("metadata") or {},
            )
            for m in result.get("matches", [])
        ]
