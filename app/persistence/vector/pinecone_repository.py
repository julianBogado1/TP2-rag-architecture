from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document

EMBEDDING_DIMENSION = 384


class PineconeRepository:
    def __init__(self, api_key: str, index_name: str) -> None:
        self._pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self._initialize_index()
        self._index = self._pc.Index(index_name)

    def _initialize_index(self) -> None:
        if self.index_name not in self._pc.list_indexes().names():
            self._pc.create_index(
                name=self.index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def upsert(self, doc_id: str, embedding: list[float], metadata: dict) -> None:
        self._index.upsert(vectors=[(doc_id, embedding, metadata)])

    def upsert_batch(self, records: list[tuple[str, list[float], dict]]) -> None:
        self._index.upsert(vectors=records)

    def clear(self) -> None:
        self._index.delete(delete_all=True)

    def query(self, embedding: list[float], top_k: int = 5) -> list[Document]:
        results = self._index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [
            Document(page_content=res["metadata"]["text"], metadata=res["metadata"])
            for res in results["matches"]
        ]
