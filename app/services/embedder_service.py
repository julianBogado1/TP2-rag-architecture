import logging

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.persistence.vector.pinecone_repository import PineconeRepository

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedderService:
    def __init__(
        self,
        vector_repo: PineconeRepository,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self._vector_repo = vector_repo
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )

    def embed_and_index(self, chunks: list[Document], batch_size: int = 100) -> int:
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.page_content for chunk in batch]
            vectors = self._embeddings.embed_documents(texts)
            records = [
                (
                    f"{batch[j].metadata['song_id']}_{batch[j].metadata.get('start_index', j)}",
                    vectors[j],
                    batch[j].metadata,
                )
                for j in range(len(batch))
            ]
            self._vector_repo.upsert_batch(records)
            total += len(batch)
            logger.info(f"Embedded and indexed {total} / {len(chunks)} chunks.")
        return total
