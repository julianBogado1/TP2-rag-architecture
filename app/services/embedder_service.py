import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.exceptions import VectorStoreError
from app.persistence.vector.pinecone_repository import PineconeRepository

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbedderService:
    def __init__(
        self,
        vector_repo: PineconeRepository,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 100,
    ) -> None:
        self._vector_repo = vector_repo
        self._batch_size = batch_size
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )

    def embed_and_index(self, chunks) -> int:
        total = 0
        chunk_iter = iter(chunks)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = None
            while batch := list(islice(chunk_iter, self._batch_size)):
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
                try:
                    if pending is not None:
                        pending.result()
                    pending = pool.submit(self._vector_repo.upsert_batch, records)
                except VectorStoreError:
                    logger.error(
                        f"Upsert failed after {total} chunks indexed; "
                        "rerun is idempotent (deterministic ids)."
                    )
                    raise
                total += len(batch)
                logger.info(f"Indexed {total} chunks so far.")
            try:
                if pending is not None:
                    pending.result()
            except VectorStoreError:
                logger.error(
                    f"Upsert failed after {total} chunks indexed; "
                    "rerun is idempotent (deterministic ids)."
                )
                raise
        return total

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text using the same model used at index time."""
        # NOTE: uses HuggingFaceEmbeddings.embed_query, NOT .embed_documents. For
        # all-MiniLM-L6-v2 both produce the same vector, but for asymmetric models
        # (E5, BGE, etc. — they prepend "query: "/"passage: ") this would diverge
        # from the index-time path and silently degrade retrieval quality. If the
        # embedding model is swapped and recall drops, check this first.
        return self._embeddings.embed_query(text)
