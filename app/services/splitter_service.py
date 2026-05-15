import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SplitterService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        logger.info(f"Split {len(docs)} documents into {len(chunks)} chunks.")
        return chunks
