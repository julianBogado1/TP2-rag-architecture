import logging
from dataclasses import dataclass

from app.services.loader_service import LoaderService
from app.services.splitter_service import SplitterService
from app.services.embedder_service import EmbedderService

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    total_docs: int
    # In the streaming design chunks are embedded as they are produced, so
    # total_chunks always equals total_indexed (kept for API/back-compat).
    total_chunks: int
    total_indexed: int


class IndexingPipeline:
    def __init__(
        self,
        loader: LoaderService,
        splitter: SplitterService,
        embedder: EmbedderService,
    ) -> None:
        self._loader = loader
        self._splitter = splitter
        self._embedder = embedder

    def run(self, song_ids: list[int] | None = None) -> IndexingResult:
        logger.info("Indexing pipeline started.")
        doc_count = 0

        def counted_docs():
            nonlocal doc_count
            for doc in self._loader.load(song_ids):
                doc_count += 1
                yield doc

        chunks = self._splitter.split(counted_docs())
        total_indexed = self._embedder.embed_and_index(chunks)
        logger.info(f"Pipeline complete. {doc_count} docs → {total_indexed} chunks indexed.")
        return IndexingResult(
            total_docs=doc_count,
            total_chunks=total_indexed,
            total_indexed=total_indexed,
        )


