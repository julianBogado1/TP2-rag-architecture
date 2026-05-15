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

    def run(self) -> IndexingResult:
        logger.info("Indexing pipeline started.")
        docs = self._loader.load()
        chunks = self._splitter.split(docs)
        total_indexed = self._embedder.embed_and_index(chunks)
        logger.info(f"Pipeline complete. Indexed {total_indexed} chunks.")
        return IndexingResult(
            total_docs=len(docs),
            total_chunks=len(chunks),
            total_indexed=total_indexed,
        )


