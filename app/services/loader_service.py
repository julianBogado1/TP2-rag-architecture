import logging

from langchain_core.documents import Document

from app.persistence.mongo.dataset_repository import DatasetRepository

logger = logging.getLogger(__name__)


class LoaderService:
    def __init__(self, repo: DatasetRepository) -> None:
        self._repo = repo

    def load(self) -> list[Document]:
        songs = self._repo.get_all()
        docs = [
            Document(
                page_content=song.lyrics,
                metadata={
                    "song_id": song.song_id,
                    "track_name": song.title,
                    "artist_name": song.artist,
                    "genres": [song.tag],
                    "popularity": song.views,
                    "release_date": str(song.year),
                    "audio_features_chunk": [],
                },
            )
            for song in songs
        ]
        logger.info(f"Loaded {len(docs)} documents from MongoDB.")
        return docs
