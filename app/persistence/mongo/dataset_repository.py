from pymongo.database import Database
from app.models.dataset_song_dto import DatasetSongDTO


class DatasetRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def drop_collection(self) -> None:
        self._db["dataset"].drop()

    def insert_many_songs(self, songs: list[DatasetSongDTO]) -> int:
        if not songs:
            return 0
        docs = [song.model_dump() for song in songs]
        result = self._db["dataset"].insert_many(docs)
        return len(result.inserted_ids)

    def count(self) -> int:
        return self._db["dataset"].count_documents({})
