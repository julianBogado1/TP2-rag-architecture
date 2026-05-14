from pymongo.database import Database
from pymongo.collection import Collection
from app.models.dataset_song_dto import DatasetSongDTO


class DatasetRepository:
    COLLECTION = "dataset"

    def __init__(self, db: Database) -> None:
        self._collection: Collection = db[self.COLLECTION]

    def drop_collection(self) -> None:
        self._collection.drop()

    def insert_many_songs(self, songs: list[DatasetSongDTO]) -> int:
        if not songs:
            return 0
        docs = [song.model_dump() for song in songs]
        result = self._collection.insert_many(docs)
        return len(result.inserted_ids)

    def count(self) -> int:
        return self._collection.count_documents({})

    def get_all(self) -> list[DatasetSongDTO]:
        docs = self._collection.find({}, {"_id": 0})
        return [DatasetSongDTO(**doc) for doc in docs]

    def get_by_id(self, song_id: int) -> DatasetSongDTO | None:
        doc = self._collection.find_one({"song_id": song_id}, {"_id": 0})
        if doc is None:
            return None
        return DatasetSongDTO(**doc)

    def get_by_artist(self, artist: str) -> list[DatasetSongDTO]:
        docs = self._collection.find({"artist": artist}, {"_id": 0})
        return [DatasetSongDTO(**doc) for doc in docs]

    def get_by_genre(self, genre: str) -> list[DatasetSongDTO]:
        docs = self._collection.find({"tag": genre}, {"_id": 0})
        return [DatasetSongDTO(**doc) for doc in docs]

    def get_by_year(self, year: int) -> list[DatasetSongDTO]:
        docs = self._collection.find({"year": year}, {"_id": 0})
        return [DatasetSongDTO(**doc) for doc in docs]

    def get_by_title(self, title: str) -> list[DatasetSongDTO]:
        docs = self._collection.find({"title": title}, {"_id": 0})
        return [DatasetSongDTO(**doc) for doc in docs]
