from pymongo.database import Database
from pymongo.collection import Collection
from app.models.song_document import SongDocument

_IN_CHUNK = 5000


class SongRepository:
    COLLECTION = "songs"

    def __init__(self, db: Database) -> None:
        self._collection: Collection = db[self.COLLECTION]

    def ensure_indexes(self) -> None:
        self._collection.create_index("song_id", unique=True)

    def _chunked(self, ids: list[int]):
        for i in range(0, len(ids), _IN_CHUNK):
            yield ids[i:i + _IN_CHUNK]

    def drop_collection(self) -> None:
        self._collection.drop()

    def insert_many(self, songs: list[SongDocument]) -> int:
        if not songs:
            return 0
        docs = [song.model_dump() for song in songs]
        result = self._collection.insert_many(docs)
        return len(result.inserted_ids)

    def count(self) -> int:
        return self._collection.count_documents({})

    def get_all(self):
        for doc in self._collection.find({}, {"_id": 0}):
            yield SongDocument(**doc)

    def get_by_id(self, song_id: int) -> SongDocument | None:
        doc = self._collection.find_one({"song_id": song_id}, {"_id": 0})
        if doc is None:
            return None
        return SongDocument(**doc)

    def get_by_artist(self, artist: str) -> list[SongDocument]:
        docs = self._collection.find({"artist": artist}, {"_id": 0})
        return [SongDocument(**doc) for doc in docs]

    def get_by_genre(self, tag: str) -> list[SongDocument]:
        docs = self._collection.find({"tag": tag}, {"_id": 0})
        return [SongDocument(**doc) for doc in docs]

    def get_by_year(self, year: int) -> list[SongDocument]:
        docs = self._collection.find({"year": year}, {"_id": 0})
        return [SongDocument(**doc) for doc in docs]

    def get_by_ids(self, song_ids: list[int]) -> list[SongDocument]:
        if not song_ids:
            return []
        out: list[SongDocument] = []
        for chunk in self._chunked(song_ids):
            out.extend(
                SongDocument(**doc)
                for doc in self._collection.find(
                    {"song_id": {"$in": chunk}}, {"_id": 0}
                )
            )
        return out

    def get_by_ids_stream(self, song_ids: list[int]):
        for chunk in self._chunked(song_ids):
            for doc in self._collection.find(
                {"song_id": {"$in": chunk}}, {"_id": 0}
            ):
                yield SongDocument(**doc)
