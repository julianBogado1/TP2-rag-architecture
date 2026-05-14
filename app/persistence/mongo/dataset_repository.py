from pymongo.database import Database
from app.models.dataset_song_dto import DatasetSongDTO


def drop_collection(db: Database) -> None:
    db["dataset"].drop()


def insert_many_songs(db: Database, songs: list[DatasetSongDTO]) -> int:
    if not songs:
        return 0
    docs = [song.model_dump() for song in songs]
    result = db["dataset"].insert_many(docs)
    return len(result.inserted_ids)


def count(db: Database) -> int:
    return db["dataset"].count_documents({})
