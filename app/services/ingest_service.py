from dataclasses import dataclass
from pymongo.database import Database
from datasets import load_dataset
from app.models.dataset_song_dto import DatasetSongDTO
from app.persistence.mongo import dataset_repository


@dataclass(frozen=True)
class IngestResult:
    total_inserted: int
    batches: int


class IngestService:
    def __init__(self, db: Database) -> None:
        self._db = db

    def run(self, max_songs: int, batch_size: int = 500) -> IngestResult:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        dataset_repository.drop_collection(self._db)

        stream = load_dataset(
            "sebastiandizon/genius-song-lyrics",
            split="train",
            streaming=True,
        )

        batch: list[DatasetSongDTO] = []
        total_inserted = 0
        batches = 0

        for i, row in enumerate(stream):
            if i >= max_songs:
                break
            batch.append(DatasetSongDTO(
                song_id=row["id"],
                title=row["title"],
                tag=row["tag"],
                artist=row["artist"],
                year=row["year"],
                views=row["views"],
                features=row.get("features"),
                lyrics=row["lyrics"],
                language_cld3=row.get("language_cld3"),
                language_ft=row.get("language_ft"),
                language=row.get("language"),
            ))

            if len(batch) >= batch_size:
                total_inserted += dataset_repository.insert_many_songs(self._db, batch)
                batches += 1
                print(f"Inserted batch {batches} (total: {total_inserted} / {max_songs})")
                batch = []

        if batch:
            total_inserted += dataset_repository.insert_many_songs(self._db, batch)
            batches += 1
            print(f"Inserted batch {batches} (total: {total_inserted} / {max_songs})")

        return IngestResult(total_inserted=total_inserted, batches=batches)
