from dataclasses import dataclass
from pymongo.database import Database
from datasets import load_dataset
from tqdm import tqdm
from app.models.song_document import SongDocument
from app.persistence.mongo.song_repository import SongRepository
from app.services.song_matcher import SongMatcher


@dataclass(frozen=True)
class SongIngestResult:
    total_inserted: int
    batches: int
    spotify_matches: int


class SongIngestService:
    def __init__(self, db: Database) -> None:
        self._repository = SongRepository(db)

    def run(self, max_songs: int, batch_size: int = 500) -> SongIngestResult:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        matcher = SongMatcher()
        matcher.build_index(load_dataset("maharshipandya/spotify-tracks-dataset", split="train", streaming=True))

        self._repository.drop_collection()

        batch: list[SongDocument] = []
        total_inserted = 0
        batches = 0
        spotify_matches = 0

        genius = load_dataset("sebastiandizon/genius-song-lyrics", split="train", streaming=True)
        progress = tqdm(genius, total=max_songs, desc="Ingesting songs", unit="song")

        for i, row in enumerate(progress):
            if i >= max_songs:
                break

            try:
                sp = matcher.lookup(row)
                if sp:
                    spotify_matches += 1
                    progress.set_postfix(matches=spotify_matches, inserted=total_inserted)

                batch.append(SongDocument(
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
                    track_id=sp["track_id"] if sp else None,
                    album_name=sp["album_name"] if sp else None,
                    popularity=sp["popularity"] if sp else None,
                    duration_ms=sp["duration_ms"] if sp else None,
                    explicit=sp["explicit"] if sp else None,
                    danceability=sp["danceability"] if sp else None,
                    energy=sp["energy"] if sp else None,
                    key=sp["key"] if sp else None,
                    loudness=sp["loudness"] if sp else None,
                    mode=sp["mode"] if sp else None,
                    speechiness=sp["speechiness"] if sp else None,
                    acousticness=sp["acousticness"] if sp else None,
                    instrumentalness=sp["instrumentalness"] if sp else None,
                    liveness=sp["liveness"] if sp else None,
                    valence=sp["valence"] if sp else None,
                    tempo=sp["tempo"] if sp else None,
                    time_signature=sp["time_signature"] if sp else None,
                    track_genre=sp["track_genre"] if sp else None,
                ))
            except Exception as e:
                print(f"Skipping row {i}: {e}")
                continue

            if len(batch) >= batch_size:
                batches += 1
                try:
                    total_inserted += self._repository.insert_many(batch)
                    progress.set_postfix(matches=spotify_matches, inserted=total_inserted)
                except Exception as e:
                    tqdm.write(f"Error in batch {batches}: {e}")
                batch = []

        if batch:
            batches += 1
            try:
                total_inserted += self._repository.insert_many(batch)
                progress.set_postfix(matches=spotify_matches, inserted=total_inserted)
            except Exception as e:
                tqdm.write(f"Error in batch {batches}: {e}")

        return SongIngestResult(total_inserted=total_inserted, batches=batches, spotify_matches=spotify_matches)
