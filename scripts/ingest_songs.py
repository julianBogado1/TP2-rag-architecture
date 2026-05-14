#!/usr/bin/env python3
"""Ingest a partition of the genius-song-lyrics dataset into MongoDB.

Usage:
    .venv/bin/python scripts/ingest_songs.py <max_songs>

Example:
    .venv/bin/python scripts/ingest_songs.py 5000
"""

import sys
from pathlib import Path
from dotenv import dotenv_values
from pymongo import MongoClient
from app.services.ingest_service import IngestService

ROOT = Path(__file__).resolve().parent.parent
config = dotenv_values(ROOT / ".env")

MONGO_URI = config["MONGO_URI"]
MONGO_DB_NAME = config["MONGO_DB_NAME"]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: ingest_songs.py <max_songs>")
        sys.exit(1)

    try:
        max_songs = int(sys.argv[1])
        if max_songs < 1:
            raise ValueError
    except ValueError:
        print("Error: <max_songs> must be a positive integer")
        sys.exit(1)
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]

    print(f"Ingesting up to {max_songs} songs into '{MONGO_DB_NAME}.dataset'...")
    result = IngestService(db).run(max_songs)
    print(f"\nDone. Inserted {result.total_inserted} songs in {result.batches} batches.")
    client.close()


if __name__ == "__main__":
    main()
