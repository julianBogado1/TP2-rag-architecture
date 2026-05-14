#!/usr/bin/env python3
"""Fetch all songs from the dataset collection.

Usage:
    .venv/bin/python scripts/fetch_songs.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from pymongo import MongoClient
from app.persistence.mongo.dataset_repository import DatasetRepository

config = dotenv_values(ROOT / ".env")
MONGO_URI = config["MONGO_URI"]
MONGO_DB_NAME = config["MONGO_DB_NAME"]


def main() -> None:
    client = MongoClient(MONGO_URI)
    repo = DatasetRepository(client[MONGO_DB_NAME])
    songs = repo.get_all()
    for song in songs:
        print(song)
    print(f"\nTotal: {len(songs)} songs")
    client.close()


if __name__ == "__main__":
    main()
