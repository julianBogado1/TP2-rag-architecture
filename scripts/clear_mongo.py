#!/usr/bin/env python3
"""Drop the songs collection from MongoDB.

Usage:
    python scripts/clear_mongo.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from pymongo import MongoClient
from app.persistence.mongo.song_repository import SongRepository

config = dotenv_values(ROOT / ".env")

client = MongoClient(config["MONGO_URI"])
repo = SongRepository(client[config["MONGO_DB_NAME"]])
repo.drop_collection()
client.close()

print("Songs collection dropped.")
