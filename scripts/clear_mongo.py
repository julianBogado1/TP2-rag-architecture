#!/usr/bin/env python3
"""Drop the songs collection from MongoDB. Destructive — requires confirmation.

Usage:
    python scripts/clear_mongo.py --yes        # non-interactive
    python scripts/clear_mongo.py              # prompts for confirmation
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from pymongo import MongoClient
from app.persistence.mongo.song_repository import SongRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop the songs collection (destructive).")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    args = parser.parse_args()

    if not args.yes:
        if input("This DROPS the entire songs collection. Type DROP to confirm: ").strip() != "DROP":
            print("Aborted.")
            return

    config = dotenv_values(ROOT / ".env")
    client = MongoClient(config["MONGO_URI"])
    repo = SongRepository(client[config["MONGO_DB_NAME"]])
    repo.drop_collection()
    client.close()
    print("Songs collection dropped.")


if __name__ == "__main__":
    main()
