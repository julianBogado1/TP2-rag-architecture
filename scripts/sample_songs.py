#!/usr/bin/env python3
"""Sample 200k usable song_ids from MongoDB.

Usage:
    .venv/bin/python scripts/sample_songs.py [--strategy random|balanced] [--count 200000]

Outputs: scripts/sample_200k_ids.json
"""
import sys
import json
import argparse
from pathlib import Path
from math import ceil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from pymongo import MongoClient

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["random", "balanced"], default="balanced")
    p.add_argument("--count", type=int, default=200_000)
    return p.parse_args()


def sample_random(col, n: int) -> list[int]:
    pipeline = [
        {"$match": {"lyrics": {"$nin": [None, ""]}, "tag": {"$nin": [None, ""]}}},
        {"$sample": {"size": n}},
        {"$project": {"_id": 0, "song_id": 1}},
    ]
    return [doc["song_id"] for doc in col.aggregate(pipeline)]


def sample_balanced(col, n: int) -> list[int]:
    genres = [
        g["_id"] for g in col.aggregate([
            {"$match": {"lyrics": {"$nin": [None, ""]}, "tag": {"$nin": [None, ""]}}},
            {"$group": {"_id": "$tag", "count": {"$sum": 1}}},
            {"$match": {"_id": {"$ne": None}}},
        ])
    ]
    per_genre = ceil(n / len(genres))
    print(f"  {len(genres)} genres — {per_genre} songs/genre target")

    ids: list[int] = []
    for genre in genres:
        pipeline = [
            {"$match": {"tag": genre, "lyrics": {"$nin": [None, ""]}}},
            {"$sample": {"size": per_genre}},
            {"$project": {"_id": 0, "song_id": 1}},
        ]
        batch = [doc["song_id"] for doc in col.aggregate(pipeline)]
        ids.extend(batch)

    # trim to exactly n if over
    return ids[:n]


def main():
    args = parse_args()
    config = dotenv_values(ROOT / ".env")
    for key in ("MONGO_URI", "MONGO_DB_NAME"):
        if key not in config:
            print(f"Error: {key} missing from .env", file=sys.stderr)
            sys.exit(1)

    client = MongoClient(config["MONGO_URI"])
    col = client[config["MONGO_DB_NAME"]]["songs"]

    print(f"Sampling {args.count:,} songs with strategy='{args.strategy}'...")
    if args.strategy == "random":
        ids = sample_random(col, args.count)
    else:
        ids = sample_balanced(col, args.count)

    client.close()

    out_path = ROOT / "scripts" / "sample_200k_ids.json"
    with open(out_path, "w") as f:
        json.dump(ids, f)

    print(f"Saved {len(ids):,} song_ids to {out_path}")


if __name__ == "__main__":
    main()
