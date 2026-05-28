#!/usr/bin/env python3
"""Audit MongoDB songs collection for data quality.

Usage:
    .venv/bin/python scripts/audit_mongo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from pymongo import MongoClient

config = dotenv_values(ROOT / ".env")
for key in ("MONGO_URI", "MONGO_DB_NAME"):
    if key not in config:
        print(f"Error: {key} missing from .env", file=sys.stderr)
        sys.exit(1)

client = MongoClient(config["MONGO_URI"])
db = client[config["MONGO_DB_NAME"]]
col = db["songs"]

# --- totals ---
total = col.count_documents({})
print(f"Total documents : {total:,}")

if total == 0:
    print("Warning: collection is empty.", file=sys.stderr)
    client.close()
    sys.exit(0)

# --- missing required fields ---
missing_lyrics    = col.count_documents({"$or": [{"lyrics": None}, {"lyrics": ""}]})
missing_title     = col.count_documents({"$or": [{"title": None},  {"title": ""}]})
missing_artist    = col.count_documents({"$or": [{"artist": None}, {"artist": ""}]})
missing_tag       = col.count_documents({"$or": [{"tag": None},    {"tag": ""}]})
has_spotify       = col.count_documents({"track_id": {"$ne": None}})

print(f"\n--- Field coverage ---")
print(f"  Missing lyrics  : {missing_lyrics:,}  ({missing_lyrics/total*100:.1f}%)")
print(f"  Missing title   : {missing_title:,}")
print(f"  Missing artist  : {missing_artist:,}")
print(f"  Missing tag     : {missing_tag:,}")
print(f"  With Spotify    : {has_spotify:,}  ({has_spotify/total*100:.1f}%)")

# --- genre distribution ---
print(f"\n--- Genre distribution (top 30) ---")
pipeline = [
    {"$group": {"_id": "$tag", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}},
    {"$limit": 30},
]
genres = list(col.aggregate(pipeline))
distinct_genres = col.distinct("tag")
print(f"  Distinct genres : {len(distinct_genres)}")
print(f"  {'Genre':<30} {'Count':>10}")
print(f"  {'-'*40}")
for g in genres:
    print(f"  {str(g['_id']):<30} {g['count']:>10,}")

# --- usable songs (have lyrics, title, tag) ---
usable = col.count_documents({
    "lyrics": {"$nin": [None, ""]},
    "title":  {"$nin": [None, ""]},
    "tag":    {"$nin": [None, ""]},
})
print(f"\n--- Summary ---")
print(f"  Usable songs (lyrics+title+tag present) : {usable:,}")

client.close()
