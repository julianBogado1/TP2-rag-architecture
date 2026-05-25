#!/usr/bin/env python3
"""Build ground truth test cases from MongoDB audio features.

Queries the songs collection for each mood category and writes
data/gt_test_cases.json. Also upserts the neutral eval user.

Usage:
    python scripts/build_gt.py
"""
import json
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings

TEST_CASES_TEMPLATE = [
    {
        "label": "Happy",
        "prompt": "I want happy upbeat songs",
        "target_feature": "valence",
        "operator": "gt",
        "threshold": 0.7,
    },
    {
        "label": "Sad",
        "prompt": "Give me sad melancholic songs",
        "target_feature": "valence",
        "operator": "lt",
        "threshold": 0.3,
    },
    {
        "label": "Energetic",
        "prompt": "I need high energy songs for working out",
        "target_feature": "energy",
        "operator": "gt",
        "threshold": 0.8,
    },
    {
        "label": "Calm",
        "prompt": "Something relaxing and calm to focus",
        "target_feature": "energy",
        "operator": "lt",
        "threshold": 0.3,
    },
    {
        "label": "Danceable",
        "prompt": "Songs to dance at a party tonight",
        "target_feature": "danceability",
        "operator": "gt",
        "threshold": 0.7,
    },
    {
        "label": "Acoustic",
        "prompt": "Acoustic songs, no electric instruments",
        "target_feature": "acousticness",
        "operator": "gt",
        "threshold": 0.7,
    },
    {
        "label": "Instrumental",
        "prompt": "Instrumental music, no vocals please",
        "target_feature": "instrumentalness",
        "operator": "gt",
        "threshold": 0.5,
    },
]

EVAL_USER = {
    "user_id": "eval_neutral_user",
    "favourite_genres": [],
    "favourite_artists": [],
    "favourite_songs": [],
    "preferred_language": "en",
    "disliked_genres": [],
    "disliked_artists": [],
    "listening_history": [],
}


def main() -> None:
    mongo = MongoClient(settings.mongo_uri)
    db = mongo[settings.mongo_db_name]
    songs_col = db["songs"]
    users_col = db["users"]

    users_col.update_one(
        {"user_id": EVAL_USER["user_id"]},
        {"$set": EVAL_USER},
        upsert=True,
    )
    print(f"Upserted eval user: {EVAL_USER['user_id']}")

    output = []
    for tmpl in TEST_CASES_TEMPLATE:
        feature = tmpl["target_feature"]
        op = "$gt" if tmpl["operator"] == "gt" else "$lt"
        threshold = tmpl["threshold"]

        query = {feature: {op: threshold, "$ne": None}}
        song_ids = [
            str(doc["song_id"])
            for doc in songs_col.find(query, {"song_id": 1, "_id": 0})
        ]

        entry = {
            "label": tmpl["label"],
            "prompt": tmpl["prompt"],
            "user_id": EVAL_USER["user_id"],
            "target_feature": feature,
            "operator": tmpl["operator"],
            "threshold": threshold,
            "gt_song_ids": song_ids,
        }
        output.append(entry)
        print(f"  {tmpl['label']:14s} ({feature} {tmpl['operator']} {threshold}): {len(song_ids)} songs")

    out_path = Path(__file__).parent.parent / "data" / "gt_test_cases.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWritten to {out_path}")

    mongo.close()


if __name__ == "__main__":
    main()
