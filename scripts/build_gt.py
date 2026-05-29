#!/usr/bin/env python3
"""Build ground truth test cases from MongoDB.

Two families of test cases (each tagged with a "kind"):

  - kind="audio":  relevance = a soft single audio-feature threshold. The `$ne: None`
    clause means these GT sets contain only audio-bearing songs, so audio is evaluated
    only against songs that have audio features.
  - kind="genre":  relevance = `tag` membership (one case per canonical Genre). Covers
    every song, audio and no-audio alike ("ask for pop -> get pop").

Every GT set is intersected with the indexed ids file (default
scripts/sample_200k_ids.json) when present, so the reported |GT| reflects only songs
that could actually be retrieved.

Writes data/gt_test_cases.json and upserts the neutral eval user.

Usage:
    python scripts/build_gt.py [--ids-file scripts/sample_200k_ids.json]
"""
import sys
import json
import argparse
from pathlib import Path
from pymongo import MongoClient
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from app.core.config import settings
from app.models.genre import Genre
from app.models.user_profile import UserProfileData
from app.persistence.mongo.user_profile_repository import UserProfileRepository

DEFAULT_IDS_FILE = ROOT / "scripts" / "sample_200k_ids.json"

# kind="audio" — soft single thresholds (larger, more representative GT sets than the
# old strict thresholds). The $ne: None query clause keeps these audio-bearing only.
AUDIO_CASES = [
    {"label": "Happy",        "prompt": "I want happy upbeat songs",            "target_feature": "valence",          "operator": "gt", "threshold": 0.6},
    {"label": "Sad",          "prompt": "Give me sad melancholic songs",        "target_feature": "valence",          "operator": "lt", "threshold": 0.4},
    {"label": "Energetic",    "prompt": "I need high energy songs for working out", "target_feature": "energy",       "operator": "gt", "threshold": 0.6},
    {"label": "Calm",         "prompt": "Something relaxing and calm to focus",  "target_feature": "energy",           "operator": "lt", "threshold": 0.4},
    {"label": "Danceable",    "prompt": "Songs to dance at a party tonight",     "target_feature": "danceability",     "operator": "gt", "threshold": 0.6},
    {"label": "Acoustic",     "prompt": "Acoustic songs, no electric instruments", "target_feature": "acousticness",  "operator": "gt", "threshold": 0.6},
    {"label": "Instrumental", "prompt": "Instrumental music, no vocals please",  "target_feature": "instrumentalness", "operator": "gt", "threshold": 0.5},
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


def genre_case(genre: str) -> dict:
    """Build the (DB-free) template for one genre test case."""
    return {
        "kind": "genre",
        "label": f"Genre:{genre}",
        "prompt": f"I want {genre} songs",
        "user_id": EVAL_USER["user_id"],
        "tag": genre,
    }


def audio_case(tmpl: dict) -> dict:
    """Build the (DB-free) template for one audio test case."""
    return {
        "kind": "audio",
        "label": tmpl["label"],
        "prompt": tmpl["prompt"],
        "user_id": EVAL_USER["user_id"],
        "target_feature": tmpl["target_feature"],
        "operator": tmpl["operator"],
        "threshold": tmpl["threshold"],
    }


def restrict_to_indexed(song_ids: list[str], indexed: set[str] | None) -> list[str]:
    """Keep only ids that were actually indexed. No-op when `indexed` is None."""
    if indexed is None:
        return song_ids
    return [sid for sid in song_ids if sid in indexed]


def load_indexed_ids(ids_path: Path) -> set[str] | None:
    """Load the indexed song_ids as a set of strings, or None if the file is absent."""
    if not ids_path.exists():
        print(f"WARNING: ids file {ids_path} not found — GT will NOT be restricted to the indexed corpus.")
        return None
    raw = json.loads(ids_path.read_text())
    return {str(x) for x in raw}


def _audio_query(tmpl: dict) -> dict:
    op = "$gt" if tmpl["operator"] == "gt" else "$lt"
    return {tmpl["target_feature"]: {op: tmpl["threshold"], "$ne": None}}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ids-file", default=str(DEFAULT_IDS_FILE),
                   help="JSON list of indexed song_ids; GT is intersected with it when present")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    indexed = load_indexed_ids(Path(args.ids_file))

    mongo = MongoClient(settings.mongo_uri)
    try:
        db = mongo[settings.mongo_db_name]
        songs_col = db["songs"]
        user_repo = UserProfileRepository(db)

        user_repo.upsert(UserProfileData(**EVAL_USER))
        print(f"Upserted eval user: {EVAL_USER['user_id']}\n")

        output = []

        for tmpl in AUDIO_CASES:
            entry = audio_case(tmpl)
            ids = [str(d["song_id"]) for d in songs_col.find(_audio_query(tmpl), {"song_id": 1, "_id": 0})]
            entry["gt_song_ids"] = restrict_to_indexed(ids, indexed)
            output.append(entry)
            print(f"  [audio] {entry['label']:14s} ({tmpl['target_feature']} {tmpl['operator']} {tmpl['threshold']}): {len(entry['gt_song_ids'])} songs")

        for genre in (g.value for g in Genre):
            entry = genre_case(genre)
            ids = [str(d["song_id"]) for d in songs_col.find({"tag": genre}, {"song_id": 1, "_id": 0})]
            entry["gt_song_ids"] = restrict_to_indexed(ids, indexed)
            output.append(entry)
            print(f"  [genre] {entry['label']:14s} (tag = {genre}): {len(entry['gt_song_ids'])} songs")

        out_path = ROOT / "data" / "gt_test_cases.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"\nWritten {len(output)} test cases to {out_path}")
    finally:
        mongo.close()


if __name__ == "__main__":
    main()
