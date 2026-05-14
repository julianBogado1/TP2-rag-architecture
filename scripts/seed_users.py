#!/usr/bin/env python3
"""Seed 10 example user profiles into MongoDB."""

import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import dotenv_values
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent.parent
config = dotenv_values(ROOT / ".env")

MONGO_URI = config["MONGO_URI"]
MONGO_DB_NAME = config["MONGO_DB_NAME"]

USERS: list[dict] = [
    {
        "user_id": "user_001",
        "favourite_genres": ["rock", "alternative"],
        "favourite_artists": ["Radiohead", "Arctic Monkeys"],
        "favourite_songs": ["Creep", "Do I Wanna Know?", "Karma Police"],
        "preferred_language": "en",
        "disliked_genres": ["reggaeton"],
        "disliked_artists": [],
        "listening_history": [
            {"song": "Creep", "artist": "Radiohead", "listened_at": datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)},
            {"song": "Do I Wanna Know?", "artist": "Arctic Monkeys", "listened_at": datetime(2024, 3, 2, 14, 30, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_002",
        "favourite_genres": ["pop", "indie pop"],
        "favourite_artists": ["Billie Eilish", "Lorde"],
        "favourite_songs": ["bad guy", "Royals", "Ocean Eyes"],
        "preferred_language": "en",
        "disliked_genres": ["heavy metal"],
        "listening_history": [
            {"song": "bad guy", "artist": "Billie Eilish", "listened_at": datetime(2024, 3, 5, 9, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_003",
        "favourite_genres": ["hip-hop", "rap"],
        "favourite_artists": ["Kendrick Lamar", "J. Cole"],
        "favourite_songs": ["HUMBLE.", "No Role Modelz", "Alright"],
        "preferred_language": "en",
        "disliked_genres": ["country"],
        "disliked_artists": ["Nickelback"],
        "listening_history": [
            {"song": "HUMBLE.", "artist": "Kendrick Lamar", "listened_at": datetime(2024, 2, 20, 18, 0, tzinfo=timezone.utc)},
            {"song": "No Role Modelz", "artist": "J. Cole", "listened_at": datetime(2024, 2, 21, 20, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_004",
        "favourite_genres": ["electronic", "techno"],
        "favourite_artists": ["Daft Punk", "Aphex Twin"],
        "favourite_songs": ["Get Lucky", "Around the World", "Windowlicker"],
        "preferred_language": "fr",
        "listening_history": [
            {"song": "Get Lucky", "artist": "Daft Punk", "listened_at": datetime(2024, 1, 15, 23, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_005",
        "favourite_genres": ["latin", "reggaeton"],
        "favourite_artists": ["Bad Bunny", "J Balvin"],
        "favourite_songs": ["Tití Me Preguntó", "Mi Gente", "Dakiti"],
        "preferred_language": "es",
        "disliked_genres": ["classical"],
        "listening_history": [
            {"song": "Tití Me Preguntó", "artist": "Bad Bunny", "listened_at": datetime(2024, 3, 10, 16, 0, tzinfo=timezone.utc)},
            {"song": "Dakiti", "artist": "Bad Bunny", "listened_at": datetime(2024, 3, 11, 17, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_006",
        "favourite_genres": ["jazz", "soul"],
        "favourite_artists": ["Miles Davis", "John Coltrane"],
        "favourite_songs": ["So What", "A Love Supreme", "Kind of Blue"],
        "preferred_language": "en",
        "disliked_genres": ["trap"],
        "disliked_artists": [],
        "listening_history": [
            {"song": "So What", "artist": "Miles Davis", "listened_at": datetime(2024, 3, 8, 11, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_007",
        "favourite_genres": ["k-pop", "pop"],
        "favourite_artists": ["BTS", "BLACKPINK"],
        "favourite_songs": ["Dynamite", "Butter", "Pink Venom"],
        "preferred_language": "ko",
        "disliked_genres": ["death metal"],
        "listening_history": [
            {"song": "Dynamite", "artist": "BTS", "listened_at": datetime(2024, 3, 3, 8, 0, tzinfo=timezone.utc)},
            {"song": "Pink Venom", "artist": "BLACKPINK", "listened_at": datetime(2024, 3, 4, 12, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_008",
        "favourite_genres": ["classical", "ambient"],
        "favourite_artists": ["Johann Sebastian Bach", "Brian Eno"],
        "favourite_songs": ["Clair de Lune", "Music for Airports", "Goldberg Variations"],
        "preferred_language": "de",
        "disliked_genres": ["rap", "reggaeton"],
        "listening_history": [
            {"song": "Clair de Lune", "artist": "Claude Debussy", "listened_at": datetime(2024, 2, 28, 20, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_009",
        "favourite_genres": ["metal", "progressive rock"],
        "favourite_artists": ["Tool", "Mastodon"],
        "favourite_songs": ["Schism", "The Grudge", "Blood and Thunder"],
        "preferred_language": "en",
        "disliked_genres": ["pop", "k-pop"],
        "listening_history": [
            {"song": "Schism", "artist": "Tool", "listened_at": datetime(2024, 3, 6, 22, 0, tzinfo=timezone.utc)},
            {"song": "Blood and Thunder", "artist": "Mastodon", "listened_at": datetime(2024, 3, 7, 21, 0, tzinfo=timezone.utc)},
        ],
    },
    {
        "user_id": "user_010",
        "favourite_genres": ["r&b", "soul", "funk"],
        "favourite_artists": ["Frank Ocean", "Solange"],
        "favourite_songs": ["Nights", "Cranes in the Sky", "Pyramids"],
        "preferred_language": "en",
        "disliked_genres": [],
        "disliked_artists": [],
        "listening_history": [
            {"song": "Nights", "artist": "Frank Ocean", "listened_at": datetime(2024, 3, 9, 19, 0, tzinfo=timezone.utc)},
            {"song": "Cranes in the Sky", "artist": "Solange", "listened_at": datetime(2024, 3, 9, 20, 0, tzinfo=timezone.utc)},
        ],
    },
]


def main():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collection = db["user_profiles"]

    collection.delete_many({"user_id": {"$in": [u["user_id"] for u in USERS]}})
    result = collection.insert_many(USERS)
    print(f"Inserted {len(result.inserted_ids)} user profiles into '{MONGO_DB_NAME}.user_profiles'")
    client.close()


if __name__ == "__main__":
    main()
