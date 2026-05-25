#!/usr/bin/env python3
"""Delete all vectors from the Pinecone index.

Usage:
    python scripts/clear_pinecone.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values
from app.persistence.vector.pinecone_repository import PineconeRepository

config = dotenv_values(ROOT / ".env")

PINECONE_API_KEY = config["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = config["PINECONE_INDEX_NAME"]


def main() -> None:
    print(f"Clearing all vectors from index '{PINECONE_INDEX_NAME}'...")
    repo = PineconeRepository(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME)
    repo.clear()
    print("Done. Index is empty.")


if __name__ == "__main__":
    main()
