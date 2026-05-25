"""Diagnostic: prints Pinecone index stats and a sample query result.

Run:
    .venv/bin/python scripts/debug_pinecone.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.persistence.vector.pinecone_repository import PineconeRepository


def main() -> None:
    repo = PineconeRepository(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
    )
    stats = repo._index.describe_index_stats()
    print("=== INDEX STATS ===")
    print(stats)
    print()

    # Sample query with a zero vector — should return ANY vectors that exist
    # regardless of similarity. If this is empty, the index is empty.
    print("=== SAMPLE QUERY (zero vector, top_k=5) ===")
    result = repo._index.query(vector=[0.0] * 384, top_k=5, include_metadata=True)
    print("raw result type:", type(result).__name__)
    print("raw result:", result)
    print()
    print("matches via repo.query():")
    for m in repo.query([0.0] * 384, top_k=5):
        print(f"  {m.chunk_id}  score={m.score:.4f}  meta keys={list(m.metadata.keys())}")


if __name__ == "__main__":
    main()
