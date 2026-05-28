import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import settings
from pinecone import Pinecone

pc = Pinecone(api_key=settings.pinecone_api_key)
index = pc.Index(settings.pinecone_index_name)

unique_songs = set()
total_vectors = 0

for ids_batch in index.list(limit=100):
    for item in ids_batch:
        song_id = item.id.rsplit("_", 1)[0]
        unique_songs.add(song_id)
    total_vectors += len(ids_batch)
    print(f"\rVectors scanned: {total_vectors} | Unique songs: {len(unique_songs)}", end="", flush=True)

print()

print(f"Total vectors : {total_vectors}")
print(f"Unique songs  : {len(unique_songs)}")
