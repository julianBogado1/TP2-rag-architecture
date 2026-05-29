#!/usr/bin/env python3
"""Fetch every vector from the Pinecone index, reduce to 2D with PCA, and scatter-plot it.

Points are coloured by their primary genre.

Run:
    .venv/bin/python scripts/plot_embeddings_pca.py [output.png]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from pinecone import Pinecone
from sklearn.decomposition import PCA

from app.core.config import settings

# Pinecone caps a single fetch at 1000 ids; keep batches under that.
_PINECONE_FETCH_BATCH = 1000
_PCA_COMPONENTS = 2
# Genres beyond this many (by frequency) collapse into one "other" colour so the
# legend stays readable; the long tail of rare genres would otherwise be noise.
_MAX_LEGEND_GENRES = 12
_POINT_SIZE = 8
_POINT_ALPHA = 0.6
_FIGURE_SIZE = (12, 9)
_DEFAULT_OUTPUT = ROOT / "embeddings_pca.png"
_UNKNOWN_GENRE = "unknown"


def fetch_embeddings() -> tuple[np.ndarray, list[str]]:
    """All embeddings + primary genre from the active Pinecone index."""
    index = Pinecone(api_key=settings.pinecone_api_key).Index(settings.pinecone_index_name)
    ids = [vid for batch in index.list(limit=100) for vid in batch]

    vectors: list[list[float]] = []
    genres: list[str] = []
    for start in range(0, len(ids), _PINECONE_FETCH_BATCH):
        window = ids[start : start + _PINECONE_FETCH_BATCH]
        fetched = index.fetch(ids=window).vectors
        for vid in window:
            record = fetched.get(vid)
            if record is None:
                continue
            vectors.append(record.values)
            genres.append(_primary_genre(record.metadata or {}))
        print(f"\rFetched {len(vectors)}/{len(ids)} vectors", end="", flush=True)
    print()
    return np.asarray(vectors, dtype=np.float32), genres


def _primary_genre(metadata: dict) -> str:
    genre_list = metadata.get("genres") or []
    return genre_list[0] if genre_list else _UNKNOWN_GENRE


def _legend_labels(genres: list[str]) -> list[str]:
    """Keep the _MAX_LEGEND_GENRES most common genres; fold the rest into "other"."""
    counts: dict[str, int] = {}
    for genre in genres:
        counts[genre] = counts.get(genre, 0) + 1
    # Most frequent first so the colour budget goes to the genres actually present.
    # Alternative: alphabetical — but then rare genres could crowd out common ones.
    kept = {g for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:_MAX_LEGEND_GENRES]}
    return [g if g in kept else "other" for g in genres]


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUTPUT

    print(f"Fetching embeddings from Pinecone index '{settings.pinecone_index_name}'...")
    embeddings, genres = fetch_embeddings()
    if len(embeddings) == 0:
        print("No vectors found — is the index populated?")
        return
    print(f"Fetched {len(embeddings)} vectors of dim {embeddings.shape[1]}.")

    pca = PCA(n_components=_PCA_COMPONENTS)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_

    labels = _legend_labels(genres)
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=_FIGURE_SIZE)
    for label, color in zip(unique_labels, colors):
        mask = np.array([lbl == label for lbl in labels])
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=_POINT_SIZE, alpha=_POINT_ALPHA, color=color, label=label,
        )
    plt.xlabel(f"PC1 ({explained[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained[1]:.1%} variance)")
    plt.title(f"PCA of {len(embeddings)} embeddings — {settings.pinecone_index_name}")
    plt.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path} "
          f"(PC1+PC2 explain {explained[:2].sum():.1%} of variance).")


if __name__ == "__main__":
    main()
