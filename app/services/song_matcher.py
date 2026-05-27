import re
import unicodedata
from typing import Iterable

# Suffixes after " - " that name the *same* recording and are safe to strip.
# live / remix / acoustic / instrumental are deliberately absent: those are
# different recordings with different audio features and must NOT be merged.
COSMETIC_SUFFIXES = (
    "remaster", "remastered", "radio edit", "single version", "album version",
    "mono", "stereo", "bonus track", "deluxe", "anniversary edition",
)

# Token used to separate distinct artists packed into one string.
_FEAT_PAREN = re.compile(r"[\(\[]\s*feat[^)\]]*[\)\]]", re.IGNORECASE)
_FEAT_TRAILING = re.compile(r"\s+feat\.?\s+.*$", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ARTIST_SPLIT = re.compile(
    r"\s*(?:[;&,]|\bfeat\.?\b|\bfeaturing\b|\bwith\b|\bvs\.?\b| x )\s*",
    re.IGNORECASE,
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _strip_cosmetic_suffix(s: str) -> str:
    head, sep, tail = s.rpartition(" - ")
    if sep and any(kw in tail.lower() for kw in COSMETIC_SUFFIXES):
        return head
    return s


def normalize_title(title: str) -> str:
    """Accent/case-fold a title and drop feat segments + cosmetic suffixes."""
    s = _strip_accents(title)
    s = _FEAT_PAREN.sub(" ", s)
    s = _FEAT_TRAILING.sub("", s)
    s = _strip_cosmetic_suffix(s)
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def normalize_artist(artist: str) -> str:
    s = _strip_accents(artist).lower()
    return _NON_ALNUM.sub(" ", s).strip()


def split_artists(artists: str) -> set[str]:
    parts = (normalize_artist(p) for p in _ARTIST_SPLIT.split(artists))
    return {p for p in parts if p}


class SongMatcher:
    """Deterministic multi-key join from Genius rows to Spotify track rows."""

    def __init__(self) -> None:
        self._index: dict[tuple[str, str], dict] = {}

    def build_index(self, rows: Iterable[dict]) -> None:
        for row in rows:
            if row["track_name"] is None or row["artists"] is None:
                continue
            title = normalize_title(row["track_name"])
            if not title:
                continue
            for artist in split_artists(row["artists"]):
                key = (title, artist)
                existing = self._index.get(key)
                # Collision -> keep the more popular row.
                # Alternative: keep first-seen (simpler, but order-dependent).
                if existing is None or (row.get("popularity") or 0) > (existing.get("popularity") or 0):
                    self._index[key] = row

    def lookup(self, genius_row: dict) -> dict | None:
        title = normalize_title(genius_row["title"])
        if not title:
            return None
        for artist in split_artists(genius_row["artist"]):
            match = self._index.get((title, artist))
            if match is not None:
                return match
        return None
