from enum import StrEnum


class Genre(StrEnum):
    """Canonical genre vocabulary — mirrors the `tag` values present in the songs collection."""
    COUNTRY = "country"
    MISC    = "misc"
    POP     = "pop"
    RAP     = "rap"
    RB      = "rb"
    ROCK    = "rock"
