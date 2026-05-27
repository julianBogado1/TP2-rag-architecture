from app.services.song_matcher import (
    SongMatcher,
    normalize_artist,
    normalize_title,
    split_artists,
)


def _spotify_row(track_name: str, artists: str, popularity: int = 50) -> dict:
    return {
        "track_id": "sp1",
        "artists": artists,
        "track_name": track_name,
        "popularity": popularity,
        "track_genre": "pop",
    }


# --- normalize_title -------------------------------------------------------


def test_normalize_title_lowercases_and_strips_accents():
    assert normalize_title("Déjà Vu") == "deja vu"


def test_normalize_title_drops_punctuation_and_collapses_space():
    assert normalize_title("  Hello,   World!! ") == "hello world"


def test_normalize_title_strips_feat_parenthetical():
    assert normalize_title("Get Lucky (feat. Pharrell Williams)") == "get lucky"


def test_normalize_title_strips_trailing_feat():
    assert normalize_title("Stay feat. Justin Bieber") == "stay"


def test_normalize_title_strips_cosmetic_suffix():
    assert normalize_title("Hotel California - Remastered 2011") == "hotel california"
    assert normalize_title("Yesterday - Mono") == "yesterday"


def test_normalize_title_preserves_live_and_remix():
    assert normalize_title("Bohemian Rhapsody - Live") == "bohemian rhapsody live"
    assert normalize_title("One More Time - Remix") == "one more time remix"
    assert normalize_title("Layla - Acoustic") == "layla acoustic"


# --- normalize_artist / split_artists -------------------------------------


def test_normalize_artist_folds_accents_and_case():
    assert normalize_artist("Beyoncé") == "beyonce"


def test_split_artists_on_semicolon():
    assert split_artists("Daft Punk;Pharrell Williams") == {"daft punk", "pharrell williams"}


def test_split_artists_on_ampersand_and_feat():
    assert split_artists("Calvin Harris & Dua Lipa") == {"calvin harris", "dua lipa"}
    assert split_artists("Eminem feat. Rihanna") == {"eminem", "rihanna"}


# --- SongMatcher -----------------------------------------------------------


def test_lookup_exact_normalized_match():
    matcher = SongMatcher()
    matcher.build_index([_spotify_row("My Song", "My Artist")])
    assert matcher.lookup({"title": "my song", "artist": "my artist"})["track_id"] == "sp1"


def test_lookup_miss_returns_none():
    matcher = SongMatcher()
    matcher.build_index([_spotify_row("My Song", "My Artist")])
    assert matcher.lookup({"title": "Other Song", "artist": "My Artist"}) is None


def test_lookup_matches_any_one_of_multiple_spotify_artists():
    matcher = SongMatcher()
    matcher.build_index([_spotify_row("Get Lucky", "Daft Punk;Pharrell Williams")])
    assert matcher.lookup({"title": "Get Lucky", "artist": "Pharrell Williams"}) is not None
    assert matcher.lookup({"title": "Get Lucky", "artist": "Daft Punk"}) is not None


def test_lookup_matches_genius_collab_via_any_artist():
    matcher = SongMatcher()
    matcher.build_index([_spotify_row("Closer", "The Chainsmokers")])
    assert matcher.lookup({"title": "Closer", "artist": "The Chainsmokers & Halsey"}) is not None


def test_lookup_matches_across_track_decorations():
    matcher = SongMatcher()
    matcher.build_index([_spotify_row("Hotel California - Remastered", "Eagles")])
    assert matcher.lookup({"title": "Hotel California", "artist": "Eagles"}) is not None


def test_collision_keeps_highest_popularity():
    matcher = SongMatcher()
    low = _spotify_row("Song", "Artist", popularity=10)
    low["track_id"] = "low"
    high = _spotify_row("Song", "Artist", popularity=90)
    high["track_id"] = "high"
    matcher.build_index([low, high])
    assert matcher.lookup({"title": "Song", "artist": "Artist"})["track_id"] == "high"
