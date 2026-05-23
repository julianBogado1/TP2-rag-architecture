import pytest
from pydantic import ValidationError
from app.models.song_document import SongDocument


def _genius_fields() -> dict:
    return dict(
        song_id=1,
        title="Stronger",
        tag="rap",
        artist="Kanye West",
        year=2007,
        views=500000,
        lyrics="Work it harder, make it better",
    )


def test_song_document_genius_only_has_null_spotify_fields():
    song = SongDocument(**_genius_fields())
    assert song.song_id == 1
    assert song.title == "Stronger"
    assert song.track_id is None
    assert song.popularity is None
    assert song.danceability is None
    assert song.energy is None
    assert song.valence is None
    assert song.tempo is None
    assert song.track_genre is None


def test_song_document_with_spotify_fields():
    song = SongDocument(
        **_genius_fields(),
        track_id="spotify123",
        album_name="Graduation",
        popularity=90,
        duration_ms=312000,
        explicit=False,
        danceability=0.75,
        energy=0.85,
        key=5,
        loudness=-4.5,
        mode=1,
        speechiness=0.08,
        acousticness=0.05,
        instrumentalness=0.0,
        liveness=0.12,
        valence=0.65,
        tempo=104.0,
        time_signature=4,
        track_genre="rap",
    )
    assert song.track_id == "spotify123"
    assert song.popularity == 90
    assert song.danceability == 0.75
    assert song.track_genre == "rap"


def test_song_document_missing_required_genius_field_raises():
    with pytest.raises(ValidationError):
        SongDocument(
            song_id=1,
            tag="rap",
            artist="Kanye West",
            year=2007,
            views=500000,
            lyrics="some lyrics",
            # title missing
        )
