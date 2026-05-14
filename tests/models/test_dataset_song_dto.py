import pytest
from pydantic import ValidationError
from app.models.dataset_song_dto import DatasetSongDTO


def test_dataset_song_required_fields_only():
    song = DatasetSongDTO(
        song_id=1,
        title="Test Song",
        tag="rap",
        artist="Test Artist",
        year=2020,
        views=1000,
        lyrics="Some lyrics here",
    )
    assert song.song_id == 1
    assert song.title == "Test Song"
    assert song.features is None
    assert song.language_cld3 is None
    assert song.language_ft is None
    assert song.language is None


def test_dataset_song_with_all_fields():
    song = DatasetSongDTO(
        song_id=38,
        title="Barry Bonds",
        tag="rap",
        artist="Kanye West",
        year=2007,
        views=280626,
        features='{"Lil Wayne"}',
        lyrics="[Verse 1: Kanye West]...",
        language_cld3="en",
        language_ft="en",
        language="en",
    )
    assert song.song_id == 38
    assert song.features == '{"Lil Wayne"}'
    assert song.language == "en"


def test_dataset_song_missing_required_field_raises():
    with pytest.raises(ValidationError):
        DatasetSongDTO(
            song_id=1,
            tag="rap",
            artist="Artist",
            year=2020,
            views=0,
            lyrics="lyrics",
            # title missing
        )
