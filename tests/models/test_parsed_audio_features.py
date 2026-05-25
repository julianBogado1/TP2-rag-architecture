import pytest
from pydantic import ValidationError
from app.models.parsed_audio_features import ParsedAudioFeatures


def test_construct_with_all_fields():
    p = ParsedAudioFeatures(
        valence=0.8, energy=0.7, danceability=0.6,
        acousticness=0.1, instrumentalness=0.0, tempo_norm=0.5,
    )
    assert p.valence == 0.8
    assert p.tempo_norm == 0.5


def test_missing_field_raises():
    with pytest.raises(ValidationError):
        ParsedAudioFeatures(valence=0.8, energy=0.7, danceability=0.6,
                            acousticness=0.1, instrumentalness=0.0)
