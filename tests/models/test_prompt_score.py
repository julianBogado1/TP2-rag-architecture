import pytest
from app.models.prompt_score import PromptScore, PromptAudioFeatures


def _audio():
    return PromptAudioFeatures(valence=0.5, energy=0.5, danceability=0.5,
                                acousticness=0.5, instrumentalness=0.5, tempo_norm=0.5)


def _base_kwargs(**overrides):
    base = dict(
        happy=0.5, sad=0.0, energetic=0.5, calm=0.0, nostalgic=0.0,
        romantic=0.0, assertive=0.0, deep=0.0, playful=0.0,
        wants_recent_songs=0.0, wants_popular_songs=0.0, wants_obscure_songs=0.0,
        wants_lyrics_focus=0.0, wants_mood_focus=0.0,
        semantic_query="q", audio_features=_audio(),
    )
    return base | overrides


@pytest.mark.parametrize("raw,expected", [
    ("es", "es"),
    ("ES", "es"),
    (" en ", "en"),
    ("en", "en"),
    ("Pt", "pt"),
    (None, None),
])
def test_preferred_language_normalises_valid_iso(raw, expected):
    score = PromptScore(**_base_kwargs(preferred_language=raw))
    assert score.preferred_language == expected


@pytest.mark.parametrize("bad", [
    "Spanish",
    "english",
    "spa",
    "",
    "  ",
    "es-AR",
    "e1",
    123,
])
def test_preferred_language_invalid_format_becomes_none(bad):
    score = PromptScore(**_base_kwargs(preferred_language=bad))
    assert score.preferred_language is None
