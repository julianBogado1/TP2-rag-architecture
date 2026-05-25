from datetime import date
from app.models.song_candidate import AudioFeatures, SongMetadata
from app.models.ranked_song import RankedSongCandidate, ScoreBreakdown
from app.services.retrieval.top_n_selector_service import TopNSelectorService


def _ranked(song_id, score, artist="X", title=None):
    return RankedSongCandidate(
        song_id=song_id, track_name=title or f"t{song_id}", artist_name=artist,
        score_total=score,
        score_breakdown=ScoreBreakdown(score_lyrics=score, score_audio=0, score_profile=0,
                                        score_popularity=0, score_recency=0),
        evidence_chunks=[],
        metadata=SongMetadata(
            track_name=title or f"t{song_id}", artist_name=artist, genres=["pop"],
            popularity=50, release_date=date(2024, 1, 1),
            audio_features=AudioFeatures(valence=0.5, energy=0.5, danceability=0.5,
                                          acousticness=0.5, instrumentalness=0.0, tempo_norm=0.5),
        ),
    )


def test_sorts_descending_by_score():
    svc = TopNSelectorService(max_per_artist=10)
    out = svc.select([_ranked("1", 0.2), _ranked("2", 0.8), _ranked("3", 0.5)], top_n=3)
    assert [r.song_id for r in out] == ["2", "3", "1"]


def test_caps_per_artist():
    svc = TopNSelectorService(max_per_artist=2)
    candidates = [
        _ranked("1", 0.9, artist="A"),
        _ranked("2", 0.8, artist="A"),
        _ranked("3", 0.7, artist="A"),
        _ranked("4", 0.6, artist="B"),
    ]
    out = svc.select(candidates, top_n=10)
    assert [r.song_id for r in out] == ["1", "2", "4"]


def test_dedup_by_normalized_title():
    svc = TopNSelectorService(max_per_artist=10)
    candidates = [
        _ranked("1", 0.9, title="Friday Lights"),
        _ranked("2", 0.8, title="friday lights"),
        _ranked("3", 0.7, title="Other"),
    ]
    out = svc.select(candidates, top_n=10)
    assert [r.song_id for r in out] == ["1", "3"]
