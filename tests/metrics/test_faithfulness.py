from datetime import date
from app.metrics.faithfulness import faithfulness_score, _build_context, _match_song
from app.core.exceptions import LLMProviderError
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.ranked_song import TopRecommendedSong, ScoreBreakdown
from app.models.song_candidate import SongMetadata, AudioFeatures


def _top_song(track="t", artist="a"):
    return TopRecommendedSong(
        song_id="1", track_name=track, artist_name=artist, score_total=0.5,
        score_breakdown=ScoreBreakdown(score_lyrics=0.5, score_audio=0.0, score_profile=0.0,
                                       score_popularity=0.0, score_recency=0.0),
        evidence_chunks=["lyric evidence"],
        metadata=SongMetadata(track_name=track, artist_name=artist, genres=["pop"],
                              popularity=50, release_date=date(2024, 1, 1), audio_features=None),
    )


def _rec(track="t", artist="a"):
    return SongRecommendation(rank=1, track_name=track, artist_name=artist, explanation="grounded claim",
                              matched_mood=["happy"], matched_audio_features=["valence"])


class _Judgment:
    def __init__(self, supported):
        self.supported = supported


class _FakeLLM:
    def __init__(self, behavior):
        self.behavior = behavior

    def parse_structured(self, *args, **kwargs):
        if self.behavior == "error":
            raise LLMProviderError("boom")
        if self.behavior == "none":
            return None
        return _Judgment(True)


def test_supported_explanation_scores_one():
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    assert faithfulness_score(resp, [_top_song()], _FakeLLM("supported")) == 1.0


def test_all_judge_errors_returns_zero_not_penalized():
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    assert faithfulness_score(resp, [_top_song()], _FakeLLM("error")) == 0.0


class _MixedLLM:
    def __init__(self):
        self.calls = 0

    def parse_structured(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise LLMProviderError("boom")
        return _Judgment(True)


def test_failed_judge_excluded_from_denominator():
    # one judge errors, one is supported -> 1 supported / 1 judged = 1.0
    # (old behaviour wrongly gave 1/2 = 0.5 by counting the error as unsupported)
    resp = RecommendationResponse(message="m", recommendations=[_rec(track="t1"), _rec(track="t2")])
    top = [_top_song(track="t1"), _top_song(track="t2")]
    assert faithfulness_score(resp, top, _MixedLLM()) == 1.0


def _audio_song(track="t", artist="a"):
    return TopRecommendedSong(
        song_id="1", track_name=track, artist_name=artist, score_total=0.5,
        score_breakdown=ScoreBreakdown(score_lyrics=0.5, score_audio=0.0, score_profile=0.0,
                                       score_popularity=0.0, score_recency=0.0),
        evidence_chunks=["la la la"],
        metadata=SongMetadata(track_name=track, artist_name=artist, genres=["rock"],
                              popularity=50, release_date=date(2024, 1, 1),
                              audio_features=AudioFeatures(valence=0.8, energy=0.9, danceability=0.7,
                                                           acousticness=0.1, instrumentalness=0.0,
                                                           tempo_norm=0.6)),
    )


def test_rank_fallback_matches_when_name_differs():
    # LLM rephrased the track name; key lookup misses, rank position recovers it.
    rec = SongRecommendation(rank=1, track_name="Totally Different Title", artist_name="Someone Else",
                             explanation="x", matched_mood=[], matched_audio_features=[])
    song = _audio_song(track="t", artist="a")
    assert _match_song(rec, {("t", "a"): song}, [song]) is song


def test_context_includes_structured_facts():
    rec = _rec()
    ctx = _build_context(_audio_song(), rec)
    assert "Lyrics:" in ctx
    assert "Genres: rock" in ctx
    assert "energy=0.90" in ctx          # audio facts now visible to the judge
    assert "Matched moods: happy" in ctx
