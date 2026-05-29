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


class _FakeClaim:
    def __init__(self, supported):
        self.supported = supported


class _FakeJudgment:
    """Mimics the production _Judgment: a list of per-claim verdicts."""
    def __init__(self, *supported_flags):
        self.claims = [_FakeClaim(s) for s in supported_flags]


class _FakeLLM:
    """Returns a queued _FakeJudgment per call, or raises on the first call for infra cases."""
    def __init__(self, *judgments, raise_first=False):
        self.judgments = list(judgments)
        self.raise_first = raise_first
        self.calls = 0

    def parse_structured(self, *args, **kwargs):
        self.calls += 1
        if self.raise_first and self.calls == 1:
            raise LLMProviderError("boom")
        return self.judgments.pop(0)


def test_all_claims_supported_scores_one():
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    llm = _FakeLLM(_FakeJudgment(True, True))
    assert faithfulness_score(resp, [_top_song()], llm) == 1.0


def test_partial_claims_scored_as_fraction():
    # 2 of 3 claims grounded -> 2/3 (old all-or-nothing behaviour gave 0.0)
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    llm = _FakeLLM(_FakeJudgment(True, True, False))
    assert abs(faithfulness_score(resp, [_top_song()], llm) - 2 / 3) < 1e-9


def test_micro_average_pools_claims_across_recommendations():
    # rec1 -> 2/2 supported, rec2 -> 1/2 supported  => 3 supported / 4 total = 0.75
    resp = RecommendationResponse(message="m", recommendations=[_rec(track="t1"), _rec(track="t2")])
    top = [_top_song(track="t1"), _top_song(track="t2")]
    llm = _FakeLLM(_FakeJudgment(True, True), _FakeJudgment(True, False))
    assert faithfulness_score(resp, top, llm) == 0.75


def test_all_judge_errors_returns_zero_not_penalized():
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    llm = _FakeLLM(raise_first=True)
    assert faithfulness_score(resp, [_top_song()], llm) == 0.0


def test_failed_judge_excluded_from_denominator():
    # one rec's judge errors, the other returns 1/1 supported -> 1 supported / 1 total = 1.0
    # (the errored rec contributes nothing to numerator or denominator)
    resp = RecommendationResponse(message="m", recommendations=[_rec(track="t1"), _rec(track="t2")])
    top = [_top_song(track="t1"), _top_song(track="t2")]
    llm = _FakeLLM(_FakeJudgment(True), raise_first=True)
    assert faithfulness_score(resp, top, llm) == 1.0


def test_empty_claim_list_contributes_nothing():
    # judge extracts no claims from the only rec -> no claims judged -> 0.0
    resp = RecommendationResponse(message="m", recommendations=[_rec()])
    llm = _FakeLLM(_FakeJudgment())
    assert faithfulness_score(resp, [_top_song()], llm) == 0.0


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
