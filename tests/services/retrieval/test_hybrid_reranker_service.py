from datetime import date
from math import isclose
from app.models.song_candidate import CandidateSong, AudioFeatures
from app.models.song_recommendation_request import (
    SongRecommendationRequest, MetadataFilters, RankingWeights,
)
from app.models.user_profile import UserProfileData
from app.models.prompt_score import PromptScore, PromptAudioFeatures
from app.services.retrieval.hybrid_reranker_service import HybridRerankerService


def _candidate(song_id="1", valence=0.85, popularity=80, year=2024,
               artist="X", genres=("pop",), lyrics_sim=0.8):
    return CandidateSong(
        song_id=song_id, track_name=f"t{song_id}", artist_name=artist,
        genres=list(genres), popularity=popularity, release_date=date(year, 1, 1),
        best_lyrics_chunks=["..."], best_lyrics_similarity=lyrics_sim,
        audio_features=AudioFeatures(valence=valence, energy=0.7, danceability=0.7,
                                      acousticness=0.1, instrumentalness=0.0, tempo_norm=0.5),
    )


def _request(profile=None, weights=None, audio=None):
    audio = audio or PromptAudioFeatures(valence=0.85, energy=0.7, danceability=0.7,
                                          acousticness=0.1, instrumentalness=0.0, tempo_norm=0.5)
    score = PromptScore(
        happy=0.5, sad=0.0, energetic=0.5, calm=0.0, nostalgic=0.0,
        romantic=0.0, assertive=0.0, deep=0.0, playful=0.0,
        wants_recent_songs=0.0, wants_popular_songs=0.0, wants_obscure_songs=0.0,
        wants_lyrics_focus=0.0, wants_mood_focus=0.0,
        semantic_query="q", audio_features=audio,
    )
    return SongRecommendationRequest(
        user_id="u", raw_prompt="r",
        prompt_score=score,
        user_profile=profile or UserProfileData(
            user_id="u", favourite_genres=[], favourite_artists=[],
            favourite_songs=[], preferred_language="es",
        ),
        semantic_query="q", target_audio_features=audio,
        metadata_filters=MetadataFilters(),
        ranking_weights=weights or RankingWeights(),
        top_k_retrieval=10, top_n_output=10,
    )


def test_score_breakdown_sums_to_total():
    svc = HybridRerankerService()
    ranked = svc.rerank([_candidate()], _request())
    r = ranked[0]
    total = (r.score_breakdown.score_lyrics + r.score_breakdown.score_audio
             + r.score_breakdown.score_profile + r.score_breakdown.score_popularity
             + r.score_breakdown.score_recency)
    assert isclose(total, r.score_total, abs_tol=1e-6)


def test_favourite_genre_boosts_profile():
    svc = HybridRerankerService()
    profile = UserProfileData(user_id="u", favourite_genres=["pop"], favourite_artists=[],
                              favourite_songs=[], preferred_language="es")
    ranked = svc.rerank([_candidate()], _request(profile=profile))
    assert ranked[0].score_breakdown.score_profile > 0


def test_disliked_artist_zero_profile():
    svc = HybridRerankerService()
    profile = UserProfileData(user_id="u", favourite_genres=["pop"], favourite_artists=[],
                              favourite_songs=[], preferred_language="es",
                              disliked_artists=["X"])
    ranked = svc.rerank([_candidate(artist="X")], _request(profile=profile))
    assert ranked[0].score_breakdown.score_profile == 0.0


def test_recency_decays_with_age():
    svc = HybridRerankerService()
    new = svc.rerank([_candidate(year=2025)], _request())[0]
    old = svc.rerank([_candidate(year=2000)], _request())[0]
    assert new.score_breakdown.score_recency > old.score_breakdown.score_recency


def test_negative_popularity_weight_demotes_popular():
    svc = HybridRerankerService()
    weights = RankingWeights(w_lyrics=0.0, w_audio=0.0, w_profile=0.0,
                              w_popularity=-1.0, w_recency=0.0)
    popular = svc.rerank([_candidate(popularity=90)], _request(weights=weights))[0]
    obscure = svc.rerank([_candidate(popularity=10)], _request(weights=weights))[0]
    assert popular.score_total < obscure.score_total


def _no_audio_candidate(song_id="1", lyrics_sim=0.8, popularity=80, year=2024, artist="X"):
    return CandidateSong(
        song_id=song_id, track_name=f"t{song_id}", artist_name=artist,
        genres=["pop"], popularity=popularity, release_date=date(year, 1, 1),
        best_lyrics_chunks=["..."], best_lyrics_similarity=lyrics_sim,
        audio_features=None,
    )


def test_no_audio_song_scores_zero_on_audio_axis():
    svc = HybridRerankerService()
    r = svc.rerank([_no_audio_candidate()], _request())[0]
    assert r.score_breakdown.score_audio == 0.0
    total = (r.score_breakdown.score_lyrics + r.score_breakdown.score_audio
             + r.score_breakdown.score_profile + r.score_breakdown.score_popularity
             + r.score_breakdown.score_recency)
    assert isclose(total, r.score_total, abs_tol=1e-6)


def test_no_audio_renormalizes_remaining_weights():
    svc = HybridRerankerService()
    weights = RankingWeights(w_lyrics=0.55, w_audio=0.30, w_profile=0.10,
                              w_popularity=0.03, w_recency=0.02)
    r = svc.rerank([_no_audio_candidate(lyrics_sim=0.8)], _request(weights=weights))[0]
    # lyrics contribution is scaled up by 1/(1 - w_audio) so the song isn't
    # penalized for missing audio (would be 0.55*0.8 = 0.44 without renormalization).
    assert r.score_breakdown.score_lyrics > 0.55 * 0.8


# --- Prompt-target nullability -------------------------------------------


def test_null_target_audio_drops_audio_axis():
    # All six prompt axes None → treat like a lyrics-only request.
    svc = HybridRerankerService()
    audio = PromptAudioFeatures()  # all axes default to None
    r = svc.rerank([_candidate()], _request(audio=audio))[0]
    assert r.score_breakdown.score_audio == 0.0


def test_partial_target_audio_scores_only_present_axes():
    # Only valence is set; the similarity must still compute (single-axis cosine == 1.0)
    # and not crash on the None axes.
    svc = HybridRerankerService()
    audio = PromptAudioFeatures(valence=0.85)
    r = svc.rerank([_candidate(valence=0.85)], _request(audio=audio))[0]
    assert r.score_breakdown.score_audio > 0


def test_target_with_some_nulls_does_not_raise():
    svc = HybridRerankerService()
    audio = PromptAudioFeatures(valence=0.5, energy=None, danceability=0.5,
                                 acousticness=None, instrumentalness=None, tempo_norm=0.5)
    # Just exercising the code path — must not raise on None axes.
    svc.rerank([_candidate()], _request(audio=audio))
