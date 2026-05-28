from app.core.config import Settings
from app.services.retrieval.recommendation_request_builder import RecommendationRequestBuilder


def _settings():
    return Settings(
        mongo_uri="x", mongo_db_name="x",
        pinecone_api_key="x", openai_api_key="x",
        _env_file=None,
    )


def test_builds_request_with_defaults(sample_request_context, sample_prompt_score, sample_user_profile):
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, sample_prompt_score, sample_user_profile)
    assert req.user_id == "user_001"
    assert req.semantic_query == "happy upbeat songs with positive mood"
    assert req.top_k_retrieval == 150
    assert req.top_n_output == 10


def test_mood_focus_bumps_audio_weight(sample_request_context, sample_prompt_score, sample_user_profile):
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, sample_prompt_score, sample_user_profile)
    # sample_prompt_score has wants_mood_focus=0.75 (> 0.7)
    assert req.ranking_weights.w_audio == 0.45
    assert req.ranking_weights.w_lyrics == 0.40


def test_disliked_genres_become_filter(sample_request_context, sample_prompt_score, sample_user_profile):
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, sample_prompt_score, sample_user_profile)
    assert req.metadata_filters.genres_not_in == ["metal"]


def test_obscure_mode_inverts_popularity_weight(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"wants_obscure_songs": 0.8})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    # Weights are renormalized after presets, so the value is no longer exactly -0.05,
    # but it stays negative (popularity inverted) — that's the obscure-mode behavior.
    assert req.ranking_weights.w_popularity < 0


def test_weights_renormalized_to_sum_one(sample_request_context, sample_prompt_score, sample_user_profile):
    builder = RecommendationRequestBuilder(_settings())
    for update in ({"wants_lyrics_focus": 0.9}, {"wants_obscure_songs": 0.8}, {"wants_mood_focus": 0.9}):
        score = sample_prompt_score.model_copy(update=update)
        w = builder.build(sample_request_context, score, sample_user_profile).ranking_weights
        total = w.w_lyrics + w.w_audio + w.w_profile + w.w_popularity + w.w_recency
        assert abs(total - 1.0) < 1e-6


def test_recent_score_does_not_set_release_date_filter(sample_request_context, sample_prompt_score, sample_user_profile):
    # Explicit-only: an inferred wants_recent_songs score never sets release_date_min.
    score = sample_prompt_score.model_copy(update={"wants_recent_songs": 0.8})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.release_date_min is None
    assert req.metadata_filters.release_date_max is None


def test_popular_score_does_not_set_min_popularity_filter(sample_request_context, sample_prompt_score, sample_user_profile):
    # Explicit-only: an inferred wants_popular_songs score never sets min_popularity.
    score = sample_prompt_score.model_copy(update={"wants_popular_songs": 0.8})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.min_popularity is None


def test_wanted_genres_become_genres_in(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"wanted_genres": ["rap"]})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.genres_in == ["rap"]


def test_unwanted_genres_union_with_profile_dislikes(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"unwanted_genres": ["rap"]})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    # profile.disliked_genres = ["metal"]; both should be present in the merged set.
    assert set(req.metadata_filters.genres_not_in) == {"rap", "metal"}


def test_wanted_artists_become_artist_in(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"wanted_artists": ["Taylor Swift", "Ed Sheeran"]})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.artist_in == ["Taylor Swift", "Ed Sheeran"]


def test_unwanted_songs_become_songs_not_in(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"unwanted_songs": ["Friday"]})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.songs_not_in == ["Friday"]


def test_unwanted_strips_conflicting_wanted(sample_request_context, sample_prompt_score, sample_user_profile):
    # If the LLM emits the same item in both lists, unwanted wins.
    score = sample_prompt_score.model_copy(update={"wanted_genres": ["rap", "pop"], "unwanted_genres": ["rap"]})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.genres_in == ["pop"]
    assert "rap" in req.metadata_filters.genres_not_in
