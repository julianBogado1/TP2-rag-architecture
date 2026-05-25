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
    assert req.ranking_weights.w_popularity == -0.05


def test_recent_songs_filter(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"wants_recent_songs": 0.8})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.release_date_min is not None


def test_popular_songs_filter(sample_request_context, sample_prompt_score, sample_user_profile):
    score = sample_prompt_score.model_copy(update={"wants_popular_songs": 0.8})
    builder = RecommendationRequestBuilder(_settings())
    req = builder.build(sample_request_context, score, sample_user_profile)
    assert req.metadata_filters.min_popularity == 30
