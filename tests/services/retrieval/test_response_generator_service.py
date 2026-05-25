from datetime import date
from app.models.song_candidate import AudioFeatures, SongMetadata
from app.models.ranked_song import TopRecommendedSong, ScoreBreakdown
from app.models.recommendation_response import RecommendationResponse, SongRecommendation
from app.models.song_recommendation_request import (
    SongRecommendationRequest, MetadataFilters, RankingWeights,
)
from app.services.retrieval.response_generator_service import ResponseGeneratorService
from tests.fakes.fake_llm_client import FakeLLMClient


def _top_song():
    return TopRecommendedSong(
        song_id="42", track_name="Friday Lights", artist_name="Luna Reyes",
        score_total=0.8,
        score_breakdown=ScoreBreakdown(score_lyrics=0.4, score_audio=0.3, score_profile=0.05,
                                        score_popularity=0.03, score_recency=0.02),
        evidence_chunks=["chorus..."],
        metadata=SongMetadata(
            track_name="Friday Lights", artist_name="Luna Reyes", genres=["pop"],
            popularity=78, release_date=date(2024, 1, 1),
            audio_features=AudioFeatures(valence=0.88, energy=0.82, danceability=0.79,
                                          acousticness=0.08, instrumentalness=0.0, tempo_norm=0.5),
        ),
    )


def test_generate_calls_llm_with_context(sample_prompt_score, sample_user_profile):
    req = SongRecommendationRequest(
        user_id="u", raw_prompt="happy plz",
        prompt_score=sample_prompt_score, user_profile=sample_user_profile,
        semantic_query="happy", target_audio_features=sample_prompt_score.audio_features,
        metadata_filters=MetadataFilters(), ranking_weights=RankingWeights(),
        top_k_retrieval=10, top_n_output=1,
    )
    expected = RecommendationResponse(message="msg", recommendations=[
        SongRecommendation(rank=1, track_name="Friday Lights", artist_name="Luna Reyes",
                           explanation="x", matched_mood=["happy"], matched_audio_features=["valence"])
    ])
    fake = FakeLLMClient(returns={"RecommendationResponse": expected})
    svc = ResponseGeneratorService(llm_client=fake, model="gpt-4o-mini")

    result = svc.generate([_top_song()], req)

    assert result is expected
    assert fake.calls[0].method == "generate_structured"
    assert fake.calls[0].context is not None
    assert fake.calls[0].schema is RecommendationResponse
