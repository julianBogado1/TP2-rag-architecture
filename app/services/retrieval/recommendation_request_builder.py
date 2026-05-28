from datetime import date, timedelta
from app.core.config import Settings
from app.models.request_context import RequestContext
from app.models.prompt_score import PromptScore
from app.models.user_profile import UserProfileData
from app.models.song_recommendation_request import (
    SongRecommendationRequest, MetadataFilters, RankingWeights,
)

# Activation thresholds for prompt-score-driven branches
WANTS_POPULAR_THRESHOLD       = 0.6
WANTS_OBSCURE_THRESHOLD       = 0.6
WANTS_RECENT_THRESHOLD        = 0.6
WANTS_LYRICS_FOCUS_THRESHOLD  = 0.7
WANTS_MOOD_FOCUS_THRESHOLD    = 0.7

# Filter values applied when an activation threshold fires
MIN_POPULARITY_WHEN_POPULAR   = 30
RECENT_WINDOW_DAYS            = 365 * 5

# Preset weights when a focus mode kicks in (override defaults)
LYRICS_FOCUS_W_LYRICS = 0.70
LYRICS_FOCUS_W_AUDIO  = 0.20
MOOD_FOCUS_W_LYRICS   = 0.40
MOOD_FOCUS_W_AUDIO    = 0.45
OBSCURE_W_POPULARITY  = -0.05


class RecommendationRequestBuilder:
    """Pure function: builds a SongRecommendationRequest from context, score, profile."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        ctx: RequestContext,
        score: PromptScore,
        profile: UserProfileData,
    ) -> SongRecommendationRequest:
        return SongRecommendationRequest(
            user_id=ctx.user_id,
            raw_prompt=ctx.raw_prompt,
            prompt_score=score,
            user_profile=profile,
            semantic_query=score.semantic_query,
            target_audio_features=score.audio_features,
            metadata_filters=self._build_filters(score, profile),
            ranking_weights=self._adjust_weights(score),
            top_k_retrieval=self._settings.retrieval_top_k,
            top_n_output=self._settings.output_top_n,
        )

    def _build_filters(self, score: PromptScore, profile: UserProfileData) -> MetadataFilters:
        today = date.today()
        return MetadataFilters(
            genres_in          = score.extracted_genres or None,
            genres_not_in      = profile.disliked_genres or None,
            artist_not_in      = profile.disliked_artists or None,
            preferred_language = score.preferred_language or None,
            min_popularity     = MIN_POPULARITY_WHEN_POPULAR if score.wants_popular_songs > WANTS_POPULAR_THRESHOLD else None,
            release_date_min   = today - timedelta(days=RECENT_WINDOW_DAYS) if score.wants_recent_songs > WANTS_RECENT_THRESHOLD else None,
            release_date_max   = None,
        )

    def _adjust_weights(self, score: PromptScore) -> RankingWeights:
        # Alternative: hard presets — `if score.wants_lyrics_focus > T: return LYRICS_HEAVY_PRESET`;
        # avoids renormalization but loses smooth interpolation between modes.
        s = self._settings
        weights = RankingWeights(
            w_lyrics=s.default_w_lyrics,
            w_audio=s.default_w_audio,
            w_profile=s.default_w_profile,
            w_popularity=s.default_w_popularity,
            w_recency=s.default_w_recency,
        )
        if score.wants_lyrics_focus > WANTS_LYRICS_FOCUS_THRESHOLD:
            weights.w_lyrics = LYRICS_FOCUS_W_LYRICS
            weights.w_audio  = LYRICS_FOCUS_W_AUDIO
        if score.wants_mood_focus > WANTS_MOOD_FOCUS_THRESHOLD:
            weights.w_lyrics = MOOD_FOCUS_W_LYRICS
            weights.w_audio  = MOOD_FOCUS_W_AUDIO
        if score.wants_obscure_songs > WANTS_OBSCURE_THRESHOLD:
            weights.w_popularity = OBSCURE_W_POPULARITY
        # Presets shift individual weights off the 1.0 sum; renormalize so score_total
        # stays comparable across requests (guard the degenerate near-zero sum).
        total = (weights.w_lyrics + weights.w_audio + weights.w_profile
                 + weights.w_popularity + weights.w_recency)
        if abs(total - 1.0) > 1e-9:
            weights.w_lyrics /= total
            weights.w_audio /= total
            weights.w_profile /= total
            weights.w_popularity /= total
            weights.w_recency /= total
        return weights
