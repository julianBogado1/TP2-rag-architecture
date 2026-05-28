from datetime import date
from math import exp, sqrt
from app.models.song_candidate import CandidateSong, SongMetadata, AudioFeatures
from app.models.ranked_song import RankedSongCandidate, ScoreBreakdown
from app.models.song_recommendation_request import SongRecommendationRequest
from app.models.user_profile import UserProfileData
from app.models.prompt_score import PromptAudioFeatures

# Profile-affinity contribution weights (sum to ~1.0 at the upper bound)
PROFILE_FAVOURITE_ARTIST_BONUS = 0.50
PROFILE_FAVOURITE_GENRE_BONUS  = 0.30
PROFILE_FAVOURITE_SONG_BONUS   = 0.10
PROFILE_DISLIKED_ARTIST_PENALTY = -0.50
PROFILE_DISLIKED_GENRE_PENALTY  = -0.30

# Popularity normalisation
POPULARITY_CEILING = 100

# Recency decay (exponential)
RECENCY_DECAY_YEARS = 5.0


class HybridRerankerService:
    """Weighted-sum reranker: lyrics + audio + profile + popularity + recency."""

    _AUDIO_FEATURE_ORDER = ("valence", "energy", "danceability",
                            "acousticness", "instrumentalness", "tempo_norm")

    def rerank(
        self,
        candidates: list[CandidateSong],
        req: SongRecommendationRequest,
    ) -> list[RankedSongCandidate]:
        target = req.target_audio_features
        target_present = self._has_any_axis(target)
        w = req.ranking_weights
        out: list[RankedSongCandidate] = []
        for c in candidates:
            lyrics_sim = c.best_lyrics_similarity
            profile    = self._profile_affinity(c, req.user_profile)
            pop_score  = self._popularity_score(c.popularity)
            rec_score  = self._recency_score(c.release_date)
            # Drop the audio axis when either side has no usable audio signal:
            # candidate has no Spotify features, or the prompt didn't imply any axis.
            if c.audio_features is None or not target_present:
                denom = 1.0 - w.w_audio
                scale = (1.0 / denom) if abs(denom) > 1e-9 else 1.0
                breakdown = ScoreBreakdown(
                    score_lyrics     = w.w_lyrics     * lyrics_sim * scale,
                    score_audio      = 0.0,
                    score_profile    = w.w_profile    * profile    * scale,
                    score_popularity = w.w_popularity * pop_score  * scale,
                    score_recency    = w.w_recency    * rec_score  * scale,
                )
            else:
                audio_sim = self._audio_similarity(target, c.audio_features)
                breakdown = ScoreBreakdown(
                    score_lyrics     = w.w_lyrics     * lyrics_sim,
                    score_audio      = w.w_audio      * audio_sim,
                    score_profile    = w.w_profile    * profile,
                    score_popularity = w.w_popularity * pop_score,
                    score_recency    = w.w_recency    * rec_score,
                )
            total = (breakdown.score_lyrics + breakdown.score_audio
                     + breakdown.score_profile + breakdown.score_popularity
                     + breakdown.score_recency)
            out.append(RankedSongCandidate(
                song_id=c.song_id, track_name=c.track_name, artist_name=c.artist_name,
                score_total=total, score_breakdown=breakdown,
                evidence_chunks=c.best_lyrics_chunks,
                metadata=SongMetadata(
                    track_name=c.track_name, artist_name=c.artist_name,
                    genres=c.genres, popularity=c.popularity,
                    release_date=c.release_date, audio_features=c.audio_features,
                ),
            ))
        return out

    @classmethod
    def _has_any_axis(cls, target: PromptAudioFeatures | None) -> bool:
        if target is None:
            return False
        return any(getattr(target, name) is not None for name in cls._AUDIO_FEATURE_ORDER)

    def _audio_similarity(self, target: PromptAudioFeatures, cand_audio: AudioFeatures) -> float:
        # Alternative: 1.0 - sum(abs(p - c)) / k — Manhattan inverted; cheaper but ignores direction.
        # Score only the axes the prompt actually implied; silent axes drop out entirely
        # (rather than being treated as 0, which would penalise candidates without cause).
        dims = [(getattr(target, n), getattr(cand_audio, n)) for n in self._AUDIO_FEATURE_ORDER
                if getattr(target, n) is not None]
        if not dims:
            return 0.0
        p_vec = [p for p, _ in dims]
        c_vec = [c for _, c in dims]
        dot = sum(p * c for p, c in zip(p_vec, c_vec))
        norm_p = sqrt(sum(p * p for p in p_vec))
        norm_c = sqrt(sum(c * c for c in c_vec))
        if norm_p == 0.0 or norm_c == 0.0:
            return 0.0
        return max(0.0, dot / (norm_p * norm_c))

    def _profile_affinity(self, cand: CandidateSong, profile: UserProfileData) -> float:
        # Alternative: step function — 1.0 if artist in favs else 0.5 if any genre overlaps else 0.0.
        # Cheaper, but loses partial-match granularity.
        score = 0.0
        if cand.artist_name in profile.favourite_artists:    score += PROFILE_FAVOURITE_ARTIST_BONUS
        if set(cand.genres) & set(profile.favourite_genres): score += PROFILE_FAVOURITE_GENRE_BONUS
        if cand.track_name in profile.favourite_songs:       score += PROFILE_FAVOURITE_SONG_BONUS
        if cand.artist_name in profile.disliked_artists:     score += PROFILE_DISLIKED_ARTIST_PENALTY
        if set(cand.genres) & set(profile.disliked_genres):  score += PROFILE_DISLIKED_GENRE_PENALTY
        return max(0.0, min(1.0, score))

    @staticmethod
    def _popularity_score(popularity: int) -> float:
        # Already linear — no alternative needed.
        return min(popularity, POPULARITY_CEILING) / float(POPULARITY_CEILING)

    @staticmethod
    def _recency_score(release_date: date) -> float:
        # Alternative: max(0.0, 1.0 - years_old / 20) — linear decay; less harsh on old songs.
        # date.today() per call so a long-running process doesn't freeze the reference year.
        years_old = date.today().year - release_date.year
        return exp(-years_old / RECENCY_DECAY_YEARS)
