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

    TODAY_YEAR = date.today().year
    _AUDIO_FEATURE_ORDER = ("valence", "energy", "danceability",
                            "acousticness", "instrumentalness", "tempo_norm")

    def rerank(
        self,
        candidates: list[CandidateSong],
        req: SongRecommendationRequest,
    ) -> list[RankedSongCandidate]:
        prompt_vec = self._to_audio_vec(req.target_audio_features)
        w = req.ranking_weights
        out: list[RankedSongCandidate] = []
        for c in candidates:
            lyrics_sim = c.best_lyrics_similarity
            audio_sim  = self._audio_similarity(prompt_vec, c.audio_features)
            profile    = self._profile_affinity(c, req.user_profile)
            pop_score  = self._popularity_score(c.popularity)
            rec_score  = self._recency_score(c.release_date)
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

    def _to_audio_vec(self, f: PromptAudioFeatures) -> list[float]:
        return [getattr(f, name) for name in self._AUDIO_FEATURE_ORDER]

    def _audio_similarity(self, prompt_vec: list[float], cand_audio: AudioFeatures) -> float:
        # Alternative: 1.0 - sum(abs(p - c)) / 6 — Manhattan inverted; cheaper but ignores direction.
        cand_vec = [getattr(cand_audio, name) for name in self._AUDIO_FEATURE_ORDER]
        dot = sum(p * c for p, c in zip(prompt_vec, cand_vec))
        norm_p = sqrt(sum(p * p for p in prompt_vec))
        norm_c = sqrt(sum(c * c for c in cand_vec))
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

    @classmethod
    def _recency_score(cls, release_date: date) -> float:
        # Alternative: max(0.0, 1.0 - years_old / 20) — linear decay; less harsh on old songs.
        years_old = cls.TODAY_YEAR - release_date.year
        return exp(-years_old / RECENCY_DECAY_YEARS)
