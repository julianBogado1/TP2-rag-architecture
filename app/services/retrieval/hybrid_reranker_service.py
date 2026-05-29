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
            # Additive polynomial: each axis contributes weight·score only when it is
            # both *wanted* (prompt implied it) and *present* (song has the data).
            # An absent axis contributes 0 and its weight is NOT redistributed — so a
            # no-audio song earns nothing on audio rather than getting its lyrics
            # weight boosted, and a real audio match keeps the edge it deserves.
            if target_present and c.audio_features is not None:
                audio_score = w.w_audio * self._audio_similarity(target, c.audio_features)
            else:
                audio_score = 0.0
            breakdown = ScoreBreakdown(
                score_lyrics     = w.w_lyrics     * lyrics_sim,
                score_audio      = audio_score,
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
        # Normalized inverted Euclidean distance (Clase 5, SIMILITUD): audio features are
        # absolute values in [0,1] where the *value* is the signal, so we measure closeness
        # in value — not cosine direction, which is degenerate here (a single positive axis,
        # or a "low" target vs a "high" song, both yield cosine ~1.0 and rank nothing).
        # Score only the axes the prompt implied; silent axes drop out entirely.
        dims = [(getattr(target, n), getattr(cand_audio, n)) for n in self._AUDIO_FEATURE_ORDER
                if getattr(target, n) is not None]
        if not dims:
            return 0.0
        # each axis is in [0,1] -> max distance over n axes is sqrt(n); divide it out so
        # audio_sim = 1 - RMSE stays in [0,1]: 1.0 = exact match, 0.0 = maximally far.
        rmse = sqrt(sum((p - c) ** 2 for p, c in dims) / len(dims))
        return max(0.0, 1.0 - rmse)

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
