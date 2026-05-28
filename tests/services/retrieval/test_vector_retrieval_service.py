import json
from app.models.song_recommendation_request import SongRecommendationRequest, MetadataFilters, RankingWeights
from app.models.prompt_score import PromptScore, PromptAudioFeatures
from app.models.user_profile import UserProfileData
from app.services.retrieval.vector_retrieval_service import VectorRetrievalService
from tests.fakes.fake_pinecone_repository import FakePineconeRepository


def _meta(song_id, popularity=70, language="es", genres=("pop",), valence=0.8, tempo=120):
    return {
        "song_id": song_id,
        "track_name": f"track_{song_id}",
        "artist_name": "artist",
        "genres": list(genres),
        "release_date": "2024",
        "song_characteristics_chunk": json.dumps({"popularity": popularity, "language": language}),
        "audio_features_chunk": json.dumps({"valence": valence, "energy": 0.7, "danceability": 0.7, "acousticness": 0.1, "instrumentalness": 0.0, "tempo": tempo}),
    }


def _profile():
    return UserProfileData(
        user_id="u", favourite_genres=[], favourite_artists=[],
        favourite_songs=[], preferred_language="es",
    )


def _request(filters=None, top_k=10):
    audio = PromptAudioFeatures(valence=0.5, energy=0.5, danceability=0.5,
                                 acousticness=0.5, instrumentalness=0.5, tempo_norm=0.5)
    score = PromptScore(
        happy=0.5, sad=0.0, energetic=0.5, calm=0.0, nostalgic=0.0,
        romantic=0.0, assertive=0.0, deep=0.0, playful=0.0,
        wants_recent_songs=0.0, wants_popular_songs=0.0, wants_obscure_songs=0.0,
        wants_lyrics_focus=0.0, wants_mood_focus=0.0,
        semantic_query="q", audio_features=audio,
    )
    return SongRecommendationRequest(
        user_id="u", raw_prompt="r",
        prompt_score=score, user_profile=_profile(),
        semantic_query="q", target_audio_features=audio,
        metadata_filters=filters or MetadataFilters(),
        ranking_weights=RankingWeights(),
        top_k_retrieval=top_k, top_n_output=10,
    )


def test_returns_candidate_chunks_with_parsed_audio():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([("1_0", [1.0, 0.0, 0.0], _meta(1))])
    svc = VectorRetrievalService(fake_repo)

    result = svc.retrieve(_request(top_k=5), [1.0, 0.0, 0.0])

    assert len(result) == 1
    cand = result[0]
    assert cand.chunk_id == "1_0"
    assert cand.song_id == "1"
    assert cand.metadata.popularity == 70
    assert cand.metadata.audio_features.valence == 0.8
    assert cand.metadata.audio_features.tempo_norm == 120 / 250.0


def test_song_with_audio_parses_all_six_features():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([("1_0", [1.0, 0.0, 0.0], _meta(1, valence=0.6, tempo=100))])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(top_k=5), [1.0, 0.0, 0.0])
    af = result[0].metadata.audio_features
    assert af is not None
    assert af.valence == 0.6
    assert af.tempo_norm == 100 / 250.0


def test_filters_disliked_genres():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta(1, genres=["pop"])),
        ("2_0", [1.0, 0.0, 0.0], _meta(2, genres=["metal"])),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(genres_not_in=["metal"])), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["1"]


def test_filters_min_popularity():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta(1, popularity=10)),
        ("2_0", [1.0, 0.0, 0.0], _meta(2, popularity=80)),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(min_popularity=50)), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["2"]


def test_filters_language():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta(1, language="es")),
        ("2_0", [1.0, 0.0, 0.0], _meta(2, language="en")),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(preferred_language="es")), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["1"]


def test_malformed_json_kept_without_audio():
    fake_repo = FakePineconeRepository()
    bad_meta = _meta(1)
    bad_meta["audio_features_chunk"] = "{ not json"
    fake_repo.upsert_batch([("1_0", [1.0, 0.0, 0.0], bad_meta)])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(), [1.0, 0.0, 0.0])
    assert len(result) == 1
    assert result[0].metadata.audio_features is None


def test_incomplete_audio_kept_without_audio():
    fake_repo = FakePineconeRepository()
    bad_meta = _meta(1)
    bad_meta["audio_features_chunk"] = json.dumps({"valence": 0.5})  # missing other 4
    fake_repo.upsert_batch([("1_0", [1.0, 0.0, 0.0], bad_meta)])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(), [1.0, 0.0, 0.0])
    assert len(result) == 1
    assert result[0].metadata.audio_features is None


def _meta_named(song_id, artist_name, track_name="t", genres=("pop",)):
    m = _meta(song_id, genres=genres)
    m["artist_name"] = artist_name
    m["track_name"] = track_name
    return m


def test_filters_artist_in_keeps_only_matching():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta_named(1, "Taylor Swift")),
        ("2_0", [1.0, 0.0, 0.0], _meta_named(2, "Metallica")),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(artist_in=["Taylor Swift"])), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["1"]


def test_filters_artist_in_is_case_insensitive():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta_named(1, "Taylor Swift")),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(artist_in=["taylor swift"])), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["1"]


def test_filters_artist_not_in_is_case_insensitive():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta_named(1, "Taylor Swift")),
        ("2_0", [1.0, 0.0, 0.0], _meta_named(2, "Metallica")),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(artist_not_in=["TAYLOR SWIFT"])), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["2"]


def test_filters_songs_not_in_excludes_by_track_name():
    fake_repo = FakePineconeRepository()
    fake_repo.upsert_batch([
        ("1_0", [1.0, 0.0, 0.0], _meta_named(1, "A", track_name="Friday")),
        ("2_0", [1.0, 0.0, 0.0], _meta_named(2, "B", track_name="Monday")),
    ])
    svc = VectorRetrievalService(fake_repo)
    result = svc.retrieve(_request(filters=MetadataFilters(songs_not_in=["friday"])), [1.0, 0.0, 0.0])
    assert [c.song_id for c in result] == ["2"]
