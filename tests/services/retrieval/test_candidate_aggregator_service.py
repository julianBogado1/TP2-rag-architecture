from datetime import date
from app.models.song_candidate import CandidateChunk, SongMetadata, AudioFeatures
from app.models.song_document import SongDocument
from app.services.retrieval.candidate_aggregator_service import CandidateAggregatorService
from tests.fakes.fake_song_repository import FakeSongRepository


def _chunk(chunk_id, song_id, sim, valence=0.8):
    return CandidateChunk(
        chunk_id=chunk_id, song_id=str(song_id),
        lyrics_chunk="",
        lyrics_similarity=sim,
        metadata=SongMetadata(
            track_name=f"t{song_id}", artist_name=f"a{song_id}",
            genres=["pop"], popularity=50, release_date=date(2024, 1, 1),
            audio_features=AudioFeatures(valence=valence, energy=0.5, danceability=0.5,
                                          acousticness=0.5, instrumentalness=0.0, tempo_norm=0.5),
        ),
    )


def _song(song_id, lyrics):
    return SongDocument(song_id=song_id, title=f"t{song_id}", tag="pop", artist=f"a{song_id}",
                        year=2024, views=0, lyrics=lyrics)


def test_groups_by_song_and_fetches_lyrics():
    chunks = [
        _chunk("1_0", 1, 0.8), _chunk("1_1", 1, 0.7),
        _chunk("2_0", 2, 0.6),
    ]
    song_repo = FakeSongRepository(songs=[_song(1, "lyrics one"), _song(2, "lyrics two")])
    svc = CandidateAggregatorService(song_repo=song_repo, max_evidence_chunks=3)

    cands = svc.aggregate(chunks)

    assert len(cands) == 2
    by_id = {c.song_id: c for c in cands}
    assert by_id["1"].best_lyrics_similarity == 0.8
    assert "lyrics one" in by_id["1"].best_lyrics_chunks[0]


def test_empty_chunks_returns_empty():
    svc = CandidateAggregatorService(song_repo=FakeSongRepository(), max_evidence_chunks=3)
    assert svc.aggregate([]) == []


def test_non_integer_song_id_is_dropped_with_warning(caplog):
    import logging
    chunks = [_chunk("abc_0", "abc", 0.9), _chunk("2_0", 2, 0.6)]
    song_repo = FakeSongRepository(songs=[_song(2, "lyrics two")])
    svc = CandidateAggregatorService(song_repo=song_repo, max_evidence_chunks=3)

    with caplog.at_level(logging.WARNING):
        cands = svc.aggregate(chunks)

    # non-integer id is still returned as a candidate (it was grouped), but it is
    # excluded from the Mongo lyrics lookup and a warning is logged.
    assert any("non-integer song_id" in rec.message for rec in caplog.records)
    by_id = {c.song_id: c for c in cands}
    assert by_id["abc"].best_lyrics_chunks == []
    assert "lyrics two" in by_id["2"].best_lyrics_chunks[0]


def test_caps_evidence_chunks_to_max():
    chunks = [_chunk(f"1_{i}", 1, 0.5) for i in range(10)]
    song_repo = FakeSongRepository(songs=[_song(1, "\n".join(f"line {i}" for i in range(10)))])
    svc = CandidateAggregatorService(song_repo=song_repo, max_evidence_chunks=3)

    cands = svc.aggregate(chunks)

    assert len(cands) == 1
    assert len(cands[0].best_lyrics_chunks) == 3
