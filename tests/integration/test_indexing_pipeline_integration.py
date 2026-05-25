from unittest.mock import MagicMock
from app.lambdas.indexing_pipeline import IndexingPipeline
from app.services.loader_service import LoaderService
from app.services.splitter_service import SplitterService
from app.services.embedder_service import EmbedderService
from app.models.song_document import SongDocument
from tests.fakes.fake_song_repository import FakeSongRepository
from tests.fakes.fake_pinecone_repository import FakePineconeRepository


def test_indexing_with_fakes():
    songs = [
        SongDocument(song_id=i, title=f"t{i}", tag="pop", artist=f"a{i}",
                     year=2024, views=0,
                     lyrics="line one\nline two\nline three\n" * 30,
                     language="es",
                     popularity=70, danceability=0.7, energy=0.7, valence=0.8,
                     acousticness=0.1, instrumentalness=0.0, tempo=120.0)
        for i in range(1, 4)
    ]
    song_repo = FakeSongRepository(songs=songs)
    vec_repo = FakePineconeRepository()

    embedder = EmbedderService.__new__(EmbedderService)
    embedder._vector_repo = vec_repo
    embedder._batch_size = 10
    fake_embed = MagicMock()
    fake_embed.embed_documents.return_value = [[0.1] * 384] * 100
    embedder._embeddings = fake_embed

    pipeline = IndexingPipeline(
        loader=LoaderService(song_repo),
        splitter=SplitterService(chunk_size=200, chunk_overlap=20),
        embedder=embedder,
    )

    result = pipeline.run()

    assert result.total_docs == 3
    assert result.total_chunks > 3
    assert result.total_indexed == result.total_chunks
