from langchain_core.documents import Document
from app.services.splitter_service import SplitterService


def test_splits_long_document_within_chunk_size():
    text = "palabra " * 200  # ~1600 chars, well over 400
    doc = Document(page_content=text, metadata={"song_id": 1})
    splitter = SplitterService(chunk_size=400, chunk_overlap=50)

    chunks = list(splitter.split([doc]))

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 400 for c in chunks)


def test_short_document_stays_single_chunk():
    doc = Document(page_content="short lyric line", metadata={"song_id": 2})
    splitter = SplitterService(chunk_size=400, chunk_overlap=50)

    chunks = list(splitter.split([doc]))

    assert len(chunks) == 1
    assert chunks[0].page_content == "short lyric line"
