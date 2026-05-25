from app.models.song_document import SongDocument


class FakeSongRepository:
    def __init__(self, songs: list[SongDocument] | None = None) -> None:
        self._by_id: dict[int, SongDocument] = {s.song_id: s for s in (songs or [])}

    def get_all(self) -> list[SongDocument]:
        return list(self._by_id.values())

    def get_by_id(self, song_id: int) -> SongDocument | None:
        return self._by_id.get(song_id)

    def get_by_ids(self, song_ids: list[int]) -> list[SongDocument]:
        return [self._by_id[i] for i in song_ids if i in self._by_id]

    def count(self) -> int:
        return len(self._by_id)
