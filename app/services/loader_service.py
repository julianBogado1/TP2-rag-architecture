import json
import logging

from langchain_core.documents import Document

from app.persistence.mongo.song_repository import SongRepository

logger = logging.getLogger(__name__)


class LoaderService:
    def __init__(self, repo: SongRepository) -> None:
        self._repo = repo

    def load(self, song_ids: list[int] | None = None):
        source = self._repo.get_by_ids_stream(song_ids) if song_ids is not None else self._repo.get_all()
        for song in source:
            yield Document(
                page_content=song.lyrics,
                metadata={
                    "song_id": song.song_id,
                    "track_name": song.title,
                    "artist_name": song.artist,
                    "genres": [song.tag],
                    "release_date": str(song.year),
                    "song_characteristics_chunk": json.dumps(self._build_song_characteristics(song)),
                    "audio_features_chunk": json.dumps(self._build_audio_features(song)),
                },
            )
    
    @staticmethod
    def _build_song_characteristics(song) -> dict:
        fields = {
            "popularity": song.popularity,
            "duration_ms": song.duration_ms,
            "explicit": song.explicit,
            "tempo": song.tempo,
            "time_signature": song.time_signature,
            "language": song.language
        }
        return {k: v for k, v in fields.items() if v is not None}

    @staticmethod
    def _build_audio_features(song) -> dict:
        fields = {
            "danceability": song.danceability,
            "energy": song.energy,
            "key": song.key,
            "loudness": song.loudness,
            "mode": song.mode,
            "speechiness": song.speechiness,
            "acousticness": song.acousticness,
            "instrumentalness": song.instrumentalness,
            "liveness": song.liveness,
            "valence": song.valence
        }
        return {k: v for k, v in fields.items() if v is not None}
