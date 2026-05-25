from pydantic import BaseModel


class ParsedAudioFeatures(BaseModel):
    valence: float
    energy: float
    danceability: float
    acousticness: float
    instrumentalness: float
    tempo_norm: float
