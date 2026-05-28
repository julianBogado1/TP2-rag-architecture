from app.metrics.answer_relevance import answer_relevance_score
from app.models.recommendation_response import RecommendationResponse, SongRecommendation


class _FakeLLM:
    def __init__(self, result):
        self._result = result

    def parse_structured(self, *args, **kwargs):
        return self._result


class _FakeEmbedder:
    def embed_query(self, text):
        return [1.0, 0.0]


def _response():
    return RecommendationResponse(
        message="here are some songs",
        recommendations=[
            SongRecommendation(rank=1, track_name="t", artist_name="a", explanation="e",
                               matched_mood=["happy"], matched_audio_features=["valence"]),
        ],
    )


def test_none_parse_result_returns_zero_not_crash():
    score = answer_relevance_score("happy songs", _response(), _FakeEmbedder(), _FakeLLM(None))
    assert score == 0.0


def test_empty_questions_returns_zero():
    class _Empty:
        questions = []
    score = answer_relevance_score("happy songs", _response(), _FakeEmbedder(), _FakeLLM(_Empty()))
    assert score == 0.0
