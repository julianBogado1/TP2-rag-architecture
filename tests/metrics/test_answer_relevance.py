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


class _RecordingLLM:
    """Captures the user prompt and kwargs the reverse-question call receives."""
    def __init__(self, result):
        self._result = result
        self.user = None
        self.kwargs = None

    def parse_structured(self, model, system, user, schema, **kwargs):
        self.user = user
        self.kwargs = kwargs
        return self._result


class _Questions:
    questions = ["q1"]


def test_reverse_question_input_is_message_only_no_song_names():
    # A: song names are semantic noise that drag reverse questions off the mood/activity.
    # The reverse-question generator should see only the opener message.
    llm = _RecordingLLM(_Questions())
    answer_relevance_score("happy songs", _response(), _FakeEmbedder(), llm)
    assert llm.user == "here are some songs"
    assert "by a" not in llm.user          # "t by a" song string must be gone


def test_reverse_question_call_is_deterministic_temperature_zero():
    # B: reproducibility — same config must give the same questions.
    llm = _RecordingLLM(_Questions())
    answer_relevance_score("happy songs", _response(), _FakeEmbedder(), llm)
    assert llm.kwargs.get("temperature") == 0


class _ScaledEmbedder:
    """Prompt -> [1,0]; any question -> a vector whose cosine with the prompt is 0.4."""
    def embed_query(self, text):
        if text == "happy songs":
            return [1.0, 0.0]
        return [0.4, 0.916515138991168]  # unit vector, cos with [1,0] = 0.4


def test_raw_cosine_is_rescaled_onto_zero_one():
    # raw mean cosine 0.4 on a model band of [0.0, 0.80] -> 0.4 / 0.80 = 0.5
    score = answer_relevance_score("happy songs", _response(), _ScaledEmbedder(), _FakeLLM(_Questions()))
    assert abs(score - 0.5) < 1e-9
