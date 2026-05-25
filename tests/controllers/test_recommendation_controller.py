from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.controllers.recommendation_controller import build_recommendation_router
from app.core.exceptions import UserNotFoundError, LLMProviderError
from app.models.recommendation_response import RecommendationResponse, SongRecommendation


class StubOrchestrator:
    def __init__(self, *, raise_user_not_found=False, raise_llm=False, response=None):
        self._raise_user_not_found = raise_user_not_found
        self._raise_llm = raise_llm
        self._response = response

    def recommend(self, user_id, raw_prompt):
        if self._raise_user_not_found:
            raise UserNotFoundError(user_id)
        if self._raise_llm:
            raise LLMProviderError("boom")
        return self._response


def _app(orch):
    app = FastAPI()
    app.include_router(build_recommendation_router(orch))
    return TestClient(app)


def test_happy_path():
    resp_obj = RecommendationResponse(message="hi", recommendations=[
        SongRecommendation(rank=1, track_name="t", artist_name="a", explanation="e",
                           matched_mood=["happy"], matched_audio_features=["valence"]),
    ])
    client = _app(StubOrchestrator(response=resp_obj))
    r = client.post("/recommend", json={"user_id": "u", "raw_prompt": "hola"})
    assert r.status_code == 200
    assert r.json()["message"] == "hi"


def test_user_not_found_returns_404():
    client = _app(StubOrchestrator(raise_user_not_found=True))
    r = client.post("/recommend", json={"user_id": "u", "raw_prompt": "hola"})
    assert r.status_code == 404


def test_llm_error_returns_502():
    client = _app(StubOrchestrator(raise_llm=True))
    r = client.post("/recommend", json={"user_id": "u", "raw_prompt": "hola"})
    assert r.status_code == 502


def test_empty_prompt_rejected_422():
    client = _app(StubOrchestrator(response=None))
    r = client.post("/recommend", json={"user_id": "u", "raw_prompt": ""})
    assert r.status_code == 422
