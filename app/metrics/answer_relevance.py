from pydantic import BaseModel
from app.core.llm_client import OpenAILLMClient
from app.models.recommendation_response import RecommendationResponse
from app.services.embedder_service import EmbedderService
from app.services.retrieval.response_generator_service import SKIP_LLM_MESSAGE

_MODEL = "gpt-4o-mini"
_N_QUESTIONS = 3

_SYSTEM = (
    "Given a music recommendation response, generate exactly "
    f"{_N_QUESTIONS} different questions that a user could have asked to receive "
    "this response. Focus on the mood, genre, or activity described."
)


class _ReverseQuestions(BaseModel):
    questions: list[str]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def answer_relevance_score(
    raw_prompt: str,
    response: RecommendationResponse,
    embedder: EmbedderService,
    llm: OpenAILLMClient,
) -> float:
    """
    Answer Relevance: cosine similarity between original prompt and LLM-generated
    reverse questions from the response.

    AR = (1/N) × Σ cos(embed(generated_question_i), embed(original_prompt))
    """
    if not response.message or response.message == SKIP_LLM_MESSAGE:
        return 0.0

    summary = response.message + " " + ", ".join(
        f"{r.track_name} by {r.artist_name}" for r in response.recommendations
    )
    result = llm.parse_structured(_MODEL, _SYSTEM, summary, _ReverseQuestions)
    if result is None or not result.questions:
        return 0.0

    prompt_vec = embedder.embed_query(raw_prompt)
    scores = [
        _cosine(embedder.embed_query(q), prompt_vec)
        for q in result.questions
    ]
    return sum(scores) / len(scores) if scores else 0.0
