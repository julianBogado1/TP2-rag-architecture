from pydantic import BaseModel
from app.core.llm_client import OpenAILLMClient
from app.models.recommendation_response import RecommendationResponse
from app.services.embedder_service import EmbedderService
from app.services.retrieval.response_generator_service import SKIP_LLM_MESSAGE

_MODEL = "gpt-4o-mini"
_N_QUESTIONS = 3

# Raw cosine on all-MiniLM-L6-v2 is compressed: a clearly-relevant question/prompt pair
# tops out ~0.80 and an unrelated pair sits ~0.05, so a raw mean never fills a 0-1 scale
# and looks artificially low next to the other 0-1 metrics. We rescale the meaningful
# band [_FLOOR, _CEILING] (measured on this model) onto [0,1]. This is a documented,
# monotonic transform — it preserves ranking and keeps unrelated answers near 0; it does
# NOT inflate the floor the way a high-offset embedder (e.g. ada-002) would. Re-measure
# these anchors if the embedding model changes.
_FLOOR = 0.0
_CEILING = 0.80

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

    AR = (1/N) × Σ cos(embed(generated_question_i), embed(original_prompt)),
    then min-max rescaled from the model's [_FLOOR, _CEILING] cosine band onto [0,1]
    so the score is comparable to the other 0-1 metrics.
    """
    if not response.message or response.message == SKIP_LLM_MESSAGE:
        return 0.0

    # Reverse-engineer questions from the opener message only. Including the song
    # names ("Track by Artist") drags the generated questions toward specific tracks
    # and away from the mood/genre/activity the prompt actually expressed. temperature=0
    # makes the questions (and thus the score) reproducible across runs.
    result = llm.parse_structured(
        _MODEL, _SYSTEM, response.message, _ReverseQuestions, temperature=0
    )
    if result is None or not result.questions:
        return 0.0

    prompt_vec = embedder.embed_query(raw_prompt)
    scores = [
        _cosine(embedder.embed_query(q), prompt_vec)
        for q in result.questions
    ]
    if not scores:
        return 0.0
    raw = sum(scores) / len(scores)
    # rescale the model's meaningful cosine band onto a full 0-1 scale
    return max(0.0, min(1.0, (raw - _FLOOR) / (_CEILING - _FLOOR)))
