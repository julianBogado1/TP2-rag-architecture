from app.core.llm_client import LLMClient
from app.models.prompt_score import PromptScore


_SYSTEM_PROMPT = """You convert a user's natural-language music request into a
structured score. For every mood/intent field in the schema, output a float in
[0, 1] reflecting how strongly the prompt expresses that mood. For audio_features,
infer plausible target values (happy upbeat prompt -> high valence/energy/dance;
melancholic prompt -> low valence, low energy, high acousticness). Set
semantic_query to a clean English rephrasing optimized for embedding search:
strip emojis, fix Spanglish, expand abbreviations. Extract any explicit
artist/genre/language mentions; otherwise leave them null. Never invent data the
user didn't imply: output 0.0 for moods the prompt is silent on."""


class PromptParserService:
    """Calls the LLM with a structured-output schema to turn raw prompt → PromptScore."""

    def __init__(self, llm_client: LLMClient, model: str) -> None:
        self._llm = llm_client
        self._model = model

    def parse(self, raw_prompt: str) -> PromptScore:
        return self._llm.parse_structured(
            model=self._model,
            system=_SYSTEM_PROMPT,
            user=raw_prompt,
            schema=PromptScore,
        )
