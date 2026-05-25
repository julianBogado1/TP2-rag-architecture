from app.models.prompt_score import PromptScore
from app.services.retrieval.prompt_parser_service import PromptParserService
from tests.fakes.fake_llm_client import FakeLLMClient


def test_parse_returns_prompt_score(sample_prompt_score):
    fake = FakeLLMClient(returns={"PromptScore": sample_prompt_score})
    svc = PromptParserService(llm_client=fake, model="gpt-4o-mini")

    result = svc.parse("quiero canciones felices :)")

    assert isinstance(result, PromptScore)
    assert result.happy == sample_prompt_score.happy
    assert fake.calls[0].method == "parse_structured"
    assert fake.calls[0].user == "quiero canciones felices :)"
    assert fake.calls[0].schema is PromptScore
