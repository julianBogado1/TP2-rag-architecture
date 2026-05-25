from pydantic import BaseModel
from tests.fakes.fake_llm_client import FakeLLMClient


class Out(BaseModel):
    value: int


def test_fake_returns_precanned_and_records_call():
    fake = FakeLLMClient(returns={"Out": Out(value=42)})
    result = fake.parse_structured(model="m", system="s", user="u", schema=Out)
    assert result.value == 42
    assert len(fake.calls) == 1
    assert fake.calls[0].schema is Out
    assert fake.calls[0].user == "u"


def test_generate_structured_records_context():
    fake = FakeLLMClient(returns={"Out": Out(value=7)})
    ctx = Out(value=1)
    result = fake.generate_structured(model="m", system="s", context=ctx, schema=Out)
    assert result.value == 7
    assert fake.calls[0].method == "generate_structured"
    assert fake.calls[0].context is ctx
