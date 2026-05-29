from types import SimpleNamespace

import pytest
from openai import OpenAIError
from pydantic import BaseModel

from app.core.exceptions import LLMProviderError
from app.core.llm_client import LLMClient, OpenAILLMClient


class _Schema(BaseModel):
    value: str


def _completion_with_parsed(parsed):
    """Build a fake OpenAI completion exposing .choices[0].message.parsed."""
    message = SimpleNamespace(parsed=parsed)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeOpenAIClient:
    """Minimal stand-in for the openai SDK client used by OpenAILLMClient."""

    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(parse=self._parse)
            )
        )

    def _parse(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result


def test_llm_client_protocol_methods_exist():
    assert hasattr(LLMClient, "parse_structured")
    assert hasattr(LLMClient, "generate_structured")


def test_openai_client_constructs():
    client = OpenAILLMClient(api_key="sk-test")
    assert client is not None


def test_parse_structured_returns_none_when_parsed_is_none():
    client = OpenAILLMClient(api_key="sk-test")
    client._client = _FakeOpenAIClient(result=_completion_with_parsed(None))

    result = client.parse_structured("gpt-x", "sys", "usr", _Schema)

    assert result is None


def test_parse_structured_returns_parsed_value():
    client = OpenAILLMClient(api_key="sk-test")
    parsed = _Schema(value="ok")
    client._client = _FakeOpenAIClient(result=_completion_with_parsed(parsed))

    result = client.parse_structured("gpt-x", "sys", "usr", _Schema)

    assert result is parsed


def test_parse_structured_wraps_openai_error():
    client = OpenAILLMClient(api_key="sk-test")
    client._client = _FakeOpenAIClient(error=OpenAIError("boom"))

    with pytest.raises(LLMProviderError):
        client.parse_structured("gpt-x", "sys", "usr", _Schema)


def test_generate_structured_returns_none_when_parsed_is_none():
    client = OpenAILLMClient(api_key="sk-test")
    client._client = _FakeOpenAIClient(result=_completion_with_parsed(None))

    result = client.generate_structured("gpt-x", "sys", _Schema(value="ctx"), _Schema)

    assert result is None


def test_generate_structured_wraps_openai_error():
    client = OpenAILLMClient(api_key="sk-test")
    client._client = _FakeOpenAIClient(error=OpenAIError("boom"))

    with pytest.raises(LLMProviderError):
        client.generate_structured("gpt-x", "sys", _Schema(value="ctx"), _Schema)


class _CapturingOpenAIClient(_FakeOpenAIClient):
    """Records the kwargs passed to the SDK parse() call."""
    def __init__(self, *, result=None):
        super().__init__(result=result)
        self.captured: dict = {}

    def _parse(self, **kwargs):
        self.captured = kwargs
        return self._result


def test_parse_structured_forwards_temperature_when_given():
    client = OpenAILLMClient(api_key="sk-test")
    fake = _CapturingOpenAIClient(result=_completion_with_parsed(_Schema(value="ok")))
    client._client = fake

    client.parse_structured("gpt-x", "sys", "usr", _Schema, temperature=0)

    assert fake.captured["temperature"] == 0


def test_parse_structured_omits_temperature_by_default():
    # Default path must not pin temperature, preserving existing callers' behaviour.
    client = OpenAILLMClient(api_key="sk-test")
    fake = _CapturingOpenAIClient(result=_completion_with_parsed(_Schema(value="ok")))
    client._client = fake

    client.parse_structured("gpt-x", "sys", "usr", _Schema)

    assert "temperature" not in fake.captured
