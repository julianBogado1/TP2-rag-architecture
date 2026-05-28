from typing import Protocol, TypeVar
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from app.core.exceptions import LLMProviderError

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def parse_structured(
        self, model: str, system: str, user: str, schema: type[T]
    ) -> T | None: ...

    def generate_structured(
        self, model: str, system: str, context: BaseModel, schema: type[T]
    ) -> T | None: ...


class OpenAILLMClient:
    def __init__(
        self, api_key: str, timeout: float = 30.0, max_retries: int = 2
    ) -> None:
        self._client = OpenAI(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def parse_structured(
        self, model: str, system: str, user: str, schema: type[T]
    ) -> T | None:
        try:
            completion = self._client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
            )
        except OpenAIError as e:
            raise LLMProviderError(str(e)) from e
        return completion.choices[0].message.parsed

    def generate_structured(
        self, model: str, system: str, context: BaseModel, schema: type[T]
    ) -> T | None:
        try:
            completion = self._client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": context.model_dump_json()},
                ],
                response_format=schema,
            )
        except OpenAIError as e:
            raise LLMProviderError(str(e)) from e
        return completion.choices[0].message.parsed
