from typing import Protocol, TypeVar
from pydantic import BaseModel
from openai import OpenAI

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def parse_structured(
        self, model: str, system: str, user: str, schema: type[T]
    ) -> T: ...

    def generate_structured(
        self, model: str, system: str, context: BaseModel, schema: type[T]
    ) -> T: ...


class OpenAILLMClient:
    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def parse_structured(self, model: str, system: str, user: str, schema: type[T]) -> T:
        completion = self._client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        return completion.choices[0].message.parsed

    def generate_structured(
        self, model: str, system: str, context: BaseModel, schema: type[T]
    ) -> T:
        completion = self._client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": context.model_dump_json()},
            ],
            response_format=schema,
        )
        return completion.choices[0].message.parsed
