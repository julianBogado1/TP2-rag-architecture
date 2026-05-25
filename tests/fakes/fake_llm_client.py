from dataclasses import dataclass, field
from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class FakeLLMCall:
    method: str
    model: str
    system: str
    user: str | None
    context: BaseModel | None
    schema: type


@dataclass
class FakeLLMClient:
    returns: dict[str, BaseModel] = field(default_factory=dict)
    calls: list[FakeLLMCall] = field(default_factory=list)

    def parse_structured(self, model: str, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(FakeLLMCall("parse_structured", model, system, user, None, schema))
        return self.returns[schema.__name__]

    def generate_structured(self, model: str, system: str, context: BaseModel, schema: type[T]) -> T:
        self.calls.append(FakeLLMCall("generate_structured", model, system, None, context, schema))
        return self.returns[schema.__name__]
