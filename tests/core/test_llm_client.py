from app.core.llm_client import LLMClient, OpenAILLMClient


def test_llm_client_protocol_methods_exist():
    assert hasattr(LLMClient, "parse_structured")
    assert hasattr(LLMClient, "generate_structured")


def test_openai_client_constructs():
    client = OpenAILLMClient(api_key="sk-test")
    assert client is not None
