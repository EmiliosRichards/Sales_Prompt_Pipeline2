from types import SimpleNamespace

from src.core.schemas import WebsiteTextSummaryLLM
from src.llm_clients.openai_client import OpenAIClient


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        openai_api_key="test-key",
        openai_timeout_seconds=30,
        openai_model_name="gpt-test",
        openai_model_name_sales_insights="gpt-test-sales",
        openai_service_tier="flex",
        openai_flex_max_retries=1,
        openai_flex_fallback_to_auto=False,
        openai_prompt_cache=False,
        openai_prompt_cache_retention="24h",
        openai_reasoning_effort=None,
    )


def test_openai_client_uses_explicit_schema_name(monkeypatch):
    client = OpenAIClient(_config())
    captured = {}

    response = SimpleNamespace(
        output_text='{"summary":"Summary","extracted_company_name_from_summary":"Acme","key_topics_mentioned":["ai"]}',
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        status="completed",
        output=[],
    )

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client.client.responses, "create", _fake_create)

    result = client.generate_structured_output_with_retry(
        user_prompt="prompt",
        response_model=WebsiteTextSummaryLLM,
        schema_name="website_text_summary",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert captured["text"]["format"]["name"] == "website_text_summary"
    assert result.parsed_output.summary == "Summary"
    assert result.usage["total_tokens"] == 18


def test_openai_client_surfaces_refusal_without_throwing(monkeypatch):
    client = OpenAIClient(_config())
    refusal_response = SimpleNamespace(
        output_text="",
        usage=SimpleNamespace(input_tokens=3, output_tokens=0, total_tokens=3),
        status="completed",
        output=[SimpleNamespace(type="refusal", refusal="safety")],
    )

    monkeypatch.setattr(client.client.responses, "create", lambda **kwargs: refusal_response)

    result = client.generate_structured_output_with_retry(
        user_prompt="prompt",
        response_model=WebsiteTextSummaryLLM,
        schema_name="website_text_summary",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert result.parsed_output is None
    assert result.refusal == "safety"
    assert "Model refusal" in (result.provider_error or "")


def test_openai_client_accepts_optional_generation_kwargs(monkeypatch):
    client = OpenAIClient(_config())
    response = SimpleNamespace(
        output_text='{"summary":"Summary","extracted_company_name_from_summary":"Acme","key_topics_mentioned":[]}',
        usage=SimpleNamespace(input_tokens=2, output_tokens=2, total_tokens=4),
        status="completed",
        output=[],
    )
    monkeypatch.setattr(client.client.responses, "create", lambda **kwargs: response)

    result = client.generate_structured_output_with_retry(
        user_prompt="prompt",
        response_model=WebsiteTextSummaryLLM,
        schema_name="website_text_summary",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
        temperature=0.2,
        max_output_tokens=256,
    )

    assert result.parsed_output is not None
