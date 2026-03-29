from types import SimpleNamespace

from src.core.schemas import PartnerMatchOnlyOutput, PhoneRankingOutput
from src.llm_clients.common import CompatLLMResponse, UsageMetadata
from src.llm_clients.gemini_client import GeminiClient


def test_gemini_structured_output_parses_from_raw_text():
    client = GeminiClient.__new__(GeminiClient)
    client.config = SimpleNamespace(llm_temperature_extraction=0.2, llm_max_tokens=128)
    client.generate_content_with_retry = lambda **kwargs: CompatLLMResponse(
        text='{"match_score":"High","matched_partner_name":"Partner A","match_rationale_features":["industry fit"]}',
        usage_metadata=UsageMetadata(prompt_token_count=5, candidates_token_count=4, total_token_count=9),
        candidates=[{"text": "json"}],
    )

    result = GeminiClient.generate_structured_output_with_retry(
        client,
        user_prompt="prompt",
        response_model=PartnerMatchOnlyOutput,
        schema_name="partner_match_only",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert result.parsed_output is not None
    assert result.parsed_output.matched_partner_name == "Partner A"
    assert result.usage["total_tokens"] == 9


def test_gemini_structured_output_reports_parse_error():
    client = GeminiClient.__new__(GeminiClient)
    client.config = SimpleNamespace(llm_temperature_extraction=0.2, llm_max_tokens=128)
    client.generate_content_with_retry = lambda **kwargs: CompatLLMResponse(
        text="not json",
        usage_metadata=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
        candidates=[{"text": "not json"}],
    )

    result = GeminiClient.generate_structured_output_with_retry(
        client,
        user_prompt="prompt",
        response_model=PartnerMatchOnlyOutput,
        schema_name="partner_match_only",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert result.parsed_output is None
    assert result.status == "parse_error"
    assert "Failed to parse Gemini structured output" in (result.provider_error or "")


def test_gemini_structured_output_repairs_malformed_json_once():
    client = GeminiClient.__new__(GeminiClient)
    client.config = SimpleNamespace(llm_temperature_extraction=0.2, llm_max_tokens=128)
    responses = iter([
        CompatLLMResponse(
            text='{"match_score":"High","matched_partner_name":"Partner A","match_rationale_features":["broken"',
            usage_metadata=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
            candidates=[{"text": "broken"}],
        ),
        CompatLLMResponse(
            text='{"match_score":"High","matched_partner_name":"Partner A","match_rationale_features":["fixed"]}',
            usage_metadata=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
            candidates=[{"text": "fixed"}],
        ),
    ])
    client.generate_content_with_retry = lambda **kwargs: next(responses)

    result = GeminiClient.generate_structured_output_with_retry(
        client,
        user_prompt="prompt",
        response_model=PartnerMatchOnlyOutput,
        schema_name="partner_match_only",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert result.parsed_output is not None
    assert result.parsed_output.matched_partner_name == "Partner A"
    assert result.retry_count == 1


def test_gemini_structured_output_handles_schema_mismatch_without_throwing():
    client = GeminiClient.__new__(GeminiClient)
    client.config = SimpleNamespace(llm_temperature_extraction=0.2, llm_max_tokens=128)
    responses = iter([
        CompatLLMResponse(
            text='{"number":"+49123","reasoning":"wrong shape"}',
            usage_metadata=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
            candidates=[{"text": "wrong-shape"}],
        ),
        CompatLLMResponse(
            text='{"number":"+49123","reasoning":"still wrong shape"}',
            usage_metadata=UsageMetadata(prompt_token_count=1, candidates_token_count=1, total_token_count=2),
            candidates=[{"text": "wrong-shape"}],
        ),
    ])
    client.generate_content_with_retry = lambda **kwargs: next(responses)

    result = GeminiClient.generate_structured_output_with_retry(
        client,
        user_prompt="prompt",
        response_model=PhoneRankingOutput,
        schema_name="phone_ranking",
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
        system_prompt="Return JSON",
    )

    assert result.parsed_output is None
    assert result.status == "parse_error"
    assert "Failed to parse Gemini structured output" in (result.provider_error or "")
