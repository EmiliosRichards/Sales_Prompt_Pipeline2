from pathlib import Path
from types import SimpleNamespace

from src.core.schemas import (
    B2BAnalysisOutput,
    DetailedCompanyAttributes,
    DetailedCompanyAttributesLLM,
    GermanShortSummaryOutput,
    PartnerMatchOnlyOutput,
    SalesPitchLLMOutput,
    WebsiteTextSummary,
    WebsiteTextSummaryLLM,
)
from src.extractors.llm_tasks.b2b_capacity_check_task import check_b2b_and_capacity
from src.extractors.llm_tasks.extract_attributes_task import extract_detailed_attributes
from src.extractors.llm_tasks.generate_sales_pitch_task import generate_sales_pitch
from src.extractors.llm_tasks.german_short_summary_from_description_task import generate_german_short_summary_from_description
from src.extractors.llm_tasks.match_partner_task import match_partner
from src.extractors.llm_tasks.summarize_task import generate_website_summary


class FakeStructuredClient:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output
        self.calls = []

    def generate_structured_output_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed_output=self.parsed_output,
            raw_text="{}",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            provider_error=None,
            refusal=None,
        )


class FakeMalformedPartnerClient(FakeStructuredClient):
    def __init__(self, raw_text):
        super().__init__(parsed_output=None)
        self.raw_text = raw_text

    def generate_structured_output_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            parsed_output=None,
            raw_text=self.raw_text,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            provider_error="Failed to parse Gemini structured output for PartnerMatchOnlyOutput.",
            refusal=None,
        )


def _write_prompt(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def _config(tmp_path: Path):
    return SimpleNamespace(
        PROMPT_PATH_WEBSITE_SUMMARIZER=_write_prompt(tmp_path, "summary.txt", "{{SCRAPED_WEBSITE_TEXT_PLACEHOLDER}}"),
        PROMPT_PATH_ATTRIBUTE_EXTRACTOR=_write_prompt(tmp_path, "attributes.txt", "{{WEBSITE_SUMMARY_TEXT_PLACEHOLDER}}"),
        PROMPT_PATH_GERMAN_PARTNER_MATCHING=_write_prompt(
            tmp_path,
            "partner.txt",
            "{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}\n{{GOLDEN_PARTNER_SUMMARIES_PLACEHOLDER}}",
        ),
        PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION=_write_prompt(
            tmp_path,
            "pitch.txt",
            "{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}\n{{MATCHED_GOLDEN_PARTNER_JSON_PLACEHOLDER}}\n{{PREVIOUS_MATCH_RATIONALE_PLACEHOLDER}}",
        ),
        PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION=_write_prompt(
            tmp_path,
            "german_short.txt",
            "{{INPUT_DESCRIPTION_PLACEHOLDER}}",
        ),
        filename_company_name_max_len=20,
        llm_max_tokens=256,
        llm_max_tokens_summary=128,
        llm_temperature_extraction=0.2,
        llm_temperature_creative=0.5,
        llm_model_name="gemini-test",
        llm_model_name_sales_insights="gemini-sales-test",
        openai_model_name_sales_insights="openai-sales-test",
        LLM_MAX_INPUT_CHARS_FOR_SUMMARY=1000,
        LLM_MAX_INPUT_CHARS_FOR_DESCRIPTION_DE_SUMMARY=1000,
        llm_max_tokens_description_de_summary=128,
    )


def test_generate_website_summary_uses_llm_specific_schema(tmp_path):
    client = FakeStructuredClient(
        WebsiteTextSummaryLLM(
            summary="Short summary",
            extracted_company_name_from_summary="Acme",
            key_topics_mentioned=["ai"],
        )
    )
    config = _config(tmp_path)

    result, _, _ = generate_website_summary(
        gemini_client=client,
        config=config,
        original_url="https://acme.test",
        scraped_text="Acme website text",
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert result.original_url == "https://acme.test"
    assert client.calls[-1]["response_model"] is WebsiteTextSummaryLLM


def test_extract_attributes_uses_llm_specific_schema(tmp_path):
    client = FakeStructuredClient(
        DetailedCompanyAttributesLLM(
            industry="Manufacturing",
            products_services_offered=["Widgets"],
        )
    )
    config = _config(tmp_path)
    summary = WebsiteTextSummary(original_url="https://acme.test", summary="Summary")

    result, _, _ = extract_detailed_attributes(
        gemini_client=client,
        config=config,
        summary_obj=summary,
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert result.input_summary_url == "https://acme.test"
    assert client.calls[-1]["response_model"] is DetailedCompanyAttributesLLM


def test_b2b_and_partner_tasks_use_explicit_structured_models(tmp_path):
    config = _config(tmp_path)

    b2b_client = FakeStructuredClient(
        B2BAnalysisOutput(
            is_b2b="Yes",
            is_b2b_reason="B2B",
            serves_1000_customers="Yes",
            serves_1000_customers_reason="Scale",
        )
    )
    b2b_result, _, _ = check_b2b_and_capacity(
        gemini_client=b2b_client,
        config=config,
        company_text="Company text",
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )
    assert b2b_result.is_b2b == "Yes"
    assert b2b_client.calls[-1]["response_model"] is B2BAnalysisOutput

    partner_client = FakeStructuredClient(
        PartnerMatchOnlyOutput(
            match_score="High",
            matched_partner_name="Partner A",
            match_rationale_features=["fit"],
        )
    )
    target_attributes = DetailedCompanyAttributes(input_summary_url="https://acme.test")
    partner_result, _, _ = match_partner(
        gemini_client=partner_client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[{"name": "Partner A", "summary": "Summary"}],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )
    assert partner_result.matched_partner_name == "Partner A"
    assert partner_client.calls[-1]["response_model"] is PartnerMatchOnlyOutput


def test_match_partner_salvages_partial_structured_output(tmp_path):
    config = _config(tmp_path)
    client = FakeMalformedPartnerClient(
        '{"match_score":"High","matched_partner_name":"Partner Salvaged","match_rationale_features":["broken"'
    )
    target_attributes = DetailedCompanyAttributes(input_summary_url="https://acme.test")

    partner_result, raw_text, _ = match_partner(
        gemini_client=client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[{"name": "Partner Salvaged", "summary": "Summary"}],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert partner_result is not None
    assert partner_result.matched_partner_name == "Partner Salvaged"
    assert partner_result.match_rationale_features == []
    assert '"matched_partner_name":"Partner Salvaged"' in raw_text


def test_sales_pitch_task_enriches_programmatic_fields_and_uses_sales_schema(tmp_path):
    client = FakeStructuredClient(
        SalesPitchLLMOutput(
            match_score="High",
            match_rationale_features=["fit"],
            phone_sales_line="Pitch text",
        )
    )
    config = _config(tmp_path)
    target_attributes = DetailedCompanyAttributes(input_summary_url="https://acme.test")
    website_summary = WebsiteTextSummary(original_url="https://acme.test", summary="Summary")

    result, _, _ = generate_sales_pitch(
        gemini_client=client,
        config=config,
        target_attributes=target_attributes,
        matched_partner={"name": "Partner A", "summary": "Partner summary", "avg_leads_per_day": 9, "rank": 2},
        website_summary_obj=website_summary,
        previous_match_rationale=["fit"],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert result.matched_partner_name == "Partner A"
    assert result.analyzed_company_url == "https://acme.test"
    assert client.calls[-1]["response_model"] is SalesPitchLLMOutput
    assert client.calls[-1]["model_name_override"] == "gemini-sales-test"


def test_german_short_summary_task_uses_explicit_schema(tmp_path):
    client = FakeStructuredClient(GermanShortSummaryOutput(german_summary="Kurze deutsche Zusammenfassung"))
    config = _config(tmp_path)

    result, _, _ = generate_german_short_summary_from_description(
        gemini_client=client,
        config=config,
        description_text="English company description",
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert "Zusammenfassung" in result
    assert client.calls[-1]["response_model"] is GermanShortSummaryOutput
