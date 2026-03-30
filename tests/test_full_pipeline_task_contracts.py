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
from src.extractors.llm_tasks.generate_sales_pitch_task import (
    build_no_match_sales_pitch_output,
    build_weak_match_sales_pitch_output,
    generate_sales_pitch,
)
from src.extractors.llm_tasks.german_short_summary_from_description_task import generate_german_short_summary_from_description
from src.extractors.llm_tasks.match_partner_task import (
    assess_partner_match,
    build_partner_shortlist,
    match_partner,
    should_accept_partner_match,
)
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
        MAX_GOLDEN_PARTNERS_IN_PROMPT=3,
        partner_match_sparse_top_k=3,
        partner_match_dense_top_k=3,
        partner_match_fused_top_k=3,
        partner_match_rrf_k=60,
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
            match_confidence="strong",
            matched_partner_id="partner-a",
            matched_partner_name="Partner A",
            match_rationale_features=["fit"],
        )
    )
    target_attributes = DetailedCompanyAttributes(input_summary_url="https://acme.test")
    partner_result, _, _ = match_partner(
        gemini_client=partner_client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[{"partner_id": "partner-a", "name": "Partner A", "summary": "Summary"}],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )
    assert partner_result.matched_partner_name == "Partner A"
    assert partner_result.matched_partner_id == "partner-a"
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


def test_partner_shortlist_prioritizes_target_audience_even_cross_industry():
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        industry="Healthcare software",
        products_services_offered=["Scheduling workflows"],
        customer_target_segments=["HR teams in hospitals", "hospital administration"],
    )

    shortlist = build_partner_shortlist(
        target_attributes=target_attributes,
        golden_partner_summaries=[
            {
                "partner_id": "same-industry",
                "name": "Same Industry Wrong Audience",
                "industry": "Healthcare software",
                "services_products": "Patient engagement software",
                "target_audience": "Patients and consumers",
                "business_model": "SaaS",
                "summary": "Healthcare software for patients and consumers",
                "rank": "2",
            },
            {
                "partner_id": "cross-industry",
                "name": "Cross Industry Right Audience",
                "industry": "Workforce operations platform",
                "services_products": "Shift planning and compliance workflows",
                "target_audience": "HR teams, hospital administration, operations leads",
                "business_model": "SaaS",
                "summary": "Workflow platform for HR teams and hospital administration",
                "rank": "9",
            },
            {
                "partner_id": "generic-software",
                "name": "Generic Software",
                "industry": "Software",
                "services_products": "Automation platform",
                "target_audience": "General SMEs",
                "business_model": "SaaS",
                "summary": "Generic automation platform for SMEs",
                "rank": "1",
            },
        ],
        sparse_top_k=3,
        dense_top_k=3,
        fused_top_k=2,
        rrf_k=60,
    )

    assert [item["name"] for item in shortlist] == [
        "Cross Industry Right Audience",
        "Same Industry Wrong Audience",
    ]
    assert "Shared target audience" in " ".join(shortlist[0]["shortlist_overlap_features"])
    assert "hr_workforce_buyers" in shortlist[0]["shortlist_shared_taxonomy_tags"]["audience"]


def test_partner_shortlist_penalizes_false_friend_domain_matches():
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        industry="Shared mobility software",
        products_services_offered=["Vehicle booking", "Fleet management platform"],
        customer_target_segments=["Mobility managers", "fleet operators"],
    )

    shortlist = build_partner_shortlist(
        target_attributes=target_attributes,
        golden_partner_summaries=[
            {
                "partner_id": "mobility-fit",
                "name": "Mobility Fit",
                "industry": "Mobility operations platform",
                "services_products": "Shared mobility software with vehicle booking and fleet management",
                "target_audience": "Mobility managers, fleet operators, municipal utilities",
                "business_model": "SaaS",
                "summary": "Shared mobility and fleet platform",
                "rank": "5",
            },
            {
                "partner_id": "doc-false-friend",
                "name": "Document False Friend",
                "industry": "Document management software",
                "services_products": "Document workflow automation, tagging, archive",
                "target_audience": "SMEs and back-office teams",
                "business_model": "SaaS",
                "summary": "Document workflow platform",
                "rank": "1",
            },
        ],
        sparse_top_k=2,
        dense_top_k=2,
        fused_top_k=2,
        rrf_k=60,
    )

    assert [item["name"] for item in shortlist] == [
        "Mobility Fit",
        "Document False Friend",
    ]
    assert shortlist[0]["shortlist_taxonomy_penalty"] == 0
    assert shortlist[1]["shortlist_taxonomy_penalty"] > 0


def test_match_partner_only_prompts_with_shortlisted_candidates(tmp_path):
    config = _config(tmp_path)
    config.MAX_GOLDEN_PARTNERS_IN_PROMPT = 2
    client = FakeStructuredClient(
        PartnerMatchOnlyOutput(
            match_score="High",
            match_confidence="strong",
            matched_partner_id="cross-industry",
            matched_partner_name="Cross Industry Right Audience",
            match_rationale_features=["Gemeinsame Zielgruppe: HR-Teams in Kliniken"],
        )
    )
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        industry="Healthcare software",
        products_services_offered=["Scheduling workflows"],
        customer_target_segments=["HR teams in hospitals", "hospital administration"],
    )

    match_partner(
        gemini_client=client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[
            {
                "partner_id": "same-industry",
                "name": "Same Industry Wrong Audience",
                "industry": "Healthcare software",
                "services_products": "Patient engagement software",
                "target_audience": "Patients and consumers",
                "business_model": "SaaS",
                "summary": "Healthcare software for patients and consumers",
                "rank": "2",
            },
            {
                "partner_id": "cross-industry",
                "name": "Cross Industry Right Audience",
                "industry": "Workforce operations platform",
                "services_products": "Shift planning and compliance workflows",
                "target_audience": "HR teams, hospital administration, operations leads",
                "business_model": "SaaS",
                "summary": "Workflow platform for HR teams and hospital administration",
                "rank": "9",
            },
            {
                "partner_id": "generic-software",
                "name": "Generic Software",
                "industry": "Software",
                "services_products": "Automation platform",
                "target_audience": "General SMEs",
                "business_model": "SaaS",
                "summary": "Generic automation platform for SMEs",
                "rank": "1",
            },
        ],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    user_prompt = client.calls[-1]["user_prompt"]
    assert "Cross Industry Right Audience" in user_prompt
    assert "Same Industry Wrong Audience" in user_prompt
    assert "Generic Software" not in user_prompt


def test_match_partner_persists_shortlist_metadata(tmp_path):
    config = _config(tmp_path)
    client = FakeStructuredClient(
        PartnerMatchOnlyOutput(
            match_score="High",
            match_confidence="strong",
            matched_partner_id="cross-industry",
            matched_partner_name="Cross Industry Right Audience",
        )
    )
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        customer_target_segments=["HR teams in hospitals"],
    )

    partner_result, _, _ = match_partner(
        gemini_client=client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[
            {
                "partner_id": "cross-industry",
                "name": "Cross Industry Right Audience",
                "target_audience": "HR teams in hospitals",
                "services_products": "Shift planning",
            },
            {
                "partner_id": "generic",
                "name": "Generic Software",
                "target_audience": "SMEs",
                "services_products": "Automation platform",
            },
        ],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert partner_result is not None
    assert "Cross Industry Right Audience" in (partner_result.shortlisted_partner_names or [])
    assert "cross-industry" in (partner_result.shortlisted_partner_ids or [])


def test_match_partner_canonicalizes_partner_name_from_shortlist(tmp_path):
    config = _config(tmp_path)
    client = FakeStructuredClient(
        PartnerMatchOnlyOutput(
            match_score="High",
            match_confidence="strong",
            matched_partner_name="uberblick.io",
            match_rationale_features=["Gemeinsame Zielgruppe: Fachabteilungen mit dokumentenlastigen Prozessen"],
        )
    )
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        customer_target_segments=["Fachabteilungen mit dokumentenlastigen Prozessen"],
    )

    partner_result, _, _ = match_partner(
        gemini_client=client,
        config=config,
        target_attributes=target_attributes,
        golden_partner_summaries=[
            {"partner_id": "uberblick", "name": "Uberblick.io", "summary": "Document workflow platform", "target_audience": "Fachabteilungen"},
            {"partner_id": "generic", "name": "Generic Software", "summary": "Generic platform", "target_audience": "SMEs"},
        ],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert partner_result is not None
    assert partner_result.matched_partner_name == "Uberblick.io"


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
        matched_partner={"partner_id": "partner-a", "name": "Partner A", "summary": "Partner summary", "avg_leads_per_day": 9, "rank": 2},
        website_summary_obj=website_summary,
        partner_match_output=PartnerMatchOnlyOutput(
            match_score="High",
            match_confidence="strong",
            overlap_type="audience",
            matched_partner_id="partner-a",
            matched_partner_name="Partner A",
            target_evidence=["operations leads"],
            partner_evidence=["operations managers"],
        ),
        previous_match_rationale=["fit"],
        llm_context_dir=str(tmp_path / "ctx"),
        llm_requests_dir=str(tmp_path / "req"),
        file_identifier_prefix="row1",
        triggering_input_row_id=1,
        triggering_company_name="Acme",
    )

    assert result.matched_partner_name == "Partner A"
    assert result.matched_partner_id == "partner-a"
    assert result.match_confidence == "strong"
    assert result.analyzed_company_url == "https://acme.test"
    assert client.calls[-1]["response_model"] is SalesPitchLLMOutput
    assert client.calls[-1]["model_name_override"] == "gemini-sales-test"


def test_partner_match_gate_rejects_generic_medium_match():
    accepted, reason = should_accept_partner_match(
        PartnerMatchOnlyOutput(
            match_score="Medium",
            matched_partner_name="NxtLog",
            match_rationale_features=[
                "Gemeinsame Ausrichtung auf digitale Softwarelösungen",
                "Fokus auf Automatisierung und Effizienzsteigerung",
            ],
        ),
        matched_partner_data={"name": "NxtLog", "summary": "Logistics software summary"},
    )

    assert accepted is False
    assert reason == "medium_match_too_generic"


def test_partner_match_gate_accepts_specific_high_match():
    accepted, reason = should_accept_partner_match(
        PartnerMatchOnlyOutput(
            match_score="High",
            matched_partner_name="Digitalagentur1",
            match_rationale_features=[
                "Gemeinsame Branche: Digitale Agentur und IT-Dienstleistungen",
                "Die Zielgruppe umfasst Marken und Einzelhändler mit E-Commerce-Fokus",
            ],
        ),
        matched_partner_data={"name": "Digitalagentur1", "summary": "Digital agency summary"},
    )

    assert accepted is True
    assert reason == "accepted_high_match"


def test_no_match_pitch_builder_creates_personalized_pitch_without_partner():
    target_attributes = DetailedCompanyAttributes(
        input_summary_url="https://acme.test",
        industry="RFID- und ERP-Lösungen für Intralogistik",
        products_services_offered=["RFID-Lösungen", "Cloud ERP"],
        customer_target_segments=["Unternehmen in Intralogistik und Produktion"],
    )
    website_summary = WebsiteTextSummary(original_url="https://acme.test", summary="Acme liefert RFID- und ERP-Lösungen.")

    result = build_no_match_sales_pitch_output(
        target_attributes=target_attributes,
        website_summary_obj=website_summary,
        rejection_reason="medium_match_too_generic",
        company_name="Acme",
    )

    assert result.matched_partner_name is None
    assert "{programmatic placeholder}" not in (result.phone_sales_line or "")
    assert "Acme" in (result.phone_sales_line or "")
    assert "passenden Ansprechpartnern" in (result.phone_sales_line or "")
    assert "medium_match_too_generic" in " ".join(result.match_rationale_features or [])


def test_assess_partner_match_accepts_weak_with_validated_evidence():
    decision = assess_partner_match(
        PartnerMatchOnlyOutput(
            match_score="Medium",
            match_confidence="weak",
            overlap_type="use_case",
            matched_partner_id="workflow-partner",
            matched_partner_name="Workflow Partner",
            match_rationale_features=["Teilweise Überschneidung bei dokumentenlastigen Prozessen"],
            target_evidence=["contract automation software"],
            partner_evidence=["document automation and workflow management"],
        ),
        matched_partner_data={
            "partner_id": "workflow-partner",
            "name": "Workflow Partner",
            "services_products": "document automation and workflow management",
            "target_audience": "Legal departments",
        },
        target_attributes=DetailedCompanyAttributes(
            input_summary_url="https://acme.test",
            products_services_offered=["contract automation software"],
        ),
    )

    assert decision["decision"] == "weak_match"
    assert decision["reason"] == "accepted_weak_match"


def test_assess_partner_match_rejected_strong_becomes_no_match_confidence():
    decision = assess_partner_match(
        PartnerMatchOnlyOutput(
            match_score="High",
            match_confidence="strong",
            matched_partner_id="partner-a",
            matched_partner_name="Partner A",
            match_rationale_features=["Gemeinsame Zielgruppe: Industrieunternehmen"],
            target_evidence=[],
            partner_evidence=[],
        ),
        matched_partner_data={
            "partner_id": "partner-a",
            "name": "Partner A",
            "shortlist_score": 0.0,
            "shortlist_overlap_features": [],
        },
        target_attributes=DetailedCompanyAttributes(
            input_summary_url="https://acme.test",
            customer_target_segments=["Kliniken"],
        ),
    )

    assert decision["decision"] == "no_match"
    assert decision["reason"] == "rejected_strong_missing_evidence"
    assert decision["match_confidence"] == "no_match"


def test_weak_match_pitch_builder_preserves_partner_metadata():
    target_attributes = DetailedCompanyAttributes(input_summary_url="https://acme.test")
    website_summary = WebsiteTextSummary(original_url="https://acme.test", summary="Kurze Beschreibung")
    partner_match = PartnerMatchOnlyOutput(
        match_score="Medium",
        match_confidence="weak",
        overlap_type="audience",
        matched_partner_id="partner-a",
        matched_partner_name="Partner A",
        target_evidence=["HR teams"],
        partner_evidence=["HR managers"],
    )

    result = build_weak_match_sales_pitch_output(
        target_attributes=target_attributes,
        matched_partner={"partner_id": "partner-a", "name": "Partner A", "summary": "Partner summary", "rank": 3},
        website_summary_obj=website_summary,
        partner_match_output=partner_match,
        rejection_reason="accepted_weak_match",
        company_name="Acme",
    )

    assert result.matched_partner_id == "partner-a"
    assert result.matched_partner_name == "Partner A"
    assert result.match_confidence == "weak"
    assert "Acme" in (result.phone_sales_line or "")


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
