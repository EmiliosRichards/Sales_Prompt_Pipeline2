import pandas as pd

from src.core.schemas import ConsolidatedPhoneNumber, ConsolidatedPhoneNumberSource
from src.phone_retrieval.processing.outcome_analyzer import determine_final_row_outcome_and_fault


def _sample_number() -> ConsolidatedPhoneNumber:
    return ConsolidatedPhoneNumber(
        number="+4930123456",
        classification="Primary",
        sources=[
            ConsolidatedPhoneNumberSource(
                type="Sales",
                source_path="/kontakt",
                original_full_url="https://example.com/kontakt",
            )
        ],
    )


def test_consolidated_numbers_without_exportable_top_numbers_are_downgraded():
    reason, fault = determine_final_row_outcome_and_fault(
        index=1,
        row_summary=pd.Series(dtype=object),
        df_status_snapshot={"ScrapingStatus": "Success"},
        company_contact_details_summary=None,
        unique_sorted_consolidated_numbers=[_sample_number()],
        canonical_url_summary="https://example.com",
        true_base_scraper_status_map={"https://example.com": "Success"},
        true_base_to_pathful_map={"https://example.com": ["https://example.com/kontakt"]},
        canonical_site_pathful_scraper_status={"https://example.com/kontakt": "Success"},
        canonical_site_raw_llm_outputs={},
        canonical_site_regex_candidates_found_status={"https://example.com": True},
        canonical_site_llm_exception_details={},
        has_exportable_top_numbers=False,
        rerank_requested=True,
        rerank_error="ValidationError",
    )

    assert reason == "Contact_Consolidated_RerankFailed_NoOperationalCallList"
    assert fault == "LLM Issue"


def test_exportable_top_numbers_keep_success_status():
    reason, fault = determine_final_row_outcome_and_fault(
        index=1,
        row_summary=pd.Series(dtype=object),
        df_status_snapshot={"ScrapingStatus": "Success"},
        company_contact_details_summary=None,
        unique_sorted_consolidated_numbers=[_sample_number()],
        canonical_url_summary="https://example.com",
        true_base_scraper_status_map={"https://example.com": "Success"},
        true_base_to_pathful_map={"https://example.com": ["https://example.com/kontakt"]},
        canonical_site_pathful_scraper_status={"https://example.com/kontakt": "Success"},
        canonical_site_raw_llm_outputs={},
        canonical_site_regex_candidates_found_status={"https://example.com": True},
        canonical_site_llm_exception_details={},
        has_exportable_top_numbers=True,
        rerank_requested=True,
        rerank_error=None,
    )

    assert reason == "Contact_Successfully_Extracted"
    assert fault == "N/A"
