"""
Wrapper for the phone number retrieval pipeline.

This wrapper is used by the full pipeline to run phone retrieval for a single URL.
"""
import asyncio
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import AppConfig
from src.core.schemas import ConsolidatedPhoneNumber
from src.phone_retrieval.extractors.llm_extractor import GeminiLLMExtractor
from src.phone_retrieval.processing.pipeline_flow import execute_pipeline_flow
from src.phone_retrieval.utils.helpers import initialize_dataframe_columns

logger = logging.getLogger(__name__)

def retrieve_phone_numbers_for_url(
    url: str,
    company_name: str,
    app_config: AppConfig,
    run_output_dir: Optional[str] = None,
    llm_context_dir: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Tuple[Optional[List[ConsolidatedPhoneNumber]], str, Dict[str, Any]]:
    """
    Retrieves phone numbers for a given URL by running a simplified, in-memory version
    of the phone retrieval pipeline.

    Args:
        url: The URL to scrape for phone numbers.
        company_name: The name of the company associated with the URL.
        app_config: The application configuration object.

    Returns:
        A tuple containing:
        - A list of ConsolidatedPhoneNumber objects, or None if an error occurs.
        - A status string indicating the outcome of the retrieval process.
    """
    # This is a simplified integration. A more robust solution would involve
    # creating a shared configuration and a more modular pipeline structure.
    # For now, we will create a temporary in-memory setup.

    # Create a temporary DataFrame to drive the pipeline
    import pandas as pd
    df = pd.DataFrame([{"GivenURL": url, "CompanyName": company_name}])
    df = initialize_dataframe_columns(df)

    # Use the provided app_config for the phone retrieval
    llm_extractor = GeminiLLMExtractor(config=app_config)

    # We need to mock some of the pipeline's dependencies, like the failure writer
    class MockWriter:
        def writerow(self, row):
            pass

    # Default to a temp folder if the full pipeline didn't pass run directories.
    # This must be Windows-safe (no hardcoded /tmp).
    base_out = run_output_dir or os.path.join(tempfile.gettempdir(), "sales_prompt_pipeline_phone_retrieval")
    base_ctx = llm_context_dir or os.path.join(base_out, "llm_context")
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(base_ctx, exist_ok=True)
    effective_run_id = run_id or "temp_run"

    # Execute the core pipeline flow from the phone_retrieval module
    try:
        run_metrics = {
            "scraping_stats": {
                "urls_processed_for_scraping": 0, "scraping_failure_invalid_url": 0,
                "new_canonical_sites_scraped": 0, "scraping_failure_already_processed": 0,
                "scraping_failure_error": 0, "scraping_success": 0,
                "total_pages_scraped_overall": 0, "pages_scraped_by_type": {},
                "total_successful_canonical_scrapes": 0,
            },
            "tasks": {}, "errors_encountered": [],
            "regex_extraction_stats": {
                "sites_processed_for_regex": 0, "sites_with_regex_candidates": 0,
                "total_regex_candidates_found": 0,
            },
            "llm_processing_stats": {
                "sites_processed_for_llm": 0, "llm_calls_success": 0,
                "llm_calls_failure_prompt_missing": 0, "llm_calls_failure_processing_error": 0,
                "llm_no_candidates_to_process": 0, "total_llm_extracted_numbers_raw": 0,
                "llm_successful_calls_with_token_data": 0, "total_llm_prompt_tokens": 0,
                "total_llm_completion_tokens": 0, "total_llm_tokens_overall": 0,
                "sites_already_attempted_llm_or_skipped": set(),
            },
            "data_processing_stats": {
                "rows_successfully_processed_pass1": 0, "rows_failed_pass1": 0,
                "row_level_failure_summary": {}, "unique_true_base_domains_consolidated": 0,
            }
        }
        df_processed, _, _, final_consolidated_data, _, _, _, _ = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            llm_extractor=llm_extractor,
            run_output_dir=base_out,
            llm_context_dir=base_ctx,
            run_id=effective_run_id,
            failure_writer=MockWriter(),
            run_metrics=run_metrics,
            original_phone_col_name_for_profile=None
        )

        meta: Dict[str, Any] = {}
        try:
            if df_processed is not None and not df_processed.empty:
                row0 = df_processed.iloc[0].to_dict()
                keep = [
                    "Top_Number_1", "Top_Type_1", "Top_SourceURL_1",
                    "Top_Number_2", "Top_Type_2", "Top_SourceURL_2",
                    "Top_Number_3", "Top_Type_3", "Top_SourceURL_3",
                    "Primary_Number_1", "Primary_Type_1", "Primary_SourceURL_1",
                    "Secondary_Number_1", "Secondary_Type_1", "Secondary_SourceURL_1",
                    "Secondary_Number_2", "Secondary_Type_2", "Secondary_SourceURL_2",
                    "RegexCandidateSnippets", "BestMatchedPhoneNumbers", "OtherRelevantNumbers",
                    "LLMExtractedNumbers", "LLMContextPath",
                    "Final_Row_Outcome_Reason", "Determined_Fault_Category",
                    "ScrapingStatus",
                    # Person-associated contact fields (optional)
                    "BestPersonContactName", "BestPersonContactRole", "BestPersonContactDepartment", "BestPersonContactNumber",
                    "PersonContacts",
                    # Main office / switchboard backup + ranking trace
                    "MainOffice_Number", "MainOffice_Type", "MainOffice_SourceURL",
                    "LLMPhoneRanking",
                    "LLMPhoneRankingError",
                ]
                meta = {k: row0.get(k) for k in keep if k in row0}
        except Exception:
            meta = {}

        if final_consolidated_data:
            domain_bundle = next(iter(final_consolidated_data.values()), None)
            if domain_bundle and domain_bundle.company_contact_details:
                numbers = domain_bundle.company_contact_details.consolidated_numbers or []
                if len(numbers) > 0:
                    return numbers, "Success", meta
                # ContactDetails exists but no consolidated numbers made it through.
                # Treat this as "no numbers found" rather than a successful retrieval.
                # Fall through to interpret via run_metrics below.
        
        if run_metrics["regex_extraction_stats"]["total_regex_candidates_found"] == 0:
            return None, "No_Candidates_Found", meta

    except Exception as e:
        logger.error(f"Error during phone number retrieval for {url}: {e}", exc_info=True)
        return None, "Error", {}

    return None, "No_Main_Line_Found", meta