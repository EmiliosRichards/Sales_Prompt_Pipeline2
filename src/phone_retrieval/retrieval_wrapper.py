"""
Wrapper for the phone number retrieval pipeline.
"""
import asyncio
import logging
from typing import List, Optional, Tuple

from src.core.config import AppConfig
from src.core.schemas import ConsolidatedPhoneNumber, DomainExtractionBundle
from src.phone_retrieval.extractors.llm_extractor import GeminiLLMExtractor
from src.phone_retrieval.processing.pipeline_flow import execute_pipeline_flow

logger = logging.getLogger(__name__)

def retrieve_phone_numbers_for_url(url: str, company_name: str, app_config: AppConfig) -> Tuple[Optional[List[ConsolidatedPhoneNumber]], str]:
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

    # Use the provided app_config for the phone retrieval
    llm_extractor = GeminiLLMExtractor(config=app_config)

    # We need to mock some of the pipeline's dependencies, like the failure writer
    class MockWriter:
        def writerow(self, row):
            pass

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
        _, _, _, final_consolidated_data, _, _, _, _ = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            llm_extractor=llm_extractor,
            run_output_dir="/tmp/phone_retrieval",
            llm_context_dir="/tmp/phone_retrieval",
            run_id="temp_run",
            failure_writer=MockWriter(),
            run_metrics=run_metrics,
            original_phone_col_name_for_profile=None
        )

        if final_consolidated_data:
            domain_bundle = next(iter(final_consolidated_data.values()), None)
            if domain_bundle and domain_bundle.company_contact_details:
                return domain_bundle.company_contact_details.consolidated_numbers, "Success"
        
        if run_metrics["regex_extraction_stats"]["total_regex_candidates_found"] == 0:
            return None, "No_Candidates_Found"

    except Exception as e:
        logger.error(f"Error during phone number retrieval for {url}: {e}", exc_info=True)
        return None, "Error"

    return None, "No_Main_Line_Found"