"""
Wrapper for the phone number retrieval pipeline.
"""
import asyncio
import logging
from typing import List, Optional

from src.core.config import AppConfig
from src.core.schemas import ConsolidatedPhoneNumber, DomainExtractionBundle
from src.phone_retrieval.extractors.llm_extractor import GeminiLLMExtractor
from src.phone_retrieval.processing.pipeline_flow import execute_pipeline_flow

logger = logging.getLogger(__name__)

def retrieve_phone_numbers_for_url(url: str, company_name: str) -> Optional[List[ConsolidatedPhoneNumber]]:
    """
    Retrieves phone numbers for a given URL by running a simplified, in-memory version
    of the phone retrieval pipeline.

    Args:
        url: The URL to scrape for phone numbers.
        company_name: The name of the company associated with the URL.

    Returns:
        A list of ConsolidatedPhoneNumber objects, or None if an error occurs.
    """
    # This is a simplified integration. A more robust solution would involve
    # creating a shared configuration and a more modular pipeline structure.
    # For now, we will create a temporary in-memory setup.

    # Create a temporary DataFrame to drive the pipeline
    import pandas as pd
    df = pd.DataFrame([{"GivenURL": url, "CompanyName": company_name}])

    # Use a temporary, in-memory configuration for the phone retrieval
    app_config = AppConfig()
    llm_extractor = GeminiLLMExtractor(config=app_config)

    # We need to mock some of the pipeline's dependencies, like the failure writer
    class MockWriter:
        def writerow(self, row):
            pass

    # Execute the core pipeline flow from the phone_retrieval module
    try:
        _, _, _, final_consolidated_data, _, _, _, _ = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            llm_extractor=llm_extractor,
            run_output_dir="/tmp/phone_retrieval",  # Temporary, not used
            llm_context_dir="/tmp/phone_retrieval", # Temporary, not used
            run_id="temp_run",
            failure_writer=MockWriter(),
            run_metrics={},
            original_phone_col_name_for_profile=None
        )

        # Extract the consolidated numbers for the given URL
        if final_consolidated_data:
            # The key for final_consolidated_data is the canonical base URL.
            # We need to derive it to look up the results.
            from src.phone_retrieval.data_handling.consolidator import get_canonical_base_url
            canonical_url = get_canonical_base_url(url)
            if canonical_url and canonical_url in final_consolidated_data:
                domain_bundle = final_consolidated_data[canonical_url]
                if domain_bundle.company_contact_details:
                    return domain_bundle.company_contact_details.consolidated_numbers

    except Exception as e:
        logger.error(f"Error during phone number retrieval for {url}: {e}", exc_info=True)

    return None