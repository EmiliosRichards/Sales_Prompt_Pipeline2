# General-purpose utility functions
import logging
import csv
import json
import re # Added for sanitize_filename_component
from typing import Optional, Any
from datetime import datetime
from urllib.parse import urlparse
import phonenumbers
from phonenumbers import NumberParseException

# It's better to pass constants like TARGET_COUNTRY_CODES_INT if they are needed,
# or import them directly if they are truly global and stable.
# For now, assuming direct import from where they will reside.
from src.core.constants import TARGET_COUNTRY_CODES_INT
# Assuming normalize_url will also be a utility or imported within functions that need it.
# For now, if get_input_canonical_url needs it, it should be imported there or passed.
# We will import it from scraper_logic for now as it was in main_pipeline
from src.scraper.scraper_logic import normalize_url


logger = logging.getLogger(__name__)

from src.core.config import AppConfig # Added for setup_output_directories
from typing import Dict, Tuple # Added for type hints
import os # Added for path operations

def resolve_path(path_str: str, base_file_path: str) -> str:
    """
    Resolves a given path string to an absolute path.
    If the path_str is already absolute, it's returned directly.
    Otherwise, it's resolved relative to the directory of base_file_path.
    """
    if not os.path.isabs(path_str):
        project_root_dir = os.path.dirname(os.path.abspath(base_file_path))
        return os.path.join(project_root_dir, path_str)
    return path_str

def initialize_run_metrics(run_id: Optional[str]) -> Dict[str, Any]:
    """Initializes and returns the base structure for run metrics."""
    return {
        "run_id": run_id,
        "total_duration_seconds": None,
        "tasks": {},
        "data_processing_stats": {
            "input_rows_count": 0,
            "rows_successfully_processed_pass1": 0,
            "rows_failed_pass1": 0,
            "row_level_failure_summary": {},
            "input_unique_company_names": 0,
            "input_unique_canonical_urls": 0,
            "input_company_names_with_duplicates_count": 0,
            "input_canonical_urls_with_duplicates_count": 0,
            "input_rows_with_duplicate_company_name": 0,
            "input_rows_with_duplicate_canonical_url": 0,
            "input_rows_considered_duplicates_overall": 0,
            "unique_true_base_domains_consolidated": 0,
            "rows_in_attrition_report": 0
        },
        "scraping_stats": {
            "urls_processed_for_scraping": 0,
            "scraping_success": 0,
            "scraping_failure_invalid_url": 0,
            "scraping_failure_already_processed": 0,
            "scraping_failure_error": 0,
            "new_canonical_sites_scraped": 0,
            "total_pages_scraped_overall": 0,
            "pages_scraped_by_type": {},
            "total_successful_canonical_scrapes": 0,
            "total_urls_fetched_by_scraper": 0,
        },
        "regex_extraction_stats": {
            "sites_processed_for_regex": 0,
            "sites_with_regex_candidates": 0,
            "total_regex_candidates_found": 0,
        },
        "llm_processing_stats": {
            "sites_processed_for_llm": 0,
            "llm_calls_success": 0,
            "llm_calls_failure_prompt_missing": 0,
            "llm_calls_failure_processing_error": 0,
            "llm_no_candidates_to_process": 0,
            "total_llm_extracted_numbers_raw": 0,
            "total_llm_prompt_tokens": 0,
            "total_llm_completion_tokens": 0,
            "total_llm_tokens_overall": 0,
            "llm_successful_calls_with_token_data": 0,
        },
        "report_generation_stats": {
            "detailed_report_rows": 0,
            "summary_report_rows": 0,
            "tertiary_report_rows": 0,
            "canonical_domain_summary_rows": 0,
        },
        "errors_encountered": []
    }

def setup_output_directories(app_config: AppConfig, run_id: str, base_file_path: str) -> Tuple[str, str, str]:
    """
    Sets up the output directories for the pipeline run.
    Returns paths for the run-specific output, LLM context, and LLM requests directories.
    """
    output_base_dir_abs = app_config.output_base_dir
    if not os.path.isabs(output_base_dir_abs):
        project_root_dir_local = os.path.dirname(os.path.abspath(base_file_path))
        output_base_dir_abs = os.path.join(project_root_dir_local, output_base_dir_abs)
        
    run_output_dir = os.path.join(output_base_dir_abs, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    
    llm_context_dir = os.path.join(run_output_dir, app_config.llm_context_subdir)
    os.makedirs(llm_context_dir, exist_ok=True)

    llm_requests_dir = os.path.join(run_output_dir, "llm_requests")
    os.makedirs(llm_requests_dir, exist_ok=True)
    
    logger.info(f"Output directories created: run_output_dir={run_output_dir}, llm_context_dir={llm_context_dir}, llm_requests_dir={llm_requests_dir}")
    
    return run_output_dir, llm_context_dir, llm_requests_dir
import pandas as pd # Added for DataFrame operations
from collections import Counter # Added for duplicate counting
from typing import List # Added for type hints

def precompute_input_duplicate_stats(df: pd.DataFrame, app_config: AppConfig, run_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Pre-computes statistics about duplicate company names and URLs in the input DataFrame.
    Updates run_metrics with these stats and returns the DataFrame (potentially with temporary columns removed).
    """
    logger.info("Starting pre-computation of input duplicate counts...")
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, app_config.INPUT_COLUMN_PROFILES['default'])
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')
    url_col_key = active_profile.get('GivenURL', 'GivenURL')

    input_company_names_list: List[str] = []
    input_derived_canonical_urls_list: List[str] = []
    for row_tuple in df.itertuples(index=False):
        company_name_val = str(getattr(row_tuple, company_name_col_key, "MISSING_COMPANY_NAME_INPUT")).strip()
        input_company_names_list.append(company_name_val)
        given_url_val = getattr(row_tuple, url_col_key, None)
        derived_input_canonical = get_input_canonical_url(given_url_val) # Uses existing helper
        input_derived_canonical_urls_list.append(derived_input_canonical if derived_input_canonical else "MISSING_OR_INVALID_URL_INPUT")

    company_name_counts = Counter(input_company_names_list)
    input_canonical_url_counts = Counter(input_derived_canonical_urls_list)
    
    run_metrics["data_processing_stats"]["input_unique_company_names"] = len(company_name_counts)
    run_metrics["data_processing_stats"]["input_unique_canonical_urls"] = len(input_canonical_url_counts)
    
    num_company_names_with_duplicates = sum(1 for name, count in company_name_counts.items() if count > 1 and name != "MISSING_COMPANY_NAME_INPUT")
    num_urls_with_duplicates = sum(1 for url, count in input_canonical_url_counts.items() if count > 1 and url != "MISSING_OR_INVALID_URL_INPUT")
    run_metrics["data_processing_stats"]["input_company_names_with_duplicates_count"] = num_company_names_with_duplicates
    run_metrics["data_processing_stats"]["input_canonical_urls_with_duplicates_count"] = num_urls_with_duplicates
    
    total_rows_with_dup_company = sum(count for name, count in company_name_counts.items() if count > 1 and name != "MISSING_COMPANY_NAME_INPUT")
    total_rows_with_dup_url = sum(count for url, count in input_canonical_url_counts.items() if count > 1 and url != "MISSING_OR_INVALID_URL_INPUT")
    run_metrics["data_processing_stats"]["input_rows_with_duplicate_company_name"] = total_rows_with_dup_company
    run_metrics["data_processing_stats"]["input_rows_with_duplicate_canonical_url"] = total_rows_with_dup_url
    
    rows_considered_duplicates_overall = 0
    # Use temporary columns for efficient checking, then drop them.
    df['temp_input_canonical_url_for_dup_count'] = input_derived_canonical_urls_list
    df['temp_input_company_name_for_dup_count'] = input_company_names_list
    
    for _, row_data in df.iterrows():
        is_dup_company = company_name_counts[row_data['temp_input_company_name_for_dup_count']] > 1 and row_data['temp_input_company_name_for_dup_count'] != "MISSING_COMPANY_NAME_INPUT"
        is_dup_url = input_canonical_url_counts[row_data['temp_input_canonical_url_for_dup_count']] > 1 and row_data['temp_input_canonical_url_for_dup_count'] != "MISSING_OR_INVALID_URL_INPUT"
        if is_dup_company or is_dup_url:
            rows_considered_duplicates_overall += 1
            
    run_metrics["data_processing_stats"]["input_rows_considered_duplicates_overall"] = rows_considered_duplicates_overall
    df.drop(columns=['temp_input_canonical_url_for_dup_count', 'temp_input_company_name_for_dup_count'], inplace=True)
    logger.info("Input duplicate pre-computation complete.")
    return df
def initialize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the DataFrame has all required columns for the pipeline,
    initializing them with default values if they are missing.
    """
    df_length = len(df)
    required_cols: Dict[str, Any] = {
        'ScrapingStatus': '', 
        'RegexCandidateSnippets': lambda: [[] for _ in range(df_length)],
        'BestMatchedPhoneNumbers': lambda: [[] for _ in range(df_length)], 
        'OtherRelevantNumbers': lambda: [[] for _ in range(df_length)],
        'ConfidenceScore': None, 
        'LLMExtractedNumbers': lambda: [[] for _ in range(df_length)], 
        'LLMContextPath': '', 
        'Notes': '',
        'Top_Number_1': None, 'Top_Type_1': None, 'Top_SourceURL_1': None,
        'Top_Number_2': None, 'Top_Type_2': None, 'Top_SourceURL_2': None,
        'Top_Number_3': None, 'Top_Type_3': None, 'Top_SourceURL_3': None,
        'Final_Row_Outcome_Reason': pd.Series([None] * df_length, dtype=object),
        'Determined_Fault_Category': pd.Series([None] * df_length, dtype=object)
    }
    for col, default_val in required_cols.items():
        if col not in df.columns:
            # Special handling for 'TargetCountryCodes' if it were in required_cols,
            # but it's usually added during load_and_preprocess_data or expected to exist.
            # If it needs to be part of this initialization, it would be:
            # if col == 'TargetCountryCodes': df[col] = pd.Series([[] for _ in range(df_length)], dtype=object)
            df[col] = default_val() if callable(default_val) else default_val
    
    # Ensure other commonly used columns exist, even if they are mostly populated later
    if 'GivenPhoneNumber' not in df.columns: 
        df['GivenPhoneNumber'] = None
    if 'Description' not in df.columns: 
        df['Description'] = None
    # 'CanonicalEntryURL' is typically added by pipeline_flow, but ensure it if checking here.
    # if 'CanonicalEntryURL' not in df.columns:
    #     df['CanonicalEntryURL'] = None


    logger.debug("DataFrame columns initialized/verified.")
    return df

def is_target_country_number_reliable(phone_number_str: str) -> bool:
    if not phone_number_str or not isinstance(phone_number_str, str):
        return False
    try:
        parsed_num = phonenumbers.parse(phone_number_str, None)
        return parsed_num.country_code in TARGET_COUNTRY_CODES_INT
    except NumberParseException:
        logger.debug(f"NumberParseException for '{phone_number_str}' during target country check.")
        return False

def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def log_row_failure(
    failure_log_writer: Optional[Any], # csv.writer object
    input_row_identifier: Any,
    company_name: str,
    given_url: Optional[str],
    stage_of_failure: str,
    error_reason: str,
    log_timestamp: str,
    error_details: str = "",
    associated_pathful_canonical_url: Optional[str] = None
) -> None:
    """Helper function to write a row-specific failure to the CSV log."""
    if failure_log_writer:
        try:
            sanitized_reason = str(error_reason).replace('\n', ' ').replace('\r', '')
            sanitized_details = str(error_details).replace('\n', ' ').replace('\r', '')
            row_to_write = [
                log_timestamp,
                input_row_identifier,
                company_name,
                given_url if given_url is not None else "",
                stage_of_failure,
                sanitized_reason,
                sanitized_details,
                associated_pathful_canonical_url if associated_pathful_canonical_url is not None else ""
            ]
            failure_log_writer.writerow(row_to_write)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to write to failure_log_csv: {e}. Row ID: {input_row_identifier}, Stage: {stage_of_failure}, Timestamp: {log_timestamp}", exc_info=True)
    else:
        logger.warning(f"Attempted to log row failure but failure_log_writer is None. Row ID: {input_row_identifier}, Stage: {stage_of_failure}, Timestamp: {log_timestamp}")

def get_input_canonical_url(url_string: Optional[str]) -> Optional[str]:
    """
    Normalizes a given URL string and extracts its netloc (domain part)
    to serve as an 'input canonical URL' for pre-computation duplicate checks.
    """
    if not url_string or not isinstance(url_string, str):
        return None
    
    temp_url = url_string.strip()
    if not temp_url: # Handle empty string after strip
        return None

    # Ensure a scheme is present for consistent parsing by normalize_url and for netloc extraction
    parsed_initial = urlparse(temp_url)
    if not parsed_initial.scheme:
        temp_url = "http://" + temp_url
    
    try:
        normalized = normalize_url(temp_url)
        if not normalized:
            logger.debug(f"normalize_url returned None for input: {temp_url} (original: {url_string})")
            return None
        parsed_normalized_url = urlparse(normalized)
        return parsed_normalized_url.netloc if parsed_normalized_url.netloc else None
    except Exception as e:
        logger.debug(f"Could not parse/normalize input URL '{url_string}' for canonical pre-check: {e}")
        return None

def sanitize_filename_component(name_part: str, max_len: int = 50) -> str:
    """Replaces problematic characters in a string to make it suitable for a filename component.
    
    Args:
        name_part (str): The string to sanitize.
        max_len (int): The maximum allowed length for the sanitized component. Default is 50.
    """
    if not isinstance(name_part, str):
        name_part = str(name_part)
    name_part = name_part.replace(' ', '_') # Replace spaces with underscores
    # Remove characters that are generally problematic in filenames across OS
    # \ / : * ? " < > |
    name_part = re.sub(r'[\\/:*?"<>|]', '', name_part)
    # Truncate if longer than max_len
    if len(name_part) > max_len:
        name_part = name_part[:max_len]
    return name_part