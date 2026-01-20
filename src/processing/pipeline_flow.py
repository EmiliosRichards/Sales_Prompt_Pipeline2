"""
Core pipeline execution flow for processing company data.

This module orchestrates the main data processing pipeline, which involves:
1.  Iterating through input company data (typically from a DataFrame).
2.  Processing and validating input URLs.
3.  Scraping website content for valid URLs.
4.  A 3-stage LLM (Large Language Model) process:
    a.  Generate a summary of the website content.
    b.  Extract detailed company attributes from the summary.
    c.  Generate sales insights by comparing extracted attributes against
        golden partner profiles.
5.  Collecting metrics, logging failures, and preparing outputs.

The pipeline is designed to be resilient, handling errors at each stage and
continuing processing for other rows where possible. It also tracks various
data points for reporting and analysis, such as scraper statuses and LLM
call statistics.
"""
import pandas as pd
import asyncio
import json
import os
import time
from datetime import datetime
import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import Counter

from src.core.config import AppConfig
from src.core.schemas import (
    WebsiteTextSummary, DetailedCompanyAttributes, GoldenPartnerMatchOutput, PartnerMatchOnlyOutput, B2BAnalysisOutput
)
from src.data_handling.consolidator import get_canonical_base_url
from src.llm_clients.gemini_client import GeminiClient
from src.scraper import scrape_website
from src.scraper import scraper_logic as full_scraper_logic
from src.extractors.llm_tasks.summarize_task import generate_website_summary
from src.extractors.llm_tasks.extract_attributes_task import extract_detailed_attributes
from src.extractors.llm_tasks.match_partner_task import match_partner
from src.extractors.llm_tasks.generate_sales_pitch_task import generate_sales_pitch
from src.extractors.llm_tasks.b2b_capacity_check_task import check_b2b_and_capacity
from src.utils.helpers import log_row_failure, sanitize_filename_component
from src.processing.url_processor import process_input_url
from src.phone_retrieval.retrieval_wrapper import retrieve_phone_numbers_for_url
from src.utils.helpers import should_attempt_phone_retrieval


def _normalize_phone_str(val: Any) -> str:
    """Normalize phone-like values from pandas cells into a usable string (or '')."""
    try:
        if val is None:
            return ""
        if isinstance(val, float) and pd.isna(val):
            return ""
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return ""
        if s.startswith("'") and len(s) > 1:
            s = s[1:].strip()
        return s
    except Exception:
        return ""


def _is_blank_cell(val: Any) -> bool:
    return _normalize_phone_str(val) == ""


def _pick_first_usable_phone(row: pd.Series, df: pd.DataFrame, col_names: List[str]) -> str:
    """Return the first phone value that looks usable (or '' if none)."""
    for col in col_names:
        if not col:
            continue
        if col not in df.columns:
            continue
        s = _normalize_phone_str(row.get(col))
        if s and not should_attempt_phone_retrieval(s):
            return s
    return ""


def _format_avg_leads_per_day(val: Any) -> Optional[str]:
    """Format avg leads per day for insertion into sales pitch text."""
    try:
        if val is None:
            return None
        if isinstance(val, float) and pd.isna(val):
            return None
        # Pydantic may coerce strings to float already, but be defensive.
        if isinstance(val, (int, float)):
            # Most partners use whole numbers; keep it tidy.
            return f"{val:.0f}"
        s = str(val).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return None
        try:
            f = float(s)
            return f"{f:.0f}"
        except Exception:
            return s
    except Exception:
        return None


def _fill_programmatic_placeholder(pitch: Optional[str], avg_leads_per_day: Any) -> Optional[str]:
    if not pitch:
        return pitch
    marker = "{programmatic placeholder}"
    if marker not in pitch:
        return pitch
    formatted = _format_avg_leads_per_day(avg_leads_per_day)
    if not formatted:
        return pitch
    return pitch.replace(marker, formatted)
from src.reporting.live_csv_reporter import LiveCsvReporter
from src.reporting.live_jsonl_reporter import LiveJsonlReporter

logger = logging.getLogger(__name__)

# Define a type alias for the complex return tuple for better readability
PipelineOutput = Tuple[
    pd.DataFrame,
    List[GoldenPartnerMatchOutput],
    List[Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, Optional[str]],
    Dict[str, int]
]

def execute_pipeline_flow(
    df: pd.DataFrame,
    app_config: AppConfig,
    gemini_client: GeminiClient,
    run_output_dir: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    run_id: str,
    failure_writer: Any,  # csv.writer object
    run_metrics: Dict[str, Any],
    golden_partner_summaries: List[Dict[str, Any]],
    skip_prequalification: bool = False,
    pitch_from_description: bool = False,
    live_reporter: Optional[LiveCsvReporter] = None,
    row_queue: Optional[Any] = None,
    row_queue_meta: Optional[Dict[str, Any]] = None,
) -> PipelineOutput:
    """
    Executes the core data processing flow of the pipeline.

    The flow per input row is:
    1. URL Validation: Input URL is processed and validated.
    2. Scrape Website: Content is scraped from the validated URL.
    3. LLM - Summarize: Scraped text is summarized by an LLM.
    4. LLM - Extract Attributes: Detailed attributes are extracted from the summary by an LLM.
    5. LLM - Compare & Sales Line: Attributes are compared to golden partner profiles,
       and sales insights are generated by an LLM.

    Failures at any step are logged, and the pipeline attempts to continue with the next row.

    Args:
        df: Input DataFrame containing company data.
        app_config: Application configuration object.
        gemini_client: Client for interacting with the Gemini LLM.
        run_output_dir: Directory where scraper outputs (like HTML files) are stored.
        llm_context_dir: Directory where LLM interaction context (prompts, responses)
                         is stored for debugging and analysis.
        llm_requests_dir: Directory where LLM request payloads are stored.
        run_id: Unique identifier for the current pipeline run.
        failure_writer: A CSV writer object for logging row-level failures.
        run_metrics: A dictionary that will be updated with various processing metrics.
        golden_partner_summaries: A list of dictionaries, where each dictionary
                                  contains the name and summary of a "golden partner."

    Returns:
        A tuple containing:
        - df (pd.DataFrame): The input DataFrame, potentially updated with statuses.
        - all_golden_partner_match_outputs (List[GoldenPartnerMatchOutput]):
          The primary output; a list of results from the final LLM comparison stage.
        - attrition_data_list (List[Dict[str, Any]]): A list of dictionaries,
          each representing a logged failure.
        - canonical_domain_journey_data (Dict[str, Dict[str, Any]]): Data tracking
          processing attempts and outcomes per canonical domain.
        - true_base_scraper_status (Dict[str, str]): Maps canonical true base URLs
          to their overall scraper status.
        - true_base_to_pathful_map (Dict[str, List[str]]): Maps canonical true base
          URLs to a list of actual pathful URLs scraped under them.
        - input_to_canonical_map (Dict[str, Optional[str]]): Maps original input URLs
          to their determined canonical true base URL.
        - row_level_failure_counts (Dict[str, int]): A summary count of failures
          by type.
    """
    globally_processed_urls: Set[str] = set()  # Tracks URLs to avoid re-scraping
    # Stores scraper status for each specific pathful URL attempted
    canonical_site_pathful_scraper_status: Dict[str, str] = {}
    # Maps original input URL to its determined canonical true base domain
    input_to_canonical_map: Dict[str, Optional[str]] = {}

    all_golden_partner_match_outputs: List[GoldenPartnerMatchOutput] = []
    attrition_data_list: List[Dict[str, Any]] = [] # For detailed failure logging
    row_level_failure_counts: Dict[str, int] = Counter()

    # Data structures for Canonical Domain Journey Report
    # Tracks processing details per unique canonical domain encountered
    canonical_domain_journey_data: Dict[str, Dict[str, Any]] = {}

    pipeline_loop_start_time = time.time()
    rows_processed_count = 0
    rows_failed_count = 0

    # --- Live Reporter Setup ---
    # In single-process mode, we write a live CSV from inside this function.
    # In parallel mode, we stream rows to the master via row_queue and the master writes the live CSV/JSONL.
    augmented_live_reporter: Optional[LiveCsvReporter] = None
    augmented_jsonl_reporter: Optional[LiveJsonlReporter] = None
    sales_jsonl_reporter: Optional[LiveJsonlReporter] = None

    if live_reporter is None and row_queue is None:
        # SalesOutreach live outputs
        live_report_header = list(df.columns) + [
            'CanonicalEntryURL', 'found_number', 'PhoneNumber_Status', 'is_b2b', 'serves_1000',
            'is_b2b_reason', 'serves_1000_reason', 'description', 'Industry',
            'Products/Services Offered', 'USP/Key Selling Points', 'Customer Target Segments',
            'Business Model', 'Company Size Inferred', 'Innovation Level Indicators',
            'Website Clarity Notes', 'B2B Indicator', 'Phone Outreach Suitability',
            'Target Group Size Assessment', 'sales_pitch', 'matched_golden_partner',
            'match_reasoning', 'Matched Partner Description', 'Avg Leads Per Day', 'Rank'
        ]
        live_report_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live.csv")
        live_reporter = LiveCsvReporter(filepath=live_report_path, header=live_report_header)
        sales_jsonl_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live.jsonl")
        sales_jsonl_reporter = LiveJsonlReporter(filepath=sales_jsonl_path)

        # Augmented-input live outputs (original columns + enrichment columns).
        augmented_header = list(df.columns) + [
            'CanonicalEntryURL', 'found_number', 'PhoneNumber_Status',
            'description', 'Industry',
            'Products/Services Offered', 'USP/Key Selling Points', 'Customer Target Segments',
            'Business Model', 'Company Size Inferred', 'Innovation Level Indicators',
            'Website Clarity Notes', 'B2B Indicator', 'Phone Outreach Suitability',
            'Target Group Size Assessment',
            'matched_golden_partner', 'match_reasoning',
            'sales_pitch'
        ]
        # Deduplicate while preserving order
        seen_cols = set()
        augmented_header = [c for c in augmented_header if not (c in seen_cols or seen_cols.add(c))]

        augmented_live_path = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.csv")
        augmented_jsonl_path = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.jsonl")
        augmented_live_reporter = LiveCsvReporter(filepath=augmented_live_path, header=augmented_header)
        augmented_jsonl_reporter = LiveJsonlReporter(filepath=augmented_jsonl_path)
    # --- End Live Reporter Setup ---

    # Pre-fetch company name and URL column names from AppConfig
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(
        app_config.input_file_profile_name,
        app_config.INPUT_COLUMN_PROFILES['default']
    )
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')
    url_col_key = active_profile.get('GivenURL', 'GivenURL')
    # NOTE: Profiles are input->canonical maps; by default we operate on canonical columns.
    # Phone can exist under multiple names depending on which stage produced the file.
    phone_col_key = active_profile.get('PhoneNumber', 'PhoneNumber')
    original_phone_column_name = None
    try:
        original_phone_column_name = active_profile.get('_original_phone_column_name')
    except Exception:
        original_phone_column_name = None

    for i, (index, row_series) in enumerate(df.iterrows()):
        rows_processed_count += 1
        row: pd.Series = row_series
        company_name_str: str = str(row.get(company_name_col_key, f"MissingCompanyName_Row_{index}"))
        given_url_original: Optional[str] = row.get(url_col_key)
        # Prefer "already-found" phone columns first, but also allow raw/original phone columns like "Company Phone".
        # This ensures we still generate a pitch when a phone exists in another column even if phone extraction found none.
        candidate_phone_cols: List[str] = [
            phone_col_key,               # canonical phone for full pipeline
            "GivenPhoneNumber",          # common canonical for phone-only outputs
            "PhoneNumber_Found",         # common augmented column from phone_extract
        ]
        if original_phone_column_name and original_phone_column_name not in candidate_phone_cols:
            candidate_phone_cols.append(original_phone_column_name)
        # Always consider the raw "Company Phone" column if present, even if the profile points elsewhere.
        if "Company Phone" not in candidate_phone_cols:
            candidate_phone_cols.append("Company Phone")

        usable_input_phone = _pick_first_usable_phone(row, df, candidate_phone_cols)
        phone_number_original: Optional[str] = usable_input_phone if usable_input_phone else None
        given_url_original_str: str = str(given_url_original) if given_url_original else "MissingURL"

        current_row_number_for_log: int = i + 1  # 1-based for logging
        log_identifier = f"[RowID: {index}, Company: {company_name_str}, URL: {given_url_original_str}]"
        logger.info(f"{log_identifier} --- Processing row {current_row_number_for_log}/{len(df)} ---")

        current_row_scraper_status: str = "Not_Run"
        final_canonical_entry_url: Optional[str] = None  # Pathful canonical URL from scraper
        true_base_domain_for_row: Optional[str] = None  # True base domain

        website_summary_obj: Optional[WebsiteTextSummary] = None
        detailed_attributes_obj: Optional[DetailedCompanyAttributes] = None
        final_match_output: Optional[GoldenPartnerMatchOutput] = None
        b2b_analysis_obj: Optional[B2BAnalysisOutput] = None

        # --- 1. URL Processing ---
        processed_url, url_status = process_input_url(
            given_url_original, app_config.url_probing_tlds, log_identifier
        )
        if url_status == "InvalidURL":
            df.at[index, 'ScrapingStatus'] = 'InvalidURL'
            current_row_scraper_status = 'InvalidURL'
            run_metrics["scraping_stats"]["scraping_failure_invalid_url"] += 1
            log_row_failure(
                failure_writer, index, company_name_str, given_url_original_str,
                "URL_Validation_InvalidOrMissing",
                f"Invalid or missing URL: {processed_url}", datetime.now().isoformat(),
                json.dumps({"original_url": given_url_original_str, "processed_url": processed_url})
            )
            row_level_failure_counts["URL_Validation_InvalidOrMissing"] += 1
            rows_failed_count += 1
            all_golden_partner_match_outputs.append(
                GoldenPartnerMatchOutput(
                    analyzed_company_url=given_url_original_str,
                    analyzed_company_attributes=DetailedCompanyAttributes(
                        input_summary_url=given_url_original_str
                    ),
                    match_rationale_features=[f"Failed at URL Validation: {url_status}"],
                    scrape_status=current_row_scraper_status
                )
            )
            continue
        # --- End URL Processing ---

        try:
            assert processed_url is not None
            scraped_text_to_use = None

            if pitch_from_description:
                # Use description text directly; skip scraping
                # Try Description-like columns
                candidate_desc_cols = ['Combined_Description', 'Description']
                # Use the already-resolved active_profile to infer mapped Description
                if active_profile:
                    for original_name, mapped_name in active_profile.items():
                        if not original_name.startswith('_') and mapped_name == 'Description':
                            candidate_desc_cols.append('Description')
                            break
                for col in candidate_desc_cols:
                    if col in df.columns:
                        try:
                            val = df.at[index, col]
                            if isinstance(val, str) and val.strip():
                                scraped_text_to_use = val
                                break
                        except Exception:
                            continue
                df.at[index, 'ScrapingStatus'] = 'Used_Description_Only'
                current_row_scraper_status = 'Used_Description_Only'
                true_base_domain_for_row = get_canonical_base_url(processed_url)
                df.at[index, 'CanonicalEntryURL'] = true_base_domain_for_row
            else:
                run_metrics["scraping_stats"]["urls_processed_for_scraping"] += 1
                scrape_task_start_time = time.time()

                # --- 2. Scrape Website ---
                logger.info(f"{log_identifier} Starting website scraping for: {processed_url}")
                # Ensure the scraper module uses this run's AppConfig (it has a module-level config_instance).
                # This also enables per-run options like reuse_scraped_content_if_available / cache dirs.
                try:
                    full_scraper_logic.config_instance = app_config
                except Exception:
                    pass
                _, scraper_status, final_canonical_entry_url, collected_summary_text = asyncio.run(
                    scrape_website(processed_url, run_output_dir, company_name_str, globally_processed_urls, index, run_id)
                )
                run_metrics["tasks"].setdefault("scrape_website_total_duration_seconds", 0)
                run_metrics["tasks"]["scrape_website_total_duration_seconds"] += (time.time() - scrape_task_start_time)

                df.at[index, 'ScrapingStatus'] = scraper_status
                true_base_domain_for_row = get_canonical_base_url(final_canonical_entry_url) \
                    if final_canonical_entry_url else None
                df.at[index, 'CanonicalEntryURL'] = true_base_domain_for_row # Store true_base
                current_row_scraper_status = scraper_status
                canonical_site_pathful_scraper_status[
                    final_canonical_entry_url if final_canonical_entry_url else processed_url
                ] = scraper_status

                scraped_text_to_use = collected_summary_text

            # If scraping yielded no text for any reason, try the fallback description.
            if not scraped_text_to_use or not scraped_text_to_use.strip():
                logger.warning(f"{log_identifier} No text available from scraping (Status: {current_row_scraper_status}). Attempting to use fallback description.")
                # Special-case profile: for 'altenpflege_products' prefer contacts + truncated pdf_text
                if getattr(app_config, 'input_file_profile_name', '') == 'altenpflege_products':
                    try:
                        contacts_val = ''
                        pdf_val = ''
                        if 'contacts' in df.columns:
                            cv = df.at[index, 'contacts']
                            contacts_val = cv if isinstance(cv, str) else ''
                        if 'pdf_text' in df.columns:
                            pv = df.at[index, 'pdf_text']
                            if isinstance(pv, str):
                                # limit to avoid bloat
                                pdf_val = pv.strip()[:4000]
                        combined = "\n\n".join([s for s in [contacts_val.strip(), pdf_val] if s])
                        if combined.strip():
                            scraped_text_to_use = combined
                            current_row_scraper_status = 'Used_Fallback_Description'
                            df.at[index, 'ScrapingStatus'] = current_row_scraper_status
                            logger.info(f"{log_identifier} Used contacts + pdf_text as fallback.")
                    except Exception:
                        pass
                # Try to use a fallback description column that exists for other profiles
                # Prefer commonly used names, then fall back to whatever the profile maps for 'Description'
                possible_fallback_cols = [
                    'Combined_Description',
                    'Description',
                    'Beschreibung',
                    'beschreibung',
                    'categories',
                    'products',
                    'pdf_text'
                ]
                # Add any profile-mapped name for Description if provided
                try:
                    active_profile_local = app_config.INPUT_COLUMN_PROFILES.get(
                        app_config.input_file_profile_name,
                        app_config.INPUT_COLUMN_PROFILES.get('default', {})
                    )
                    if isinstance(active_profile_local, dict):
                        for original_name, mapped_name in active_profile_local.items():
                            if not original_name.startswith('_') and mapped_name == 'Description':
                                # Consider both the renamed column 'Description' and the original source name
                                if original_name not in possible_fallback_cols:
                                    possible_fallback_cols.append(original_name)
                except Exception:
                    pass

                fallback_text = None
                for col_name in possible_fallback_cols:
                    if col_name in df.columns:
                        try:
                            val = df.at[index, col_name]
                            if isinstance(val, str) and val.strip():
                                fallback_text = val
                                break
                        except Exception:
                            continue
                # If both categories and products exist, concatenate them for richer context
                if not fallback_text:
                    try:
                        cat = df.at[index, 'categories'] if 'categories' in df.columns else ''
                        prod = df.at[index, 'products'] if 'products' in df.columns else ''
                        combo = "\n\n".join([s for s in [str(cat).strip(), str(prod).strip()] if s])
                        if combo.strip():
                            fallback_text = combo
                    except Exception:
                        pass
                # If description and pdf_text both exist but description was empty, try pdf_text alone
                if not fallback_text and 'pdf_text' in df.columns:
                    try:
                        pv = df.at[index, 'pdf_text']
                        if isinstance(pv, str) and pv.strip():
                            fallback_text = pv
                    except Exception:
                        pass

                if fallback_text and isinstance(fallback_text, str) and fallback_text.strip():
                    scraped_text_to_use = fallback_text
                    current_row_scraper_status = 'Used_Fallback_Description'
                    df.at[index, 'ScrapingStatus'] = current_row_scraper_status
                    logger.info(f"{log_identifier} Successfully used fallback description.")
                else:
                    # If there's no scraped text AND no fallback, optionally synthesize minimal context
                    # so the LLM can still proceed (especially when --skip-prequalification is used).
                    if skip_prequalification:
                        industry_val = None
                        for cand in [
                            'Industry', 'Kategorie', 'category', 'industry']:
                            if cand in df.columns:
                                try:
                                    v = df.at[index, cand]
                                    if isinstance(v, str) and v.strip():
                                        industry_val = v.strip()
                                        break
                                except Exception:
                                    pass
                        synthesized = f"{company_name_str} — Website: {given_url_original_str}."
                        if industry_val:
                            synthesized += f" Industry: {industry_val}."
                        logger.warning(f"{log_identifier} No scraped/fallback text. Using synthesized minimal context for LLM calls.")
                        scraped_text_to_use = synthesized
                    else:
                        logger.warning(f"{log_identifier} No fallback description available. Skipping LLM calls.")
                        log_row_failure(
                            failure_writer, index, company_name_str, given_url_original_str,
                            f"LLM_Input_NoTextAvailable",
                            f"Scraper status was '{current_row_scraper_status}', and no text was collected or available as a fallback.",
                            datetime.now().isoformat(),
                            json.dumps({
                                "pathful_canonical_url": final_canonical_entry_url,
                                "true_base_domain": true_base_domain_for_row
                            }),
                            associated_pathful_canonical_url=final_canonical_entry_url
                        )
                        row_level_failure_counts["LLM_Input_NoTextAvailable"] += 1
                        rows_failed_count += 1
                        all_golden_partner_match_outputs.append(
                            GoldenPartnerMatchOutput(
                                analyzed_company_url=given_url_original_str,
                                analyzed_company_attributes=DetailedCompanyAttributes(
                                    input_summary_url=given_url_original_str
                                ),
                                match_rationale_features=["No text collected from website scraping and no fallback"],
                                scrape_status=current_row_scraper_status
                            )
                        )
                        continue
            
            logger.info(f"{log_identifier} Collected {len(scraped_text_to_use)} characters for LLM processing.")

            # Ensure variables exist regardless of optional branches (skip-prequalification / skip-pitch)
            b2b_analysis_obj = None
            final_match_output = None

            # --- 3a. Pre-qualification (optional) ---
            llm_file_prefix_row = sanitize_filename_component(
                f"Row{index}_{company_name_str[:20]}_{str(time.time())[-5:]}", max_len=50
            )
            if not skip_prequalification:
                b2b_capacity_tuple = check_b2b_and_capacity(
                    gemini_client=gemini_client,
                    config=app_config,
                    company_text=scraped_text_to_use, # Use full text for this check
                    llm_context_dir=llm_context_dir,
                    llm_requests_dir=llm_requests_dir,
                    file_identifier_prefix=llm_file_prefix_row,
                    triggering_input_row_id=index,
                    triggering_company_name=company_name_str
                )
                b2b_analysis_obj = b2b_capacity_tuple[0]
                if b2b_capacity_tuple[2]: # token_stats
                    run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += b2b_capacity_tuple[2].get("prompt_tokens", 0)
                    run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += b2b_capacity_tuple[2].get("completion_tokens", 0)
                    run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += b2b_capacity_tuple[2].get("total_tokens", 0)
                    run_metrics["llm_processing_stats"]["llm_calls_b2b_capacity_check"] = run_metrics["llm_processing_stats"].get("llm_calls_b2b_capacity_check", 0) + 1

                if not b2b_analysis_obj or b2b_analysis_obj.is_b2b != "Yes" or b2b_analysis_obj.serves_1000_customers == "No":
                    failure_reason = "B2B_Capacity_Check_Failed"
                    if b2b_analysis_obj:
                        df.at[index, 'is_b2b'] = b2b_analysis_obj.is_b2b
                        df.at[index, 'serves_1000'] = b2b_analysis_obj.serves_1000_customers
                        df.at[index, 'is_b2b_reason'] = b2b_analysis_obj.is_b2b_reason
                        df.at[index, 'serves_1000_reason'] = b2b_analysis_obj.serves_1000_customers_reason
                        if b2b_analysis_obj.is_b2b != "Yes":
                            failure_reason = "Not_B2B"
                        elif b2b_analysis_obj.serves_1000_customers == "No":
                            failure_reason = "Capacity_Under_1000"

                    logger.warning(f"{log_identifier} Company failed B2B/capacity check. Reason: {failure_reason}")
                    log_row_failure(
                        failure_writer, index, company_name_str, given_url_original_str,
                        f"PreQual_{failure_reason}", "Company did not pass pre-qualification checks.",
                        datetime.now().isoformat(),
                        json.dumps({"raw_response": b2b_capacity_tuple[1] or "N/A"})
                    )
                    row_level_failure_counts[f"PreQual_{failure_reason}"] += 1
                    rows_failed_count += 1
                    all_golden_partner_match_outputs.append(
                        GoldenPartnerMatchOutput(
                            analyzed_company_url=given_url_original_str,
                            analyzed_company_attributes=DetailedCompanyAttributes(
                                input_summary_url=given_url_original_str
                            ),
                            match_rationale_features=[f"Pre-qualification failed: {failure_reason}"],
                            scrape_status=current_row_scraper_status
                        )
                    )
                    if df.at[index, 'ScrapingStatus'] == 'Used_Fallback_Description':
                        continue
                    else:
                        continue
                
                df.at[index, 'is_b2b'] = b2b_analysis_obj.is_b2b
                df.at[index, 'serves_1000'] = b2b_analysis_obj.serves_1000_customers
                df.at[index, 'is_b2b_reason'] = b2b_analysis_obj.is_b2b_reason
                df.at[index, 'serves_1000_reason'] = b2b_analysis_obj.serves_1000_customers_reason
                logger.info(f"{log_identifier} Company passed B2B/capacity check.")
            else:
                # Skip pre-qualification; mark as unknown
                df.at[index, 'is_b2b'] = 'Unknown'
                df.at[index, 'serves_1000'] = 'Unknown'
                df.at[index, 'is_b2b_reason'] = None
                df.at[index, 'serves_1000_reason'] = None

            # --- 3. LLM Call 1: Generate Website Summary ---
            summary_obj_tuple = generate_website_summary(
                gemini_client=gemini_client,
                config=app_config,
                original_url=given_url_original_str,
                scraped_text=scraped_text_to_use,
                llm_context_dir=llm_context_dir,
                llm_requests_dir=llm_requests_dir,
                file_identifier_prefix=llm_file_prefix_row,
                triggering_input_row_id=index,
                triggering_company_name=company_name_str
            )
            website_summary_obj = summary_obj_tuple[0]
            if summary_obj_tuple[2]:  # token_stats
                run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += \
                    summary_obj_tuple[2].get("prompt_tokens", 0)
                run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += \
                    summary_obj_tuple[2].get("completion_tokens", 0)
                run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += \
                    summary_obj_tuple[2].get("total_tokens", 0)
                run_metrics["llm_processing_stats"]["llm_calls_summary_generation"] = \
                    run_metrics["llm_processing_stats"].get("llm_calls_summary_generation", 0) + 1

            if not website_summary_obj or not website_summary_obj.summary:
                logger.warning(f"{log_identifier} LLM Call 1 (Summarization) failed. Raw: {summary_obj_tuple[1]}")
                log_row_failure(
                    failure_writer, index, company_name_str, given_url_original_str,
                    "LLM_Summarization_Failed", "Failed to generate website summary.",
                    datetime.now().isoformat(),
                    json.dumps({"raw_response": summary_obj_tuple[1] or "N/A"})
                )
                row_level_failure_counts["LLM_Summarization_Failed"] += 1
                rows_failed_count += 1
                all_golden_partner_match_outputs.append(
                    GoldenPartnerMatchOutput(
                        analyzed_company_url=given_url_original_str,
                        analyzed_company_attributes=DetailedCompanyAttributes(
                            input_summary_url=given_url_original_str
                        ),
                        match_rationale_features=["LLM Summarization Failed"],
                        scrape_status=current_row_scraper_status
                    )
                )
                continue
            logger.info(f"{log_identifier} LLM Call 1 (Summarization) successful.")

            # --- 4. LLM Call 2: Extract Detailed Attributes ---
            attributes_obj_tuple = extract_detailed_attributes(
                gemini_client=gemini_client,
                config=app_config,
                summary_obj=website_summary_obj,
                llm_context_dir=llm_context_dir,
                llm_requests_dir=llm_requests_dir,
                file_identifier_prefix=llm_file_prefix_row,
                triggering_input_row_id=index,
                triggering_company_name=company_name_str
            )
            detailed_attributes_obj = attributes_obj_tuple[0]
            if attributes_obj_tuple[2]:  # token_stats
                run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += \
                    attributes_obj_tuple[2].get("prompt_tokens", 0)
                run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += \
                    attributes_obj_tuple[2].get("completion_tokens", 0)
                run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += \
                    attributes_obj_tuple[2].get("total_tokens", 0)
                run_metrics["llm_processing_stats"]["llm_calls_attribute_extraction"] = \
                    run_metrics["llm_processing_stats"].get("llm_calls_attribute_extraction", 0) + 1

            if not detailed_attributes_obj:
                logger.warning(f"{log_identifier} LLM Call 2 (Attribute Extraction) failed. Raw: {attributes_obj_tuple[1]}")
                log_row_failure(
                    failure_writer, index, company_name_str, given_url_original_str,
                    "LLM_AttributeExtraction_Failed", "Failed to extract detailed attributes.",
                    datetime.now().isoformat(),
                    json.dumps({"raw_response": attributes_obj_tuple[1] or "N/A"})
                )
                row_level_failure_counts["LLM_AttributeExtraction_Failed"] += 1
                rows_failed_count += 1
                all_golden_partner_match_outputs.append(
                    GoldenPartnerMatchOutput(
                        analyzed_company_url=given_url_original_str,
                        analyzed_company_attributes=DetailedCompanyAttributes(
                            input_summary_url=website_summary_obj.original_url
                            if website_summary_obj else given_url_original_str
                        ),
                        match_rationale_features=["LLM Attribute Extraction Failed"],
                        scrape_status=current_row_scraper_status
                    )
                )
                continue
            logger.info(f"{log_identifier} LLM Call 2 (Attribute Extraction) successful.")

            phone_status = "Not_Processed"
            if not pitch_from_description:
                phone_meta = None
                if not getattr(app_config, "enable_phone_retrieval_in_full_pipeline", True):
                    # Do not perform phone retrieval within the full pipeline.
                    # Still treat a usable input phone as the "found_number" so reports are consistent.
                    if phone_number_original is not None and not should_attempt_phone_retrieval(str(phone_number_original)):
                        phone_status = "Provided_In_Input"
                        num = str(phone_number_original).strip()
                        if num.startswith("'") and len(num) > 1:
                            num = num[1:].strip()
                        if num:
                            df.at[index, 'found_number'] = num
                            # Populate Top_/Primary_ compatibility fields from provided input.
                            df.at[index, "Top_Number_1"] = num
                            df.at[index, "Top_Type_1"] = "Provided_In_Input"
                            df.at[index, "Top_SourceURL_1"] = ""
                            df.at[index, "Primary_Number_1"] = num
                            df.at[index, "Primary_Type_1"] = "Provided_In_Input"
                            df.at[index, "Primary_SourceURL_1"] = ""
                    else:
                        phone_status = "Skipped_Phone_Retrieval"
                        logger.info(f"{log_identifier} Phone retrieval skipped (ENABLE_PHONE_RETRIEVAL_IN_FULL_PIPELINE=False).")
                elif app_config.force_phone_extraction or should_attempt_phone_retrieval(phone_number_original):
                    if app_config.force_phone_extraction:
                        logger.info(f"{log_identifier} Force flag enabled. Attempting phone retrieval.")
                    else:
                        logger.info(f"{log_identifier} No phone number in input. Attempting retrieval.")
                    retrieved_numbers, phone_status, phone_meta = retrieve_phone_numbers_for_url(
                        given_url_original_str,
                        company_name_str,
                        app_config,
                        run_output_dir=os.path.join(run_output_dir, "phone_retrieval"),
                        llm_context_dir=os.path.join(llm_context_dir, "phone_retrieval"),
                        run_id=run_id
                    )
                    # Best-effort: copy phone retrieval debug fields into this run's df for live CSV/JSONL.
                    if phone_meta:
                        for k, v in phone_meta.items():
                            if k in df.columns:
                                df.at[index, k] = v
                    if retrieved_numbers:
                        primary_numbers = [n for n in retrieved_numbers if n.classification == 'Primary']
                        secondary_numbers = [n for n in retrieved_numbers if n.classification == 'Secondary']
                        
                        if primary_numbers:
                            best_number = primary_numbers[0].number
                            phone_status = "Found_Primary"
                        elif secondary_numbers:
                            best_number = secondary_numbers[0].number
                            phone_status = "Found_Secondary"
                        else:
                            best_number = None
                            phone_status = "No_Main_Line_Found"
                        
                        if best_number:
                            df.at[index, 'found_number'] = best_number
                            # If the wrapper returned detailed fields, prefer those; otherwise fill minimal compatibility.
                            if _is_blank_cell(df.at[index, "Top_Number_1"]):
                                df.at[index, "Top_Number_1"] = best_number
                            if _is_blank_cell(df.at[index, "Top_Type_1"]):
                                df.at[index, "Top_Type_1"] = "Retrieved"
                            if _is_blank_cell(df.at[index, "Top_SourceURL_1"]):
                                df.at[index, "Top_SourceURL_1"] = ""
                            if primary_numbers and _is_blank_cell(df.at[index, "Primary_Number_1"]):
                                df.at[index, "Primary_Number_1"] = primary_numbers[0].number
                                if _is_blank_cell(df.at[index, "Primary_Type_1"]):
                                    df.at[index, "Primary_Type_1"] = "Retrieved"
                                if _is_blank_cell(df.at[index, "Primary_SourceURL_1"]):
                                    df.at[index, "Primary_SourceURL_1"] = ""
                            if secondary_numbers and _is_blank_cell(df.at[index, "Secondary_Number_1"]):
                                df.at[index, "Secondary_Number_1"] = secondary_numbers[0].number
                                if _is_blank_cell(df.at[index, "Secondary_Type_1"]):
                                    df.at[index, "Secondary_Type_1"] = "Retrieved"
                                if _is_blank_cell(df.at[index, "Secondary_SourceURL_1"]):
                                    df.at[index, "Secondary_SourceURL_1"] = ""
                            logger.info(f"{log_identifier} Found best number: {best_number} (Status: {phone_status})")
                    else:
                        logger.warning(f"{log_identifier} Phone number retrieval failed with status: {phone_status}")
                else:
                    phone_status = "Provided_In_Input"
                    logger.info(f"{log_identifier} Phone number retrieval skipped due to acceptable input value: '{phone_number_original}'")
                    # Populate found_number from the usable input phone so downstream report columns
                    # (and pitch gating) have a consistent "number to call" field.
                    if phone_number_original is not None:
                        num = str(phone_number_original).strip()
                        if num.startswith("'") and len(num) > 1:
                            num = num[1:].strip()
                        if num:
                            df.at[index, 'found_number'] = num
                            # Populate Top_/Primary_ compatibility fields from provided input.
                            df.at[index, "Top_Number_1"] = num
                            df.at[index, "Top_Type_1"] = "Provided_In_Input"
                            df.at[index, "Top_SourceURL_1"] = ""
                            df.at[index, "Primary_Number_1"] = num
                            df.at[index, "Primary_Type_1"] = "Provided_In_Input"
                            df.at[index, "Primary_SourceURL_1"] = ""
            df.at[index, 'PhoneNumber_Status'] = phone_status

            # Decide whether we have a usable phone number for outreach (found or usable input fallback).
            found_number_val_raw = df.at[index, 'found_number'] if 'found_number' in df.columns else None
            # Pandas may store missing values as NaN floats; treat those as empty.
            if found_number_val_raw is None or (isinstance(found_number_val_raw, float) and pd.isna(found_number_val_raw)):
                found_number_val = ""
            else:
                found_number_val = str(found_number_val_raw).strip()
                if found_number_val.lower() in {"nan", "none", "null"}:
                    found_number_val = ""
                if found_number_val.startswith("'") and len(found_number_val) > 1:
                    found_number_val = found_number_val[1:].strip()

            input_number_val = _normalize_phone_str(phone_number_original)

            has_phone_for_outreach = bool(found_number_val) or (input_number_val and not should_attempt_phone_retrieval(input_number_val))

            # If we didn't find a phone via retrieval but have a usable phone in the input (incl. Company Phone fallback),
            # use it for outreach and reporting so downstream is consistent.
            if not found_number_val and input_number_val and not should_attempt_phone_retrieval(input_number_val):
                df.at[index, 'found_number'] = input_number_val
                if phone_status in {"No_Numbers_Found", "No_Main_Line_Found", "Skipped_Phone_Retrieval", "Not_Processed"}:
                    df.at[index, 'PhoneNumber_Status'] = "Provided_In_Input_CompanyPhone"
                found_number_val = input_number_val
                # Populate Top_/Primary_ compatibility fields from fallback input phone.
                df.at[index, "Top_Number_1"] = input_number_val
                df.at[index, "Top_Type_1"] = "Provided_In_Input"
                df.at[index, "Top_SourceURL_1"] = ""
                df.at[index, "Primary_Number_1"] = input_number_val
                df.at[index, "Primary_Type_1"] = "Provided_In_Input"
                df.at[index, "Primary_SourceURL_1"] = ""

            # NOTE: We always create a GoldenPartnerMatchOutput for the row (even when skipping),
            # so downstream outputs can show a reason rather than being blank.
            final_match_output: Optional[GoldenPartnerMatchOutput] = None

            if not has_phone_for_outreach:
                # Skip partner matching + sales pitch if we have no phone number to call.
                final_match_output = GoldenPartnerMatchOutput(
                    analyzed_company_url=given_url_original_str,
                    analyzed_company_attributes=detailed_attributes_obj,
                    summary=website_summary_obj.summary if website_summary_obj else None,
                    match_rationale_features=["Skipped pitch generation: no usable phone number found"],
                    scrape_status=current_row_scraper_status,
                )
                all_golden_partner_match_outputs.append(final_match_output)
            else:
                # --- 5. LLM Call 3: Match Partner ---
                partner_match_tuple = match_partner(
                    gemini_client=gemini_client,
                    config=app_config,
                    target_attributes=detailed_attributes_obj,
                    golden_partner_summaries=golden_partner_summaries,
                    llm_context_dir=llm_context_dir,
                    llm_requests_dir=llm_requests_dir,
                    file_identifier_prefix=llm_file_prefix_row,
                    triggering_input_row_id=index,
                    triggering_company_name=company_name_str,
                )
                partner_match_output = partner_match_tuple[0]
                if partner_match_tuple[2]:  # token_stats
                    run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += partner_match_tuple[2].get("prompt_tokens", 0)
                    run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += partner_match_tuple[2].get("completion_tokens", 0)
                    run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += partner_match_tuple[2].get("total_tokens", 0)
                    run_metrics["llm_processing_stats"]["llm_calls_partner_matching"] = run_metrics["llm_processing_stats"].get("llm_calls_partner_matching", 0) + 1

                if (
                    not partner_match_output
                    or not partner_match_output.matched_partner_name
                    or partner_match_output.matched_partner_name == "No suitable match found"
                ):
                    logger.warning(
                        f"{log_identifier} LLM Call 3 (Partner Matching) failed or found no suitable match. "
                        f"Raw: {partner_match_tuple[1]}"
                    )
                    log_row_failure(
                        failure_writer, index, company_name_str, given_url_original_str,
                        "LLM_PartnerMatching_FailedOrNoMatch", "Failed to find a suitable partner match.",
                        datetime.now().isoformat(),
                        json.dumps({"raw_response": partner_match_tuple[1] or "N/A"})
                    )
                    row_level_failure_counts["LLM_PartnerMatching_FailedOrNoMatch"] += 1
                    final_match_output = GoldenPartnerMatchOutput(
                        analyzed_company_url=detailed_attributes_obj.input_summary_url,
                        analyzed_company_attributes=detailed_attributes_obj,
                        summary=website_summary_obj.summary if website_summary_obj else None,
                        match_rationale_features=["LLM Partner Matching Failed or No Match Found"],
                        scrape_status=current_row_scraper_status,
                    )
                    all_golden_partner_match_outputs.append(final_match_output)
                else:
                    logger.info(
                        f"{log_identifier} LLM Call 3 (Partner Matching) successful. "
                        f"Matched with: {partner_match_output.matched_partner_name}"
                    )

                    # --- 6. LLM Call 4: Generate Sales Pitch ---
                    matched_partner_data = next(
                        (p for p in golden_partner_summaries if p.get('name') == partner_match_output.matched_partner_name),
                        None
                    )
                    if not matched_partner_data:
                        logger.error(
                            f"{log_identifier} Could not find full data for matched partner: "
                            f"{partner_match_output.matched_partner_name}"
                        )
                        final_match_output = GoldenPartnerMatchOutput(
                            analyzed_company_url=detailed_attributes_obj.input_summary_url,
                            analyzed_company_attributes=detailed_attributes_obj,
                            summary=website_summary_obj.summary if website_summary_obj else None,
                            match_rationale_features=["Matched partner data missing; pitch not generated"],
                            scrape_status=current_row_scraper_status,
                        )
                        all_golden_partner_match_outputs.append(final_match_output)
                    else:
                        sales_pitch_tuple = generate_sales_pitch(
                            gemini_client=gemini_client,
                            config=app_config,
                            target_attributes=detailed_attributes_obj,
                            matched_partner=matched_partner_data,
                            website_summary_obj=website_summary_obj,
                            previous_match_rationale=(partner_match_output.match_rationale_features or []) if partner_match_output else [],
                            llm_context_dir=llm_context_dir,
                            llm_requests_dir=llm_requests_dir,
                            file_identifier_prefix=llm_file_prefix_row,
                            triggering_input_row_id=index,
                            triggering_company_name=company_name_str,
                        )
                        final_match_output = sales_pitch_tuple[0]
                        if sales_pitch_tuple[2]:  # token_stats
                            run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += sales_pitch_tuple[2].get("prompt_tokens", 0)
                            run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += sales_pitch_tuple[2].get("completion_tokens", 0)
                            run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += sales_pitch_tuple[2].get("total_tokens", 0)
                            run_metrics["llm_processing_stats"]["llm_calls_sales_pitch_generation"] = run_metrics["llm_processing_stats"].get("llm_calls_sales_pitch_generation", 0) + 1

                        if not final_match_output:
                            logger.warning(
                                f"{log_identifier} LLM Call 4 (Sales Pitch Generation) failed. "
                                f"Raw: {sales_pitch_tuple[1]}"
                            )
                            log_row_failure(
                                failure_writer, index, company_name_str, given_url_original_str,
                                "LLM_SalesPitchGeneration_Failed", "Failed to generate sales pitch.",
                                datetime.now().isoformat(),
                                json.dumps({"raw_response": sales_pitch_tuple[1] or "N/A"})
                            )
                            row_level_failure_counts["LLM_SalesPitchGeneration_Failed"] += 1
                            final_match_output = GoldenPartnerMatchOutput(
                                analyzed_company_url=detailed_attributes_obj.input_summary_url,
                                analyzed_company_attributes=detailed_attributes_obj,
                                summary=website_summary_obj.summary if website_summary_obj else None,
                                match_rationale_features=["LLM Sales Pitch Generation Failed"],
                                scrape_status=current_row_scraper_status,
                            )
                            all_golden_partner_match_outputs.append(final_match_output)
                        else:
                            logger.info(f"{log_identifier} LLM Call 4 (Sales Pitch Generation) successful.")
                            # Fill the programmatic placeholder with the golden partner's "Avg Leads Per Day"
                            # so live CSV/JSONL outputs have the final pitch text.
                            try:
                                final_match_output.phone_sales_line = _fill_programmatic_placeholder(
                                    final_match_output.phone_sales_line,
                                    final_match_output.avg_leads_per_day,
                                )
                            except Exception:
                                pass
                            final_match_output.scrape_status = current_row_scraper_status
                            final_match_output.analyzed_company_attributes = detailed_attributes_obj
                            final_match_output.summary = website_summary_obj.summary if website_summary_obj else None
                            all_golden_partner_match_outputs.append(final_match_output)

            input_to_canonical_map[given_url_original_str] = true_base_domain_for_row

            if true_base_domain_for_row:
                if true_base_domain_for_row not in canonical_domain_journey_data:
                    canonical_domain_journey_data[true_base_domain_for_row] = {
                        "Input_Row_IDs": set(), "Input_CompanyNames": set(),
                        "Input_GivenURLs": set(), "Pathful_URLs_Attempted_List": set(),
                        "Overall_Scraper_Status_For_Domain": "Unknown",
                        "LLM_Stages_Attempted": 0, "LLM_Stages_Succeeded": 0
                    }
                journey_entry = canonical_domain_journey_data[true_base_domain_for_row]
                journey_entry["Input_Row_IDs"].add(index)
                journey_entry["Input_CompanyNames"].add(company_name_str)
                journey_entry["Input_GivenURLs"].add(given_url_original_str)
                if final_canonical_entry_url:
                    journey_entry["Pathful_URLs_Attempted_List"].add(final_canonical_entry_url)
                # This status might be overwritten by subsequent rows for the same domain;
                # A more robust aggregation might be needed if per-row status varies widely.
                journey_entry["Overall_Scraper_Status_For_Domain"] = current_row_scraper_status
                
                journey_entry["LLM_Stages_Attempted"] = 3 # Assumes all 3 are attempted if scraping succeeds
                current_succeeded_stages = 0
                if website_summary_obj: current_succeeded_stages +=1
                if detailed_attributes_obj: current_succeeded_stages +=1
                if final_match_output: current_succeeded_stages +=1
                # Maximize succeeded stages if multiple rows hit the same domain
                journey_entry["LLM_Stages_Succeeded"] = max(
                    journey_entry.get("LLM_Stages_Succeeded", 0), current_succeeded_stages
                )

            # --- Live Reporting ---
            # After all processing for a row is complete (or has failed), gather all data and write outputs.
            #
            # IMPORTANT: Use the *current* dataframe row (not the iterrows() snapshot) so any df.at[...] updates
            # (Top_Number_*, Primary_/Secondary_*, Regex/LLM debug fields, etc.) are included in:
            # - single-process live CSV/JSONL
            # - parallel worker streaming payloads (row_queue)
            final_row_data = df.loc[index].to_dict()
            # Update with data gathered during the pipeline flow
            final_row_data.update({
                'CanonicalEntryURL': true_base_domain_for_row,
                'PhoneNumber_Status': df.at[index, 'PhoneNumber_Status'],
                'found_number': df.at[index, 'found_number'] if 'found_number' in df.columns else None,
                # Use df columns (works even when --skip-prequalification is enabled)
                'is_b2b': df.at[index, 'is_b2b'] if 'is_b2b' in df.columns else None,
                'serves_1000': df.at[index, 'serves_1000'] if 'serves_1000' in df.columns else None,
                'is_b2b_reason': df.at[index, 'is_b2b_reason'] if 'is_b2b_reason' in df.columns else None,
                'serves_1000_reason': df.at[index, 'serves_1000_reason'] if 'serves_1000_reason' in df.columns else None,
                'description': website_summary_obj.summary if website_summary_obj else None,
                'Industry': detailed_attributes_obj.industry if detailed_attributes_obj else None,
                'Products/Services Offered': "; ".join(detailed_attributes_obj.products_services_offered) if detailed_attributes_obj and detailed_attributes_obj.products_services_offered else None,
                'USP/Key Selling Points': "; ".join(detailed_attributes_obj.usp_key_selling_points) if detailed_attributes_obj and detailed_attributes_obj.usp_key_selling_points else None,
                'Customer Target Segments': "; ".join(detailed_attributes_obj.customer_target_segments) if detailed_attributes_obj and detailed_attributes_obj.customer_target_segments else None,
                'Business Model': detailed_attributes_obj.business_model if detailed_attributes_obj else None,
                'Company Size Inferred': detailed_attributes_obj.company_size_category_inferred if detailed_attributes_obj else None,
                'Innovation Level Indicators': detailed_attributes_obj.innovation_level_indicators_text if detailed_attributes_obj else None,
                'Website Clarity Notes': detailed_attributes_obj.website_clarity_notes if detailed_attributes_obj else None,
                'B2B Indicator': detailed_attributes_obj.b2b_indicator if detailed_attributes_obj else None,
                'Phone Outreach Suitability': detailed_attributes_obj.phone_outreach_suitability if detailed_attributes_obj else None,
                'Target Group Size Assessment': detailed_attributes_obj.target_group_size_assessment if detailed_attributes_obj else None,
                'sales_pitch': final_match_output.phone_sales_line if final_match_output else None,
                'matched_golden_partner': final_match_output.matched_partner_name if final_match_output else None,
                'match_reasoning': "; ".join(final_match_output.match_rationale_features) if final_match_output and final_match_output.match_rationale_features else None,
                'Matched Partner Description': final_match_output.matched_partner_description if final_match_output else None,
                'Avg Leads Per Day': final_match_output.avg_leads_per_day if final_match_output else None,
                'Rank': final_match_output.rank if final_match_output else None
            })
            # --- Live Reporting ---
            if live_reporter is not None:
                live_reporter.append_row(final_row_data)
            if sales_jsonl_reporter is not None:
                sales_jsonl_reporter.append_obj(final_row_data)
            if augmented_live_reporter is not None:
                augmented_live_reporter.append_row(final_row_data)
            if augmented_jsonl_reporter is not None:
                augmented_jsonl_reporter.append_obj(final_row_data)

            if row_queue is not None:
                payload = dict(final_row_data)
                if row_queue_meta:
                    payload.update({f"__meta_{k}": v for k, v in row_queue_meta.items()})
                try:
                    row_queue.put({
                        "type": "row",
                        "run_id": run_id,
                        "ts": datetime.now().isoformat(),
                        "data": payload
                    })
                except Exception:
                    # Queue errors should not kill the worker; best effort only.
                    pass
            # --- End Live Reporting ---

            logger.info(f"{log_identifier} Row {current_row_number_for_log} processing complete.")

        except Exception as e_row_processing:
            logger.error(
                f"{log_identifier} Unhandled error for row {current_row_number_for_log}: {e_row_processing}",
                exc_info=True
            )
            run_metrics["errors_encountered"].append(
                f"Row error for {company_name_str} (URL: {given_url_original_str}): {str(e_row_processing)}"
            )
            log_row_failure(
                failure_writer, index, company_name_str, given_url_original_str,
                "RowProcessing_UnhandledException", "Unhandled exception in main loop",
                datetime.now().isoformat(),
                json.dumps({
                    "exception_type": type(e_row_processing).__name__,
                    "exception_message": str(e_row_processing)
                }),
                associated_pathful_canonical_url=final_canonical_entry_url
            )
            row_level_failure_counts["RowProcessing_UnhandledException"] += 1
            rows_failed_count += 1
            all_golden_partner_match_outputs.append(
                GoldenPartnerMatchOutput(
                    analyzed_company_url=given_url_original_str,
                    analyzed_company_attributes=DetailedCompanyAttributes(
                        input_summary_url=given_url_original_str
                    ),
                    match_rationale_features=[f"Unhandled Exception: {str(e_row_processing)}"],
                    scrape_status=current_row_scraper_status
                )
            )

    run_metrics["tasks"]["pipeline_main_loop_duration_seconds"] = time.time() - pipeline_loop_start_time
    run_metrics["data_processing_stats"]["rows_successfully_processed_main_flow"] = \
        rows_processed_count - rows_failed_count
    run_metrics["data_processing_stats"]["rows_failed_main_flow"] = rows_failed_count
    run_metrics["data_processing_stats"]["row_level_failure_summary"] = dict(row_level_failure_counts)
    logger.info(f"Main processing loop complete. Processed {rows_processed_count} rows.")

    true_base_scraper_status: Dict[str, str] = {}
    true_base_to_pathful_map: Dict[str, List[str]] = {}

    # Populate true_base_scraper_status and true_base_to_pathful_map
    # This aggregates status from individual pathful URL scrapes to their true base domain.
    for pathful_url, status in canonical_site_pathful_scraper_status.items():
        true_base = get_canonical_base_url(pathful_url)
        if true_base:
            true_base_to_pathful_map.setdefault(true_base, []).append(pathful_url)
            # Prioritize "Success" status for a domain if any of its pages succeeded.
            # Otherwise, take the status of one of its pages (could be an error state).
            if true_base not in true_base_scraper_status or status == "Success":
                true_base_scraper_status[true_base] = status
            elif true_base_scraper_status[true_base] != "Success" and "Error" not in status:
                # Prefer a non-error status if current is an error and new one isn't
                true_base_scraper_status[true_base] = status
            # If current is non-success, non-error, and new is error, keep current.
            # If both are errors, the last one processed for that true_base will stick.

    # If we wrote an augmented live file in single-process mode, also write a stable final copy
    # for downstream consumption (mirrors the parallel writer behavior).
    try:
        if augmented_live_reporter is not None:
            import shutil
            live_path = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.csv")
            final_path = os.path.join(run_output_dir, f"input_augmented_{run_id}.csv")
            if os.path.exists(live_path):
                shutil.copyfile(live_path, final_path)
            live_jsonl = os.path.join(run_output_dir, f"input_augmented_{run_id}_live.jsonl")
            final_jsonl = os.path.join(run_output_dir, f"input_augmented_{run_id}.jsonl")
            if os.path.exists(live_jsonl):
                shutil.copyfile(live_jsonl, final_jsonl)
        if sales_jsonl_reporter is not None:
            import shutil
            live_jsonl = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_live.jsonl")
            final_jsonl = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}.jsonl")
            if os.path.exists(live_jsonl):
                shutil.copyfile(live_jsonl, final_jsonl)
    except Exception:
        pass

    logger.info("Pipeline flow execution finished.")
    
    return (
        df,
        all_golden_partner_match_outputs,
        attrition_data_list,
        canonical_domain_journey_data,
        true_base_scraper_status,
        true_base_to_pathful_map,
        input_to_canonical_map,
        dict(row_level_failure_counts)
    )