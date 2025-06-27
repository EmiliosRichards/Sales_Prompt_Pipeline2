"""
Manages the collection, calculation, and writing of pipeline run metrics.

This module provides functionality to aggregate various statistics gathered
during a pipeline run, calculate derived metrics (e.g., averages), and
format them into a human-readable Markdown report. The report includes
information on task durations, data processing statistics, scraping performance,
LLM processing details, report generation counts, and error summaries.
"""
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Counter as TypingCounter # Renamed to avoid conflict with variable
from collections import Counter # For actual counting

# It's good practice to get the logger for the current module
logger_module = logging.getLogger(__name__)


def write_run_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
    run_id: str,
    pipeline_start_time: float,
    attrition_data_list_for_metrics: List[Dict[str, Any]],
    canonical_domain_journey_data: Dict[str, Dict[str, Any]],
    # logger: logging.Logger # Logger instance is passed, but consider using module logger if appropriate
) -> None:
    """
    Writes the collected run metrics to a Markdown (.md) file.

    This function takes a dictionary of metrics, along with other relevant data
    from the pipeline run, and generates a comprehensive report detailing
    performance, data flow, and any errors encountered.

    Args:
        metrics (Dict[str, Any]): A dictionary containing various metrics collected
            throughout the pipeline run. Expected keys include 'tasks',
            'scraping_stats', 'regex_extraction_stats', 'llm_processing_stats',
            'data_processing_stats', 'report_generation_stats', 'errors_encountered'.
        output_dir (str): The directory where the metrics file will be saved.
        run_id (str): The unique identifier for the current pipeline run.
        pipeline_start_time (float): The timestamp (seconds since epoch) when
            the pipeline started.
        attrition_data_list_for_metrics (List[Dict[str, Any]]): A list of
            dictionaries, where each dictionary represents a row that did not
            yield a contact, used for attrition analysis.
        canonical_domain_journey_data (Dict[str, Dict[str, Any]]): A dictionary
            mapping canonical domains to their processing journey and outcomes.
        logger (logging.Logger): The logger instance to use for logging messages
                                 within this function. (Note: This is passed in,
                                 alternatively, the module-level logger could be used).
    """
    # Calculate total duration and define file path
    metrics["total_duration_seconds"] = round(time.time() - pipeline_start_time, 2)
    metrics_file_path = os.path.join(output_dir, f"run_metrics_{run_id}.md")

    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            # --- General Run Information ---
            f.write(f"# Pipeline Run Metrics: {run_id}\n\n")
            f.write(f"**Run ID:** {metrics.get('run_id', 'N/A')}\n")
            f.write(f"**Total Run Duration:** {metrics.get('total_duration_seconds', 0.0):.2f} seconds\n")
            f.write(f"**Pipeline Start Time:** {datetime.fromtimestamp(pipeline_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Pipeline End Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # --- Task Durations ---
            f.write("## Task Durations (seconds):\n")
            if metrics.get("tasks"):
                for task_name, duration in metrics["tasks"].items():
                    f.write(f"- **{task_name.replace('_', ' ').title()}:** {float(duration):.2f}\n")
            else:
                f.write("- No task durations recorded.\n")
            f.write("\n")

            # --- Average Task Durations ---
            f.write("### Average Task Durations (per relevant item):\n")
            tasks_data = metrics.get("tasks", {})
            scraping_stats_data = metrics.get("scraping_stats", {})
            regex_stats_data = metrics.get("regex_extraction_stats", {})
            llm_stats_data = metrics.get("llm_processing_stats", {})
            data_proc_stats_data = metrics.get("data_processing_stats", {})

            # Helper for average calculation and writing
            def _write_average_metric(label: str, total_duration_key: str, count_key: str, count_source_dict: Dict, unit: str = "seconds"):
                total_duration = tasks_data.get(total_duration_key, 0.0)
                item_count = count_source_dict.get(count_key, 0)
                if item_count > 0:
                    average = total_duration / item_count
                    f.write(f"- **{label}:** {average:.2f} {unit}\n")
                else:
                    f.write(f"- {label}: N/A (No items processed for this metric)\n")

            _write_average_metric(
                "Average Scrape Website Duration (per New Canonical Site Scraped)",
                "scrape_website_total_duration_seconds", "new_canonical_sites_scraped", scraping_stats_data
            )
            _write_average_metric(
                "Average Regex Extraction Duration (per Site Processed for Regex)",
                "regex_extraction_total_duration_seconds", "sites_processed_for_regex", regex_stats_data
            )
            _write_average_metric(
                "Average LLM Extraction Duration (per Site Processed for LLM)",
                "llm_extraction_total_duration_seconds", "sites_processed_for_llm", llm_stats_data
            )
            _write_average_metric(
                "Average Pass 1 Main Loop Duration (per Input Row)",
                "pass1_main_loop_duration_seconds", "input_rows_count", data_proc_stats_data
            )
            f.write("\n")

            # --- Data Processing Statistics ---
            f.write("## Data Processing Statistics:\n")
            stats = metrics.get("data_processing_stats", {})
            f.write(f"- **Input Rows Processed (Initial Load):** {stats.get('input_rows_count', 0)}\n")
            f.write(f"- **Rows Successfully Processed (Pass 1):** {stats.get('rows_successfully_processed_pass1', 0)}\n")
            f.write(f"- **Rows Failed During Processing (Pass 1):** {stats.get('rows_failed_pass1', 0)} "
                    "(Input rows that did not complete Pass 1 successfully due to errors such as "
                    "invalid URL, scraping failure, or critical processing exceptions for that row, "
                    "preventing LLM processing or final data consolidation for that specific input.)\n")
            f.write(f"- **Unique True Base Domains Consolidated:** {stats.get('unique_true_base_domains_consolidated', 0)}\n")
            f.write("\n")

            # --- Input Data Duplicate Analysis ---
            f.write("## Input Data Duplicate Analysis:\n")
            dp_stats = metrics.get("data_processing_stats", {})
            f.write(f"- **Total Input Rows Analyzed for Duplicates:** {dp_stats.get('input_rows_count', 0)}\n")
            f.write(f"- **Unique Input CompanyNames Found:** {dp_stats.get('input_unique_company_names', 0)}\n")
            f.write(f"- **Input CompanyNames Appearing More Than Once:** {dp_stats.get('input_company_names_with_duplicates_count', 0)}\n")
            f.write(f"- **Total Input Rows with a Duplicate CompanyName:** {dp_stats.get('input_rows_with_duplicate_company_name', 0)}\n")
            f.write(f"- **Unique Input Canonical URLs Found:** {dp_stats.get('input_unique_canonical_urls', 0)}\n")
            f.write(f"- **Input Canonical URLs Appearing More Than Once:** {dp_stats.get('input_canonical_urls_with_duplicates_count', 0)}\n")
            f.write(f"- **Total Input Rows with a Duplicate Input Canonical URL:** {dp_stats.get('input_rows_with_duplicate_canonical_url', 0)}\n")
            f.write(f"- **Total Input Rows Considered Duplicates (either CompanyName or URL):** {dp_stats.get('input_rows_considered_duplicates_overall', 0)}\n\n")

            attrition_input_company_duplicates = 0
            attrition_input_url_duplicates = 0
            attrition_overall_input_duplicates = 0
            if attrition_data_list_for_metrics:
                for attrition_row in attrition_data_list_for_metrics:
                    if attrition_row.get("Is_Input_CompanyName_Duplicate") == "Yes":
                        attrition_input_company_duplicates += 1
                    if attrition_row.get("Is_Input_CanonicalURL_Duplicate") == "Yes":
                        attrition_input_url_duplicates += 1
                    if attrition_row.get("Is_Input_Row_Considered_Duplicate") == "Yes":
                        attrition_overall_input_duplicates += 1
            
            f.write("### Input Duplicates within Attrition Report:\n")
            total_attrition_rows = len(attrition_data_list_for_metrics)
            f.write(f"- **Total Rows in Attrition Report:** {total_attrition_rows}\n")
            f.write(f"- **Attrition Rows with Original Input CompanyName Duplicate:** {attrition_input_company_duplicates}\n")
            f.write(f"- **Attrition Rows with Original Input Canonical URL Duplicate:** {attrition_input_url_duplicates}\n")
            f.write(f"- **Attrition Rows Considered Overall Original Input Duplicates:** {attrition_overall_input_duplicates}\n\n")

            # --- Scraping Statistics ---
            f.write("## Scraping Statistics:\n")
            stats = metrics.get("scraping_stats", {})
            f.write(f"- **URLs Processed for Scraping:** {stats.get('urls_processed_for_scraping', 0)}\n")
            f.write(f"- **New Canonical Sites Scraped:** {stats.get('new_canonical_sites_scraped', 0)}\n")
            f.write(f"- **Scraping Successes:** {stats.get('scraping_success', 0)}\n")
            f.write(f"- **Scraping Failures (Invalid URL):** {stats.get('scraping_failure_invalid_url', 0)}\n")
            f.write(f"- **Scraping Failures (Already Processed):** {stats.get('scraping_failure_already_processed', 0)}\n")
            f.write(f"- **Scraping Failures (Other Errors):** {stats.get('scraping_failure_error', 0)}\n")
            f.write(f"- **Total Pages Scraped Overall:** {stats.get('total_pages_scraped_overall', 0)}\n")
            f.write(f"- **Total Unique URLs Successfully Fetched:** {stats.get('total_urls_fetched_by_scraper', 0)}\n")
            f.write(f"- **Total Successfully Scraped Canonical Sites:** {stats.get('total_successful_canonical_scrapes', 0)}\n")

            total_successful_scrapes = stats.get('total_successful_canonical_scrapes', 0)
            if total_successful_scrapes > 0:
                avg_pages_per_site = stats.get('total_pages_scraped_overall', 0) / total_successful_scrapes
                f.write(f"- **Average Pages Scraped per Successfully Scraped Canonical Site:** {avg_pages_per_site:.2f}\n")
            else:
                f.write("- Average Pages Scraped per Successfully Scraped Canonical Site: N/A (No successful canonical scrapes)\n")

            f.write("- **Pages Scraped by Type:**\n")
            pages_by_type = stats.get("pages_scraped_by_type", {})
            if pages_by_type:
                for page_type, count in sorted(pages_by_type.items()):
                    f.write(f"  - *{page_type.replace('_', ' ').title()}:* {count}\n")
            else:
                f.write("  - No page type data recorded.\n")
            f.write("\n")

            # --- Regex Extraction Statistics ---
            f.write("## Regex Extraction Statistics:\n")
            stats = metrics.get("regex_extraction_stats", {})
            f.write(f"- **Canonical Sites Processed for Regex:** {stats.get('sites_processed_for_regex', 0)}\n")
            f.write(f"- **Canonical Sites with Regex Candidates Found:** {stats.get('sites_with_regex_candidates', 0)}\n")
            f.write(f"- **Total Regex Candidates Found:** {stats.get('total_regex_candidates_found', 0)}\n\n")

            # --- LLM Processing Statistics ---
            f.write("## LLM Processing Statistics:\n")
            stats = metrics.get("llm_processing_stats", {})
            f.write(f"- **Canonical Sites Sent for LLM Processing:** {stats.get('sites_processed_for_llm', 0)}\n")
            f.write(f"- **LLM Calls Successful:** {stats.get('llm_calls_success', 0)}\n")
            f.write(f"- **LLM Calls Failed (Prompt Missing):** {stats.get('llm_calls_failure_prompt_missing', 0)}\n")
            f.write(f"- **LLM Calls Failed (Processing Error):** {stats.get('llm_calls_failure_processing_error', 0)}\n")
            f.write(f"- **Canonical Sites with No Regex Candidates (Skipped LLM):** {stats.get('llm_no_candidates_to_process', 0)}\n")
            f.write(f"- **Total LLM Extracted Phone Number Objects (Raw):** {stats.get('total_llm_extracted_numbers_raw', 0)}\n")
            f.write(f"- **LLM Successful Calls with Token Data:** {stats.get('llm_successful_calls_with_token_data', 0)}\n")
            f.write(f"- **Total LLM Prompt Tokens:** {stats.get('total_llm_prompt_tokens', 0)}\n")
            f.write(f"- **Total LLM Completion Tokens:** {stats.get('total_llm_completion_tokens', 0)}\n")
            f.write(f"- **Total LLM Tokens Overall:** {stats.get('total_llm_tokens_overall', 0)}\n")

            successful_calls_for_avg = stats.get('llm_successful_calls_with_token_data', 0)
            if successful_calls_for_avg > 0:
                _write_average_metric(
                    "Average Prompt Tokens per Successful Call",
                    "total_llm_prompt_tokens", "llm_successful_calls_with_token_data", stats, unit="tokens"
                ) # Note: total_llm_prompt_tokens is not in tasks_data, it's in llm_stats_data (aliased as stats here)
                # Correcting the source for token data:
                avg_prompt_tokens = stats.get('total_llm_prompt_tokens', 0) / successful_calls_for_avg
                avg_completion_tokens = stats.get('total_llm_completion_tokens', 0) / successful_calls_for_avg
                avg_total_tokens = stats.get('total_llm_tokens_overall', 0) / successful_calls_for_avg
                f.write(f"- **Average Prompt Tokens per Successful Call:** {avg_prompt_tokens:.2f} tokens\n")
                f.write(f"- **Average Completion Tokens per Successful Call:** {avg_completion_tokens:.2f} tokens\n")
                f.write(f"- **Average Total Tokens per Successful Call:** {avg_total_tokens:.2f} tokens\n")
            else:
                f.write("- Average token counts not available (no successful calls with token data).\n")
            f.write("\n")

            # --- Report Generation Statistics ---
            f.write("## Report Generation Statistics:\n")
            stats = metrics.get("report_generation_stats", {})
            f.write(f"- **Detailed Report Rows Created:** {stats.get('detailed_report_rows', 0)}\n")
            f.write(f"- **Summary Report Rows Created:** {stats.get('summary_report_rows', 0)}\n")
            f.write(f"- **Tertiary Report Rows Created:** {stats.get('tertiary_report_rows', 0)}\n")
            f.write(f"- **Canonical Domain Summary Rows Created:** {stats.get('canonical_domain_summary_rows', 0)}\n")
            f.write(f"- **Prospect Analysis CSV Rows Created:** {stats.get('prospect_analysis_csv_rows', 0)}\n\n")


            # --- Canonical Domain Processing Summary ---
            f.write("## Canonical Domain Processing Summary:\n")
            if canonical_domain_journey_data:
                total_canonical_domains = len(canonical_domain_journey_data)
                f.write(f"- **Total Unique Canonical Domains Processed:** {total_canonical_domains}\n")

                outcome_counts: TypingCounter[str] = Counter()
                fault_counts: TypingCounter[str] = Counter()

                for domain_data in canonical_domain_journey_data.values():
                    outcome = domain_data.get("Final_Domain_Outcome_Reason", "Unknown_Outcome")
                    fault = domain_data.get("Primary_Fault_Category_For_Domain", "Unknown_Fault")
                    outcome_counts[outcome] += 1
                    if fault != "N/A":
                        fault_counts[fault] += 1
                
                f.write("### Outcomes for Canonical Domains:\n")
                if outcome_counts:
                    for outcome, count in sorted(outcome_counts.items()):
                        f.write(f"  - **{outcome.replace('_', ' ').title()}:** {count}\n")
                else:
                    f.write("  - No outcome data recorded for canonical domains.\n")
                f.write("\n")

                f.write("### Primary Fault Categories for Canonical Domains (where applicable):\n")
                if fault_counts:
                    for fault, count in sorted(fault_counts.items()):
                        f.write(f"  - **{fault.replace('_', ' ').title()}:** {count}\n")
                else:
                    f.write("  - No fault data recorded for canonical domains or all succeeded.\n")
                f.write("\n")
            else:
                f.write("- No canonical domain journey data available to summarize.\n\n")

            # --- Summary of Row-Level Failures ---
            f.write(f"## Summary of Row-Level Failures (from `failed_rows_{run_id}.csv`):\n")
            row_failures_summary = metrics.get("data_processing_stats", {}).get("row_level_failure_summary", {})

            if row_failures_summary:
                # Define categories for grouping failures
                failure_categories: Dict[str, List[str]] = {
                    "Scraping": ["Scraping_"],
                    "LLM": ["LLM_"],
                    "URL Validation": ["URL_Validation_"],
                    "Regex Extraction": ["Regex_Extraction_"],
                    "Row Processing": ["RowProcessing_"], # General row processing issues
                    "Data Issues": ["NoCanonicalURLDetermined", "InputURLInvalid"], # Issues with input data
                    "Other": [] # Catch-all for unclassified
                }
                grouped_failures: Dict[str, Dict[str, Any]] = {
                    cat: {"total": 0, "details": {}} for cat in failure_categories
                }
                grouped_failures["Other"] = {"total": 0, "details": {}} # Ensure 'Other' exists

                for stage, count in sorted(row_failures_summary.items()):
                    matched_category = False
                    for category_name, prefixes in failure_categories.items():
                        if any(stage.startswith(p) for p in prefixes) or (not prefixes and category_name == "Other"): # Check prefixes
                             # Special handling for exact matches if needed, e.g. "NoCanonicalURLDetermined"
                            if stage in prefixes: # Exact match for items like "NoCanonicalURLDetermined"
                                grouped_failures[category_name]["total"] += count
                                grouped_failures[category_name]["details"][stage] = count
                                matched_category = True
                                break
                            elif any(stage.startswith(p) for p in prefixes if p.endswith("_")): # Prefix match
                                grouped_failures[category_name]["total"] += count
                                grouped_failures[category_name]["details"][stage] = count
                                matched_category = True
                                break
                    if not matched_category:
                        grouped_failures["Other"]["total"] += count
                        grouped_failures["Other"]["details"][stage] = count
                
                for category_name, data in grouped_failures.items():
                    if data["total"] > 0:
                        f.write(f"- **Total {category_name} Failures:** {data['total']}\n")
                        for stage, count in sorted(data["details"].items()):
                            f.write(f"  - *{stage.replace('_', ' ').title()}:* {count}\n")
                f.write("\n")
            else:
                f.write("- No row-level failures recorded with specific stages.\n")
            f.write("\n")
            
            # --- Global Pipeline Errors ---
            f.write("## Global Pipeline Errors:\n")
            if metrics.get("errors_encountered"):
                for error_msg in metrics["errors_encountered"]:
                    f.write(f"- {error_msg}\n")
            else:
                f.write("- No significant global pipeline errors recorded.\n")
            f.write("\n")

            # --- Input Row Attrition Summary ---
            f.write("## Input Row Attrition Summary:\n")
            if attrition_data_list_for_metrics:
                total_input_rows = metrics.get("data_processing_stats", {}).get("input_rows_count", 0)
                rows_not_yielding_contact = len(attrition_data_list_for_metrics)
                # Ensure rows_yielding_contact is not negative if total_input_rows is unexpectedly small
                rows_yielding_contact = max(0, total_input_rows - rows_not_yielding_contact)


                f.write(f"- **Total Input Rows Processed:** {total_input_rows}\n")
                f.write(f"- **Input Rows Yielding at Least One Contact (Estimated):** {rows_yielding_contact}\n")
                f.write(f"- **Input Rows Not Yielding Any Contact (Recorded in Attrition Report):** {rows_not_yielding_contact}\n\n")

                if rows_not_yielding_contact > 0:
                    f.write("### Reasons for Non-Extraction (Fault Categories from Attrition Data):\n")
                    # Ensure fault_category_counts is TypingCounter for type hints
                    fault_category_counts: TypingCounter[str] = Counter()
                    for item in attrition_data_list_for_metrics:
                        fault = item.get("Determined_Fault_Category", "Unknown_Fault_In_Attrition")
                        fault_category_counts[fault] += 1
                    
                    for fault, count in sorted(fault_category_counts.items()):
                        f.write(f"  - **{fault.replace('_', ' ').title()}:** {count}\n")
                    f.write("\n")
            else:
                f.write("- No input rows recorded in the attrition report (all rows presumably yielded contacts or failed critically before attrition tracking).\n")
            f.write("\n")

        logger_module.info(f"Run metrics successfully written to {metrics_file_path}")
    except IOError as e:
        logger_module.error(f"Failed to write run metrics to {metrics_file_path}: {e}", exc_info=True)
    except Exception as e_global:
        logger_module.error(f"An unexpected error occurred while writing metrics to {metrics_file_path}: {e_global}", exc_info=True)