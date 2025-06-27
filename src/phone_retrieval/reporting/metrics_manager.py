import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Counter # Added Counter

def write_run_metrics(
    metrics: Dict[str, Any], 
    output_dir: str, 
    run_id: str, 
    pipeline_start_time: float, 
    attrition_data_list_for_metrics: List[Dict[str, Any]], 
    canonical_domain_journey_data: Dict[str, Dict[str, Any]],
    logger: logging.Logger # Added logger instance
) -> None:
    """Writes the collected run metrics to a Markdown file."""
    metrics["total_duration_seconds"] = time.time() - pipeline_start_time
    metrics_file_path = os.path.join(output_dir, f"run_metrics_{run_id}.md")

    try:
        with open(metrics_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pipeline Run Metrics: {run_id}\n\n")
            f.write(f"**Run ID:** {metrics.get('run_id', 'N/A')}\n")
            f.write(f"**Total Run Duration:** {metrics.get('total_duration_seconds', 0):.2f} seconds\n")
            f.write(f"**Pipeline Start Time:** {datetime.fromtimestamp(pipeline_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Pipeline End Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Task Durations (seconds):\n")
            if metrics.get("tasks"):
                for task_name, duration in metrics["tasks"].items():
                    f.write(f"- **{task_name.replace('_', ' ').title()}:** {duration:.2f}\n")
            else:
                f.write("- No task durations recorded.\n")
            f.write("\n")

            f.write("### Average Task Durations (per relevant item):\n")

            tasks_data = metrics.get("tasks", {})
            scraping_stats_data = metrics.get("scraping_stats", {})
            regex_stats_data = metrics.get("regex_extraction_stats", {})
            llm_stats_data = metrics.get("llm_processing_stats", {})
            data_proc_stats_data = metrics.get("data_processing_stats", {})

            # Average Scrape Duration
            total_scrape_duration = tasks_data.get("scrape_website_total_duration_seconds", 0)
            new_canonical_sites_scraped = scraping_stats_data.get("new_canonical_sites_scraped", 0)
            if new_canonical_sites_scraped > 0:
                avg_scrape_duration = total_scrape_duration / new_canonical_sites_scraped
                f.write(f"- **Average Scrape Website Duration (per New Canonical Site Scraped):** {avg_scrape_duration:.2f} seconds\n")
            else:
                f.write("- Average Scrape Website Duration (per New Canonical Site Scraped): N/A (No new canonical sites scraped)\n")

            # Average Regex Extraction Duration
            total_regex_duration = tasks_data.get("regex_extraction_total_duration_seconds", 0)
            sites_processed_for_regex = regex_stats_data.get("sites_processed_for_regex", 0)
            if sites_processed_for_regex > 0:
                avg_regex_duration = total_regex_duration / sites_processed_for_regex
                f.write(f"- **Average Regex Extraction Duration (per Site Processed for Regex):** {avg_regex_duration:.2f} seconds\n")
            else:
                f.write("- Average Regex Extraction Duration (per Site Processed for Regex): N/A (No sites processed for regex)\n")

            # Average LLM Extraction Duration
            total_llm_duration = tasks_data.get("llm_extraction_total_duration_seconds", 0)
            sites_processed_for_llm = llm_stats_data.get("sites_processed_for_llm", 0)
            if sites_processed_for_llm > 0:
                avg_llm_duration = total_llm_duration / sites_processed_for_llm
                f.write(f"- **Average LLM Extraction Duration (per Site Processed for LLM):** {avg_llm_duration:.2f} seconds\n")
            else:
                f.write("- Average LLM Extraction Duration (per Site Processed for LLM): N/A (No sites processed for LLM)\n")

            # Average Pass 1 Main Loop Duration
            total_pass1_duration = tasks_data.get("pass1_main_loop_duration_seconds", 0)
            input_rows_count = data_proc_stats_data.get("input_rows_count", 0)
            if input_rows_count > 0:
                avg_pass1_duration = total_pass1_duration / input_rows_count
                f.write(f"- **Average Pass 1 Main Loop Duration (per Input Row):** {avg_pass1_duration:.2f} seconds\n")
            else:
                f.write("- Average Pass 1 Main Loop Duration (per Input Row): N/A (No input rows)\n")
            
            f.write("\n") 

            f.write("## Data Processing Statistics:\n")
            stats = metrics.get("data_processing_stats", {})
            f.write(f"- **Input Rows Processed (Initial Load):** {stats.get('input_rows_count', 0)}\n")
            f.write(f"- **Rows Successfully Processed (Pass 1):** {stats.get('rows_successfully_processed_pass1', 0)}\n")
            f.write(f"- **Rows Failed During Processing (Pass 1):** {stats.get('rows_failed_pass1', 0)} (Input rows that did not complete Pass 1 successfully due to errors such as invalid URL, scraping failure, or critical processing exceptions for that row, preventing LLM processing or final data consolidation for that specific input.)\n")
            f.write(f"- **Unique True Base Domains Consolidated:** {stats.get('unique_true_base_domains_consolidated', 0)}\n")
            
            f.write("\n## Input Data Duplicate Analysis:\n")
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

            if stats.get('total_successful_canonical_scrapes', 0) > 0:
                avg_pages_per_site = stats.get('total_pages_scraped_overall', 0) / stats.get('total_successful_canonical_scrapes', 1) 
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

            f.write("## Regex Extraction Statistics:\n")
            stats = metrics.get("regex_extraction_stats", {})
            f.write(f"- **Canonical Sites Processed for Regex:** {stats.get('sites_processed_for_regex', 0)}\n")
            f.write(f"- **Canonical Sites with Regex Candidates Found:** {stats.get('sites_with_regex_candidates', 0)}\n")
            f.write(f"- **Total Regex Candidates Found:** {stats.get('total_regex_candidates_found', 0)}\n\n")

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
                avg_prompt_tokens = stats.get('total_llm_prompt_tokens', 0) / successful_calls_for_avg
                avg_completion_tokens = stats.get('total_llm_completion_tokens', 0) / successful_calls_for_avg
                avg_total_tokens = stats.get('total_llm_tokens_overall', 0) / successful_calls_for_avg
                f.write(f"- **Average Prompt Tokens per Successful Call:** {avg_prompt_tokens:.2f}\n")
                f.write(f"- **Average Completion Tokens per Successful Call:** {avg_completion_tokens:.2f}\n")
                f.write(f"- **Average Total Tokens per Successful Call:** {avg_total_tokens:.2f}\n")
            else:
                f.write("- Average token counts not available (no successful calls with token data).\n")
            f.write("\n")

            f.write("## Report Generation Statistics:\n")
            stats = metrics.get("report_generation_stats", {})
            f.write(f"- **Detailed Report Rows Created:** {stats.get('detailed_report_rows', 0)}\n")
            f.write(f"- **Summary Report Rows Created:** {stats.get('summary_report_rows', 0)}\n")
            f.write(f"- **Tertiary Report Rows Created:** {stats.get('tertiary_report_rows', 0)}\n")
            f.write(f"- **Canonical Domain Summary Rows Created:** {stats.get('canonical_domain_summary_rows', 0)}\n\n")

            f.write("## Canonical Domain Processing Summary:\n")
            if canonical_domain_journey_data:
                total_canonical_domains = len(canonical_domain_journey_data)
                f.write(f"- **Total Unique Canonical Domains Processed:** {total_canonical_domains}\n")

                outcome_counts: Dict[str, int] = Counter()
                fault_counts: Dict[str, int] = Counter()

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

            f.write("## Summary of Row-Level Failures (from `failed_rows_{run_id}.csv`):\n")
            row_failures_summary = metrics.get("data_processing_stats", {}).get("row_level_failure_summary", {})

            if row_failures_summary:
                grouped_failures: Dict[str, Dict[str, Any]] = {
                    "Scraping": {"total": 0, "details": {}},
                    "LLM": {"total": 0, "details": {}},
                    "URL Validation": {"total": 0, "details": {}},
                    "Regex Extraction": {"total": 0, "details": {}},
                    "Row Processing": {"total": 0, "details": {}},
                    "Other": {"total": 0, "details": {}}
                }
                failure_category_map: Dict[str, str] = {
                    "Scraping_": "Scraping",
                    "LLM_": "LLM",
                    "URL_Validation_": "URL Validation",
                    "Regex_Extraction_": "Regex Extraction",
                    "RowProcessing_": "Row Processing"
                }

                for stage, count in sorted(row_failures_summary.items()):
                    matched_category = False
                    for prefix, category_name in failure_category_map.items():
                        if stage.startswith(prefix):
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
            
            f.write("## Global Pipeline Errors:\n") 
            if metrics.get("errors_encountered"):
                for error_msg in metrics["errors_encountered"]:
                    f.write(f"- {error_msg}\n")
            else:
                f.write("- No significant global pipeline errors recorded.\n")
            f.write("\n")

            f.write("## Input Row Attrition Summary:\n")
            if attrition_data_list_for_metrics:
                total_input_rows = metrics.get("data_processing_stats", {}).get("input_rows_count", 0)
                rows_not_yielding_contact = len(attrition_data_list_for_metrics)
                rows_yielding_contact = total_input_rows - rows_not_yielding_contact

                f.write(f"- **Total Input Rows Processed:** {total_input_rows}\n")
                f.write(f"- **Input Rows Yielding at Least One Contact:** {rows_yielding_contact}\n")
                f.write(f"- **Input Rows Not Yielding Any Contact:** {rows_not_yielding_contact}\n\n")

                if rows_not_yielding_contact > 0:
                    f.write("### Reasons for Non-Extraction (Fault Categories):\n")
                    fault_category_counts: Dict[str, int] = Counter() # Use Counter here
                    for item in attrition_data_list_for_metrics:
                        fault = item.get("Determined_Fault_Category", "Unknown")
                        fault_category_counts[fault] += 1
                    
                    for fault, count in sorted(fault_category_counts.items()):
                        f.write(f"  - **{fault}:** {count}\n")
                    f.write("\n")
            else:
                f.write("- No input rows recorded in the attrition report (all rows presumably yielded contacts or failed critically before attrition tracking).\n")
            f.write("\n")

        logger.info(f"Run metrics successfully written to {metrics_file_path}")
    except IOError as e:
        logger.error(f"Failed to write run metrics to {metrics_file_path}: {e}", exc_info=True)
    except Exception as e_global:
        logger.error(f"An unexpected error occurred while writing metrics to {metrics_file_path}: {e_global}", exc_info=True)