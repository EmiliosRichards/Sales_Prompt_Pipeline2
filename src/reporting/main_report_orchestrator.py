"""
Orchestrates the generation of various reports for the contact pipeline.
i
This module is responsible for calling individual report generation functions
to produce outputs such as CSV analysis, attrition reports, and domain summaries.
It manages the flow of data to these reporting functions and updates run metrics
accordingly.

Note:
    Several older report generation functions related to a previous data
    consolidation logic (using `final_consolidated_data_by_true_base`)
    are currently commented out in the main `generate_all_reports` function.
    These are preserved for potential future reference or adaptation but are
    not active in the current primary workflow, which focuses on reports
    derived from `all_golden_partner_match_outputs`.
"""
import pandas as pd
import logging
import os
import time # Moved import to the top
from typing import List, Dict, Any, Optional

from src.core.config import AppConfig
from src.core.schemas import (
    GoldenPartnerMatchOutput
)
from .report_generator import (
    write_row_attrition_report,
    write_canonical_domain_summary_report,
)
from .csv_reporter import write_sales_outreach_report
from .slack_notifier import send_slack_notification

logger = logging.getLogger(__name__)


def generate_all_reports(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    attrition_data_list: List[Dict[str, Any]],
    canonical_domain_journey_data: Dict[str, Any],
    input_to_canonical_map: Dict[str, Optional[str]],
    all_golden_partner_match_outputs: List[GoldenPartnerMatchOutput],
    true_base_scraper_status: Dict[str, str],  # Kept for potential use by legacy reports
    original_phone_col_name_for_profile: Optional[str],  # May become unused if augmented report is fully deprecated
    original_input_file_path: str,
    sales_prompt_path: Optional[str] = None,
    run_command: Optional[str] = None
) -> None:
    """
    Orchestrates the generation of all standard pipeline reports.

    This function coordinates the creation of various output files, including
    the primary prospect analysis CSV, attrition reports, and domain journey summaries.
    It also updates the `run_metrics` dictionary with statistics related to
    report generation.

    Note:
        Several older report generation functions (`_generate_detailed_report`,
        `_generate_tertiary_report`, `_generate_summary_report`,
        `_generate_augmented_input_report`, `_generate_final_processed_contacts_report`)
        are currently not called directly. They relied on a previous data structure
        (`final_consolidated_data_by_true_base`) and are preserved in this module
        for potential future reference or adaptation. The primary focus is now on
        reports generated from `all_golden_partner_match_outputs`.

    Args:
        df (pd.DataFrame): The main processed DataFrame from the pipeline.
                           (Currently primarily used by legacy/commented-out reports).
        app_config (AppConfig): The application configuration object.
        run_id (str): The unique identifier for the current pipeline run.
        run_output_dir (str): The directory where reports will be saved.
        run_metrics (Dict[str, Any]): A dictionary to store metrics about the run.
        attrition_data_list (List[Dict[str, Any]]): Data for the row attrition report.
        canonical_domain_journey_data (Dict[str, Any]): Data for the canonical domain summary.
        input_to_canonical_map (Dict[str, Optional[str]]): Mapping of input URLs to canonical URLs.
        all_golden_partner_match_outputs (List[GoldenPartnerMatchOutput]): The primary data output
            from the LLM analysis stages, used for the main prospect analysis CSV.
        true_base_scraper_status (Dict[str, str]): Status of scraping attempts for true base URLs.
                                                    (Primarily for legacy reports).
        original_phone_col_name_for_profile (Optional[str]): The name of the original phone number
            column from the input file, used by the (currently inactive) augmented input report.
        original_input_file_path (str): Path to the original input file, used by the
                                            (currently inactive) augmented input report.
        """
    logger.info("Starting main report orchestration...")
    report_generation_start_time = time.time()

    # Generate Sales Outreach Report
    if all_golden_partner_match_outputs:
        sales_outreach_report_path_csv = write_sales_outreach_report(
            output_data=all_golden_partner_match_outputs,
            output_dir=run_output_dir,
            run_id=run_id,
            original_df=df,
            output_format='csv'
        )
        sales_outreach_report_path_excel = write_sales_outreach_report(
            output_data=all_golden_partner_match_outputs,
            output_dir=run_output_dir,
            run_id=run_id,
            original_df=df,
            output_format='excel'
        )
        if sales_outreach_report_path_csv:
            logger.info(f"Sales outreach report generated at: {sales_outreach_report_path_csv}")
            run_metrics["report_generation_stats"]["sales_outreach_report_rows"] = len(all_golden_partner_match_outputs)
            send_slack_notification(
                config=app_config,
                file_path=sales_outreach_report_path_csv,
                report_name="Sales Outreach Report (CSV)",
                run_id=run_id,
                input_file=os.path.basename(original_input_file_path),
                rows_processed=len(all_golden_partner_match_outputs),
                mode=app_config.input_file_profile_name,
                run_command=run_command or ""
            )
        if sales_outreach_report_path_excel:
            send_slack_notification(
                config=app_config,
                file_path=sales_outreach_report_path_excel,
                report_name="Sales Outreach Report (Excel)",
                run_id=run_id,
                input_file=os.path.basename(original_input_file_path),
                rows_processed=len(all_golden_partner_match_outputs),
                mode=app_config.input_file_profile_name,
                run_command=run_command or ""
            )
        else:
            logger.error("Failed to generate sales outreach report.")
            run_metrics["report_generation_stats"]["sales_outreach_report_rows"] = 0
    else:
        logger.info("No GoldenPartnerMatchOutput data to generate sales outreach report.")
        run_metrics["report_generation_stats"]["sales_outreach_report_rows"] = 0



    # --- Currently Active Reports ---

    # 4. Canonical Domain Summary Report

    # 5. Row Attrition Report (Still relevant for tracking failures)
    if attrition_data_list:
        num_attrition_rows = write_row_attrition_report(run_id, attrition_data_list, run_output_dir, canonical_domain_journey_data, input_to_canonical_map, logger, app_config)
        run_metrics["data_processing_stats"]["rows_in_attrition_report"] = num_attrition_rows
    else:
        logger.info("No attrition_data_list for report.")
        run_metrics["data_processing_stats"]["rows_in_attrition_report"] = 0
        

    run_metrics["tasks"]["report_orchestration_duration_seconds"] = round(time.time() - report_generation_start_time, 2)
    logger.info("Main report orchestration finished.")

