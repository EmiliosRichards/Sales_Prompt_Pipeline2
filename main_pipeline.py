import pandas as pd
from typing import List, Dict, Optional, Any
import csv
import logging
import os
import time
# from datetime import datetime # No longer used directly for run_id generation here
# import re, urllib.parse, socket, asyncio # Already removed

from dotenv import load_dotenv
# from pathlib import Path # No longer used directly for augmented report path here

from src.data_handling.loader import load_and_preprocess_data
# from src.data_handling.consolidator import get_canonical_base_url, generate_processed_contacts_report # Moved to report orchestrator
from src.llm_clients.gemini_client import GeminiClient
from src.core.schemas import GoldenPartnerMatchOutput # For the new pipeline result
from src.core.logging_config import setup_logging
from src.core.config import AppConfig
# from src.core.constants import EXCLUDED_TYPES_FOR_TOP_CONTACTS_REPORT, FAULT_CATEGORY_MAP_DEFINITION # Used in report orchestrator
from src.data_handling.partner_data_handler import load_golden_partners, summarize_golden_partner # Added for Golden Partners
from src.utils.helpers import (
    generate_run_id,
    # get_input_canonical_url, # Used by precompute_input_duplicate_stats
    resolve_path,
    initialize_run_metrics,
    setup_output_directories,
    precompute_input_duplicate_stats,
    initialize_dataframe_columns
)
from src.reporting.metrics_manager import write_run_metrics
from src.processing.pipeline_flow import execute_pipeline_flow
from src.reporting.main_report_orchestrator import generate_all_reports # NEW

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# __file__ will refer to main_pipeline.py's location
BASE_FILE_PATH_FOR_RESOLVE = __file__

def main() -> None:
    app_config: AppConfig = AppConfig() # Initialize AppConfig globally for easy access
    """
    Main entry point for the phone validation pipeline.
    Orchestrates the entire process from data loading to report generation.
    """
    pipeline_start_time = time.time()
    
    # 1. Initialize Run ID and Metrics
    run_id = generate_run_id()
    run_metrics: Dict[str, Any] = initialize_run_metrics(run_id) # Use helper

    # 2. Setup Output Directories
    # __file__ refers to the location of main_pipeline.py
    run_output_dir, llm_context_dir, llm_requests_dir = setup_output_directories(app_config, run_id, BASE_FILE_PATH_FOR_RESOLVE)

    # 3. Setup Logging
    log_file_name = f"pipeline_run_{run_id}.log"
    log_file_path = os.path.join(run_output_dir, log_file_name)
    file_log_level_int = getattr(logging, app_config.log_level.upper(), logging.INFO)
    console_log_level_int = getattr(logging, app_config.console_log_level.upper(), logging.WARNING)
    setup_logging(file_log_level=file_log_level_int, console_log_level=console_log_level_int, log_file_path=log_file_path)
    
    logger.info(f"Logging initialized. Run ID: {run_id}")
    logger.info(f"Base output directory for this run: {run_output_dir}")

    # Resolve input file path (relative to project root if not absolute)
    # The project root is determined based on the location of this main_pipeline.py file.
    input_file_path_abs = resolve_path(app_config.input_excel_file_path, BASE_FILE_PATH_FOR_RESOLVE) # Use helper
    logger.info(f"Resolved input file path: {input_file_path_abs}")

    failure_log_csv_path = os.path.join(run_output_dir, f"failed_rows_{run_id}.csv")
    logger.info(f"Row-specific failure log for this run will be: {failure_log_csv_path}")

    logger.info("Starting phone validation pipeline...")
    if not os.path.exists(input_file_path_abs):
        logger.error(f"CRITICAL: Input file not found at resolved path: {input_file_path_abs}. Exiting.")
        # Minimal metrics write on critical early failure
        run_metrics["errors_encountered"].append(f"Input file not found: {input_file_path_abs}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir if 'run_output_dir' in locals() else ".", run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return

    # 4. Initialize LLM Extractor
    gemini_client: Optional[GeminiClient] = None # Initialize as Optional
    try:
        gemini_client = GeminiClient(config=app_config)
        logger.info("GeminiClient initialized successfully.")
    except ValueError as ve:
        logger.error(f"Failed to initialize GeminiClient: {ve}. Check GEMINI_API_KEY. Pipeline cannot proceed with LLM steps.")
        run_metrics["errors_encountered"].append(f"LLM Extractor init failed: {ve}")
        # Decide if pipeline should stop or continue without LLM
        # For now, let's assume it stops if LLM is critical.
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing GeminiClient: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"LLM Extractor init unexpected error: {e}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    
    if gemini_client is None: # Safeguard, should be caught by returns above
        logger.error("GeminiClient is None after initialization attempt. Exiting.")
        return

    # Load and Summarize Golden Partner Data
    golden_partner_summaries: List[Dict[str, Any]] = []
    golden_partners_raw: List[Dict[str, Any]] = []
    try:
        logger.info("Loading Golden Partner data...")
        # Assuming app_config.golden_partner_data_path holds the path
        golden_partner_data_path_abs = resolve_path(app_config.PATH_TO_GOLDEN_PARTNERS_DATA, BASE_FILE_PATH_FOR_RESOLVE)
        logger.info(f"Resolved Golden Partner data path: {golden_partner_data_path_abs}")

        golden_partners_raw = load_golden_partners(file_path=golden_partner_data_path_abs)

        if golden_partners_raw:
            logger.info(f"Successfully loaded {len(golden_partners_raw)} golden partners.")
            for partner_dict_data in golden_partners_raw:
                # summarize_golden_partner takes Dict[str, Any] and returns str
                # It does not use llm_extractor, config, or llm_context_dir
                summary_obj = summarize_golden_partner(partner_data=partner_dict_data)
                if summary_obj and summary_obj["summary"] != "Partner data not available or insufficient for summary.":
                    golden_partner_summaries.append(summary_obj)
                else:
                    partner_name_for_log = partner_dict_data.get('Partner Name', 'Unknown Partner')
                    logger.warning(f"Failed to generate a meaningful summary for golden partner: {partner_name_for_log}")
            logger.info(f"Generated {len(golden_partner_summaries)} golden partner summaries (strings).")
            if not golden_partner_summaries and golden_partners_raw:
                 logger.warning("No meaningful summaries generated for loaded golden partners. Sales insight generation might be impacted.")
        else:
            logger.warning("No golden partner data loaded or found. Proceeding without golden partner comparisons.")
    except Exception as e_gp:
        logger.error(f"Error during golden partner data loading or summarization: {e_gp}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Golden partner processing error: {str(e_gp)}")
        # Continue pipeline execution, but sales insights will be affected.

    # 5. Load and Preprocess Data
    df: Optional[pd.DataFrame] = None
    task_start_time = time.time()
    try:
        logger.info(f"Attempting to load data from: {input_file_path_abs}")
        df = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config)
        if df is not None:
            logger.info(f"Successfully loaded and preprocessed data. Shape: {df.shape}.")
            run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
        else:
            logger.error(f"Failed to load data from {input_file_path_abs}. DataFrame is None.")
            run_metrics["errors_encountered"].append(f"Data loading failed: DataFrame is None from {input_file_path_abs}")
            run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
            # Finalize metrics and exit
            run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
            write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
            return
    except Exception as e:
        logger.error(f"Error loading data in main: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Data loading exception: {str(e)}")
        run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
        # Finalize metrics and exit
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={})
        return
    run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
 
    if df is None: # Should be caught by returns above, but as a safeguard
        logger.error("DataFrame is None after loading attempt, cannot proceed.")
        return
    assert df is not None, "DataFrame loading failed, assertion." # Should not be reached if above checks work

    # 6. Initialize DataFrame Columns
    df = initialize_dataframe_columns(df) # Use helper

    # 7. Pre-computation of Input Duplicate Counts
    pre_comp_start_time = time.time()
    df = precompute_input_duplicate_stats(df, app_config, run_metrics) # Use helper
    logger.info(f"Input duplicate pre-computation complete. Duration: {time.time() - pre_comp_start_time:.2f}s")
    run_metrics["tasks"]["pre_computation_duplicate_counts_duration_seconds"] = time.time() - pre_comp_start_time
    
    # Initialize variables that will be populated by execute_pipeline_flow
    attrition_data_list: List[Dict[str, Any]] = []
    canonical_domain_journey_data: Dict[str, Any] = {}
    
    # 8. Execute Core Pipeline Flow
    failure_log_file_handle = None
    failure_writer = None
    try:
        failure_log_file_handle = open(failure_log_csv_path, 'w', newline='', encoding='utf-8')
        failure_writer = csv.writer(failure_log_file_handle)
        # Write header for failure log
        failure_writer.writerow(['log_timestamp', 'input_row_identifier', 'CompanyName', 'GivenURL', 'stage_of_failure', 'error_reason', 'error_details', 'Associated_Pathful_Canonical_URL'])

        logger.info("Starting core pipeline processing flow...")
        # These variables will be populated by execute_pipeline_flow
        # final_consolidated_data_by_true_base is replaced by all_match_outputs
        # true_base_scraper_status might still be relevant for overall scraping stats
        (df, all_match_outputs, attrition_data_list, canonical_domain_journey_data,
         true_base_scraper_status, true_base_to_pathful_map, input_to_canonical_map,
         row_level_failure_counts
        ) = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            gemini_client=gemini_client,
            run_output_dir=run_output_dir,
            llm_context_dir=llm_context_dir,
            llm_requests_dir=llm_requests_dir,
            run_id=run_id,
            failure_writer=failure_writer,
            run_metrics=run_metrics,
            golden_partner_summaries=golden_partner_summaries
        )
        run_metrics["data_processing_stats"]["row_level_failure_summary"] = row_level_failure_counts # Update from flow
        logger.info("Core pipeline processing flow finished.")

        # 9. Report Generation
        # All report generation logic is now encapsulated in main_report_orchestrator
        # generate_all_reports will need to be updated to handle all_match_outputs
        if df is not None:
            # Dynamically find the original phone column name from the active profile
            active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, {})
            original_phone_col_name = next((k for k, v in active_profile.items() if v == "PhoneNumber"), None)

            generate_all_reports(
                df=df,
                app_config=app_config,
                run_id=run_id,
                run_output_dir=run_output_dir,
                run_metrics=run_metrics,
                attrition_data_list=attrition_data_list,
                canonical_domain_journey_data=canonical_domain_journey_data,
                input_to_canonical_map=input_to_canonical_map,
                all_golden_partner_match_outputs=all_match_outputs, # Corrected parameter name and position
                true_base_scraper_status=true_base_scraper_status,
                original_phone_col_name_for_profile=original_phone_col_name,
                original_input_file_path=input_file_path_abs,
                sales_prompt_path=None
            )

    except Exception as pipeline_exec_error:
        logger.error(f"An unhandled error occurred during pipeline execution or reporting: {pipeline_exec_error}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Pipeline execution/reporting error: {str(pipeline_exec_error)}")
    finally:
        if failure_log_file_handle:
            try:
                failure_log_file_handle.close()
            except Exception as e_close:
                logger.error(f"Error closing failure log CSV: {e_close}")
    
    # 10. Finalize and Write Run Metrics
    run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
    write_run_metrics(
        metrics=run_metrics,
        output_dir=run_output_dir,
        run_id=run_id,
        pipeline_start_time=pipeline_start_time,
        attrition_data_list_for_metrics=attrition_data_list, # Pass the populated list
        canonical_domain_journey_data=canonical_domain_journey_data # Pass the populated dict
    )
    logger.info(f"Pipeline run {run_id} finished. Total duration: {run_metrics['total_duration_seconds']:.2f}s.")
    logger.info(f"Run metrics file: {os.path.join(run_output_dir, f'run_metrics_{run_id}.md')}")
    logger.info(f"All outputs for this run are in: {run_output_dir}")

if __name__ == '__main__':
    # Basic logging config if no handlers are configured yet (e.g., when run directly)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()