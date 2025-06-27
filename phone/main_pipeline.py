import pandas as pd
from typing import List, Dict, Optional, Any # Removed Set, Callable, Union, Tuple as they are no longer directly used here
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
from src.extractors.llm_extractor import GeminiLLMExtractor
# from src.core.schemas import CompanyContactDetails, ConsolidatedPhoneNumber # Used within pipeline_flow and report_orchestrator
from src.core.logging_config import setup_logging
from src.core.config import AppConfig
# from src.core.constants import EXCLUDED_TYPES_FOR_TOP_CONTACTS_REPORT, FAULT_CATEGORY_MAP_DEFINITION # Used in report orchestrator
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

load_dotenv()

logger = logging.getLogger(__name__)
app_config: AppConfig = AppConfig() # Initialize AppConfig globally for easy access

# __file__ will refer to main_pipeline.py's location
BASE_FILE_PATH_FOR_RESOLVE = __file__

def main() -> None:
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
    run_output_dir, llm_context_dir = setup_output_directories(app_config, run_id, BASE_FILE_PATH_FOR_RESOLVE) # Use helper

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
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir if 'run_output_dir' in locals() else ".", run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={}, logger=logger)
        return

    # 4. Initialize LLM Extractor
    llm_extractor: Optional[GeminiLLMExtractor] = None # Initialize as Optional
    try:
        llm_extractor = GeminiLLMExtractor(config=app_config)
        logger.info("GeminiLLMExtractor initialized successfully.")
    except ValueError as ve:
        logger.error(f"Failed to initialize GeminiLLMExtractor: {ve}. Check GEMINI_API_KEY. Pipeline cannot proceed with LLM steps.")
        run_metrics["errors_encountered"].append(f"LLM Extractor init failed: {ve}")
        # Decide if pipeline should stop or continue without LLM
        # For now, let's assume it stops if LLM is critical.
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={}, logger=logger)
        return
    except Exception as e:
        logger.error(f"Unexpected error initializing GeminiLLMExtractor: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"LLM Extractor init unexpected error: {e}")
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={}, logger=logger)
        return
    
    if llm_extractor is None: # Safeguard, should be caught by returns above
        logger.error("LLM Extractor is None after initialization attempt. Exiting.")
        return


    # 5. Load and Preprocess Data
    df: Optional[pd.DataFrame] = None
    original_phone_col_name_for_profile: Optional[str] = None
    new_phone_col_name_for_profile: Optional[str] = None
    task_start_time = time.time()
    try:
        logger.info(f"Attempting to load data from: {input_file_path_abs}")
        df, original_phone_col_name_for_profile, new_phone_col_name_for_profile = load_and_preprocess_data(input_file_path_abs, app_config_instance=app_config)
        if df is not None:
            logger.info(f"Successfully loaded and preprocessed data. Shape: {df.shape}. Original phone col: '{original_phone_col_name_for_profile}', New phone col: '{new_phone_col_name_for_profile}'")
            run_metrics["data_processing_stats"]["input_rows_count"] = len(df)
        else:
            logger.error(f"Failed to load data from {input_file_path_abs}. DataFrame is None.")
            run_metrics["errors_encountered"].append(f"Data loading failed: DataFrame is None from {input_file_path_abs}")
            run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
            # Finalize metrics and exit
            run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
            write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={}, logger=logger)
            return
    except Exception as e:
        logger.error(f"Error loading data in main: {e}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Data loading exception: {str(e)}")
        run_metrics["tasks"]["load_and_preprocess_data_duration_seconds"] = time.time() - task_start_time
        # Finalize metrics and exit
        run_metrics["total_duration_seconds"] = time.time() - pipeline_start_time
        write_run_metrics(metrics=run_metrics, output_dir=run_output_dir, run_id=run_id, pipeline_start_time=pipeline_start_time, attrition_data_list_for_metrics=[], canonical_domain_journey_data={}, logger=logger)
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
        # final_consolidated_data_by_true_base, true_base_scraper_status, etc.
        (df, attrition_data_list, canonical_domain_journey_data,
         final_consolidated_data_by_true_base, true_base_scraper_status,
         true_base_to_pathful_map, input_to_canonical_map, row_level_failure_counts
        ) = execute_pipeline_flow(
            df=df,
            app_config=app_config,
            llm_extractor=llm_extractor, # llm_extractor is now guaranteed to be initialized or pipeline exited
            run_output_dir=run_output_dir,
            llm_context_dir=llm_context_dir,
            run_id=run_id,
            failure_writer=failure_writer,
            run_metrics=run_metrics,
            original_phone_col_name_for_profile=original_phone_col_name_for_profile
        )
        run_metrics["data_processing_stats"]["row_level_failure_summary"] = row_level_failure_counts # Update from flow
        logger.info("Core pipeline processing flow finished.")

        # 9. Report Generation
        # All report generation logic is now encapsulated in main_report_orchestrator
        generate_all_reports(
            df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir,
            run_metrics=run_metrics,
            attrition_data_list=attrition_data_list,
            canonical_domain_journey_data=canonical_domain_journey_data,
            input_to_canonical_map=input_to_canonical_map,
            final_consolidated_data_by_true_base=final_consolidated_data_by_true_base,
            true_base_scraper_status=true_base_scraper_status,
            original_phone_col_name_for_profile=original_phone_col_name_for_profile,
            new_phone_col_name_for_profile=new_phone_col_name_for_profile,
            original_input_file_path=input_file_path_abs # Pass the absolute path
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
        canonical_domain_journey_data=canonical_domain_journey_data, # Pass the populated dict
        logger=logger
    )
    logger.info(f"Pipeline run {run_id} finished. Total duration: {run_metrics['total_duration_seconds']:.2f}s.")
    logger.info(f"Run metrics file: {os.path.join(run_output_dir, f'run_metrics_{run_id}.md')}")
    logger.info(f"All outputs for this run are in: {run_output_dir}")

if __name__ == '__main__':
    # Basic logging config if no handlers are configured yet (e.g., when run directly)
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()