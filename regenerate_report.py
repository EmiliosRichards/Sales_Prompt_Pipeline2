import argparse
import os
import pandas as pd
import json
import logging
import re # For parsing filenames
from typing import List, Dict, Optional, Any

from src.core.config import AppConfig
from src.core.schemas import GoldenPartnerMatchOutput, DetailedCompanyAttributes, B2BAnalysisOutput, WebsiteTextSummary
from src.data_handling.loader import load_and_preprocess_data
from src.core.logging_config import setup_logging
from src.utils.helpers import resolve_path # To resolve input file path

# Set up basic logging
setup_logging()
logger = logging.getLogger(__name__)


def get_row_index_from_filename(filename: str) -> Optional[int]:
    """Parses the row index from an LLM context filename."""
    match = re.search(r"Row(\d+)_", filename)
    if match:
        return int(match.group(1))
    return None


def regenerate_report(run_id: str, app_config: AppConfig):
    """
    Main function to regenerate the sales outreach report from a given run's artifacts.
    """
    logger.info(f"Starting report regeneration for run_id: {run_id}")

    # --- 1. Initialization and Path Setup ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
    output_base_dir_abs = os.path.join(project_root, app_config.output_base_dir)
    run_output_dir = os.path.join(output_base_dir_abs, run_id)
    llm_context_dir = os.path.join(run_output_dir, app_config.llm_context_subdir)
    failed_rows_path = os.path.join(run_output_dir, f"failed_rows_{run_id}.csv")

    if not os.path.isdir(run_output_dir):
        logger.error(f"Run output directory not found: {run_output_dir}")
        return

    # --- 2. Load Initial Data & Apply Slice ---
    input_file_path_abs = resolve_path(app_config.input_excel_file_path, __file__)
    logger.info(f"Loading original input data from: {input_file_path_abs}")
    original_df = load_and_preprocess_data(input_file_path_abs, app_config)
    if original_df is None:
        logger.error(f"Failed to load original input data from {input_file_path_abs}. Exiting.")
        return
    original_df.reset_index(inplace=True)
    original_df.rename(columns={'index': 'original_index'}, inplace=True)

    start_row = app_config.skip_rows_config or 0
    num_rows = app_config.nrows_config
    end_row = (start_row + num_rows) if num_rows is not None else len(original_df)
    processed_df_slice = original_df.iloc[start_row:end_row].copy()
    logger.info(f"Processing slice of original data: rows {start_row} to {end_row-1}")

    if processed_df_slice.empty:
        logger.warning("The specified processing slice resulted in an empty DataFrame.")
        return

    # --- 3. Load Run Artifacts ---
    failure_map: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(failed_rows_path):
        failures_df = pd.read_csv(failed_rows_path)
        for _, row in failures_df.iterrows():
            failure_map[row['input_row_identifier']] = row.to_dict()
        logger.info(f"Loaded {len(failure_map)} failure records.")

    artifact_map: Dict[int, Dict[str, str]] = {}
    if os.path.isdir(llm_context_dir):
        for filename in os.listdir(llm_context_dir):
            row_index = get_row_index_from_filename(filename)
            if row_index is not None:
                if row_index not in artifact_map:
                    artifact_map[row_index] = {}
                
                if "_1a_B2BCapacityCheck.json" in filename:
                    artifact_map[row_index]['b2b_check'] = os.path.join(llm_context_dir, filename)
                elif "_1_WebsiteSummary.json" in filename:
                    artifact_map[row_index]['summary'] = os.path.join(llm_context_dir, filename)
                elif "_2_AttributeExtraction.json" in filename:
                    artifact_map[row_index]['attributes'] = os.path.join(llm_context_dir, filename)
                elif "_4_SalesPitchGeneration.json" in filename:
                    artifact_map[row_index]['sales_pitch'] = os.path.join(llm_context_dir, filename)
        logger.info(f"Mapped LLM artifacts for {len(artifact_map)} rows.")

    # --- 4. Row-by-Row Progressive Enrichment ---
    report_data = []
    missing_rows_data = []
    logger.info("Starting row-by-row progressive enrichment...")

    for _, row in processed_df_slice.iterrows():
        row_index = row['original_index']
        report_row = row.to_dict() # Start with the original data
        artifacts = artifact_map.get(row_index)

        if artifacts:
            # Progressively enrich the row
            if 'b2b_check' in artifacts:
                try:
                    with open(artifacts['b2b_check'], 'r', encoding='utf-8') as f:
                        b2b_data = B2BAnalysisOutput.parse_obj(json.load(f))
                        report_row['is_b2b'] = b2b_data.is_b2b
                        report_row['is_b2b_reason'] = b2b_data.is_b2b_reason
                        report_row['serves_1000'] = b2b_data.serves_1000_customers
                        report_row['serves_1000_reason'] = b2b_data.serves_1000_customers_reason
                except Exception as e:
                    logger.warning(f"Could not parse B2B check for row {row_index}: {e}")
            
            if 'summary' in artifacts:
                 try:
                    with open(artifacts['summary'], 'r', encoding='utf-8') as f:
                        summary_data = WebsiteTextSummary.parse_obj(json.load(f))
                        report_row['description'] = summary_data.summary
                 except Exception as e:
                    logger.warning(f"Could not parse Summary for row {row_index}: {e}")

            if 'attributes' in artifacts:
                try:
                    with open(artifacts['attributes'], 'r', encoding='utf-8') as f:
                        attrs_data = DetailedCompanyAttributes.parse_obj(json.load(f))
                        report_row['Industry'] = attrs_data.industry or report_row.get('Industry')
                        report_row['Products/Services Offered'] = "; ".join(attrs_data.products_services_offered) if attrs_data.products_services_offered else ''
                        report_row['USP/Key Selling Points'] = "; ".join(attrs_data.usp_key_selling_points) if attrs_data.usp_key_selling_points else ''
                        report_row['Customer Target Segments'] = "; ".join(attrs_data.customer_target_segments) if attrs_data.customer_target_segments else ''
                except Exception as e:
                    logger.warning(f"Could not parse Attributes for row {row_index}: {e}")

            if 'sales_pitch' in artifacts:
                try:
                    with open(artifacts['sales_pitch'], 'r', encoding='utf-8') as f:
                        pitch_data = GoldenPartnerMatchOutput.parse_obj(json.load(f))
                        report_row['sales_pitch'] = pitch_data.phone_sales_line
                        report_row['matched_golden_partner'] = pitch_data.matched_partner_name
                        report_row['match_reasoning'] = "; ".join(pitch_data.match_rationale_features) if pitch_data.match_rationale_features else ""
                except Exception as e:
                    logger.warning(f"Could not parse Sales Pitch for row {row_index}: {e}")

        # Add failure information if it exists
        if row_index in failure_map:
            report_row['Regeneration_Status'] = 'Failed'
            report_row['Failure_Stage'] = failure_map[row_index].get('stage_of_failure')
            report_row['Failure_Reason'] = failure_map[row_index].get('error_reason')
        elif artifacts:
            report_row['Regeneration_Status'] = 'Success'
        else:
            # This is a missing row
            missing_rows_data.append({
                'Original Row Number': row_index + 2,
                'CompanyName': row.get('CompanyName'),
                'GivenURL': row.get('GivenURL'),
                'Reason': "No LLM artifacts or failure log entry found for this row."
            })
            continue # Don't add missing rows to the main report

        report_data.append(report_row)

    logger.info(f"Enrichment complete. Processed {len(report_data)} rows for the main report.")

    # --- 5. Generate Reports ---
    if report_data:
        logger.info(f"Generating main Sales Outreach Report...")
        final_df = pd.DataFrame(report_data)
        
        # Clean up by removing the original index column if it exists
        if 'original_index' in final_df.columns:
            final_df.drop(columns=['original_index'], inplace=True)

        csv_report_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_regenerated.csv")
        final_df.to_csv(csv_report_path, index=False, encoding='utf-8-sig')
        logger.info(f"Successfully generated CSV report: {csv_report_path}")
        
        excel_report_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_regenerated.xlsx")
        final_df.to_excel(excel_report_path, index=False)
        logger.info(f"Successfully generated Excel report: {excel_report_path}")
    else:
        logger.warning("No data available to generate a final report.")

    if missing_rows_data:
        logger.warning(f"Found {len(missing_rows_data)} missing rows. Generating Reconciliation Summary.")
        missing_df = pd.DataFrame(missing_rows_data)
        missing_report_path = os.path.join(run_output_dir, f"Reconciliation_Summary_{run_id}.csv")
        missing_df.to_csv(missing_report_path, index=False, encoding='utf-8-sig')
        logger.warning(f"Reconciliation Summary for missing rows generated: {missing_report_path}")
    else:
        logger.info("No missing rows found. Reconciliation successful.")

    logger.info(f"Report regeneration for run_id: {run_id} complete.")


def main():
    """
    Argument parsing and script entry point.
    """
    parser = argparse.ArgumentParser(description="Regenerate Sales Outreach Report from pipeline artifacts.")
    parser.add_argument("run_id", type=str, help="The unique run_id of the pipeline execution to regenerate the report for.")
    args = parser.parse_args()

    try:
        app_config = AppConfig()
        regenerate_report(args.run_id, app_config)
    except Exception as e:
        logger.error(f"An unexpected error occurred during script execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()