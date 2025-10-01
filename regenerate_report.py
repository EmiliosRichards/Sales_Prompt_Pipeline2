import argparse
import os
import pandas as pd
import numpy as np
import json
import logging
import re
from typing import Dict, Any, Optional
from collections import defaultdict
from src.phone_retrieval.data_handling.normalizer import normalize_phone_number

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def extract_json_from_text(text_output: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extracts a JSON object from a larger text block."""
    if not text_output:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_output, re.DOTALL)
    json_str = match.group(1) if match else text_output
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.debug(f"Failed to decode JSON string: {json_str[:200]}...")
        return None

def process_artifacts(llm_context_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Scans all artifacts, parses them, and aggregates required data by row index.
    A row is considered complete if its artifacts contain both a 'phone_sales_line'
    and a 'matched_partner_name'.
    """
    artifact_data = defaultdict(dict)
    abs_path = os.path.abspath(llm_context_dir)
    logger.info(f"Scanning and processing all artifacts in: {abs_path}")

    if not os.path.isdir(abs_path):
        logger.warning(f"LLM context directory not found: {abs_path}")
        return {}

    for filename in os.listdir(abs_path):
        match = re.search(r"Row(\d+)_", filename)
        if not match:
            continue
        
        row_index = int(match.group(1))
        
        file_path = os.path.join(abs_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = extract_json_from_text(content)
            if not data:
                continue

            # Check for sales pitch and partner info within the content
            if 'phone_sales_line' in data:
                artifact_data[row_index]['sales_pitch'] = data['phone_sales_line']
            
            if 'matched_partner_name' in data:
                artifact_data[row_index]['matched_golden_partner'] = data['matched_partner_name']
                artifact_data[row_index]['match_reasoning'] = "; ".join(data.get('match_rationale_features', []))
                artifact_data[row_index]['Matched Partner Description'] = data.get('matched_partner_description', '')

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            
    return artifact_data

def regenerate_final_report(run_id: str, input_file: str):
    """
    Generates a focused sales report by enriching an input file with data
    found inside sales pitch and partner match artifacts.
    """
    logger.info(f"Starting final report regeneration for run_id: {run_id}")

    # --- 1. Path Setup ---
    project_root = os.path.abspath(os.path.dirname(__file__))
    run_output_dir = os.path.join(project_root, 'output_data', run_id)
    llm_context_dir = os.path.join(run_output_dir, 'llm_context')

    # --- 2. Process all artifacts and identify completed rows ---
    all_artifact_data = process_artifacts(llm_context_dir)
    
    completed_row_indices = {
        idx for idx, data in all_artifact_data.items()
        if 'sales_pitch' in data and 'matched_golden_partner' in data
    }
    
    logger.info(f"Found {len(completed_row_indices)} rows with both required data points (sales_pitch and matched_partner).")

    if not completed_row_indices:
        logger.warning("No completed rows found after processing artifacts. Exiting.")
        return

    # --- 3. Load and Filter Original Data ---
    try:
        logger.info(f"Loading original input data from: {input_file}")
        original_df = pd.read_csv(input_file, keep_default_na=False, low_memory=False) if input_file.endswith('.csv') else pd.read_excel(input_file, keep_default_na=False)
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}"); return
    except Exception as e:
        logger.error(f"Error loading input file: {e}"); return

    original_df['original_index'] = original_df.index
    filtered_df = original_df[original_df['original_index'].isin(completed_row_indices)].copy()
    logger.info(f"Filtered input data to {len(filtered_df)} rows.")

    if filtered_df.empty:
        logger.warning("After filtering, no matching rows from the input file were found. Exiting."); return

    # --- 4. Enrich Data from Processed Artifacts ---
    report_data = []
    for _, row in filtered_df.iterrows():
        report_row = row.to_dict()
        row_index = report_row['original_index']
        
        # Merge data from artifacts
        report_row.update(all_artifact_data.get(row_index, {}))
        
        # Set B2B flags since these rows are complete
        report_row['is_b2b'] = 'Yes'
        report_row['serves_1000'] = 'Yes'
        
        report_data.append(report_row)

    # --- 5. Generate Final Report ---
    if not report_data:
        logger.warning("No data was successfully enriched. No report will be generated."); return

    final_df = pd.DataFrame(report_data)
    
    # Define and order final columns
    final_columns = [
        'Company Name', 'URL', 'original_index', 'is_b2b', 'serves_1000',
        'sales_pitch', 'matched_golden_partner', 'match_reasoning',
        'Matched Partner Description'
    ]
    # Add original columns that are not in our final list
    original_cols_to_add = [col for col in original_df.columns if col not in final_columns]
    final_df = final_df.reindex(columns=original_cols_to_add + final_columns)

    csv_report_path = os.path.join(run_output_dir, f"SalesOutreachReport_{run_id}_regenerated_final.csv")
    final_df.to_csv(csv_report_path, index=False, encoding='utf-8-sig')
    logger.info(f"Successfully generated final report: {csv_report_path}")


def regenerate_from_live_csv(run_id: str, live_csv_path: str):
    """
    Regenerates a Sales Outreach Report from a 'live' CSV file.
    This function reads the live CSV, which contains a wide array of columns,
    and transforms it to match the format of the final Sales Outreach Report.
    It outputs both a CSV and an Excel file.
    Args:
        run_id (str): The unique run_id of the pipeline execution.
        live_csv_path (str): The path to the live CSV file.
    """
    logger.info(f"Starting report regeneration from live CSV for run_id: {run_id}")
    logger.info(f"Reading live CSV from: {live_csv_path}")

    try:
        live_df = pd.read_csv(live_csv_path, low_memory=False)
    except FileNotFoundError:
        logger.error(f"Live CSV file not found at: {live_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error reading live CSV file: {e}")
        return

    # --- Debugging Step ---
    if 'PhoneNumber_Status' in live_df.columns:
        unique_statuses = live_df['PhoneNumber_Status'].unique()
        logger.info(f"Unique 'PhoneNumber_Status' values found in live CSV: {unique_statuses}")
    else:
        logger.warning("'PhoneNumber_Status' column not found in the live CSV. Cannot process phone numbers.")
        return

    # --- Phone Number Processing ---
    # 1. Drop rows that have no usable phone number
    statuses_to_drop = ['No_Candidates_Found', 'No_Main_Line_Found']
    initial_rows = len(live_df)
    live_df = live_df[~live_df['PhoneNumber_Status'].isin(statuses_to_drop)].copy()
    logger.info(f"Dropped {initial_rows - len(live_df)} rows due to no usable phone number.")

    # 2. Use np.select for conditional phone number selection
    conditions = [
        live_df['PhoneNumber_Status'].isin(['Found_Primary', 'Found_Secondary', 'Fallback_To_Input']),
        live_df['PhoneNumber_Status'] == 'Provided_In_Input'
    ]
    choices = [
        live_df['found_number'],
        live_df['PhoneNumber']
    ]
    live_df['final_phone_number'] = np.select(conditions, choices, default=np.nan)

    # 3. Normalize the phone number format using the central function
    live_df['final_phone_number'] = live_df['final_phone_number'].apply(
        lambda x: normalize_phone_number(str(x), region='DE') if pd.notna(x) else ''
    )

    # --- Column Mapping and Finalization ---
    column_mapping = {
        'CompanyName': 'Company Name',
        'GivenURL': 'URL',
        'is_b2b': 'is_b2b',
        'serves_1000': 'serves_1000',
        'is_b2b_reason': 'is_b2b_reason',
        'serves_1000_reason': 'serves_1000_reason',
        'sales_pitch': 'sales_pitch',
        'description': 'description',
        'matched_golden_partner': 'matched_golden_partner',
        'match_reasoning': 'match_reasoning',
        'Industry': 'Industry',
        'Matched Partner Description': 'Matched Partner Description',
        'Avg Leads Per Day': 'Avg Leads Per Day',
        'Rank': 'Rank',
        'B2B Indicator': 'B2B Indicator',
        'Phone Outreach Suitability': 'Phone Outreach Suitability',
        'Target Group Size Assessment': 'Target Group Size Assessment',
        'Products/Services Offered': 'Products/Services Offered',
        'USP/Key Selling Points': 'USP/Key Selling Points',
        'Customer Target Segments': 'Customer Target Segments',
        'Business Model': 'Business Model',
        'Company Size Inferred': 'Company Size Inferred',
        'Innovation Level Indicators': 'Innovation Level Indicators',
        'Website Clarity Notes': 'Website Clarity Notes',
        'ScrapingStatus': 'ScrapeStatus'
    }

    final_df = live_df.rename(columns=column_mapping)
    final_df['found_number'] = final_df['final_phone_number']

    final_columns = [
        'Company Name', 'URL', 'found_number', 'is_b2b', 'serves_1000',
        'sales_pitch', 'matched_golden_partner', 'match_reasoning',
        'Industry', 'Matched Partner Description', 'description', 'ScrapeStatus',
        'is_b2b_reason', 'serves_1000_reason', 'Avg Leads Per Day', 'Rank',
        'B2B Indicator', 'Phone Outreach Suitability', 'Target Group Size Assessment',
        'Products/Services Offered', 'USP/Key Selling Points', 'Customer Target Segments',
        'Business Model', 'Company Size Inferred', 'Innovation Level Indicators',
        'Website Clarity Notes'
    ]
    
    for col in final_columns:
        if col not in final_df.columns:
            final_df[col] = ''

    final_df = final_df[final_columns]

    # --- Generate Output Files ---
    project_root = os.path.abspath(os.path.dirname(__file__))
    regen_output_dir = os.path.join(project_root, 'regenerated_reports', run_id)
    os.makedirs(regen_output_dir, exist_ok=True)

    csv_report_path = os.path.join(regen_output_dir, f"SalesOutreachReport_{run_id}_regenerated_from_live.csv")
    final_df.to_csv(csv_report_path, index=False, encoding='utf-8-sig')
    logger.info(f"Successfully generated regenerated CSV report: {csv_report_path}")

    excel_report_path = os.path.join(regen_output_dir, f"SalesOutreachReport_{run_id}_regenerated_from_live.xlsx")
    final_df.to_excel(excel_report_path, index=False)
    logger.info(f"Successfully generated regenerated Excel report: {excel_report_path}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate Sales Outreach Reports from pipeline artifacts or live CSVs.")
    parser.add_argument("input_path", type=str, help="Path to the input file or directory.")
    parser.add_argument("--mode", type=str, choices=['artifacts', 'live_csv'], default='artifacts', help="Regeneration mode.")
    parser.add_argument("--run_id", type=str, help="Optional: The specific run_id to process. If not provided in live_csv mode, it's inferred from the filename.")

    args = parser.parse_args()

    if args.mode == 'live_csv':
        if os.path.isdir(args.input_path):
            logger.info(f"Processing all CSV files in directory: {args.input_path}")
            for filename in os.listdir(args.input_path):
                if filename.endswith('_live.csv'):
                    file_path = os.path.join(args.input_path, filename)
                    # Extract run_id from filename like 'SalesOutreachReport_20250721_104631_28000-32000_live.csv'
                    match = re.search(r'SalesOutreachReport_(.+?)_live\.csv', filename)
                    if match:
                        run_id = match.group(1)
                        regenerate_from_live_csv(run_id, file_path)
                    else:
                        logger.warning(f"Could not extract run_id from filename: {filename}. Skipping.")
        elif os.path.isfile(args.input_path):
            run_id = args.run_id
            if not run_id:
                match = re.search(r'SalesOutreachReport_(.+?)_live\.csv', os.path.basename(args.input_path))
                if match:
                    run_id = match.group(1)
                else:
                    logger.error("run_id is required for single file processing in live_csv mode and could not be inferred.")
                    return
            regenerate_from_live_csv(run_id, args.input_path)
        else:
            logger.error(f"Input path is not a valid file or directory: {args.input_path}")

    elif args.mode == 'artifacts':
        if not args.run_id:
            logger.error("run_id is required for 'artifacts' mode.")
            return
        regenerate_final_report(args.run_id, args.input_path)

if __name__ == "__main__":
    main()