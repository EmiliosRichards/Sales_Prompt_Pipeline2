import argparse
import os
import pandas as pd
import json
import logging
import re
from typing import Dict, Any, Optional
from collections import defaultdict

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


def main():
    parser = argparse.ArgumentParser(description="Regenerate a final Sales Outreach Report from pipeline artifacts.")
    parser.add_argument("run_id", type=str, help="The unique run_id of the pipeline execution.")
    parser.add_argument("input_file", type=str, help="Path to the original input file (CSV or Excel).")
    args = parser.parse_args()
    regenerate_final_report(args.run_id, args.input_file)

if __name__ == "__main__":
    main()