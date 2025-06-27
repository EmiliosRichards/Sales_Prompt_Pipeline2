import pandas as pd
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.core.config import AppConfig
from src.core.schemas import CompanyContactDetails, ConsolidatedPhoneNumber, DomainExtractionBundle
from src.core.constants import EXCLUDED_TYPES_FOR_TOP_CONTACTS_REPORT
from src.phone_retrieval.utils.helpers import get_input_canonical_url
from src.phone_retrieval.data_handling.consolidator import get_canonical_base_url, generate_processed_contacts_report
from .report_generator import (
    write_row_attrition_report,
    write_canonical_domain_summary_report,
    generate_augmented_input_report as generate_augmented_input_report_util,
    save_detailed_report,
    save_summary_report,
    save_tertiary_report
)

logger = logging.getLogger(__name__)

def generate_all_reports(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    attrition_data_list: List[Dict[str, Any]],
    canonical_domain_journey_data: Dict[str, Any],
    input_to_canonical_map: Dict[str, Optional[str]], # Allow Optional[str]
    final_consolidated_data_by_true_base: Dict[str, DomainExtractionBundle], # Updated Type Hint
    true_base_scraper_status: Dict[str, str],
    original_phone_col_name_for_profile: Optional[str],
    new_phone_col_name_for_profile: Optional[str],
    original_input_file_path: str
) -> None:
    """
    Orchestrates the generation of all standard pipeline reports.
    Updates run_metrics with report generation statistics.
    """
    logger.info("Starting main report orchestration...")
    report_generation_start_time = time.time() # type: ignore

    # 1. Detailed Flattened Report
    _generate_detailed_report(df, app_config, run_id, run_output_dir, run_metrics, final_consolidated_data_by_true_base, true_base_scraper_status)

    # 2. Tertiary (Top Contacts) Report
    _generate_tertiary_report(df, app_config, run_id, run_output_dir, run_metrics, final_consolidated_data_by_true_base, true_base_scraper_status)

    # 3. Summary Report (Processed DataFrame)
    _generate_summary_report(df, app_config, run_id, run_output_dir, run_metrics, final_consolidated_data_by_true_base, true_base_scraper_status)
    
    # 4. Canonical Domain Summary Report
    canonical_domain_summary_rows_written = write_canonical_domain_summary_report(run_id, canonical_domain_journey_data, run_output_dir, logger)
    run_metrics["report_generation_stats"]["canonical_domain_summary_rows"] = canonical_domain_summary_rows_written

    # 5. Row Attrition Report
    num_attrition_rows = write_row_attrition_report(run_id, attrition_data_list, run_output_dir, canonical_domain_journey_data, input_to_canonical_map, logger)
    run_metrics["data_processing_stats"]["rows_in_attrition_report"] = num_attrition_rows # Note: this is a data_processing_stat

    # 6. Augmented Input Report
    if original_phone_col_name_for_profile or new_phone_col_name_for_profile:
        _generate_augmented_input_report(
            processed_df=df,
            app_config=app_config,
            run_id=run_id,
            run_output_dir=run_output_dir,
            run_metrics=run_metrics,
            original_input_file_path=original_input_file_path,
            original_phone_col_name=original_phone_col_name_for_profile,
            new_phone_col_name=new_phone_col_name_for_profile
        )
    else:
        logger.info("Skipping augmented report generation: No original or new phone column name defined in profile.")

    # 7. Final Processed Contacts Report (copy of tertiary)
    _generate_final_processed_contacts_report(app_config, run_id, run_output_dir)
    
    run_metrics["tasks"]["report_orchestration_duration_seconds"] = time.time() - report_generation_start_time # type: ignore
    logger.info("Main report orchestration finished.")


def _generate_detailed_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    final_consolidated_data_by_true_base: Dict[str, DomainExtractionBundle], # Updated Type Hint
    true_base_scraper_status: Dict[str, str]
):
    logger.info("Generating Detailed Report...")
    all_flattened_rows: List[Dict[str, Any]] = []
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, app_config.INPUT_COLUMN_PROFILES['default'])
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')

    for index_report, original_row_data_report in df.iterrows():
        company_name_report = str(original_row_data_report.get(company_name_col_key, f"Row_{index_report}"))
        canonical_url_report = original_row_data_report.get('CanonicalEntryURL') # This is true_base
        
        company_contact_details_report: Optional[CompanyContactDetails] = None
        domain_bundle_detailed = final_consolidated_data_by_true_base.get(canonical_url_report) if canonical_url_report else None
        if domain_bundle_detailed:
            company_contact_details_report = domain_bundle_detailed.company_contact_details
        
        scraper_status_for_true_base_detailed = true_base_scraper_status.get(str(canonical_url_report), "Unknown") if canonical_url_report else "Unknown_NoTrueBase"

        if scraper_status_for_true_base_detailed == "Success" and company_contact_details_report and company_contact_details_report.consolidated_numbers:
            for consolidated_number_item_report in company_contact_details_report.consolidated_numbers:
                aggregated_types = sorted(list(set(s.type for s in consolidated_number_item_report.sources if s.type)))
                aggregated_source_urls = sorted(list(set(s.original_full_url for s in consolidated_number_item_report.sources)))
                llm_type_str = ", ".join(aggregated_types) if aggregated_types else "Unknown"
                llm_source_url_str = ", ".join(aggregated_source_urls) if aggregated_source_urls else "N/A"
                new_flattened_row: Dict[str, Any] = {
                    'CompanyName': company_name_report, 'Number': consolidated_number_item_report.number,
                    'LLM_Type': llm_type_str, 'LLM_Classification': consolidated_number_item_report.classification,
                    'LLM_Source_URL': llm_source_url_str, 'ScrapingStatus': scraper_status_for_true_base_detailed,
                    'TargetCountryCodes': original_row_data_report.get('TargetCountryCodes'), 'RunID': run_id
                }
                all_flattened_rows.append(new_flattened_row)
    
    detailed_columns_order = ['CompanyName', 'Number', 'LLM_Type', 'LLM_Classification', 'LLM_Source_URL', 'ScrapingStatus', 'TargetCountryCodes', 'RunID']
    if all_flattened_rows:
        df_detailed_export = pd.DataFrame(all_flattened_rows)
        classification_sort_order = ['Primary', 'Secondary', 'Support', 'Low Relevance', 'Non-Business']
        df_detailed_export['LLM_Classification_Sort'] = pd.Categorical(df_detailed_export['LLM_Classification'], categories=classification_sort_order, ordered=True)
        df_detailed_export = df_detailed_export.sort_values(by=['CompanyName', 'LLM_Classification_Sort', 'Number'], na_position='last').drop(columns=['LLM_Classification_Sort'])
        for col in detailed_columns_order:
            if col not in df_detailed_export.columns: df_detailed_export[col] = None
        df_detailed_export = df_detailed_export[detailed_columns_order]
        detailed_rows_written = save_detailed_report(df_detailed_export, detailed_columns_order, run_id, run_output_dir, logger)
        run_metrics["report_generation_stats"]["detailed_report_rows"] = detailed_rows_written
    else:
        logger.info("No data for detailed flattened report.")
        run_metrics["report_generation_stats"]["detailed_report_rows"] = 0

def _generate_tertiary_report(
    df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    final_consolidated_data_by_true_base: Dict[str, DomainExtractionBundle], # Updated Type Hint
    true_base_scraper_status: Dict[str, str]
):
    logger.info("Generating Tertiary (Top Contacts) Report...")
    all_tertiary_rows: List[Dict[str, Any]] = []
    top_contacts_aggregation_map: Dict[str, Dict[str, Any]] = {}
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, app_config.INPUT_COLUMN_PROFILES['default'])
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')
    url_col_key = active_profile.get('GivenURL', 'GivenURL')

    for true_base_domain_key_agg, domain_bundle_agg in final_consolidated_data_by_true_base.items():
        # domain_bundle_agg is DomainExtractionBundle
        ccd_object_agg = domain_bundle_agg.company_contact_details
        homepage_context_agg = domain_bundle_agg.homepage_context

        # We need ccd_object_agg to proceed with contact-related info, but can still report summary info if it's None
        # For this report, if ccd_object_agg is None, we might not have company names or URLs from it.
        # However, the primary key is true_base_domain_key_agg.
        
        matching_input_rows = df[df['CanonicalEntryURL'].astype(str) == str(true_base_domain_key_agg)]
        
        unique_original_company_names_set = set()
        if not matching_input_rows.empty:
            unique_original_company_names_set.update(matching_input_rows[company_name_col_key].dropna().astype(str))
        elif ccd_object_agg and ccd_object_agg.company_name: # Fallback if no input rows match but CCD has a name
             unique_original_company_names_set.add(str(ccd_object_agg.company_name))
        
        # If still empty, use the domain itself or a placeholder
        if not unique_original_company_names_set and homepage_context_agg and homepage_context_agg.company_name:
            unique_original_company_names_set.add(homepage_context_agg.company_name) # Use summary company name if available
        elif not unique_original_company_names_set:
             unique_original_company_names_set.add(f"Data for {true_base_domain_key_agg}")


        unique_original_given_urls_set = set()
        if not matching_input_rows.empty:
            unique_original_given_urls_set.update(matching_input_rows[url_col_key].dropna().astype(str))
        elif ccd_object_agg: # Fallback if no input rows match but CCD has original URLs
            unique_original_given_urls_set.update(map(str, ccd_object_agg.original_input_urls))
        
        report_company_name = f"{true_base_domain_key_agg} - {' - '.join(sorted(list(unique_original_company_names_set)))}"
        report_given_urls = ", ".join(sorted(list(unique_original_given_urls_set)))
        
        top_contacts_aggregation_map[true_base_domain_key_agg] = {
            "report_company_name": report_company_name, "report_given_urls": report_given_urls,
            "canonical_entry_url": true_base_domain_key_agg,
            "scraper_status": true_base_scraper_status.get(true_base_domain_key_agg, "Unknown"),
            "contact_details": ccd_object_agg, # This can be None
            "homepage_context": homepage_context_agg # This can be None
        }
    
    for aggregated_entry in top_contacts_aggregation_map.values():
        ccd_for_report = aggregated_entry["contact_details"]
        homepage_ctx_for_report = aggregated_entry["homepage_context"]
        
        new_tertiary_row: Dict[str, Any] = {
            'CompanyName': aggregated_entry["report_company_name"],
            'GivenURL': aggregated_entry["report_given_urls"],
            'CanonicalEntryURL': aggregated_entry["canonical_entry_url"],
            'ScrapingStatus': aggregated_entry["scraper_status"],
            "Company Name (from Summary)": homepage_ctx_for_report.company_name if homepage_ctx_for_report and homepage_ctx_for_report.company_name else "N/A",
            "Summary Description": homepage_ctx_for_report.summary_description if homepage_ctx_for_report and homepage_ctx_for_report.summary_description else "N/A",
            "Industry (from Summary)": homepage_ctx_for_report.industry if homepage_ctx_for_report and homepage_ctx_for_report.industry else "N/A",
            'PhoneNumber_1': None, 'PhoneNumber_2': None, 'PhoneNumber_3': None,
            'SourceURL_1': None, 'SourceURL_2': None, 'SourceURL_3': None
        }
        
        if ccd_for_report and ccd_for_report.consolidated_numbers:
            eligible_numbers = [cn for cn in ccd_for_report.consolidated_numbers if cn.classification != 'Non-Business' and not EXCLUDED_TYPES_FOR_TOP_CONTACTS_REPORT.intersection({s.type for s in cn.sources if s.type})]
            for i, cn_item in enumerate(eligible_numbers[:3]):
                types_str = ", ".join(sorted(list(set(s.type for s in cn_item.sources if s.type))))
                # Ensure original_input_company_name is accessed safely
                companies_str_set = set()
                for source_item in cn_item.sources:
                    if source_item.original_input_company_name:
                        companies_str_set.add(source_item.original_input_company_name)
                companies_str = ", ".join(sorted(list(companies_str_set)))
                
                new_tertiary_row[f'PhoneNumber_{i+1}'] = f"{cn_item.number} ({types_str}) [{companies_str if companies_str else 'UnknownCompany'}]"
                new_tertiary_row[f'SourceURL_{i+1}'] = ", ".join(sorted(list(set(s.original_full_url for s in cn_item.sources))))
        
        # Add row if it has phone numbers OR if it has summary info (as per broader interpretation of reporting all domains)
        # The original condition was: if new_tertiary_row['PhoneNumber_1'] or new_tertiary_row['PhoneNumber_2'] or new_tertiary_row['PhoneNumber_3']:
        # For now, let's keep it focused on rows with phone numbers, but this could be revisited.
        # If no phone numbers, but we want to report summary info, the condition would change.
        # The task is about *adding columns*, implying rows are determined by existing logic.
        all_tertiary_rows.append(new_tertiary_row)
    
    tertiary_report_columns_order = [
        'CompanyName', 'GivenURL', 'CanonicalEntryURL', 'ScrapingStatus',
        "Company Name (from Summary)", "Summary Description", "Industry (from Summary)",
        'PhoneNumber_1', 'PhoneNumber_2', 'PhoneNumber_3',
        'SourceURL_1', 'SourceURL_2', 'SourceURL_3'
    ]
    if all_tertiary_rows:
        df_tertiary_report = pd.DataFrame(all_tertiary_rows)
        tertiary_rows_written = save_tertiary_report(df_tertiary_report, tertiary_report_columns_order, run_id, run_output_dir, app_config.tertiary_report_file_name_template, logger)
        run_metrics["report_generation_stats"]["tertiary_report_rows"] = tertiary_rows_written
    else:
        logger.info("No data for tertiary report.")
        run_metrics["report_generation_stats"]["tertiary_report_rows"] = 0

def _generate_summary_report(
    df: pd.DataFrame, # This is the main processed df
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    final_consolidated_data_by_true_base: Dict[str, DomainExtractionBundle], # Updated Type Hint
    true_base_scraper_status: Dict[str, str]
):
    logger.info("Generating Summary Report (from main processed DataFrame)...")
    df_summary_export = df.copy() # Start with a copy of the main df
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, app_config.INPUT_COLUMN_PROFILES['default'])
    url_col_key = active_profile.get('GivenURL', 'GivenURL')


    # Logic to update 'Original_Number_Status' and 'Overall_VerificationStatus'
    # This was originally in main_pipeline.py, now adapted here for the summary report.
    for index, row_summary in df_summary_export.iterrows():
        original_norm_phone_summary = row_summary.get('NormalizedGivenPhoneNumber')
        final_reason_for_row = row_summary.get('Final_Row_Outcome_Reason', '')
        canonical_url_for_row = row_summary.get('CanonicalEntryURL')
        
        unique_consolidated_numbers_for_row: List[ConsolidatedPhoneNumber] = []
        ccd_optional: Optional[CompanyContactDetails] = None
        domain_bundle_summary = final_consolidated_data_by_true_base.get(canonical_url_for_row) if canonical_url_for_row else None
        
        if domain_bundle_summary and domain_bundle_summary.company_contact_details:
            ccd_optional = domain_bundle_summary.company_contact_details
            if ccd_optional and ccd_optional.consolidated_numbers: # Added None check for ccd_optional
                 unique_consolidated_numbers_for_row = ccd_optional.consolidated_numbers

        if original_norm_phone_summary and original_norm_phone_summary != "InvalidFormat":
            found_original = any(top_num.number == original_norm_phone_summary for top_num in unique_consolidated_numbers_for_row[:3]) # Check against top 3 for summary
            if found_original: df_summary_export.at[index, 'Original_Number_Status'] = 'Verified'
            elif unique_consolidated_numbers_for_row: df_summary_export.at[index, 'Original_Number_Status'] = 'Corrected'
            elif final_reason_for_row in ["LLM_Output_NoNumbersFound_AllAttempts", "LLM_Output_NumbersFound_NoneRelevant_AllAttempts"]:
                df_summary_export.at[index, 'Original_Number_Status'] = 'LLM_OutputEmpty_Or_NoRelevant_For_Canonical'
            elif final_reason_for_row.startswith("LLM_NoInput") or final_reason_for_row.startswith("LLM_Processing_Error") or \
                 final_reason_for_row.startswith("ScrapingFailed_Canonical") or final_reason_for_row == "Unknown_NoCanonicalURLDetermined":
                df_summary_export.at[index, 'Original_Number_Status'] = 'LLM_Not_Run_Or_NoOutput_For_Canonical'
            else: df_summary_export.at[index, 'Original_Number_Status'] = 'No Relevant Match Found by LLM'
        elif original_norm_phone_summary == "InvalidFormat": df_summary_export.at[index, 'Original_Number_Status'] = 'Original_InvalidFormat'
        else: df_summary_export.at[index, 'Original_Number_Status'] = 'Original_Not_Provided'

        overall_status = "Unverified"
        scraper_status_for_true_base_row = true_base_scraper_status.get(str(canonical_url_for_row), "Unknown") if canonical_url_for_row else "Unknown_NoTrueBase"
        if final_reason_for_row == "Contact_Successfully_Extracted": overall_status = "Verified_LLM_Match_Found"
        elif final_reason_for_row.startswith("ScrapingFailed_Canonical") or final_reason_for_row in ["Scraping_AllAttemptsFailed_Network", "Scraping_AllAttemptsFailed_AccessDenied", "Scraping_ContentNotFound_AllAttempts"]:
            overall_status = f"Unverified_Scrape_{scraper_status_for_true_base_row}"
        elif final_reason_for_row in ["LLM_Output_NumbersFound_NoneRelevant_AllAttempts", "LLM_Output_NoNumbersFound_AllAttempts"]:
            overall_status = "Unverified_LLM_NoRelevantNumbers"
        elif final_reason_for_row == "LLM_Processing_Error_AllAttempts": overall_status = "Error_LLM_Processing_For_Canonical"
        
        original_input_url_for_map = str(row_summary.get(url_col_key)) if pd.notna(row_summary.get(url_col_key)) else "None_GivenURL_Input"
        # Use get_canonical_base_url for this comparison as it was in main_pipeline
        normalized_original_input_base = get_canonical_base_url(original_input_url_for_map, log_level_for_non_domain_input=logging.INFO) if original_input_url_for_map != "None_GivenURL_Input" else None
        if canonical_url_for_row and normalized_original_input_base and normalized_original_input_base != canonical_url_for_row:
            if overall_status != "Verified_LLM_Match_Found" and not overall_status.startswith("RedirectedTo"):
                 overall_status = f"RedirectedTo[{canonical_url_for_row}]_" + overall_status
        df_summary_export.at[index, 'Overall_VerificationStatus'] = overall_status
        df_summary_export.at[index, 'ScrapingStatus_Canonical'] = scraper_status_for_true_base_row
        # LLM_Processing_Status_Canonical might need more refined logic based on pipeline_flow outputs
        df_summary_export.at[index, 'LLM_Processing_Status_Canonical'] = "Processed" if unique_consolidated_numbers_for_row else scraper_status_for_true_base_row


    summary_columns_order = ['CompanyName', 'GivenURL', 'GivenPhoneNumber', 'Original_Number_Status', 
                             'Top_Number_1', 'Top_Type_1', 'Description', 'ScrapingStatus_Canonical', 
                             'CanonicalEntryURL', 'Top_Number_1', 'Top_Type_1', 'Top_Number_2', 'Top_Type_2', 
                             'Top_Number_3', 'Top_Type_3', 'Top_SourceURL_1', 'Top_SourceURL_2', 'Top_SourceURL_3', 
                             'TargetCountryCodes', 'RunID', 'Final_Row_Outcome_Reason', 
                             'Determined_Fault_Category', 'Overall_VerificationStatus']
    unique_summary_cols_needed = list(dict.fromkeys(summary_columns_order)) # Remove duplicates while preserving order
    
    for col_name in unique_summary_cols_needed:
        if col_name not in df_summary_export.columns: 
            df_summary_export[col_name] = None # Ensure all columns exist
            
    df_summary_export['RunID'] = run_id # Ensure RunID is set for all rows
    
    # Select and reorder columns
    df_summary_export = df_summary_export[unique_summary_cols_needed]
    
    summary_rows_written = save_summary_report(df_summary_export, unique_summary_cols_needed, run_id, run_output_dir, app_config.output_excel_file_name_template, logger)
    run_metrics["report_generation_stats"]["summary_report_rows"] = summary_rows_written

def _generate_augmented_input_report(
    processed_df: pd.DataFrame,
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str,
    run_metrics: Dict[str, Any],
    original_input_file_path: str,
    original_phone_col_name: Optional[str],
    new_phone_col_name: Optional[str]
):
    logger.info("Attempting to generate Augmented Input Report...")
    try:
        original_input_df_for_aug: Optional[pd.DataFrame] = None
        if original_input_file_path.endswith('.csv'):
            original_input_df_for_aug = pd.read_csv(original_input_file_path)
        elif original_input_file_path.endswith(('.xls', '.xlsx')):
            original_input_df_for_aug = pd.read_excel(original_input_file_path, engine='openpyxl')
        
        if original_input_df_for_aug is not None and processed_df is not None:
            input_filename_path = Path(original_input_file_path)
            augmented_filename = f"{input_filename_path.stem}_augmented_{run_id}.xlsx"
            output_path_augmented_excel = Path(run_output_dir) / augmented_filename
            
            generate_augmented_input_report_util(
                original_input_df=original_input_df_for_aug,
                processed_df=processed_df,
                output_path_augmented_excel=str(output_path_augmented_excel),
                logger=logger,
                original_phone_col_name=original_phone_col_name,
                new_phone_col_name=new_phone_col_name
            )
        else:
            logger.warning("Skipping augmented report due to missing original or processed DataFrame.")

    except Exception as e_aug_report:
        logger.error(f"Error generating Augmented Input Report: {e_aug_report}", exc_info=True)
        run_metrics["errors_encountered"].append(f"Error generating Augmented Input Report: {str(e_aug_report)}")

def _generate_final_processed_contacts_report(
    app_config: AppConfig,
    run_id: str,
    run_output_dir: str
):
    logger.info("Generating 'Final Processed Contacts' Report (copy of tertiary)...")
    tertiary_output_filename = app_config.tertiary_report_file_name_template.format(run_id=run_id)
    tertiary_output_excel_path = os.path.join(run_output_dir, tertiary_output_filename)
    if os.path.exists(tertiary_output_excel_path):
        try:
            # generate_processed_contacts_report is from data_handling.consolidator
            # It copies the tertiary report to a specific name.
            generate_processed_contacts_report(tertiary_output_excel_path, app_config, run_id)
            logger.info(f"'Final Processed Contacts' report generated successfully from {tertiary_output_filename}.")
        except Exception as e_processed_report:
            logger.error(f"Error generating 'Final Processed Contacts' report: {e_processed_report}", exc_info=True)
    else:
        logger.warning(f"Tertiary report '{tertiary_output_excel_path}' not found. Skipping 'Final Processed Contacts' report generation.")

# Need to add time import for the orchestrator
import time