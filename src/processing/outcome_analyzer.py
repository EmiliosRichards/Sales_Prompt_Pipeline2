"""
Analyzes processing outcomes to determine success, failure reasons, and fault categories.

This module provides functions to determine the final status of processing for individual
input rows (URLs) and for entire canonical domains based on various data points collected
during the pipeline execution. These outcomes are crucial for reporting and understanding
pipeline performance and data quality.
"""
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple

from src.core.schemas import CompanyContactDetails, ConsolidatedPhoneNumber, PhoneNumberLLMOutput
from src.core.constants import FAULT_CATEGORY_MAP_DEFINITION


def determine_final_row_outcome_and_fault(
    index: Any,
    row_summary: pd.Series,
    df_status_snapshot: Dict[str, Any],
    company_contact_details_summary: Optional[CompanyContactDetails],
    unique_sorted_consolidated_numbers: List[ConsolidatedPhoneNumber],
    canonical_url_summary: Optional[str],
    true_base_scraper_status_map: Dict[str, str],
    true_base_to_pathful_map: Dict[str, List[str]],
    canonical_site_pathful_scraper_status: Dict[str, str],
    canonical_site_raw_llm_outputs: Dict[str, List[PhoneNumberLLMOutput]],
    canonical_site_regex_candidates_found_status: Dict[str, bool],
    canonical_site_llm_exception_details: Dict[str, str]
) -> Tuple[str, str]:
    """
    Determines the final outcome reason and fault category for an individual input row.

    This function evaluates various status flags and data collected throughout the
    processing of a single input URL to assign a definitive outcome (e.g.,
    "Contact_Successfully_Extracted", "Scraping_AllAttemptsFailed_Network") and
    a corresponding fault category (e.g., "N/A", "Website Issue").

    Args:
        index: The index of the row (e.g., from a DataFrame).
        row_summary: A pandas Series containing summary data for the row.
        df_status_snapshot: A dictionary holding the status snapshot for the DataFrame row.
        company_contact_details_summary: Optional processed contact details.
        unique_sorted_consolidated_numbers: List of unique, sorted phone numbers found.
        canonical_url_summary: The determined canonical URL for the input.
        true_base_scraper_status_map: Maps true base URLs to their scraper status.
        true_base_to_pathful_map: Maps true base URLs to a list of their pathful URLs.
        canonical_site_pathful_scraper_status: Maps pathful URLs of the canonical site
                                               to their scraper status.
        canonical_site_raw_llm_outputs: Maps pathful URLs of the canonical site
                                        to raw LLM outputs.
        canonical_site_regex_candidates_found_status: Maps canonical true base URLs
                                                      to boolean indicating if regex
                                                      candidates were found.
        canonical_site_llm_exception_details: Maps canonical true base URLs to LLM
                                              exception details.

    Returns:
        Tuple[str, str]: A tuple containing the outcome string and fault category string.
    """
    # Priority 1: Initial Input URL issues
    initial_row_scrape_status = df_status_snapshot.get('ScrapingStatus', '')
    if initial_row_scrape_status == 'InvalidURL':
        return "Input_URL_Invalid", FAULT_CATEGORY_MAP_DEFINITION["Input_URL_Invalid"]
    if initial_row_scrape_status == 'MaxRedirects_InputURL':
        return "Pipeline_Skipped_MaxRedirects_ForInputURL", \
               FAULT_CATEGORY_MAP_DEFINITION["Pipeline_Skipped_MaxRedirects_ForInputURL"]

    # Priority 2: Successful contact extraction
    if unique_sorted_consolidated_numbers:
        return "Contact_Successfully_Extracted", "N/A"

    # Priority 3: Issues with determining or scraping the canonical URL
    if not canonical_url_summary:
        if initial_row_scrape_status and \
           initial_row_scrape_status != "Success" and \
           initial_row_scrape_status != "Not_Run":
            return f"ScrapingFailure_InputURL_{initial_row_scrape_status}", \
                   FAULT_CATEGORY_MAP_DEFINITION.get(
                       f"Scraping_AllAttemptsFailed_Network", "Website Issue"
                   )
        return "Unknown_NoCanonicalURLDetermined", \
               FAULT_CATEGORY_MAP_DEFINITION["Unknown_Processing_Gap_NoContact"]

    scraper_status_for_true_base = true_base_scraper_status_map.get(
        canonical_url_summary, "Unknown"
    )

    if scraper_status_for_true_base != "Success":
        pathful_urls_for_canonical = true_base_to_pathful_map.get(canonical_url_summary, [])
        all_network_fail = True
        all_access_denied = True
        all_not_found = True
        if not pathful_urls_for_canonical:
            return "Scraping_NoPathfulURLs_ForCanonical", \
                   FAULT_CATEGORY_MAP_DEFINITION.get(
                       "Scraping_AllAttemptsFailed_Network", "Website Issue"
                   )

        for p_url in pathful_urls_for_canonical:
            p_status = canonical_site_pathful_scraper_status.get(p_url, "Unknown")
            if ("Timeout" not in p_status and "DNS" not in p_status and
                    "Network" not in p_status and "Unreachable" not in p_status):
                all_network_fail = False
            if ("403" not in p_status and "AccessDenied" not in p_status and
                    "Robots" not in p_status):
                all_access_denied = False
            if ("404" not in p_status and "NotFound" not in p_status):
                all_not_found = False
        
        if all_network_fail:
            return "Scraping_AllAttemptsFailed_Network", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_AllAttemptsFailed_Network"]
        if all_access_denied:
            return "Scraping_AllAttemptsFailed_AccessDenied", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_AllAttemptsFailed_AccessDenied"]
        if all_not_found:
            return "Scraping_ContentNotFound_AllAttempts", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_ContentNotFound_AllAttempts"]
        return f"ScrapingFailed_Canonical_{scraper_status_for_true_base}", \
               FAULT_CATEGORY_MAP_DEFINITION.get(
                   "Scraping_AllAttemptsFailed_Network", "Website Issue"
               )

    # Priority 4: Duplicate canonical URL already processed
    if initial_row_scrape_status and "Already_Processed" in initial_row_scrape_status:
        return "Canonical_Duplicate_SkippedProcessing", \
               FAULT_CATEGORY_MAP_DEFINITION["Canonical_Duplicate_SkippedProcessing"]

    # Priority 5: No regex candidates found on the canonical site
    if not canonical_site_regex_candidates_found_status.get(canonical_url_summary, False):
        return "Canonical_NoRegexCandidatesFound", \
               FAULT_CATEGORY_MAP_DEFINITION["Canonical_NoRegexCandidatesFound"]

    # Priority 6: LLM processing issues or no contact details from LLM
    if company_contact_details_summary is None:
        llm_error_detail = canonical_site_llm_exception_details.get(canonical_url_summary)
        if scraper_status_for_true_base == "Error_LLM_PromptMissing" or \
           (llm_error_detail and "PromptMissing" in llm_error_detail):
            return "LLM_Processing_Error_AllAttempts", \
                   FAULT_CATEGORY_MAP_DEFINITION["LLM_Processing_Error_AllAttempts"]
        if scraper_status_for_true_base == "Error_LLM_Processing" or llm_error_detail:
            return "LLM_Processing_Error_AllAttempts", \
                   FAULT_CATEGORY_MAP_DEFINITION["LLM_Processing_Error_AllAttempts"]
        return "LLM_NoInput_NoRegexCandidates", \
               FAULT_CATEGORY_MAP_DEFINITION["LLM_NoInput_NoRegexCandidates"]

    if not company_contact_details_summary.consolidated_numbers:
        all_raw_llm_empty_for_canonical = True
        pathful_urls_for_canonical = true_base_to_pathful_map.get(canonical_url_summary, [])
        if not pathful_urls_for_canonical:
            # If no pathful URLs, implies no LLM output could have been generated from them
            all_raw_llm_empty_for_canonical = True
        else:
            for p_url in pathful_urls_for_canonical:
                if canonical_site_raw_llm_outputs.get(p_url):
                    all_raw_llm_empty_for_canonical = False
                    break
        
        if all_raw_llm_empty_for_canonical:
            return "LLM_Output_NoNumbersFound_AllAttempts", \
                   FAULT_CATEGORY_MAP_DEFINITION["LLM_Output_NoNumbersFound_AllAttempts"]
        else:
            return "LLM_Output_NumbersFound_NoneRelevant_AllAttempts", \
                   FAULT_CATEGORY_MAP_DEFINITION["LLM_Output_NumbersFound_NoneRelevant_AllAttempts"]

    # Default fallback if no other condition met (should ideally not be reached)
    return "Unknown_Processing_Gap_NoContact", \
           FAULT_CATEGORY_MAP_DEFINITION["Unknown_Processing_Gap_NoContact"]


def determine_final_domain_outcome_and_fault(
    true_base_domain: str,
    domain_journey_entry: Dict[str, Any],
    true_base_scraper_status_map: Dict[str, str],
    true_base_to_pathful_map: Dict[str, List[str]],
    canonical_site_pathful_scraper_status: Dict[str, str],
    final_consolidated_data: Optional[CompanyContactDetails]
) -> Tuple[str, str]:
    """
    Determines the final outcome reason and fault category for a canonical domain.

    This function aggregates information related to all processing attempts for a given
    canonical domain to provide an overall outcome and fault category.

    Args:
        true_base_domain: The canonical true base domain string.
        domain_journey_entry: A dictionary containing aggregated journey data for the domain.
                              Expected keys include:
                              - "Overall_Scraper_Status_For_Domain" (str)
                              - "Regex_Candidates_Found_For_Any_Pathful" (bool)
                              - "Total_Pages_Scraped_For_Domain" (int)
                              - "LLM_Calls_Made_For_Domain" (bool)
                              - "LLM_Processing_Error_Encountered_For_Domain" (bool)
                              - "LLM_Total_Raw_Numbers_Extracted" (int)
        true_base_scraper_status_map: Maps true base URLs to their scraper status.
                                      (Used here for context, primary status from journey_entry)
        true_base_to_pathful_map: Maps true base URLs to a list of their pathful URLs.
        canonical_site_pathful_scraper_status: Maps pathful URLs of the canonical site
                                               to their scraper status.
        final_consolidated_data: Optional final consolidated contact details for the domain.

    Returns:
        Tuple[str, str]: A tuple containing the outcome string and fault category string.
    """
    # Priority 1: Successful contact extraction for the domain
    if final_consolidated_data and final_consolidated_data.consolidated_numbers:
        return "Contact_Successfully_Extracted_For_Domain", "N/A"

    overall_scraper_status_for_domain = domain_journey_entry.get(
        "Overall_Scraper_Status_For_Domain", "Unknown"
    )

    # Priority 2: Scraping failures for the domain
    if overall_scraper_status_for_domain != "Success":
        pathful_urls_for_this_domain = true_base_to_pathful_map.get(true_base_domain, [])
        all_network_fail = True
        all_access_denied = True
        all_not_found = True

        if not pathful_urls_for_this_domain:
            # If no pathful URLs were ever processed for this domain
            if "InvalidURL" in overall_scraper_status_for_domain or \
               "MaxRedirects" in overall_scraper_status_for_domain:
                return f"Domain_InputLikeFailure_{overall_scraper_status_for_domain}", \
                       FAULT_CATEGORY_MAP_DEFINITION.get("Input_URL_Invalid", "Input Data Issue")
            return "Scraping_NoPathfulURLs_Processed_ForDomain", \
                   FAULT_CATEGORY_MAP_DEFINITION.get(
                       "Scraping_AllAttemptsFailed_Network", "Website Issue"
                   )

        for p_url in pathful_urls_for_this_domain:
            p_status = canonical_site_pathful_scraper_status.get(p_url, "Unknown")
            if ("Timeout" not in p_status and "DNS" not in p_status and
                    "Network" not in p_status and "Unreachable" not in p_status):
                all_network_fail = False
            if ("403" not in p_status and "AccessDenied" not in p_status and
                    "Robots" not in p_status):
                all_access_denied = False
            if ("404" not in p_status and "NotFound" not in p_status):
                all_not_found = False
        
        if all_network_fail:
            return "Scraping_AllPathfulsFailed_Network_ForDomain", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_AllAttemptsFailed_Network"]
        if all_access_denied:
            return "Scraping_AllPathfulsFailed_AccessDenied_ForDomain", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_AllAttemptsFailed_AccessDenied"]
        if all_not_found:
            return "Scraping_AllPathfuls_ContentNotFound_ForDomain", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_ContentNotFound_AllAttempts"]
        
        return f"ScrapingFailed_Domain_{overall_scraper_status_for_domain}", \
               FAULT_CATEGORY_MAP_DEFINITION.get(
                   "Scraping_AllAttemptsFailed_Network", "Website Issue"
               )

    # Priority 3: No regex candidates found across any pages of the domain
    if not domain_journey_entry.get("Regex_Candidates_Found_For_Any_Pathful", False):
        if domain_journey_entry.get("Total_Pages_Scraped_For_Domain", 0) == 0:
            # Scraped successfully but found no relevant pages to process further
            return "Scraping_Success_NoPagesScraped_ForDomain", \
                   FAULT_CATEGORY_MAP_DEFINITION["Scraping_Success_NoRelevantContentPagesFound"]
        return "Domain_NoRegexCandidatesFound_OnAnyPage", \
               FAULT_CATEGORY_MAP_DEFINITION["Canonical_NoRegexCandidatesFound"]

    # Priority 4: LLM not called despite having regex candidates
    if not domain_journey_entry.get("LLM_Calls_Made_For_Domain", False):
        return "LLM_NotCalled_DespiteRegexCandidates_ForDomain", \
               FAULT_CATEGORY_MAP_DEFINITION["LLM_NoInput_NoRegexCandidates"]

    # Priority 5: LLM processing errors encountered for the domain
    if domain_journey_entry.get("LLM_Processing_Error_Encountered_For_Domain", False):
        return "LLM_Processing_Error_Encountered_For_Domain", \
               FAULT_CATEGORY_MAP_DEFINITION["LLM_Processing_Error_AllAttempts"]

    # Priority 6: LLM output issues (no numbers found or none relevant)
    if domain_journey_entry.get("LLM_Total_Raw_Numbers_Extracted", 0) == 0:
        return "LLM_Output_NoRawNumbersFound_ForDomain", \
               FAULT_CATEGORY_MAP_DEFINITION["LLM_Output_NoNumbersFound_AllAttempts"]
    else: # Numbers were extracted by LLM, but none made it to final consolidated list
        return "LLM_Output_RawNumbersFound_NoneConsolidated_ForDomain", \
               FAULT_CATEGORY_MAP_DEFINITION["LLM_Output_NumbersFound_NoneRelevant_AllAttempts"]

    # Default fallback if no other condition met
    return "Unknown_Domain_Processing_Gap_NoContact", \
           FAULT_CATEGORY_MAP_DEFINITION["Unknown_Processing_Gap_NoContact"]