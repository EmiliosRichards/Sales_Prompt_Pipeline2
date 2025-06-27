"""
Global constants used throughout the Intelligent Prospect Analyzer application.

This module centralizes fixed values and configurations that are unlikely to change
frequently, promoting consistency and maintainability.
"""
from typing import Set, Dict

# Target country codes (international dialing codes) for phone number processing.
# Example: 49 (Germany), 41 (Switzerland), 43 (Austria).
TARGET_COUNTRY_CODES_INT: Set[int] = {49, 41, 43}

# Set of contact information types to be excluded from the "Top Contacts" report.
# This helps in focusing the report on more direct contact methods.
EXCLUDED_TYPES_FOR_TOP_CONTACTS_REPORT: Set[str] = {
    'Unknown', 'Fax', 'Mobile', 'Date', 'ID'
}

# Defines a mapping from specific, granular fault identifiers (keys) to
# broader, more general fault categories (values). This is used for
# summarizing and reporting errors encountered during pipeline processing.
FAULT_CATEGORY_MAP_DEFINITION: Dict[str, str] = {
    "Input_URL_Invalid": "Input Data Issue",
    "Input_URL_UnsupportedScheme": "Input Data Issue",
    "Scraping_AllAttemptsFailed_Network": "Website Issue",
    "Scraping_AllAttemptsFailed_AccessDenied": "Website Issue",
    "Scraping_ContentNotFound_AllAttempts": "Website Issue",
    "Scraping_Success_NoRelevantContentPagesFound": "Website Issue",
    "Canonical_Duplicate_SkippedProcessing": "Pipeline Logic/Configuration",
    "Canonical_NoRegexCandidatesFound": "Pipeline Logic/Configuration",
    "LLM_NoInput_NoRegexCandidates": "Pipeline Logic/Configuration",
    "LLM_Output_NoNumbersFound_AllAttempts": "LLM Issue",
    "LLM_Output_NumbersFound_NoneRelevant_AllAttempts": "LLM Issue",
    "LLM_Processing_Error_AllAttempts": "LLM Issue",
    "DataConsolidation_Error_ForRow": "Pipeline Error",
    "Pipeline_Skipped_MaxRedirects_ForInputURL": "Website Issue",
    "Pipeline_Skipped_PreviouslyFailedInput": "Pipeline Logic/Configuration",  # For future use when re-processing failed inputs
    "Unknown_Processing_Gap_NoContact": "Unknown"
}