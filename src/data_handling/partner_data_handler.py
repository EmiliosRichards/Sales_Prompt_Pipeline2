"""
Handles loading and summarizing "Golden Partner" data from CSV files.

This module provides utilities to:
- Load a list of golden partners from a specified CSV file. Each partner is
  represented as a dictionary.
- Generate a concise textual summary for a single golden partner, extracting
  key information based on a predefined structure.
"""
import pandas as pd
import logging
import os # For the example usage block
import hashlib
import re
from typing import List, Dict, Any
from urllib.parse import urlparse

from ..matching.taxonomy import infer_match_taxonomy

# Configure logging
try:
    from ..core.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback basic logging configuration if core.logging_config is unavailable.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning(
        "Basic logging configured for partner_data_handler.py due to missing "
        "core.logging_config or its dependencies. This is a fallback."
    )


def load_golden_partners(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads golden partner data from a CSV or Excel file.

    Each row in the file is expected to represent a partner, with column headers
    as keys in the resulting dictionaries.

    Args:
        file_path (str): The path to the data file (CSV or Excel).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        represents a golden partner. Returns an empty list if the file is not
        found, is empty, or an error occurs during processing.
    """
    partners: List[Dict[str, Any]] = []
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            logger.error(f"Unsupported file type for golden partners: {file_path}. Please use CSV or Excel.")
            return partners
        
        # NOTE: This mapping is intentionally tolerant to small header variations between
        # different versions of the golden-partner spreadsheet.
        column_mapping = {
            'Company Name': 'name',
            # Some legacy sheets used a German "Beschreibung" column; current canonical sheet
            # stores narrative notes in "Source Document Section/Notes".
            'Beschreibung': 'description',
            'Industry': 'industry',
            # The canonical sheet currently uses "/" (not "&") in this header.
            'USP (Unique Selling Proposition) / Key Selling Points': 'usp',
            'USP (Unique Selling Proposition) & Key Selling Points': 'usp',
            'Products/Services Offered': 'services_products',
            'Customer Target Segments': 'target_audience',
            'Business Model': 'business_model',
            'Company Size Category': 'company_size',
            'Innovation Level Indicators': 'innovation_level',
            'Geographic Reach': 'geographic_reach',
            # This is the best place to store "why they are our partner" / proof points
            # without changing the downstream architecture.
            'Source Document Section/Notes': 'partnership_notes',
            'Website': 'website',
            'Email': 'email',
            'Phone': 'phone',
            'Avg Leads Per Day': 'avg_leads_per_day',
            'Rank (1-47)': 'rank'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Replace non-string values with empty strings
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else '')

        partners = [row.to_dict() for index, row in df.iterrows()]
        if partners:
            logger.info(f"Successfully loaded and processed {len(partners)} golden partners from {file_path}")
        else:
            logger.info(f"Loaded 0 golden partners from {file_path} (file might be empty or header-only).")
    except FileNotFoundError:
        logger.error(f"Golden partners file not found at {file_path}. Returning empty list.")
    except Exception as e:
        logger.error(f"An error occurred while loading golden partners from {file_path}: {e}", exc_info=True)
    return partners


def _clean_partner_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"n/a", "na", "none", "null", "nan"}:
        return ""
    return text


def _slugify_partner_name(value: Any) -> str:
    text = _clean_partner_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "unknown-partner"


def _normalize_partner_website(value: Any) -> str:
    raw = _clean_partner_text(value)
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"https://{raw}"
    try:
        parsed = urlparse(candidate)
        host = (parsed.netloc or parsed.path or "").strip().lower()
        host = host.removeprefix("www.")
        return host
    except Exception:
        return raw.lower().removeprefix("www.")


def _build_partner_id(partner_data: Dict[str, Any]) -> str:
    name_slug = _slugify_partner_name(partner_data.get("name"))
    stable_source = "|".join(
        [
            _clean_partner_text(partner_data.get("name")),
            _normalize_partner_website(partner_data.get("website")),
            _clean_partner_text(partner_data.get("rank")),
        ]
    )
    digest = hashlib.sha1(stable_source.encode("utf-8")).hexdigest()[:8]
    return f"{name_slug}-{digest}"


def _build_partner_aliases(partner_data: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []
    name = _clean_partner_text(partner_data.get("name"))
    website_host = _normalize_partner_website(partner_data.get("website"))
    if name:
        aliases.append(name)
        aliases.append(re.sub(r"\b(gmbh|ug|inc|llc|ag|kg|ltd|mbh)\b", "", name, flags=re.I).strip(" ,.-"))
    if website_host:
        aliases.append(website_host)
        aliases.append(website_host.split(".")[0])
    deduped: List[str] = []
    seen = set()
    for alias in aliases:
        cleaned = _clean_partner_text(alias)
        low = cleaned.lower()
        if cleaned and low not in seen:
            seen.add(low)
            deduped.append(cleaned)
    return deduped


def _build_partner_match_profile(partner_data: Dict[str, Any]) -> Dict[str, str]:
    return {
        "target_audience": _clean_partner_text(partner_data.get("target_audience")),
        "services_products": _clean_partner_text(partner_data.get("services_products")),
        "usp": _clean_partner_text(partner_data.get("usp")),
        "industry": _clean_partner_text(partner_data.get("industry")),
        "business_model": _clean_partner_text(partner_data.get("business_model")),
    }


def _build_partner_match_summary(match_profile: Dict[str, str]) -> str:
    ordered_fields = [
        ("industry", "Industry"),
        ("target_audience", "Target Audience"),
        ("services_products", "Services/Products"),
        ("usp", "USP"),
        ("business_model", "Business Model"),
    ]
    parts = [
        f"{label}: {match_profile[key]}"
        for key, label in ordered_fields
        if match_profile.get(key)
    ]
    return "; ".join(parts)


def _build_partner_match_taxonomy(match_profile: Dict[str, str]) -> Dict[str, List[str]]:
    return infer_match_taxonomy(
        match_profile.get("target_audience", ""),
        match_profile.get("services_products", ""),
        match_profile.get("usp", ""),
        match_profile.get("industry", ""),
        match_profile.get("business_model", ""),
    )


def summarize_golden_partner(partner_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a concise summary for a single golden partner.

    The summary includes key attributes such as name, description, industry,
    USP (Unique Selling Proposition), services/products, and target audience.
    Attributes not found or 'N/A' are omitted from the summary.

    Args:
        partner_data (Dict[str, Any]): A dictionary representing one golden
            partner. Expected keys include 'name', 'description', 'industry',
            'usp', 'services_products', 'target_audience'.

    Returns:
        Dict[str, Any]: A dictionary containing the partner's name, a
        semicolon-separated summary string, avg_leads_per_day, and rank.
    """
    # Define the order and display name for summary parts
    summary_fields = [
        ('industry', 'Industry'),
        ('usp', 'USP'),
        ('services_products', 'Services/Products'),
        ('target_audience', 'Target Audience'),
        ('business_model', 'Business Model'),
        ('company_size', 'Company Size'),
        ('innovation_level', 'Innovation Level'),
        ('geographic_reach', 'Geographic Reach'),
        ('partnership_notes', 'Partnership Notes'),
    ]

    summary_parts = []
    for key, display_name in summary_fields:
        value = partner_data.get(key)
        # Add to summary if value exists, is not None, and is not just whitespace
        if value and isinstance(value, str) and value.strip() and value.strip().lower() != 'n/a':
            summary_parts.append(f"{display_name}: {value.strip()}")

    summary_str = "; ".join(summary_parts) if summary_parts else "Partner data not available or insufficient for summary."
    partner_id = _build_partner_id(partner_data)
    partner_aliases = _build_partner_aliases(partner_data)
    match_profile = _build_partner_match_profile(partner_data)
    match_summary = _build_partner_match_summary(match_profile)
    match_taxonomy = _build_partner_match_taxonomy(match_profile)

    # Return both the compact summary string AND structured fields.
    # Downstream prompts receive JSON for each partner, so extra keys are safe and useful.
    return {
        "partner_id": partner_id,
        "name": partner_data.get("name", "Unknown Partner"),
        "partner_aliases": partner_aliases,
        "summary": summary_str,
        "match_candidate_summary": match_summary,
        "match_profile": match_profile,
        "match_taxonomy": match_taxonomy,
        "avg_leads_per_day": partner_data.get("avg_leads_per_day"),
        "rank": partner_data.get("rank"),
        # Structured fields (optional)
        "industry": partner_data.get("industry", ""),
        "usp": partner_data.get("usp", ""),
        "services_products": partner_data.get("services_products", ""),
        "target_audience": partner_data.get("target_audience", ""),
        "business_model": partner_data.get("business_model", ""),
        "company_size": partner_data.get("company_size", ""),
        "innovation_level": partner_data.get("innovation_level", ""),
        "geographic_reach": partner_data.get("geographic_reach", ""),
        "partnership_notes": partner_data.get("partnership_notes", ""),
        "website": partner_data.get("website", ""),
        "email": partner_data.get("email", ""),
        "phone": partner_data.get("phone", ""),
    }


if __name__ == '__main__':
    # This block provides an example of how to use the functions in this module.
    # It is intended for testing and demonstration purposes only.

    # Ensure logger for this example block uses the module's logger
    example_logger = logging.getLogger(__name__)
    example_logger.info("Executing example usage of partner_data_handler.py...")

    dummy_csv_path = 'dummy_golden_partners_for_handler_test.csv'
    try:
        example_logger.info(f"Creating dummy CSV for testing: {dummy_csv_path}")
        header = ['id', 'name', 'url', 'description', 'industry', 'target_audience', 'usp', 'services_products', 'extra_field']
        data = [
            ['1', 'Alpha Solutions', 'http://alpha.com', 'Leader in AI analytics', 'Tech', 'Enterprises', 'Innovative AI', 'AI Platform, Analytics Services', 'Extra1'],
            ['2', 'Beta Services', 'http://beta.com', 'Comprehensive cloud services', 'Cloud Services', 'SMEs', 'Scalability & Security', 'Cloud Hosting, Managed Services', 'Extra2'],
            ['3', 'Gamma Innovate', 'http://gamma.com', 'Develops medical devices', 'Healthcare', 'Hospitals', 'Cutting-edge tech', 'Medical Scanners, Diagnostic Tools', ''],
            ['4', 'Delta Retail', 'http://delta.com', '   ', 'Retail', 'Consumers', 'N/A', 'E-commerce Platform', 'ValueOnly'],
            ['5', None, None, None, None, None, None, None, None]
        ]
        pd.DataFrame(data, columns=header).to_csv(dummy_csv_path, index=False)

        example_logger.info(f"Attempting to load partners from: {dummy_csv_path}")
        partners_list = load_golden_partners(dummy_csv_path)

        if partners_list:
            example_logger.info(f"Loaded {len(partners_list)} partners:")
            for i, partner in enumerate(partners_list):
                example_logger.info(f"Partner {i+1}: {partner}")
                summary = summarize_golden_partner(partner)
                example_logger.info(f"Summary for Partner {i+1}: {summary}\n")
        else:
            example_logger.warning("No partners loaded or file not found during example run.")

        example_logger.info("Testing load_golden_partners with a non-existent file:")
        non_existent_partners = load_golden_partners('non_existent_file_for_test.csv')
        example_logger.info(f"Result for non-existent file (should be empty list): {non_existent_partners}")

        example_logger.info("Testing summarize_golden_partner with various direct inputs:")
        example_logger.info(f"Summary 1: {summarize_golden_partner({'name': 'Test Co', 'industry': 'Test Industry', 'description': 'A test company.'})}")
        example_logger.info(f"Summary 2 (empty dict): {summarize_golden_partner({})}")
        example_logger.info(f"Summary 3 (irrelevant keys): {summarize_golden_partner({'random_key': 'random_value'})}")
        example_logger.info(f"Summary 4 (N/A values): {summarize_golden_partner({'name': 'N/A Co', 'description': 'n/a'})}")

    except Exception as e_main:
        example_logger.error(f"Error during __main__ example execution: {e_main}", exc_info=True)
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_csv_path):
            try:
                os.remove(dummy_csv_path)
                example_logger.info(f"Cleaned up dummy CSV: {dummy_csv_path}")
            except OSError as e_remove:
                example_logger.error(f"Error removing dummy CSV {dummy_csv_path}: {e_remove}")
        example_logger.info("Example usage of partner_data_handler.py finished.")