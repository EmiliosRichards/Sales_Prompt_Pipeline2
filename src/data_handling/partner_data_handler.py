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
from typing import List, Dict, Any

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
        
        column_mapping = {
            'Company Name': 'name',
            'Beschreibung': 'description',
            'Industry': 'industry',
            'USP (Unique Selling Proposition) & Key Selling Points': 'usp',
            'Products/Services Offered': 'services_products',
            'Customer Target Segments': 'target_audience',
            'Business Model': 'business_model',
            'Company Size Category': 'company_size',
            'Innovation Level Indicators': 'innovation_level',
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
        ('innovation_level', 'Innovation Level')
    ]

    summary_parts = []
    for key, display_name in summary_fields:
        value = partner_data.get(key)
        # Add to summary if value exists, is not None, and is not just whitespace
        if value and isinstance(value, str) and value.strip() and value.strip().lower() != 'n/a':
            summary_parts.append(f"{display_name}: {value.strip()}")

    summary_str = "; ".join(summary_parts) if summary_parts else "Partner data not available or insufficient for summary."

    return {
        "name": partner_data.get("name", "Unknown Partner"),
        "summary": summary_str,
        "avg_leads_per_day": partner_data.get("avg_leads_per_day"),
        "rank": partner_data.get("rank")
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