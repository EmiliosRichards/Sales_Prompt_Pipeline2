import pandas as pd
import phonenumbers
import logging
from typing import Optional

# Configure logging
try:
    from src.core.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Basic logging configured for normalizer.py due to missing core.logging_config or its dependencies.")


def normalize_phone_number(phone_number_str: str, region: str | None = None) -> str | None:
    """
    Normalizes a given phone number string to E.164 format if valid.

    Uses the `python-phonenumbers` library to parse and validate the phone number.
    If a region is provided, it's used as a hint for parsing numbers without
    an international prefix.

    Args:
        phone_number_str (str): The phone number string to normalize.
        region (str | None, optional): A CLDR region code (e.g., "US", "DE")
            to assist in parsing. Defaults to None.

    Returns:
        str | None: The phone number in E.164 format (e.g., "+4930123456")
        if it's valid. Returns "InvalidFormat" if the number cannot be parsed
        or is considered invalid by the library. Returns None if the input
        `phone_number_str` is empty or not a string.

    Raises:
        Logs warnings for parsing errors or invalid numbers.
        Logs errors for unexpected exceptions during normalization.
    """
    if not phone_number_str or not isinstance(phone_number_str, str):
        return None
    try:
        parsed_number = phonenumbers.parse(phone_number_str, region)
        if phonenumbers.is_valid_number(parsed_number):
            return phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
        else:
            logger.info(f"Phone number '{phone_number_str}' (region: {region}) is not valid.")
            return "InvalidFormat"
    except phonenumbers.phonenumberutil.NumberParseException as e:
        logger.info(f"Could not parse phone number '{phone_number_str}' (region: {region}): {e}")
        return "InvalidFormat"
    except Exception as e:
        logger.error(f"Unexpected error normalizing phone number '{phone_number_str}': {e}")
        return "InvalidFormat"


def apply_phone_normalization(df: pd.DataFrame, phone_column: str = "GivenPhoneNumber",
                              normalized_column: str = "NormalizedGivenPhoneNumber",
                              region_column: str | None = None) -> pd.DataFrame:
    """
    Applies phone number normalization to a specified column in a DataFrame.

    Iterates over each row of the DataFrame, takes the phone number from
    `phone_column`, and attempts to normalize it using the
    `normalize_phone_number` function. The result is stored in the
    `normalized_column`. If `region_column` is provided and contains region
    codes (e.g., a list like ['DE', 'CH'] or a single string 'DE'),
    the first region is used as a hint for parsing.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        phone_column (str, optional): The name of the column containing
            the phone numbers to normalize. Defaults to "GivenPhoneNumber".
        normalized_column (str, optional): The name of the column where
            normalized phone numbers will be stored. Defaults to
            "NormalizedGivenPhoneNumber".
        region_column (str | None, optional): The name of the column
            containing region codes (e.g., 'DE', 'US') to aid parsing.
            If the column contains a list of codes, the first one is used.
            Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with the added `normalized_column`
        containing normalized phone numbers or error strings.

    Raises:
        Logs an error if the `phone_column` is not found in the DataFrame.
    """
    if phone_column not in df.columns:
        logger.error(f"Phone column '{phone_column}' not found in DataFrame.")
        df[normalized_column] = None
        return df

    def normalize_row(row: pd.Series) -> str | None:
        phone_number = row[phone_column]
        default_region: str | None = None
        if region_column and region_column in row and row[region_column]:
            if isinstance(row[region_column], list) and len(row[region_column]) > 0:
                default_region = row[region_column][0]
            elif isinstance(row[region_column], str):
                default_region = row[region_column]
        return normalize_phone_number(str(phone_number), region=default_region)

    df[normalized_column] = df.apply(normalize_row, axis=1) # type: ignore
    logger.info(f"Applied phone normalization to '{phone_column}', results in '{normalized_column}'.")
    return df