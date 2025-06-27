"""
Regex-based Phone Number Extraction Component
"""

import logging
from typing import List, Optional, Set, Dict

import phonenumbers
from phonenumbers import PhoneNumberFormat, PhoneNumberMatcher

logger = logging.getLogger(__name__)

DEFAULT_REGION = "US"

MIN_NSN_LENGTH = 7
MAX_REPEATING_DIGITS = 4
MAX_SEQUENTIAL_DIGITS = 4


def _is_placeholder_number(number_str: str) -> bool:
    """
    Checks if a given number string appears to be a common placeholder.

    This is a simplified check based on common placeholder patterns like
    sequences of zeros, "123456", or long repetitions of the same digit.

    Args:
        number_str (str): The number string (digits only ideally) to check.

    Returns:
        bool: True if the number string matches a placeholder pattern, False otherwise.
    """
    # Simplified placeholder check, can be expanded
    if number_str.startswith("000000") or number_str.startswith("123456"):
        return True
    if len(set(number_str)) == 1 and len(number_str) > 5: # e.g., 5555555
        return True
    return False

def _has_excessive_repeating_digits(number_str: str, threshold: int = MAX_REPEATING_DIGITS) -> bool:
    """
    Checks if a number string contains an excessive number of repeating identical digits.

    For example, if threshold is 4, it checks for "ddddd" (5 or more occurrences).

    Args:
        number_str (str): The number string (digits only) to check.
        threshold (int): The maximum allowed number of consecutive identical digits
                         before it's considered excessive. The check looks for
                         `threshold + 1` repetitions.

    Returns:
        bool: True if excessive repeating digits are found, False otherwise.
    """
    for digit in set(number_str):
        if number_str.count(digit * (threshold + 1)) > 0: # Checks for threshold + 1 occurrences
            return True
    return False

def _has_excessive_sequential_digits(number_str: str, threshold: int = MAX_SEQUENTIAL_DIGITS) -> bool:
    """
    Checks for excessive sequential digits (e.g., "12345" or "98765") in a number string.

    Args:
        number_str (str): The number string (must be all digits) to check.
        threshold (int): The length of the sequence to check for (e.g., if threshold is 4,
                         it looks for sequences of 4 like "1234" or "4321").
                         The check looks for sequences of `threshold + 1` digits.

    Returns:
        bool: True if a sequence of `threshold + 1` ascending or descending digits
              is found, False otherwise. Returns False if `number_str` is not all digits.
    """
    if not number_str.isdigit():
        return False
    # We are looking for sequences of length `threshold + 1`.
    # So, we need to iterate up to `len(number_str) - threshold`.
    # Example: if threshold is 4, we need 5 digits for a sequence (e.g., 12345).
    # The loop for i goes up to len - (threshold+1) to have enough room for threshold comparisons.
    # No, the original logic is: for i in range(len(number_str) - threshold):
    # This means if len=10, threshold=4, i goes up to 5 (0..5).
    # number_str[i+threshold] is the last element in a sequence of `threshold+1` items.
    # e.g. i=0, threshold=4: checks number_str[0]..number_str[4] (5 digits)
    
    # Corrected loop length for checking sequences of `threshold + 1` digits.
    # A sequence of `threshold + 1` digits has `threshold` steps.
    if len(number_str) <= threshold: # Not enough digits for a sequence of threshold+1
        return False

    for i in range(len(number_str) - threshold):
        # Ascending: e.g., 12345 (threshold=4 means 5 digits)
        is_sequential_asc = True
        for j in range(threshold): # Check `threshold` steps
            if int(number_str[i+j+1]) - int(number_str[i+j]) != 1:
                is_sequential_asc = False
                break
        if is_sequential_asc:
            return True
        
        # Descending: e.g., 54321
        is_sequential_desc = True
        for j in range(threshold): # Check `threshold` steps
            if int(number_str[i+j]) - int(number_str[i+j+1]) != 1:
                is_sequential_desc = False
                break
        if is_sequential_desc:
            return True
    return False


def _validate_number_custom(number_str: str, nsn: str) -> bool:
    """
    Applies a set of custom validation rules to a phone number.

    This includes checking if the number is a placeholder, if its National
    Significant Number (NSN) is too short, or if the NSN contains excessive
    repeating or sequential digits.

    Args:
        number_str (str): The raw matched string of the phone number.
        nsn (str): The National Significant Number (digits only, no country code).

    Returns:
        bool: True if the number passes all custom validation checks, False otherwise.
    """
    if _is_placeholder_number(number_str) or _is_placeholder_number(nsn): # Check both raw and nsn
        logger.debug(f"Number '{number_str}' (nsn: {nsn}) identified as placeholder.")
        return False
    if len(nsn) < MIN_NSN_LENGTH:
        logger.debug(f"NSN '{nsn}' for number '{number_str}' is too short (min: {MIN_NSN_LENGTH}).")
        return False
    if _has_excessive_repeating_digits(nsn): # threshold is MAX_REPEATING_DIGITS (default 4)
        logger.debug(f"NSN '{nsn}' for number '{number_str}' has excessive repeating digits.")
        return False
    if _has_excessive_sequential_digits(nsn): # threshold is MAX_SEQUENTIAL_DIGITS (default 4)
        logger.debug(f"NSN '{nsn}' for number '{number_str}' has excessive sequential digits.")
        return False
    return True


def _get_snippet(text_content: str, match_start: int, match_end: int, window_chars: int = 150) -> str:
    """
    Extracts a text snippet around a specific character match.
    The window_chars determines how many characters to include *on each side* of the match.

    Args:
        text_content (str): The full text content.
        match_start (int): The start character offset of the match.
        match_end (int): The end character offset of the match.
        window_chars (int): Number of characters to include before and after the match.

    Returns:
        str: The extracted text snippet.
    """
    snippet_start = max(0, match_start - window_chars)
    snippet_end = min(len(text_content), match_end + window_chars)
    
    # Ensure snippet doesn't break words unnecessarily if possible, by expanding to nearest space
    # This is a simple heuristic
    if snippet_start > 0:
        space_before = text_content.rfind(' ', 0, snippet_start)
        if space_before != -1:
            snippet_start = space_before + 1
            
    if snippet_end < len(text_content):
        space_after = text_content.find(' ', snippet_end)
        if space_after != -1:
            snippet_end = space_after
            
    return text_content[snippet_start:snippet_end].strip()


def extract_numbers_with_snippets_from_text(
    text_content: str,
    source_url: str,
    original_input_company_name: str, # Added
    target_country_codes: Optional[List[str]] = None,
    snippet_window_chars: int = 300 # Default to 300 chars (150 on each side)
) -> List[Dict[str, str]]:
    """
    Extracts phone numbers from text content, along with contextual snippets, source URL,
    and the original input company name. Snippets are character-based.

    Args:
        text_content (str): The text content to search within.
        source_url (str): The URL from which this text content was obtained.
        original_input_company_name (str): The company name from the original input row.
        target_country_codes (Optional[List[str]]): Hints for parsing non-international numbers.
        snippet_window_chars (int): Total characters for the snippet (half before, half after match).

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each with "number", "snippet",
                                "source_url", and "original_input_company_name".
    """
    if not text_content.strip():
        logger.info(f"Text content from {source_url} is empty or contains only whitespace.")
        return []

    results: List[Dict[str, str]] = []
    
    # Determine default parse region
    default_parse_region = DEFAULT_REGION
    if target_country_codes and target_country_codes[0] and len(target_country_codes[0]) == 2:
        default_parse_region = target_country_codes[0].upper()
    
    logger.info(f"Starting regex extraction for {source_url} with region hint {default_parse_region}, snippet window: {snippet_window_chars} chars.")

    try:
        for match in PhoneNumberMatcher(text_content, region=default_parse_region):
            number_obj = match.number
            
            if not phonenumbers.is_valid_number(number_obj):
                logger.debug(f"Number '{phonenumbers.format_number(number_obj, PhoneNumberFormat.E164)}' (raw: {match.raw_string}) from {source_url} is invalid by basic check.")
                continue

            nsn = str(number_obj.national_number)
            if not _validate_number_custom(match.raw_string, nsn):
                logger.debug(f"Custom validation failed for '{match.raw_string}' (nsn: {nsn}) from {source_url}.")
                continue
                
            e164_number = phonenumbers.format_number(number_obj, PhoneNumberFormat.E164)
            if not e164_number:
                continue
            
            # Calculate window for character-based snippet (half before, half after)
            # Ensure window_chars is even for simplicity or adjust as needed
            half_window = snippet_window_chars // 2
            snippet = _get_snippet(text_content, match.start, match.end, half_window)
            
            results.append({
                "candidate_number": e164_number,
                "snippet": snippet,
                "source_url": source_url,
                "original_input_company_name": original_input_company_name
            })
            logger.debug(f"Extracted for {source_url} (Orig Comp: {original_input_company_name}): {e164_number}, Snippet around chars {match.start}-{match.end}")

    except Exception as e:
        logger.error(f"Error during phone number matching/snippet extraction for {source_url}: {e}", exc_info=True)

    logger.info(f"Found {len(results)} candidate numbers with snippets from {source_url}.")
    return results


def extract_phone_numbers_from_file( # This function is mostly for testing/standalone use
    file_path: str,
    original_input_company_name_for_file: str = "FromFile", # Added default for standalone use
    target_country_codes: Optional[List[str]] = None,
    snippet_window_chars: int = 300
) -> List[Dict[str, str]]:
    """
    Extracts phone numbers with context snippets from a text file.
    (Primarily for testing/standalone use of this component)

    Reads text content from the file and uses `extract_numbers_with_snippets_from_text`
    to get numbers, their surrounding text snippets, the source URL (file_path),
    and an original company name.

    Args:
        file_path (str): Path to the text file.
        original_input_company_name_for_file (str): Company name to associate with this file.
        target_country_codes (Optional[List[str]]): Region hints for parsing.
        snippet_window_chars (int): Total characters for the snippet.

    Returns:
        List[Dict[str, str]]: List of dicts, each with "number", "snippet", "source_url", "original_input_company_name".
    """
    if not file_path:
        logger.warning("File path is empty, cannot extract phone numbers.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

    # When called this way, the 'source_url' for the snippets will be the file_path.
    # The main pipeline will call extract_numbers_with_snippets_from_text directly
    # with the true source URL of the scraped page.
    return extract_numbers_with_snippets_from_text(
        text_content=text_content,
        source_url=file_path,
        original_input_company_name=original_input_company_name_for_file, # Pass it through
        target_country_codes=target_country_codes,
        snippet_window_chars=snippet_window_chars
    )

# TODO: [FutureEnhancement] The __main__ block below was for direct script execution and testing
# of the regex-based phone number extraction. It includes creating a dummy test file,
# running extraction with different region hints, and cleaning up.
# Commented out as it's not part of the main pipeline execution.
# It can be uncommented for debugging or standalone testing of this component.
# if __name__ == '__main__':
#     # Basic test
#     logging.basicConfig(level=logging.DEBUG)
#     # Create a dummy file for testing
#     dummy_file_path = "dummy_phone_test.txt"
#     with open(dummy_file_path, "w") as f:
#         f.write("Contact us at +1-555-123-4567 or (202) 555-0123. \n")
#         f.write("Invalid: 12345. Placeholder: 0000000000. \n")
#         f.write("UK number: 07911 123456. German: 0171 2345678.\n")
#         f.write("Repeating: 1111111111, Sequential: 123456789.\n")
#         f.write("A valid French number: +33 1 23 45 67 89.\n")
#         f.write("Another US: 555.555.0199\n")
#
#
#     print(f"--- Testing with {dummy_file_path}, default region US ---")
#     numbers_us_default = extract_phone_numbers_from_file(dummy_file_path)
#     print(f"Extracted (US default): {numbers_us_default}")
#
#     print(f"\n--- Testing with {dummy_file_path}, target_country_codes ['GB'] ---")
#     numbers_gb_hint = extract_phone_numbers_from_file(dummy_file_path, target_country_codes=["GB"])
#     print(f"Extracted (GB hint): {numbers_gb_hint}")
#
#     print(f"\n--- Testing with {dummy_file_path}, target_country_codes ['DE'] ---")
#     numbers_de_hint = extract_phone_numbers_from_file(dummy_file_path, target_country_codes=["DE"])
#     print(f"Extracted (DE hint): {numbers_de_hint}")
#
#     # Clean up dummy file
#     import os
#     os.remove(dummy_file_path)