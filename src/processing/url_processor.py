"""
Processes and validates input URLs for the contact pipeline.

This module provides functionality to clean, normalize, and validate URLs
obtained from input data. Key operations include:
- Stripping leading/trailing whitespace.
- Ensuring a scheme (e.g., "http", "https") is present.
- Removing spaces from the domain part.
- Safely quoting URL path, query, and fragment components.
- Performing Top-Level Domain (TLD) probing for domains that appear to lack one,
  by attempting DNS resolution with common TLDs.
- Final validation to ensure the URL is well-formed and has a recognized scheme.
"""
import re
import socket
import logging
from urllib.parse import urlparse, quote, ParseResult
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def process_input_url(
    given_url_original: Optional[str],
    app_config_url_probing_tlds: List[str],
    row_identifier_for_log: str,
) -> Tuple[Optional[str], str]:
    """
    Processes an input URL by cleaning, performing TLD probing, and validating it.

    The function attempts to normalize the URL by adding a scheme if missing,
    removing spaces from the netloc, and quoting path/query/fragment.
    If the domain appears to lack a TLD (and is not 'localhost' or an IP address),
    it tries appending common TLDs from `app_config_url_probing_tlds` and
    checks for DNS resolution.

    Args:
        given_url_original: The original URL string from the input data.
        app_config_url_probing_tlds: A list of TLD strings (e.g., ["com", "org"])
                                     to try if the input URL seems to lack a TLD.
        row_identifier_for_log: A string identifier for logging, typically including
                                row index and company name, to contextualize log messages.
                                Example: "[RowID: 123, Company: ExampleCorp]"

    Returns:
        A tuple containing:
            - The processed and validated URL string if successful, otherwise None.
            - A status string: "Valid" if the URL is processed successfully,
              or "InvalidURL" if it's deemed invalid after processing.
    """
    processed_url: Optional[str] = given_url_original
    status: str = "Valid"

    if not given_url_original or not isinstance(given_url_original, str):
        logger.warning(
            f"{row_identifier_for_log} Input URL is missing or not a string: '{given_url_original}'"
        )
        return None, "InvalidURL"

    temp_url_stripped: str = given_url_original.strip()
    if not temp_url_stripped:
        logger.warning(
            f"{row_identifier_for_log} Input URL is empty after stripping: '{given_url_original}'"
        )
        return None, "InvalidURL"

    parsed_obj: ParseResult = urlparse(temp_url_stripped)
    current_scheme: str = parsed_obj.scheme
    current_netloc: str = parsed_obj.netloc
    current_path: str = parsed_obj.path
    current_params: str = parsed_obj.params
    current_query: str = parsed_obj.query
    current_fragment: str = parsed_obj.fragment

    # Ensure a scheme is present
    if not current_scheme:
        logger.info(
            f"{row_identifier_for_log} URL '{temp_url_stripped}' is schemeless. "
            f"Adding 'http://' and re-parsing."
        )
        temp_for_reparse_schemeless: str = "http://" + temp_url_stripped
        parsed_obj_schemed: ParseResult = urlparse(temp_for_reparse_schemeless)
        current_scheme = parsed_obj_schemed.scheme
        current_netloc = parsed_obj_schemed.netloc
        current_path = parsed_obj_schemed.path
        current_params = parsed_obj_schemed.params
        current_query = parsed_obj_schemed.query
        current_fragment = parsed_obj_schemed.fragment
        logger.debug(
            f"{row_identifier_for_log} After adding scheme: Netloc='{current_netloc}', Path='{current_path}'"
        )

    # Clean netloc (domain part)
    if " " in current_netloc:
        logger.info(
            f"{row_identifier_for_log} Spaces found in domain part '{current_netloc}'. Removing them."
        )
        current_netloc = current_netloc.replace(" ", "")

    # Safely quote URL components
    current_path = quote(current_path, safe='/%')
    current_query = quote(current_query, safe='=&/?+%')  # Allow common query characters
    current_fragment = quote(current_fragment, safe='/?#%')  # Allow common fragment characters

    # TLD Probing Logic for domains that seem to lack a TLD
    # (e.g., "example" instead of "example.com")
    # Skips if it's 'localhost', an IP address, or already has a TLD-like pattern.
    if current_netloc and \
       not re.search(r'\.[a-zA-Z]{2,}$', current_netloc) and \
       not current_netloc.endswith('.'):
        is_ip_address = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", current_netloc)
        if current_netloc.lower() != 'localhost' and not is_ip_address:
            logger.info(
                f"{row_identifier_for_log} Domain '{current_netloc}' appears to lack a TLD. "
                f"Attempting TLD probing with {app_config_url_probing_tlds}..."
            )
            successfully_probed_tld: bool = False
            probed_netloc_base: str = current_netloc
            
            for tld_to_try in app_config_url_probing_tlds:
                candidate_domain_to_probe: str = f"{probed_netloc_base}.{tld_to_try}"
                logger.debug(f"{row_identifier_for_log} Probing: Trying '{candidate_domain_to_probe}'")
                try:
                    socket.gethostbyname(candidate_domain_to_probe) # Attempt DNS resolution
                    current_netloc = candidate_domain_to_probe
                    logger.info(
                        f"{row_identifier_for_log} TLD probe successful. "
                        f"Using '{current_netloc}' after trying '.{tld_to_try}'."
                    )
                    successfully_probed_tld = True
                    break  # Stop probing on first success
                except socket.gaierror:
                    logger.debug(
                        f"{row_identifier_for_log} TLD probe DNS lookup failed for '{candidate_domain_to_probe}'."
                    )
                except Exception as sock_e: # Catch other potential socket errors
                    logger.warning(
                        f"{row_identifier_for_log} TLD probe for '{candidate_domain_to_probe}' "
                        f"failed with unexpected socket error: {sock_e}"
                    )
            
            if not successfully_probed_tld:
                logger.warning(
                    f"{row_identifier_for_log} TLD probing failed for base domain '{probed_netloc_base}'. "
                    f"Proceeding with original/schemed netloc: '{current_netloc}'."
                )

    # Ensure path is at least '/' if netloc is present, otherwise empty
    effective_path: str = current_path if current_path else ('/' if current_netloc else '')
    
    # Reconstruct the URL from processed components
    processed_url = urlparse('')._replace(
        scheme=current_scheme, netloc=current_netloc, path=effective_path,
        params=current_params, query=current_query, fragment=current_fragment
    ).geturl()
    
    if processed_url != given_url_original:
        logger.info(
            f"{row_identifier_for_log} URL processed: Original='{given_url_original}', "
            f"Processed='{processed_url}'"
        )
    else:
        logger.info(
            f"{row_identifier_for_log} URL: Using original='{given_url_original}' (no changes after preprocessing)."
        )

    # Final validation: must have a scheme and be a string
    if not processed_url or not isinstance(processed_url, str) or \
       not processed_url.startswith(('http://', 'https://')):
        logger.warning(
            f"{row_identifier_for_log} Final URL is invalid: '{processed_url}' "
            f"(Original input was: '{given_url_original}')"
        )
        status = "InvalidURL"
        return None, status

    return processed_url, status