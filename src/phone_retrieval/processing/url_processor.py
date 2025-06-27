import re
import socket
import logging
from urllib.parse import urlparse, quote
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def process_input_url(
    given_url_original: Optional[str],
    app_config_url_probing_tlds: list[str],
    row_identifier_for_log: str, # e.g., f"[RowID: {index}, Company: {company_name}]"
) -> Tuple[Optional[str], str]:
    """
    Processes the input URL by cleaning, performing TLD probing, and validating it.

    Args:
        given_url_original: The original URL string from the input.
        app_config_url_probing_tlds: List of TLDs to try for probing.
        row_identifier_for_log: A string identifier for logging purposes (e.g., row index and company name).

    Returns:
        A tuple containing:
            - The processed URL string if valid, otherwise None.
            - A status string: 'Valid' or 'InvalidURL'.
    """
    processed_url: Optional[str] = given_url_original
    status = "Valid"

    if given_url_original and isinstance(given_url_original, str):
        temp_url_stripped = given_url_original.strip()
        parsed_obj = urlparse(temp_url_stripped)
        current_scheme = parsed_obj.scheme
        current_netloc = parsed_obj.netloc
        current_path = parsed_obj.path
        current_params = parsed_obj.params
        current_query = parsed_obj.query
        current_fragment = parsed_obj.fragment

        if not current_scheme:
            logger.info(f"{row_identifier_for_log} URL '{temp_url_stripped}' is schemeless. Adding 'http://' and re-parsing.")
            temp_for_reparse_schemeless = "http://" + temp_url_stripped
            parsed_obj_schemed = urlparse(temp_for_reparse_schemeless)
            current_scheme = parsed_obj_schemed.scheme
            current_netloc = parsed_obj_schemed.netloc
            current_path = parsed_obj_schemed.path
            current_params = parsed_obj_schemed.params
            current_query = parsed_obj_schemed.query
            current_fragment = parsed_obj_schemed.fragment
            logger.debug(f"{row_identifier_for_log} After adding scheme: N='{current_netloc}', P='{current_path}'")

        if " " in current_netloc:
            logger.info(f"{row_identifier_for_log} Spaces found in domain part '{current_netloc}'. Removing them.")
            current_netloc = current_netloc.replace(" ", "")

        # Ensure path is quoted safely, query and fragment as well
        current_path = quote(current_path, safe='/%')
        current_query = quote(current_query, safe='=&/?+%') # Allow common query chars
        current_fragment = quote(current_fragment, safe='/?#%') # Allow common fragment chars


        # TLD Probing Logic
        if current_netloc and not re.search(r'\.[a-zA-Z]{2,}$', current_netloc) and not current_netloc.endswith('.'):
            is_ip_address = re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", current_netloc)
            if current_netloc.lower() != 'localhost' and not is_ip_address:
                logger.info(f"{row_identifier_for_log} Input domain '{current_netloc}' appears to lack a TLD. Attempting TLD probing...")
                successfully_probed_tld = False
                probed_netloc_base = current_netloc
                
                for tld_to_try in app_config_url_probing_tlds:
                    candidate_domain_to_probe = f"{probed_netloc_base}.{tld_to_try}"
                    logger.debug(f"{row_identifier_for_log} Probing TLD: Trying '{candidate_domain_to_probe}'")
                    try:
                        socket.gethostbyname(candidate_domain_to_probe)
                        current_netloc = candidate_domain_to_probe
                        logger.info(f"{row_identifier_for_log} TLD probe successful. Using '{current_netloc}' after trying '.{tld_to_try}'.")
                        successfully_probed_tld = True
                        break
                    except socket.gaierror:
                        logger.debug(f"{row_identifier_for_log} TLD probe DNS lookup failed for '{candidate_domain_to_probe}'.")
                    except Exception as sock_e:
                        logger.debug(f"{row_identifier_for_log} TLD probe for '{candidate_domain_to_probe}' failed with unexpected socket error: {sock_e}")
                
                if not successfully_probed_tld:
                    logger.warning(f"{row_identifier_for_log} TLD probing failed for base domain '{probed_netloc_base}'. Proceeding with '{current_netloc}'.")

        effective_path = current_path if current_path else ('/' if current_netloc else '')
        
        processed_url = urlparse('')._replace(
            scheme=current_scheme, netloc=current_netloc, path=effective_path,
            params=current_params, query=current_query, fragment=current_fragment
        ).geturl()
        
        if processed_url != given_url_original:
            logger.info(f"{row_identifier_for_log} URL: Original='{given_url_original}', Processed for Scraper='{processed_url}'")
        else:
            logger.info(f"{row_identifier_for_log} URL: Using original='{given_url_original}' (no changes after preprocessing).")

    if not processed_url or not isinstance(processed_url, str) or not processed_url.startswith(('http://', 'https://')):
        logger.warning(f"{row_identifier_for_log} Skipping due to invalid or missing URL after all processing: '{processed_url}' (Original input was: '{given_url_original}')")
        status = "InvalidURL"
        return None, status # Return None for URL if invalid

    return processed_url, status