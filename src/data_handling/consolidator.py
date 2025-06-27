"""
Consolidates contact data, particularly phone numbers, extracted from various sources.

This module provides functionalities to:
- Derive canonical base URLs from input URLs.
"""
import logging
from urllib.parse import urlparse, urlunparse
from typing import Optional


# Configure logging
try:
    # Attempt to use the project's centralized logging setup
    from ..core.logging_config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback to basic logging if centralized setup is not available
    # This might happen if the module is run in isolation or in a different context
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Basic logging configured for consolidator.py due to missing "
        "core.logging_config or its dependencies."
    )

# Constants for domain parsing
COMMON_TLDS = [
    'com', 'de', 'org', 'net', 'uk', 'io', 'co', 'eu', 'info', 'biz', 'at', 'ch'
]
SECONDARY_TLDS = ['co']  # For TLDs like .co.uk


def get_canonical_base_url(
    url_string: str,
    log_level_for_non_domain_input: int = logging.WARNING
) -> str | None:
    """
    Extracts the canonical base URL (scheme + netloc, with 'www.' removed).

    Examples:
        "http://www.example.com/path?query" -> "http://example.com"
        "example.com/path" -> "http://example.com"

    Args:
        url_string: The URL string to process.
        log_level_for_non_domain_input: Logging level to use if the input
            string does not appear to be a valid absolute URL or domain.
            Defaults to logging.WARNING.

    Returns:
        The canonical base URL as a string (e.g., "http://example.com"),
        or None if a base URL cannot be determined or an error occurs.
    """
    if not url_string or not isinstance(url_string, str):
        logger.warning(
            "get_canonical_base_url received empty or non-string input."
        )
        return None
    try:
        temp_url = url_string
        if not temp_url.startswith(('http://', 'https://')):
            # Check if it looks like a domain that might have had a scheme stripped
            # A simple check for a dot in the first part before any path.
            if '.' not in temp_url.split('/')[0]:
                logger.log(
                    log_level_for_non_domain_input,
                    f"Input '{url_string}' (when deriving base URL) doesn't "
                    f"appear to be a valid absolute URL or domain. This may "
                    f"be an original input value."
                )
                return None
            temp_url = 'http://' + temp_url  # Default to http if no scheme

        parsed = urlparse(temp_url)

        if not parsed.netloc:
            logger.log(
                log_level_for_non_domain_input,
                f"Could not determine network location (netloc) for input "
                f"'{url_string}' (parsed as '{temp_url}' when deriving base URL)."
            )
            return None

        netloc = parsed.netloc
        if netloc.startswith('www.'):
            netloc = netloc[4:]

        scheme = parsed.scheme if parsed.scheme else 'http'
        base_url = urlunparse((scheme, netloc, '', '', '', ''))
        return base_url
    except Exception as e:
        logger.error(
            f"Error parsing URL '{url_string}' to get base URL: {e}",
            exc_info=True
        )
        return None
