import logging
import time
from typing import Optional, Tuple, Any
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError

from ..core.config import AppConfig
from .interaction_handler import InteractionHandler

config_instance = AppConfig()
logger = logging.getLogger(__name__)

async def fetch_page_content(page: Page, url: str, input_row_id: Any, company_name_or_id: str) -> Tuple[Optional[str], Optional[int]]:
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigating to URL: {url}")
    try:
        start_time = time.time()
        response = await page.goto(url, timeout=config_instance.default_navigation_timeout * 2, wait_until='networkidle')
        navigation_time = time.time() - start_time
        logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigation took {navigation_time:.2f} seconds.")

        if response:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigation to {url} successful. Status: {response.status}")
            if response.ok:
                # After a successful navigation, handle interactions like cookie banners
                interaction_handler = InteractionHandler(page, config_instance)
                start_interaction_time = time.time()
                await interaction_handler.handle_interactions()
                interaction_time = time.time() - start_interaction_time
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Interaction handling took {interaction_time:.2f} seconds.")

                # The waiting logic has been removed to maximize speed.
                start_content_time = time.time()
                content = await page.content()
                content_time = time.time() - start_content_time
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Content fetching took {content_time:.2f} seconds.")
                logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Content fetched successfully for {url}.")
                return content, response.status
            else:
                logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] HTTP error for {url}: Status {response.status} {response.status_text}. No content fetched.")
                return None, response.status
        else:
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Failed to get a response object for {url}. Navigation might have failed silently.")
            return None, None
    except PlaywrightTimeoutError:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright navigation timeout for {url} after {config_instance.default_navigation_timeout / 1000}s.")
        return None, -1 # Specific code for timeout
    except PlaywrightError as e:
        error_message = str(e)
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright error during navigation to {url}: {error_message}")
        if "net::ERR_NAME_NOT_RESOLVED" in error_message: return None, -2 # DNS error
        elif "net::ERR_CONNECTION_REFUSED" in error_message: return None, -3 # Connection refused
        elif "net::ERR_ABORTED" in error_message: return None, -6 # Request aborted, often due to navigation elsewhere
        return None, -4 # Other Playwright error
    except Exception as e:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Unexpected error fetching page {url}: {type(e).__name__} - {e}", exc_info=True)
        return None, -5 # Generic exception