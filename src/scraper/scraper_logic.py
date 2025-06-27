import asyncio
import os
import re
import logging
import random
import time
import hashlib # Added for hashing long filenames
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from bs4 import BeautifulSoup
from bs4.element import Tag # Added for type checking
import httpx # For asynchronous robots.txt checking
from urllib.robotparser import RobotFileParser
from typing import Set, Tuple, Optional, List, Dict, Any
import tldextract # Added for DNS fallback logic

# Assuming config.py is in src.core
from ..core.config import AppConfig
from ..core.logging_config import setup_logging # For main app setup, or test setup

# Import refactored functions
from .scraper_utils import normalize_url, get_safe_filename, extract_text_from_html, find_internal_links, _classify_page_type, validate_link_status
from .page_handler import fetch_page_content
from .caching import generate_cache_key, load_from_cache, save_to_cache
from .proxy_manager import ProxyManager

# Instantiate AppConfig for scraper_logic
config_instance = AppConfig()

# Setup logger for this module
logger = logging.getLogger(__name__)


async def is_allowed_by_robots(url: str, client: httpx.AsyncClient, input_row_id: Any, company_name_or_id: str) -> bool:
    if not config_instance.respect_robots_txt:
        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] robots.txt check is disabled.")
        return True
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Fetching robots.txt from: {robots_url}")
        response = await client.get(robots_url, timeout=config_instance.robots_txt_timeout_seconds, headers={'User-Agent': config_instance.robots_txt_user_agent})
        if response.status_code == 200:
            logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Successfully fetched robots.txt for {url}, status: {response.status_code}")
            rp.parse(response.text.splitlines())
        elif response.status_code == 404:
            logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] robots.txt not found at {robots_url} (status 404), assuming allowed.")
            return True
        else:
            logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Failed to fetch robots.txt from {robots_url}, status: {response.status_code}. Assuming allowed.")
            return True
    except httpx.RequestError as e:
        logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] httpx.RequestError fetching robots.txt from {robots_url}: {e}. Assuming allowed.")
        return True
    except Exception as e:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Unexpected error processing robots.txt for {robots_url}: {e}. Assuming allowed.", exc_info=True)
        return True
    allowed = rp.can_fetch(config_instance.robots_txt_user_agent, url)
    if not allowed:
        logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Scraping disallowed by robots.txt for URL: {url} (User-agent: {config_instance.robots_txt_user_agent})")
    else:
        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Scraping allowed by robots.txt for URL: {url}")
    return allowed



async def _perform_scrape_for_entry_point(
    entry_url_to_process: str,
    playwright_context, # Existing Playwright browser context
    http_client: httpx.AsyncClient, # HTTP client for validation
    output_dir_for_run: str,
    company_name_or_id: str,
    globally_processed_urls: Set[str], # Shared across all entry point attempts for the original given_url
    input_row_id: Any
) -> Tuple[List[Dict[str, str]], str, Optional[str], str]:
    """
    Core scraping logic for a single entry point URL and its children.
    This function contains the main `while urls_to_scrape` loop.
    Returns page details, status, canonical URL, and collected text for summary.
    """
    start_time_entry = time.time()
    # final_canonical_entry_url_for_this_attempt will be the canonical URL derived *from this specific entry_url_to_process*
    # if it's successfully scraped.
    final_canonical_entry_url_for_this_attempt: Optional[str] = None
    pages_scraped_this_entry_count = 0
    high_priority_pages_scraped_after_limit_entry = 0
    
    base_scraped_content_dir = os.path.join(output_dir_for_run, config_instance.scraped_content_subdir)
    cleaned_pages_storage_dir = base_scraped_content_dir # Removed "cleaned_pages_text" subdirectory
    # os.makedirs(cleaned_pages_storage_dir, exist_ok=True) # Already created in outer function

    company_safe_name = get_safe_filename(
        company_name_or_id,
        for_url=False,
        max_len=config_instance.filename_company_name_max_len
    )
    scraped_page_details_for_this_entry: List[Dict[str, str]] = []
    collected_texts_for_summary: List[str] = []
    priority_pages_collected_count = 0
    # Define priority page types for summary collection
    # These should ideally come from AppConfig if they need to be more dynamic
    # For now, using the types specified in the task.
    priority_page_types_for_summary = {"homepage", "about", "product_service"}

    # Queue for this specific entry point attempt
    urls_to_scrape_q: List[Tuple[str, int, int]] = [(entry_url_to_process, 0, 100)]
    # processed_urls_this_entry_call tracks URLs processed starting from *this* entry_url_to_process
    # to avoid loops within its own scraping process.
    processed_urls_this_entry_call: Set[str] = {entry_url_to_process}

    # Use the passed Playwright context to create a new page for this entry attempt
    page = await playwright_context.new_page()
    page.set_default_timeout(config_instance.default_page_timeout)
    
    entry_point_status_code: Optional[int] = None # To store status of the entry point itself

    try:
        while urls_to_scrape_q:
            urls_to_scrape_q.sort(key=lambda x: (-x[2], x[1]))
            current_url_from_queue, current_depth, current_score = urls_to_scrape_q.pop(0)
            
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Dequeuing URL: '{current_url_from_queue}' (Depth: {current_depth}, Score: {current_score}, Queue: {len(urls_to_scrape_q)})")

            # Domain page limit checks
            if config_instance.scraper_max_pages_per_domain > 0 and \
               pages_scraped_this_entry_count >= config_instance.scraper_max_pages_per_domain:
                if current_score < config_instance.scraper_score_threshold_for_limit_bypass or \
                   high_priority_pages_scraped_after_limit_entry >= config_instance.scraper_max_high_priority_pages_after_limit:
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Page limit reached, skipping '{current_url_from_queue}'.")
                    continue
                else: # Bypass for high priority
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Page limit reached, but processing high-priority '{current_url_from_queue}'.")


            html_content, status_code_fetch = await fetch_page_content(page, current_url_from_queue, input_row_id, company_name_or_id)
            
            if current_url_from_queue == entry_url_to_process and current_depth == 0: # This is the fetch for the entry point itself
                entry_point_status_code = status_code_fetch


            if html_content:
                pages_scraped_this_entry_count += 1
                if pages_scraped_this_entry_count > config_instance.scraper_max_pages_per_domain and \
                   current_score >= config_instance.scraper_score_threshold_for_limit_bypass:
                    high_priority_pages_scraped_after_limit_entry +=1

                final_landed_url_raw = page.url
                final_landed_url_normalized = normalize_url(final_landed_url_raw)
                
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Page fetch: Req='{current_url_from_queue}', LandedNorm='{final_landed_url_normalized}', Status: {status_code_fetch}")

                if not final_canonical_entry_url_for_this_attempt and current_depth == 0:
                    final_canonical_entry_url_for_this_attempt = final_landed_url_normalized
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Canonical URL for this entry attempt '{entry_url_to_process}' set to: '{final_canonical_entry_url_for_this_attempt}'")
                
                if final_landed_url_normalized in globally_processed_urls:
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Landed URL '{final_landed_url_normalized}' already globally processed. Skipping content save/link extraction.")
                    continue
                
                globally_processed_urls.add(final_landed_url_normalized)
                processed_urls_this_entry_call.add(final_landed_url_normalized)

                # ... (rest of content saving and link extraction logic from original function, lines 394-433)
                cleaned_text = extract_text_from_html(html_content)
                parsed_landed_url = urlparse(final_landed_url_normalized)
                source_domain = parsed_landed_url.netloc
                safe_source_name = re.sub(r'^www\.', '', source_domain)
                safe_source_name = re.sub(r'[^\w.-]', '_', safe_source_name)
                # Truncate safe_source_name to avoid overly long directory names
                safe_source_name_truncated_dir = safe_source_name[:50]
                source_specific_output_dir = os.path.join(cleaned_pages_storage_dir, safe_source_name_truncated_dir)
                os.makedirs(source_specific_output_dir, exist_ok=True)

                landed_url_safe_name = get_safe_filename(final_landed_url_normalized, for_url=True)
                cleaned_page_filename = f"{company_safe_name}__{landed_url_safe_name}_cleaned.txt"
                cleaned_page_filepath = os.path.join(source_specific_output_dir, cleaned_page_filename)
                
                try:
                    with open(cleaned_page_filepath, 'w', encoding='utf-8') as f_cleaned_page:
                        f_cleaned_page.write(cleaned_text)
                    page_type = _classify_page_type(final_landed_url_normalized, config_instance)
                    scraped_page_details_for_this_entry.append({
                        "filepath": cleaned_page_filepath,
                        "url": final_landed_url_normalized,
                        "page_type": page_type
                    })

                    # New logic: Collect text for summary
                    if page_type in priority_page_types_for_summary and \
                       priority_pages_collected_count < getattr(config_instance, 'SCRAPER_PAGES_FOR_SUMMARY_COUNT', 3): # Default to 3 if not set
                        collected_texts_for_summary.append(cleaned_text)
                        priority_pages_collected_count += 1
                        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Collected text from '{final_landed_url_normalized}' (type: {page_type}) for summary. Count: {priority_pages_collected_count}")

                except IOError as e:
                    logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] IOError saving cleaned text for '{final_landed_url_normalized}': {e}")

                if current_depth < config_instance.max_depth_internal_links:
                    newly_found_links_with_scores = find_internal_links(html_content, final_landed_url_normalized, input_row_id, company_name_or_id)
                    added_to_queue_count = 0
                    for link_url, link_score in newly_found_links_with_scores:
                        if link_url not in globally_processed_urls and link_url not in processed_urls_this_entry_call:
                            if await validate_link_status(link_url, http_client):
                                urls_to_scrape_q.append((link_url, current_depth + 1, link_score))
                                processed_urls_this_entry_call.add(link_url)
                                added_to_queue_count += 1
                            else:
                                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Skipping invalid link: {link_url}")
                    if added_to_queue_count > 0: urls_to_scrape_q.sort(key=lambda x: (-x[2], x[1]))
            else: # html_content is None
                logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Failed to fetch content from '{current_url_from_queue}'. Status code: {status_code_fetch}.")
                if current_url_from_queue == entry_url_to_process and current_depth == 0: # Critical failure on the entry point itself
                    status_map = {-1: "TimeoutError", -2: "DNSError", -3: "ConnectionRefused", -4: "PlaywrightError", -5: "GenericScrapeError", -6: "RequestAborted"}
                    http_status_report = "UnknownScrapeError"
                    if status_code_fetch is not None:
                        if status_code_fetch > 0: http_status_report = f"HTTPError_{status_code_fetch}"
                        elif status_code_fetch in status_map: http_status_report = status_map[status_code_fetch]
                        else: http_status_report = "UnknownScrapeErrorCode"
                    else: http_status_report = "NoStatusFromServer"
                    
                    logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Critical failure on entry point '{entry_url_to_process}'. Scraper status: {http_status_report}.")
                    await page.close()
                    return [], http_status_report, None, "" # No canonical URL, empty summary text
        
        # After loop for this entry point
        await page.close()

        final_summary_input_text = ""
        if collected_texts_for_summary:
            final_summary_input_text = " ".join(collected_texts_for_summary)
            max_chars = getattr(config_instance, 'LLM_MAX_INPUT_CHARS_FOR_SUMMARY', 40000) # Default to 40k
            if len(final_summary_input_text) > max_chars:
                final_summary_input_text = final_summary_input_text[:max_chars]
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Truncated collected summary text to {max_chars} characters.")
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Final collected summary text length: {len(final_summary_input_text)} chars.")
        else:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] No priority texts collected for summary.")


        if scraped_page_details_for_this_entry:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Successfully scraped {len(scraped_page_details_for_this_entry)} pages.")
            return scraped_page_details_for_this_entry, "Success", final_canonical_entry_url_for_this_attempt, final_summary_input_text
        else: # No pages scraped for this entry point
            final_status_for_this_entry = "NoContentScraped_Overall"
            if entry_point_status_code is not None: # If the entry point itself had a specific failure status
                 status_map = {-1: "TimeoutError", -2: "DNSError", -3: "ConnectionRefused", -4: "PlaywrightError", -5: "GenericScrapeError", -6: "RequestAborted"}
                 if entry_point_status_code > 0: final_status_for_this_entry = f"HTTPError_{entry_point_status_code}"
                 elif entry_point_status_code in status_map: final_status_for_this_entry = status_map[entry_point_status_code]
                 else: final_status_for_this_entry = "UnknownScrapeErrorCode"

            logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] No content scraped. Final status for this entry: {final_status_for_this_entry}")
            return [], final_status_for_this_entry, final_canonical_entry_url_for_this_attempt, final_summary_input_text # Return canonical if set, and whatever summary text was gathered (likely empty)
    except Exception as e_entry_scrape:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] General error during scraping process: {type(e_entry_scrape).__name__} - {e_entry_scrape}", exc_info=True)
        if page.is_closed() == False : await page.close()
        # Attempt to return any summary text collected before the error
        final_summary_input_text_on_error = ""
        if collected_texts_for_summary: # Check if this list was populated before error
            final_summary_input_text_on_error = " ".join(collected_texts_for_summary)
            max_chars = getattr(config_instance, 'LLM_MAX_INPUT_CHARS_FOR_SUMMARY', 40000)
            if len(final_summary_input_text_on_error) > max_chars:
                final_summary_input_text_on_error = final_summary_input_text_on_error[:max_chars]
        return [], f"GeneralScrapingError_{type(e_entry_scrape).__name__}", final_canonical_entry_url_for_this_attempt, final_summary_input_text_on_error


async def scrape_website(
    given_url: str,
    output_dir_for_run: str,
    company_name_or_id: str,
    globally_processed_urls: Set[str],
    input_row_id: Any
) -> Tuple[List[Dict[str, str]], str, Optional[str], Optional[str]]: # Added Optional[str] for summary text
    start_time = time.time()
    log_identifier = f"[RowID: {input_row_id}, Company: {company_name_or_id}]"
    logger.info(f"{log_identifier} Starting scrape_website for original URL: {given_url}")

    # --- Caching Logic: Check before scraping ---
    if config_instance.caching_enabled:
        cache_key = generate_cache_key(given_url)
        cached_results = load_from_cache(cache_key, config_instance.cache_dir)
        if cached_results is not None:
            logger.info(f"{log_identifier} Scrape data for '{given_url}' loaded from cache.")
            # Reconstruct the full tuple that the pipeline expects
            status = "Success"  # Treat cache hit as Success
            # Assuming cached_results is a list of dicts, where each dict is a page
            # and the summary_text is stored with the first page's data or separately.
            # Based on the save logic, it seems it's not stored. Let's assume it is for now.
            canonical_url = cached_results[0].get("url") if cached_results else None
            summary_text = ""
            if cached_results and isinstance(cached_results[0], dict):
                summary_text = cached_results[0].get("summary_text", "")

            return cached_results, status, canonical_url, summary_text

    normalized_given_url = normalize_url(given_url)
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Original: '{given_url}', Normalized to: '{normalized_given_url}'")

    if not normalized_given_url or not normalized_given_url.startswith(('http://', 'https://')):
        logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Invalid URL after normalization: {normalized_given_url}")
        return [], "InvalidURL", None, None # Added None for summary text

    # Initial robots.txt check for the very first normalized URL
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as http_client:
        if not await is_allowed_by_robots(normalized_given_url, http_client, input_row_id, company_name_or_id):
            return [], "RobotsDisallowed", None, None # Added None for summary text
    
    # Prepare directories once
    base_scraped_content_dir = os.path.join(output_dir_for_run, config_instance.scraped_content_subdir)
    cleaned_pages_storage_dir = base_scraped_content_dir # Removed "cleaned_pages_text" subdirectory
    os.makedirs(cleaned_pages_storage_dir, exist_ok=True) # This now ensures base_scraped_content_dir exists

    entry_candidates_queue: asyncio.Queue[str] = asyncio.Queue()
    await entry_candidates_queue.put(normalized_given_url)
    
    # Tracks entry URLs attempted *within this specific call to scrape_website* to avoid loops from fallbacks
    attempted_entry_candidates_this_call: Set[str] = {normalized_given_url}
    
    last_dns_error_status = "DNSError_AllFallbacksExhausted" # Default if all fallbacks lead to DNS errors

    async with async_playwright() as p, httpx.AsyncClient(follow_redirects=True, verify=False) as http_client_for_validation:
        browser = None
        try:
            launch_options: Dict[str, Any] = {'headless': False}
            proxy_manager = None
            proxy_to_use = None

            if config_instance.proxy_enabled:
                proxy_manager = ProxyManager(config_instance)
                proxy_to_use = proxy_manager.get_proxy()
                if proxy_to_use:
                    logger.info(f"Using proxy: {proxy_to_use}")
                    launch_options['proxy'] = {'server': proxy_to_use}
                else:
                    logger.warning("Proxy is enabled, but no healthy proxy could be obtained. Proceeding without proxy.")

            browser = await p.chromium.launch(**launch_options)
            # Create one context to be reused by _perform_scrape_for_entry_point attempts
            # This means cookies/state might persist across fallback attempts for the same original given_url.
            # If strict isolation is needed, context creation would move inside the loop.
            # Choose a random user agent if the list is not empty
            user_agent = random.choice(config_instance.user_agents) if config_instance.user_agents else ""
            
            playwright_context = await browser.new_context(
                user_agent=user_agent,
                extra_http_headers=config_instance.scraper_default_headers,
                java_script_enabled=True,
                ignore_https_errors=True
            )

            while not entry_candidates_queue.empty():
                current_entry_url_to_attempt = await entry_candidates_queue.get()
                
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Trying entry point: {current_entry_url_to_attempt}")

                details, status, canonical_landed, collected_summary_text = await _perform_scrape_for_entry_point(
                    current_entry_url_to_attempt, playwright_context, http_client_for_validation, output_dir_for_run,
                    company_name_or_id, globally_processed_urls, input_row_id
                )

                if status != "DNSError": # Any success or non-DNS error is final for this given_url
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Entry point {current_entry_url_to_attempt} resulted in non-DNS status: {status}. Finalizing.")
                    if browser.is_connected(): await browser.close()
                    # --- Caching Logic: Save after scraping ---
                    if config_instance.caching_enabled and status == "Success":
                        cache_key = generate_cache_key(given_url)
                        # Add summary text to the data being cached
                        if details:
                            details[0]['summary_text'] = collected_summary_text
                        save_to_cache(cache_key, details, config_instance.cache_dir)
                    return details, status, canonical_landed, collected_summary_text # Propagate summary text
                
                # It was a DNSError for current_entry_url_to_attempt
                last_dns_error_status = status # Store the most recent DNS error type
                logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Entry point {current_entry_url_to_attempt} failed with DNSError. Status: {status}.")

                if config_instance.enable_dns_error_fallbacks:
                    generated_fallbacks_for_current_failed_entry: List[str] = []
                    
                    # Strategy 1: Hyphen Simplification
                    try:
                        parsed_failed_entry = tldextract.extract(current_entry_url_to_attempt)
                        domain_part = parsed_failed_entry.domain
                        suffix_part = parsed_failed_entry.suffix
                        
                        if '-' in domain_part:
                            simplified_domain_part = domain_part.split('-', 1)[0]
                            if simplified_domain_part:
                                variant1_domain = f"{simplified_domain_part}.{suffix_part}"
                                parsed_original_for_reconstruct = urlparse(current_entry_url_to_attempt)
                                variant1_url = urlunparse((parsed_original_for_reconstruct.scheme, variant1_domain, parsed_original_for_reconstruct.path, parsed_original_for_reconstruct.params, parsed_original_for_reconstruct.query, parsed_original_for_reconstruct.fragment))
                                variant1_url_normalized = normalize_url(variant1_url)
                                if variant1_url_normalized not in attempted_entry_candidates_this_call:
                                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] DNS Fallback (Hyphen): Adding '{variant1_url_normalized}' to try.")
                                    generated_fallbacks_for_current_failed_entry.append(variant1_url_normalized)
                    except Exception as e_tld_hyphen:
                        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Error during hyphen simplification for {current_entry_url_to_attempt}: {e_tld_hyphen}")

                    # Strategy 2: TLD Swap (.de to .com) on current_entry_url_to_attempt (that just DNS-failed)
                    try:
                        parsed_failed_entry_for_tld_swap = tldextract.extract(current_entry_url_to_attempt)
                        if parsed_failed_entry_for_tld_swap.suffix.lower() == 'de':
                            variant2_domain = f"{parsed_failed_entry_for_tld_swap.domain}.com"
                            parsed_original_for_reconstruct_tld = urlparse(current_entry_url_to_attempt)
                            variant2_url = urlunparse((parsed_original_for_reconstruct_tld.scheme, variant2_domain, parsed_original_for_reconstruct_tld.path, parsed_original_for_reconstruct_tld.params, parsed_original_for_reconstruct_tld.query, parsed_original_for_reconstruct_tld.fragment))
                            variant2_url_normalized = normalize_url(variant2_url)
                            if variant2_url_normalized not in attempted_entry_candidates_this_call:
                                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] DNS Fallback (TLD Swap): Adding '{variant2_url_normalized}' to try.")
                                generated_fallbacks_for_current_failed_entry.append(variant2_url_normalized)
                    except Exception as e_tld_swap_main:
                        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Error during .de to .com TLD swap for {current_entry_url_to_attempt}: {e_tld_swap_main}")

                    for fb_url in generated_fallbacks_for_current_failed_entry:
                        if fb_url not in attempted_entry_candidates_this_call: # Double check before adding
                           await entry_candidates_queue.put(fb_url)
                           attempted_entry_candidates_this_call.add(fb_url)
                else: # DNS fallbacks disabled
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] DNS fallbacks disabled. No further attempts for {current_entry_url_to_attempt}.")
                    # If this was the last item in queue (i.e. normalized_given_url and no fallbacks added)
                    # the loop will terminate and the last_dns_error_status will be returned.
            
            # If queue is exhausted
            if browser and browser.is_connected(): await browser.close()
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] All entry point attempts, including DNS fallbacks, exhausted for original URL: {given_url}. Last DNS status: {last_dns_error_status}")
            return [], last_dns_error_status, None, None # Added None for summary text

        except Exception as e_outer:
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Outer error in scrape_website for '{given_url}': {type(e_outer).__name__} - {e_outer}", exc_info=True)
            if browser and browser.is_connected(): await browser.close()
            return [], f"OuterScrapingError_{type(e_outer).__name__}", None, None # Added None for summary text
        finally:
            if browser and browser.is_connected():
                logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Ensuring browser is closed in scrape_website's final 'finally' block.")
                await browser.close()

    # Fallback if Playwright setup itself failed before entering the async with block
    logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Scrape_website ended, possibly due to Playwright launch failure for '{given_url}'.")
    return [], "ScraperSetupFailure_Outer", None, None # Added None for summary text

# TODO: [FutureEnhancement] The _test_scraper function below was for demonstrating and testing
# the scrape_website functionality directly. It includes setup for logging and test output.
# Commented out as it's not part of the main pipeline execution.
# It can be uncommented for debugging or standalone testing of the scraper logic.
async def _test_scraper():
    """
    An asynchronous test function to demonstrate and test the `scrape_website` functionality.

    Sets up logging, defines a test URL and output directory, then calls
    `scrape_website` and logs the result. This function is intended to be run
    when the script is executed directly (`if __name__ == "__main__":`).
    """
    # Ensure AppConfig is loaded with any .env overrides for testing
    global config_instance
    config_instance = AppConfig() 
    
    setup_logging(logging.DEBUG) 
    logger.info("Starting test scraper...")

    test_url = "https://www.example.com" 
    # test_url = "https://www.python.org"
    # test_url = "https://nonexistent-domain-for-testing123.com"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this script is in 'src/scraper', project_root is 'src/'
    # For 'phone_validation_pipeline' as project root, adjust path.
    # If 'src' is directly under 'phone_validation_pipeline':
    project_root = os.path.dirname(os.path.dirname(script_dir)) # Goes up to phone_validation_pipeline
    
    test_output_base = os.path.join(project_root, "test_scraper_output_data")
    test_run_id = "test_run_manual_" + time.strftime("%Y%m%d_%H%M%S")
    test_run_output_dir = os.path.join(test_output_base, test_run_id)
    
    # Ensure the main output directory for the run exists
    os.makedirs(test_run_output_dir, exist_ok=True)
    # The scrape_website function will create subdirectories like 'scraped_content/cleaned_pages_text'

    logger.info(f"Test output directory for this run: {test_run_output_dir}")
    
    # Initialize a new set for globally_processed_urls for this test run
    globally_processed_urls_for_test: Set[str] = set()

    # Adjust to expect four values from scrape_website
    scraped_items_with_type, status, canonical_url, summary_text = await scrape_website(
       test_url,
       test_run_output_dir, # This is the base for the run, scrape_website will make subdirs
       "example_company_test",
       globally_processed_urls_for_test,
       "TEST_ROW_ID_001" # Added placeholder for input_row_id
    )

    if scraped_items_with_type:
        logger.info(f"Test successful: {len(scraped_items_with_type)} page(s) scraped. Status: {status}. Canonical URL: {canonical_url}. Summary text length: {len(summary_text) if summary_text else 0}")
        # Adjust loop to handle the new tuple structure (path, url, type)
        for item_path, source_url, page_type in scraped_items_with_type:
            logger.info(f"  - Saved: {item_path} (from: {source_url}, type: {page_type})")
    else:
        logger.error(f"Test failed: Status: {status}. Canonical URL: {canonical_url}. Summary text length: {len(summary_text) if summary_text else 0}")

# TODO: [FutureEnhancement] The __main__ block below allowed direct execution of _test_scraper.
# Commented out as it's not intended for execution during normal library use.
if __name__ == "__main__":
    # This ensures that if the script is run directly, AppConfig is initialized
    # and logging is set up before _test_scraper is called.
    if not logger.hasHandlers(): 
        setup_logging(logging.INFO) 
    asyncio.run(_test_scraper())