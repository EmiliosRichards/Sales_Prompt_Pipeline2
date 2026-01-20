import asyncio
import os
import re
import logging
import time
import hashlib # Added for hashing long filenames
import shutil
from urllib.parse import urljoin, urlparse, urldefrag, urlunparse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
from bs4 import BeautifulSoup
from bs4.element import Tag # Added for type checking
import httpx # For asynchronous robots.txt checking + optional HTML fallback
from urllib.robotparser import RobotFileParser
from typing import Set, Tuple, Optional, List, Dict, Any
import tldextract # Added for DNS fallback logic

# Assuming config.py is in src.core
from src.core.config import AppConfig
from src.core.logging_config import setup_logging

# Instantiate AppConfig for scraper_logic
config_instance = AppConfig()

# Setup logger for this module
logger = logging.getLogger(__name__)

def _is_httpx_fallback_html_usable(
    html: str,
    input_row_id: Any,
    company_name_or_id: str,
    url: str
) -> Tuple[bool, str]:
    """Heuristic guardrails to avoid processing obvious block pages / junk HTML.

    Returns (usable, reason) where reason is a short machine-readable string.
    """
    if not html or not html.strip():
        return False, "empty_html"

    # Extract visible text and check min length (avoid tiny/empty boilerplate)
    text = extract_text_from_html(html)
    min_chars = getattr(config_instance, "scraper_http_fallback_min_text_chars", 200)
    if len(text) < min_chars:
        logger.info(
            f"[RowID: {input_row_id}, Company: {company_name_or_id}] httpx fallback rejected for {url}: "
            f"extracted text too short ({len(text)} < {min_chars})"
        )
        return False, f"rejected_too_short:{len(text)}<{min_chars}"

    # Block page keyword detection
    block_keywords = getattr(config_instance, "scraper_http_fallback_block_keywords", []) or []
    text_low = text.lower()
    for kw in block_keywords:
        if kw and kw in text_low:
            logger.info(
                f"[RowID: {input_row_id}, Company: {company_name_or_id}] httpx fallback rejected for {url}: "
                f"block keyword matched '{kw}'"
            )
            return False, f"rejected_block_keyword:{kw}"

    return True, "usable"

def normalize_url(url: str) -> str:
    """
    Normalizes a URL to a canonical form.
    """
    try:
        url_no_frag, _ = urldefrag(url)
        parsed = urlparse(url_no_frag)
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path
        common_indexes = ['index.html', 'index.htm', 'index.php', 'default.html', 'default.htm', 'index.asp', 'default.asp']
        for index_file in common_indexes:
            if path.endswith(f'/{index_file}'):
                path = path[:-len(index_file)]
                break
        if netloc and path and not path.startswith('/'):
            path = '/' + path
        if path != '/' and path.endswith('/'):
            path = path[:-1]
        if not path and netloc:
            path = '/'
        query = ''
        if parsed.query:
            params = parsed.query.split('&')
            ignored_params = {'fallback'}
            filtered_params = [p for p in params if (p.split('=')[0].lower() if '=' in p else p.lower()) not in ignored_params]
            if filtered_params:
                query = '&'.join(sorted(filtered_params))
        return urlparse('')._replace(scheme=scheme, netloc=netloc, path=path, params=parsed.params, query=query, fragment='').geturl()
    except Exception as e:
        logger.error(f"Error normalizing URL '{url}': {e}. Returning original URL.", exc_info=True)
        return url

def get_safe_filename(name_or_url: str, for_url: bool = False, max_len: int = 100) -> str:
    if for_url:
        logger.info(f"get_safe_filename (for_url=True): Input for filename generation='{name_or_url}'")
    original_input = name_or_url
    if for_url:
        parsed_original_url = urlparse(original_input)
        domain_part = re.sub(r'^www\.', '', parsed_original_url.netloc)
        domain_part = re.sub(r'[^\w-]', '', domain_part)[:config_instance.filename_url_domain_max_len]
        url_hash = hashlib.sha256(original_input.encode('utf-8')).hexdigest()[:config_instance.filename_url_hash_max_len]
        safe_name = f"{domain_part}_{url_hash}" # Use the sanitized domain_part
        logger.info(f"DEBUG PATH: get_safe_filename (for_url=True) output: '{safe_name}' from input '{original_input}'") # DEBUG PATH LENGTH
        return safe_name
    else:
        name_or_url = re.sub(r'^https?://', '', name_or_url)
        safe_name = re.sub(r'[^\w.-]', '_', name_or_url)
        safe_name_truncated = safe_name[:max_len]
        logger.info(f"DEBUG PATH: get_safe_filename (for_url=False) output: '{safe_name_truncated}' (original sanitized: '{safe_name}', max_len: {max_len}) from input '{original_input}'") # DEBUG PATH LENGTH
        return safe_name_truncated


def _try_copy_cached_cleaned_pages(normalized_url: str, output_dir_for_run: str) -> Optional[List[Tuple[str, str, str]]]:
    """
    Attempt to reuse existing *_cleaned.txt artifacts from one or more cache directories.
    On hit, copy them into this run's scraped_content directory and return them as scraped_pages_details.
    """
    try:
        if not getattr(config_instance, "reuse_scraped_content_if_available", False):
            return None
        cache_dirs = getattr(config_instance, "scraped_content_cache_dirs", None) or []
        if not cache_dirs:
            return None

        parsed = urlparse(normalized_url)
        host = parsed.netloc or ""
        host = host.split("@")[-1].split(":")[0]
        host = re.sub(r"^www\\.", "", host)
        if not host:
            return None
        safe_source_name = re.sub(r"[^\\w.-]", "_", host)[:50]

        candidates: List[str] = []
        for root in cache_dirs:
            root = os.path.normpath(root)
            domain_dir = os.path.join(root, safe_source_name)
            if not os.path.isdir(domain_dir):
                continue
            try:
                for fname in os.listdir(domain_dir):
                    if fname.endswith("_cleaned.txt"):
                        candidates.append(os.path.join(domain_dir, fname))
            except Exception:
                continue

        if not candidates:
            return None

        try:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        except Exception:
            pass

        base_scraped_content_dir = os.path.join(output_dir_for_run, config_instance.scraped_content_subdir)
        os.makedirs(base_scraped_content_dir, exist_ok=True)

        max_pages = int(getattr(config_instance, "scraper_pages_for_summary_count", 5) or 5)
        details: List[Tuple[str, str, str]] = []
        for fp in candidates[:max_pages]:
            try:
                dest = os.path.join(base_scraped_content_dir, f"CACHED__{os.path.basename(fp)}")
                shutil.copyfile(fp, dest)
                details.append((dest, normalized_url, "Cached"))
            except Exception:
                continue

        return details if details else None
    except Exception:
        return None

async def _fetch_html_via_httpx(
    url: str,
    input_row_id: Any,
    company_name_or_id: str
) -> Tuple[Optional[str], Optional[int], str]:
    """Best-effort HTML fetch using httpx as a fallback when Playwright navigation/content fails.

    Returns (html, status_code, result_reason).
    """
    if not getattr(config_instance, "scraper_http_fallback_enabled", True):
        return None, None, "disabled"

    timeout_s = getattr(config_instance, "scraper_http_fallback_timeout_seconds", 15)
    max_bytes = getattr(config_instance, "scraper_http_fallback_max_bytes", 2_000_000)
    headers: Dict[str, str] = {}
    try:
        if config_instance.user_agents:
            headers["User-Agent"] = config_instance.user_agents[0]
        # Merge in default headers (if configured)
        if getattr(config_instance, "scraper_default_headers", None):
            headers.update(config_instance.scraper_default_headers)
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            verify=False,
            headers=headers if headers else None,
            timeout=timeout_s,
        ) as client:
            resp = await client.get(url)
            status = resp.status_code
            if status == 404:
                return None, status, "http_404"
            # Only consider HTML-ish responses
            content_type = (resp.headers.get("content-type") or "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type and "text/" not in content_type:
                return None, status, f"non_html_content_type:{content_type[:80]}"
            # Enforce max bytes to avoid huge downloads
            content_bytes = resp.content or b""
            if max_bytes and len(content_bytes) > max_bytes:
                content_bytes = content_bytes[:max_bytes]
            text = content_bytes.decode(resp.encoding or "utf-8", errors="replace")
            if not text.strip():
                return None, status, "empty_body"
            usable, reason = _is_httpx_fallback_html_usable(text, input_row_id, company_name_or_id, url)
            if not usable:
                return None, status, reason
            logger.info(
                f"[RowID: {input_row_id}, Company: {company_name_or_id}] httpx fallback fetched HTML for {url}. Status: {status}, bytes: {len(content_bytes)}"
            )
            return text, status, "success"
    except Exception as e:
        logger.info(
            f"[RowID: {input_row_id}, Company: {company_name_or_id}] httpx fallback failed for {url}: {type(e).__name__} - {e}"
        )
        return None, None, f"exception:{type(e).__name__}"


async def fetch_page_content(
    page,
    url: str,
    input_row_id: Any,
    company_name_or_id: str
) -> Tuple[Optional[str], Optional[int], str, bool, Optional[str]]:
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigating to URL: {url}")
    httpx_attempted = False
    httpx_result: Optional[str] = None
    try:
        response = await page.goto(url, timeout=config_instance.default_navigation_timeout, wait_until='domcontentloaded')
        if response:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Navigation to {url} successful. Status: {response.status}")
            if response.ok:
                if config_instance.scraper_networkidle_timeout_ms > 0:
                    logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Waiting for networkidle on {url} (timeout: {config_instance.scraper_networkidle_timeout_ms}ms)...")
                    try:
                        await page.wait_for_load_state('networkidle', timeout=config_instance.scraper_networkidle_timeout_ms)
                        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Networkidle achieved for {url}.")
                    except PlaywrightTimeoutError:
                        logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Timeout waiting for networkidle on {url} after {config_instance.scraper_networkidle_timeout_ms}ms. Proceeding with current DOM content.")
                try:
                    content = await page.content()
                except Exception as e_content:
                    logger.info(
                        f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright page.content() failed for {url}: "
                        f"{type(e_content).__name__} - {e_content}. Trying httpx fallback."
                    )
                    httpx_attempted = True
                    fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
                    httpx_result = fb_result
                    if fb_html:
                        return fb_html, fb_status or response.status, "httpx_fallback", httpx_attempted, httpx_result
                    return None, response.status, "playwright_content_error", httpx_attempted, httpx_result
                logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Content fetched successfully for {url}.")
                return content, response.status, "playwright", httpx_attempted, httpx_result
            else:
                logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] HTTP error for {url}: Status {response.status} {response.status_text}. No content fetched.")
                # Sometimes we still have a DOM even on non-2xx; try reading it for link extraction.
                try:
                    dom_html = await page.content()
                    if dom_html and dom_html.strip():
                        return dom_html, response.status, "playwright_non_ok_dom", httpx_attempted, httpx_result
                except Exception:
                    pass
                # Try httpx fallback for non-404 errors (403/500/etc can still have HTML with phone numbers)
                if response.status != 404:
                    httpx_attempted = True
                    fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
                    httpx_result = fb_result
                    if fb_html:
                        return fb_html, fb_status or response.status, "httpx_fallback", httpx_attempted, httpx_result
                return None, response.status, "playwright_non_ok", httpx_attempted, httpx_result
        else:
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Failed to get a response object for {url}. Navigation might have failed silently.")
            httpx_attempted = True
            fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
            httpx_result = fb_result
            if fb_html:
                return fb_html, fb_status, "httpx_fallback", httpx_attempted, httpx_result
            return None, None, "playwright_no_response", httpx_attempted, httpx_result
    except PlaywrightTimeoutError:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright navigation timeout for {url} after {config_instance.default_navigation_timeout / 1000}s.")
        httpx_attempted = True
        fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
        httpx_result = fb_result
        if fb_html:
            return fb_html, fb_status, "httpx_fallback", httpx_attempted, httpx_result
        return None, -1, "playwright_timeout", httpx_attempted, httpx_result # Specific code for timeout
    except PlaywrightError as e:
        error_message = str(e)
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Playwright error during navigation to {url}: {error_message}")
        httpx_attempted = True
        fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
        httpx_result = fb_result
        if fb_html:
            return fb_html, fb_status, "httpx_fallback", httpx_attempted, httpx_result
        if "net::ERR_NAME_NOT_RESOLVED" in error_message: return None, -2, "playwright_dns", httpx_attempted, httpx_result # DNS error
        elif "net::ERR_CONNECTION_REFUSED" in error_message: return None, -3, "playwright_conn_refused", httpx_attempted, httpx_result # Connection refused
        elif "net::ERR_ABORTED" in error_message: return None, -6, "playwright_aborted", httpx_attempted, httpx_result # Request aborted, often due to navigation elsewhere
        return None, -4, "playwright_error", httpx_attempted, httpx_result # Other Playwright error
    except Exception as e:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Unexpected error fetching page {url}: {type(e).__name__} - {e}", exc_info=True)
        httpx_attempted = True
        fb_html, fb_status, fb_result = await _fetch_html_via_httpx(url, input_row_id, company_name_or_id)
        httpx_result = fb_result
        if fb_html:
            return fb_html, fb_status, "httpx_fallback", httpx_attempted, httpx_result
        return None, -5, "playwright_exception", httpx_attempted, httpx_result # Generic exception

def extract_text_from_html(html_content: str) -> str:
    if not html_content: return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=' ', strip=True)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_internal_links(html_content: str, base_url: str, input_row_id: Any, company_name_or_id: str) -> List[Tuple[str, int]]:
    if not html_content: return []
    scored_links: List[Tuple[str, int]] = []
    soup = BeautifulSoup(html_content, 'html.parser')
    normalized_base_url_str = normalize_url(base_url)
    parsed_base_url = urlparse(normalized_base_url_str)

    for link_tag in soup.find_all('a', href=True):
        if not isinstance(link_tag, Tag): continue
        href_attr = link_tag.get('href')
        current_href: Optional[str] = None
        if isinstance(href_attr, str): current_href = href_attr.strip()
        elif isinstance(href_attr, list) and href_attr and isinstance(href_attr[0], str): current_href = href_attr[0].strip()
        if not current_href: continue

        absolute_url_raw = urljoin(base_url, current_href)
        normalized_link_url = normalize_url(absolute_url_raw)
        parsed_normalized_link = urlparse(normalized_link_url)

        if parsed_normalized_link.scheme not in ['http', 'https']: continue
        if parsed_normalized_link.netloc != parsed_base_url.netloc: continue

        link_text = link_tag.get_text().lower().strip()
        link_href_lower = normalized_link_url.lower()
        initial_keyword_match = False
        if config_instance.target_link_keywords:
            if any(kw in link_text for kw in config_instance.target_link_keywords) or \
               any(kw in link_href_lower for kw in config_instance.target_link_keywords):
                initial_keyword_match = True
        if not initial_keyword_match: continue

        if config_instance.scraper_exclude_link_path_patterns:
            path_lower = parsed_normalized_link.path.lower()
            if any(p and p in path_lower for p in config_instance.scraper_exclude_link_path_patterns):
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Link '{normalized_link_url}' hard excluded by pattern in path: '{path_lower}'.")
                continue
        
        score = 0
        path_segments = [seg for seg in parsed_normalized_link.path.lower().strip('/').split('/') if seg]
        num_segments = len(path_segments)

        if config_instance.scraper_critical_priority_keywords:
            for crit_kw in config_instance.scraper_critical_priority_keywords:
                if any(seg == crit_kw for seg in path_segments):
                    current_score_val = 100
                    if num_segments > config_instance.scraper_max_keyword_path_segments:
                        current_score_val -= min(20, (num_segments - config_instance.scraper_max_keyword_path_segments) * 5)
                    score = max(score, current_score_val)
                    if score >= 100: break
            if score >= 100: pass

        if score < 90 and config_instance.scraper_high_priority_keywords:
            for high_kw in config_instance.scraper_high_priority_keywords:
                if any(seg == high_kw for seg in path_segments):
                    current_score_val = 90
                    if num_segments > config_instance.scraper_max_keyword_path_segments:
                        current_score_val -= min(20, (num_segments - config_instance.scraper_max_keyword_path_segments) * 5)
                    score = max(score, current_score_val)
                    if score >= 90: break
            if score >= 90: pass
        
        if score < 80:
            combined_keywords = list(set(config_instance.scraper_critical_priority_keywords + config_instance.scraper_high_priority_keywords))
            if combined_keywords:
                for p_kw in combined_keywords:
                    for i, seg in enumerate(path_segments):
                        if seg == p_kw:
                            current_score_val = 80 - (i * 5)
                            if num_segments > config_instance.scraper_max_keyword_path_segments:
                                current_score_val -= min(15, (num_segments - config_instance.scraper_max_keyword_path_segments) * 5)
                            score = max(score, current_score_val)
                            break 
                    if score >= 80: break
        
        if score < 50 and config_instance.target_link_keywords:
            if any(tk in seg for tk in config_instance.target_link_keywords for seg in path_segments):
                score = max(score, 50)
        
        if score < 40 and config_instance.target_link_keywords:
            if any(tk in link_text for tk in config_instance.target_link_keywords):
                score = max(score, 40)

        if score >= config_instance.scraper_min_score_to_queue:
            log_text_snippet = link_text[:50].replace('\n', ' ')
            logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Link '{normalized_link_url}' scored: {score} (Text: '{log_text_snippet}...', Path: '{parsed_normalized_link.path}') - Adding to potential queue.")
            scored_links.append((normalized_link_url, score))
        else:
            log_text_snippet = link_text[:50].replace('\n', ' ')
            logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Link '{normalized_link_url}' (score {score}) below min_score_to_queue ({config_instance.scraper_min_score_to_queue}). Path: '{parsed_normalized_link.path}', Text: '{log_text_snippet}...'. Discarding.")
            
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] From page {base_url}, found {len(scored_links)} internal links meeting score criteria.")
    return scored_links

async def is_allowed_by_robots(url: str, client: httpx.AsyncClient, input_row_id: Any, company_name_or_id: str) -> bool:
    if not config_instance.respect_robots_txt:
        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] robots.txt check is disabled.")
        return True
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Fetching robots.txt from: {robots_url}")
        response = await client.get(robots_url, timeout=10, headers={'User-Agent': config_instance.robots_txt_user_agent})
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

def _classify_page_type(url_str: str, config: AppConfig) -> str:
    """Classifies a URL based on keywords in its path."""
    if not url_str:
        return "unknown"
    
    url_lower = url_str.lower()
    # Check for specific page types based on keywords in URL path
    # Order matters if keywords overlap; more specific should come first if necessary.
    # For now, assuming simple first-match.
    
    # Path-based classification
    parsed_url = urlparse(url_lower)
    path_lower = parsed_url.path

    if any(kw in path_lower for kw in config.page_type_keywords_contact):
        return "contact"
    if any(kw in path_lower for kw in config.page_type_keywords_imprint):
        return "imprint"
    if any(kw in path_lower for kw in config.page_type_keywords_legal):
        return "legal"
    
    # Fallback if no path keywords match, check full URL for very generic terms
    # (less reliable, path is usually better indicator)
    if any(kw in url_lower for kw in config.page_type_keywords_contact): # broader check on full URL
        return "contact"
    if any(kw in url_lower for kw in config.page_type_keywords_imprint):
        return "imprint"
    if any(kw in url_lower for kw in config.page_type_keywords_legal):
        return "legal"

    # If it's just the base domain (e.g., http://example.com or http://example.com/)
    if not path_lower or path_lower == '/':
        return "homepage" # Could be a specific type or general_content

    return "general_content"


async def _perform_scrape_for_entry_point(
    entry_url_to_process: str,
    playwright_context, # Existing Playwright browser context
    output_dir_for_run: str,
    company_name_or_id: str,
    globally_processed_urls: Set[str], # Shared across all entry point attempts for the original given_url
    input_row_id: Any
) -> Tuple[List[Tuple[str, str, str]], str, Optional[str], Dict[str, Any]]:
    """
    Core scraping logic for a single entry point URL and its children.
    This function contains the main `while urls_to_scrape` loop.
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
    scraped_page_details_for_this_entry: List[Tuple[str, str, str]] = []
    
    # Queue for this specific entry point attempt
    urls_to_scrape_q: List[Tuple[str, int, int]] = [(entry_url_to_process, 0, 100)]
    # processed_urls_this_entry_call tracks URLs processed starting from *this* entry_url_to_process
    # to avoid loops within its own scraping process.
    processed_urls_this_entry_call: Set[str] = {entry_url_to_process}

    # Use the passed Playwright context to create a new page for this entry attempt
    page = await playwright_context.new_page()
    page.set_default_timeout(config_instance.default_page_timeout)
    
    entry_point_status_code: Optional[int] = None # To store status of the entry point itself

    httpx_attempted_any = False
    httpx_used_any = False
    httpx_last_result: Optional[str] = None

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


            html_content, status_code_fetch, fetch_method, httpx_attempted, httpx_result = await fetch_page_content(
                page, current_url_from_queue, input_row_id, company_name_or_id
            )
            if httpx_attempted:
                httpx_attempted_any = True
                httpx_last_result = httpx_result
            if fetch_method == "httpx_fallback" and html_content:
                httpx_used_any = True
                httpx_last_result = httpx_result or "success"
            
            if current_url_from_queue == entry_url_to_process and current_depth == 0: # This is the fetch for the entry point itself
                entry_point_status_code = status_code_fetch


            if html_content:
                pages_scraped_this_entry_count += 1
                if pages_scraped_this_entry_count > config_instance.scraper_max_pages_per_domain and \
                   current_score >= config_instance.scraper_score_threshold_for_limit_bypass:
                    high_priority_pages_scraped_after_limit_entry +=1

                final_landed_url_raw = page.url
                final_landed_url_normalized = normalize_url(final_landed_url_raw)
                
                logger.info(
                    f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] "
                    f"Page fetch: Req='{current_url_from_queue}', LandedNorm='{final_landed_url_normalized}', "
                    f"Status: {status_code_fetch}, Method: {fetch_method}"
                )

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
                method_suffix = "__httpx" if fetch_method == "httpx_fallback" else ""
                cleaned_page_filename = f"{company_safe_name}__{landed_url_safe_name}{method_suffix}_cleaned.txt"
                cleaned_page_filepath = os.path.join(source_specific_output_dir, cleaned_page_filename)
                
                try:
                    with open(cleaned_page_filepath, 'w', encoding='utf-8') as f_cleaned_page:
                        f_cleaned_page.write(cleaned_text)
                    page_type = _classify_page_type(final_landed_url_normalized, config_instance)
                    scraped_page_details_for_this_entry.append((cleaned_page_filepath, final_landed_url_normalized, page_type))
                except IOError as e:
                    logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] IOError saving cleaned text for '{final_landed_url_normalized}': {e}")

                if current_depth < config_instance.max_depth_internal_links:
                    newly_found_links_with_scores = find_internal_links(html_content, final_landed_url_normalized, input_row_id, company_name_or_id)
                    added_to_queue_count = 0
                    for link_url, link_score in newly_found_links_with_scores:
                        if link_url not in globally_processed_urls and link_url not in processed_urls_this_entry_call:
                            urls_to_scrape_q.append((link_url, current_depth + 1, link_score))
                            processed_urls_this_entry_call.add(link_url)
                            added_to_queue_count +=1
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
                    return [], http_status_report, None, {
                        "attempted": httpx_attempted_any,
                        "used": httpx_used_any,
                        "result": httpx_last_result or ("success" if httpx_used_any else "not_attempted")
                    } # No canonical URL if entry point fails critically
        
        # After loop for this entry point
        await page.close()
        if scraped_page_details_for_this_entry:
            logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] Successfully scraped {len(scraped_page_details_for_this_entry)} pages.")
            return scraped_page_details_for_this_entry, "Success", final_canonical_entry_url_for_this_attempt, {
                "attempted": httpx_attempted_any,
                "used": httpx_used_any,
                "result": httpx_last_result or ("success" if httpx_used_any else "not_attempted")
            }
        else: # No pages scraped for this entry point
            final_status_for_this_entry = "NoContentScraped_Overall"
            if entry_point_status_code is not None: # If the entry point itself had a specific failure status
                 status_map = {-1: "TimeoutError", -2: "DNSError", -3: "ConnectionRefused", -4: "PlaywrightError", -5: "GenericScrapeError", -6: "RequestAborted"}
                 if entry_point_status_code > 0: final_status_for_this_entry = f"HTTPError_{entry_point_status_code}"
                 elif entry_point_status_code in status_map: final_status_for_this_entry = status_map[entry_point_status_code]
                 else: final_status_for_this_entry = "UnknownScrapeErrorCode"

            logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] No content scraped. Final status for this entry: {final_status_for_this_entry}")
            return [], final_status_for_this_entry, final_canonical_entry_url_for_this_attempt, {
                "attempted": httpx_attempted_any,
                "used": httpx_used_any,
                "result": httpx_last_result or ("success" if httpx_used_any else "not_attempted")
            } # Return canonical if it was set by a successful landing, even if no sub-pages
    except Exception as e_entry_scrape:
        logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}, Entry: {entry_url_to_process}] General error during scraping process: {type(e_entry_scrape).__name__} - {e_entry_scrape}", exc_info=True)
        if page.is_closed() == False : await page.close()
        return [], f"GeneralScrapingError_{type(e_entry_scrape).__name__}", final_canonical_entry_url_for_this_attempt, {
            "attempted": httpx_attempted_any,
            "used": httpx_used_any,
            "result": httpx_last_result or ("success" if httpx_used_any else "not_attempted")
        }


async def scrape_website(
    given_url: str,
    output_dir_for_run: str,
    company_name_or_id: str,
    globally_processed_urls: Set[str],
    input_row_id: Any
) -> Tuple[List[Tuple[str, str, str]], str, Optional[str], Dict[str, Any]]:
    start_time = time.time()
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Starting scrape_website for original URL: {given_url}")

    normalized_given_url = normalize_url(given_url)
    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Original: '{given_url}', Normalized to: '{normalized_given_url}'")

    if not normalized_given_url or not normalized_given_url.startswith(('http://', 'https://')):
        logger.warning(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Invalid URL after normalization: {normalized_given_url}")
        return [], "InvalidURL", None, {"attempted": False, "used": False, "result": "not_attempted_invalid_url"}

    # --- Reuse cached cleaned scrape artifacts (skip network scraping) ---
    cached_pages = _try_copy_cached_cleaned_pages(normalized_given_url, output_dir_for_run)
    if cached_pages:
        try:
            globally_processed_urls.add(normalized_given_url)
        except Exception:
            pass
        logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Reusing cached scraped content (cleaned.txt) instead of re-scraping.")
        return cached_pages, "Success_CacheHit", normalized_given_url, {"attempted": False, "used": False, "result": "cache_hit"}

    # Initial robots.txt check for the very first normalized URL
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as http_client:
        if not await is_allowed_by_robots(normalized_given_url, http_client, input_row_id, company_name_or_id):
            return [], "RobotsDisallowed", None, {"attempted": False, "used": False, "result": "not_attempted_robots_disallowed"}
    
    # Prepare directories once
    base_scraped_content_dir = os.path.join(output_dir_for_run, config_instance.scraped_content_subdir)
    cleaned_pages_storage_dir = base_scraped_content_dir # Removed "cleaned_pages_text" subdirectory
    os.makedirs(cleaned_pages_storage_dir, exist_ok=True) # This now ensures base_scraped_content_dir exists

    entry_candidates_queue: asyncio.Queue[str] = asyncio.Queue()
    await entry_candidates_queue.put(normalized_given_url)
    
    # Tracks entry URLs attempted *within this specific call to scrape_website* to avoid loops from fallbacks
    attempted_entry_candidates_this_call: Set[str] = {normalized_given_url}
    
    last_dns_error_status = "DNSError_AllFallbacksExhausted" # Default if all fallbacks lead to DNS errors
    aggregated_httpx_attempted = False
    aggregated_httpx_used = False
    aggregated_httpx_last_result: Optional[str] = None

    async with async_playwright() as p:
        browser = None
        try:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )
            # Create one context to be reused by _perform_scrape_for_entry_point attempts
            # This means cookies/state might persist across fallback attempts for the same original given_url.
            # If strict isolation is needed, context creation would move inside the loop.
            playwright_context = await browser.new_context(
                user_agent=config_instance.user_agents[0] if config_instance.user_agents else 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True
            )

            while not entry_candidates_queue.empty():
                current_entry_url_to_attempt = await entry_candidates_queue.get()
                
                logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Trying entry point: {current_entry_url_to_attempt}")

                details, status, canonical_landed, httpx_meta = await _perform_scrape_for_entry_point(
                    current_entry_url_to_attempt, playwright_context, output_dir_for_run,
                    company_name_or_id, globally_processed_urls, input_row_id
                )

                try:
                    if isinstance(httpx_meta, dict):
                        aggregated_httpx_attempted = aggregated_httpx_attempted or bool(httpx_meta.get("attempted"))
                        aggregated_httpx_used = aggregated_httpx_used or bool(httpx_meta.get("used"))
                        if httpx_meta.get("result"):
                            aggregated_httpx_last_result = str(httpx_meta.get("result"))
                except Exception:
                    pass

                if status != "DNSError": # Any success or non-DNS error is final for this given_url
                    logger.info(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Entry point {current_entry_url_to_attempt} resulted in non-DNS status: {status}. Finalizing.")
                    if browser.is_connected(): await browser.close()
                    # Prefer aggregated meta (covers multiple entry attempts); fall back to the current attempt's meta.
                    return details, status, canonical_landed, {
                        "attempted": aggregated_httpx_attempted,
                        "used": aggregated_httpx_used,
                        "result": aggregated_httpx_last_result or ("success" if aggregated_httpx_used else "not_attempted")
                    }
                
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
            return [], last_dns_error_status, None, {
                "attempted": aggregated_httpx_attempted,
                "used": aggregated_httpx_used,
                "result": aggregated_httpx_last_result or ("success" if aggregated_httpx_used else "not_attempted_dns_error")
            }

        except Exception as e_outer:
            logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Outer error in scrape_website for '{given_url}': {type(e_outer).__name__} - {e_outer}", exc_info=True)
            if browser and browser.is_connected(): await browser.close()
            return [], f"OuterScrapingError_{type(e_outer).__name__}", None, {"attempted": False, "used": False, "result": f"not_attempted_outer_error:{type(e_outer).__name__}"}
        finally:
            if browser and browser.is_connected():
                logger.debug(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Ensuring browser is closed in scrape_website's final 'finally' block.")
                await browser.close()

    # Fallback if Playwright setup itself failed before entering the async with block
    logger.error(f"[RowID: {input_row_id}, Company: {company_name_or_id}] Scrape_website ended, possibly due to Playwright launch failure for '{given_url}'.")
    return [], "ScraperSetupFailure_Outer", None, {"attempted": False, "used": False, "result": "not_attempted_scraper_setup_failure"}

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

    # Adjust to expect three values from scrape_website
    scraped_items_with_type, status, canonical_url, httpx_meta = await scrape_website(
       test_url,
       test_run_output_dir, # This is the base for the run, scrape_website will make subdirs
       "example_company_test",
       globally_processed_urls_for_test,
       "TEST_ROW_ID_001" # Added placeholder for input_row_id
    )

    if scraped_items_with_type:
        logger.info(f"Test successful: {len(scraped_items_with_type)} page(s) scraped. Status: {status}. Canonical URL: {canonical_url}")
        # Adjust loop to handle the new tuple structure (path, url, type)
        for item_path, source_url, page_type in scraped_items_with_type:
            logger.info(f"  - Saved: {item_path} (from: {source_url}, type: {page_type})")
    else:
        logger.error(f"Test failed: Status: {status}. Canonical URL: {canonical_url}")

# TODO: [FutureEnhancement] The __main__ block below allowed direct execution of _test_scraper.
# Commented out as it's not intended for execution during normal library use.
if __name__ == "__main__":
    # This ensures that if the script is run directly, AppConfig is initialized
    # and logging is set up before _test_scraper is called.
    if not logger.hasHandlers(): 
        setup_logging(logging.INFO) 
    asyncio.run(_test_scraper())