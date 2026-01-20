import pandas as pd
import asyncio
import os
import json
import time
from datetime import datetime
import logging
import re
from urllib.parse import urlparse
from typing import List, Dict, Set, Optional, Any, Tuple, Callable # Added Callable
from collections import Counter

from src.core.config import AppConfig
from src.core.schemas import (
    PhoneNumberLLMOutput,
    CompanyContactDetails,
    ConsolidatedPhoneNumber,
    ConsolidatedPhoneNumberSource,
    HomepageContextOutput,
    DomainExtractionBundle,
)
from src.phone_retrieval.data_handling.consolidator import get_canonical_base_url, process_and_consolidate_contact_data
from src.phone_retrieval.extractors.llm_extractor import GeminiLLMExtractor
from src.phone_retrieval.extractors.regex_extractor import extract_numbers_with_snippets_from_text
from src.phone_retrieval.scraper import scrape_website
from src.phone_retrieval.utils.helpers import log_row_failure
from src.phone_retrieval.processing.url_processor import process_input_url
from src.phone_retrieval.processing.outcome_analyzer import determine_final_row_outcome_and_fault, determine_final_domain_outcome_and_fault

logger = logging.getLogger(__name__)

def _safe_str_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    s = str(val).strip()
    return [s] if s else []


def _summarize_sources_for_number(n: ConsolidatedPhoneNumber) -> tuple[str, str]:
    """Return (types_csv, source_urls_csv) for a ConsolidatedPhoneNumber."""
    types = sorted({(s.type or "").strip() for s in (n.sources or []) if (s.type or "").strip()})
    urls = sorted({(s.original_full_url or "").strip() for s in (n.sources or []) if (s.original_full_url or "").strip()})
    return ", ".join(types), ", ".join(urls)


def _role_score(role: str) -> int:
    r = (role or "").strip().lower()
    if not r:
        return 0
    high = ["ceo", "geschäftsführer", "geschaeftsfuehrer", "founder", "inhaber", "owner", "vorstand", "managing director"]
    sales = ["vertrieb", "sales", "business development", "account", "kundenberater", "key account"]
    tech = ["cto", "it", "technik", "engineering", "product", "produkt"]
    hr = ["hr", "personal", "recruiting"]
    if any(k in r for k in high):
        return 100
    if any(k in r for k in sales):
        return 80
    if any(k in r for k in tech):
        return 50
    if any(k in r for k in hr):
        return 30
    return 10


def _extract_person_contacts(numbers: List[ConsolidatedPhoneNumber]) -> List[Dict[str, Any]]:
    """Flatten person-associated sources into a list of contacts."""
    out: List[Dict[str, Any]] = []
    for n in numbers or []:
        for s in (n.sources or []):
            name = (getattr(s, "associated_person_name", None) or "").strip()
            role = (getattr(s, "associated_person_role", None) or "").strip()
            dept = (getattr(s, "associated_person_department", None) or "").strip()
            is_dd = getattr(s, "is_direct_dial", None)
            # Only treat as a "person contact" if we have an explicit person name OR an explicit direct-dial signal.
            # This avoids incorrectly treating generic department labels as person-associated.
            if not (name or is_dd is True):
                continue
            out.append(
                {
                    "name": name or None,
                    "role": role or None,
                    "department": dept or None,
                    "number": n.number,
                    "classification": n.classification,
                    "type": getattr(s, "type", None),
                    "source_url": getattr(s, "original_full_url", None),
                    "is_direct_dial": is_dd,
                }
            )
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for c in out:
        key = ((c.get("name") or "").lower(), (c.get("role") or "").lower(), (c.get("number") or ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _pick_best_person_contact(contacts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not contacts:
        return None

    def score(c: Dict[str, Any]) -> int:
        s = 0
        if c.get("is_direct_dial") is True:
            s += 40
        s += _role_score(c.get("role") or "")
        cls = (c.get("classification") or "").strip().lower()
        if cls == "primary":
            s += 20
        elif cls == "secondary":
            s += 10
        return s

    return sorted(contacts, key=score, reverse=True)[0]


def _rank_consolidated_numbers_for_outreach(numbers: List[ConsolidatedPhoneNumber]) -> List[ConsolidatedPhoneNumber]:
    """Rank consolidated numbers for operational calling (Top_Number_1..3)."""
    if not numbers:
        return []

    def classification_score(cls: str) -> int:
        c = (cls or "").strip().lower()
        if c == "primary":
            return 100
        if c == "secondary":
            return 80
        if c:
            return 50
        return 0

    def type_score(types_csv: str) -> int:
        t = (types_csv or "").lower()
        if "main" in t or "zentrale" in t or "headquarter" in t:
            return 30
        if "sales" in t or "vertrieb" in t:
            return 20
        if "support" in t or "service" in t:
            return 10
        if "fax" in t:
            return -30
        return 0

    def person_bonus(n: ConsolidatedPhoneNumber) -> int:
        bonus = 0
        for s in (n.sources or []):
            role = getattr(s, "associated_person_role", None) or ""
            name = getattr(s, "associated_person_name", None) or ""
            is_dd = getattr(s, "is_direct_dial", None)
            # Only boost when we have an explicit person name or explicit direct-dial.
            if not (name.strip() or is_dd is True):
                continue
            base = 20  # explicit person/direct-dial is already valuable
            bonus = max(bonus, base + _role_score(role) + (40 if is_dd is True else 0))
        return bonus

    scored: List[tuple[int, ConsolidatedPhoneNumber]] = []
    for n in numbers:
        types_csv, _ = _summarize_sources_for_number(n)
        s = classification_score(n.classification) + type_score(types_csv) + person_bonus(n)
        scored.append((s, n))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [n for _s, n in scored]


def _confidence_for_best(best: Optional[ConsolidatedPhoneNumber]) -> Optional[float]:
    if not best:
        return None
    cls = (best.classification or "").strip().lower()
    if cls == "primary":
        return 0.9
    if cls == "secondary":
        return 0.75
    if cls:
        return 0.6
    return None


def _rank_and_cap_llm_candidates(items: List[Dict[str, Any]], app_config: AppConfig) -> List[Dict[str, Any]]:
    """
    Reduce LLM load by preferring high-signal candidates and capping the total.
    Scoring heuristics:
    - Prefer contact/impressum/legal/about paths
    - Prefer snippets mentioning telephone keywords
    - Prefer numbers that appear multiple times across pages
    """
    if not items:
        return []
    max_total = int(getattr(app_config, "phone_llm_max_candidates_total", 120) or 120)
    prefer_paths = set((getattr(app_config, "phone_llm_prefer_url_path_keywords", []) or []))
    prefer_snip = set((getattr(app_config, "phone_llm_prefer_snippet_keywords", []) or []))

    # Count occurrences per candidate number.
    counts: Counter = Counter()
    for it in items:
        num = (it.get("candidate_number") or it.get("number") or "").strip()
        if num:
            counts[num] += 1

    def score(it: Dict[str, Any]) -> int:
        num = (it.get("candidate_number") or it.get("number") or "").strip()
        src = (it.get("source_url") or "").lower()
        snip = (it.get("snippet") or "").lower()
        s = 0
        if prefer_paths and any(p in src for p in prefer_paths):
            s += 100
        if prefer_snip and any(k in snip for k in prefer_snip):
            s += 30
        if num:
            s += min(50, int(counts.get(num, 1)) * 10)
        return s

    # De-dupe by (number, source_url) to avoid redundant items.
    seen_pairs: Set[tuple] = set()
    deduped: List[Dict[str, Any]] = []
    for it in items:
        num = (it.get("candidate_number") or it.get("number") or "").strip()
        src = (it.get("source_url") or "").strip()
        key = (num, src)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        deduped.append(it)

    deduped.sort(key=score, reverse=True)
    if max_total > 0:
        deduped = deduped[:max_total]
    return deduped


def _phone_results_cache_path(true_base: str, app_config: AppConfig) -> str:
    # Cache key derived from canonical base URL host; stable + filesystem-safe.
    try:
        parsed = urlparse(true_base)
        host = (parsed.netloc or "").split("@")[-1].split(":")[0]
        host = re.sub(r"^www\\.", "", host)
    except Exception:
        host = true_base
    safe = re.sub(r"[^\\w.-]", "_", host)[:80] or "unknown"
    cache_dir = os.path.normpath(getattr(app_config, "phone_results_cache_dir", "cache/phone_results_cache") or "cache/phone_results_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe}.json")


def _load_phone_results_from_cache(true_base: str, app_config: AppConfig) -> Optional[DomainExtractionBundle]:
    """
    Load a cached, consolidated phone result for a canonical base domain.

    This is an optional “resume accelerator” to avoid re-scrape / re-LLM work across runs.
    Controlled by:
    - `REUSE_PHONE_RESULTS_IF_AVAILABLE`
    - `PHONE_RESULTS_CACHE_DIR`
    """
    try:
        if not getattr(app_config, "reuse_phone_results_if_available", False):
            return None
        p = _phone_results_cache_path(true_base, app_config)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f) or {}
        nums = []
        for it in (obj.get("consolidated_numbers") or []):
            try:
                sources = [
                    ConsolidatedPhoneNumberSource(
                        type=s.get("type") or "Unknown",
                        source_path=s.get("source_path") or "",
                        original_full_url=s.get("original_full_url") or "",
                        original_input_company_name=s.get("original_input_company_name"),
                    )
                    for s in (it.get("sources") or [])
                ]
                nums.append(
                    ConsolidatedPhoneNumber(
                        number=it.get("number") or "",
                        classification=it.get("classification") or "Unknown",
                        sources=sources,
                    )
                )
            except Exception:
                continue
        details = CompanyContactDetails(
            company_name=obj.get("company_name"),
            canonical_base_url=true_base,
            consolidated_numbers=nums,
            original_input_urls=obj.get("original_input_urls") or [],
        )
        homepage_ctx = obj.get("homepage_context")
        hc = None
        if isinstance(homepage_ctx, dict):
            try:
                hc = HomepageContextOutput(**homepage_ctx)
            except Exception:
                hc = None
        return DomainExtractionBundle(company_contact_details=details, homepage_context=hc)
    except Exception:
        return None


def _save_phone_results_to_cache(true_base: str, bundle: DomainExtractionBundle, app_config: AppConfig) -> None:
    """
    Persist consolidated phone results per canonical base domain (best-effort).

    This cache is intentionally small and deterministic (1 file per domain) and is only used
    when `REUSE_PHONE_RESULTS_IF_AVAILABLE=True`.
    """
    try:
        p = _phone_results_cache_path(true_base, app_config)
        details = bundle.company_contact_details
        if not details:
            return
        nums_out = []
        for n in (details.consolidated_numbers or []):
            nums_out.append(
                {
                    "number": n.number,
                    "classification": n.classification,
                    "sources": [
                        {
                            "type": s.type,
                            "source_path": s.source_path,
                            "original_full_url": s.original_full_url,
                            "original_input_company_name": s.original_input_company_name,
                        }
                        for s in (n.sources or [])
                    ],
                }
            )
        obj = {
            "ts": datetime.now().isoformat(),
            "company_name": details.company_name,
            "canonical_base_url": details.canonical_base_url,
            "original_input_urls": details.original_input_urls,
            "consolidated_numbers": nums_out,
            "homepage_context": (bundle.homepage_context.model_dump() if bundle.homepage_context and hasattr(bundle.homepage_context, "model_dump") else None),
        }
        tmp = f"{p}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception:
        return


def execute_pipeline_flow(
    df: pd.DataFrame,
    app_config: AppConfig,
    llm_extractor: GeminiLLMExtractor,
    run_output_dir: str,
    llm_context_dir: str,
    run_id: str,
    failure_writer: Any, # csv.writer object
    run_metrics: Dict[str, Any], # To be updated within this function
    original_phone_col_name_for_profile: Optional[str], # Added
    row_queue: Optional[Any] = None,
    row_queue_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, DomainExtractionBundle], Dict[str, str], Dict[str, List[str]], Dict[str, Optional[str]], Dict[str, int]]: # Updated CompanyContactDetails to DomainExtractionBundle
    """
    Executes the core data processing flow of the pipeline.
    This includes URL processing, scraping, extraction, consolidation, and outcome determination.
    """
    globally_processed_urls: Set[str] = set()
    # canonical_site_raw_llm_outputs: Stores LLM outputs per *pathful* canonical URL
    canonical_site_raw_llm_outputs: Dict[str, List[PhoneNumberLLMOutput]] = {}
    # canonical_site_pathful_scraper_status: Stores scraper status per *pathful* canonical URL
    canonical_site_pathful_scraper_status: Dict[str, str] = {}
    # input_to_canonical_map: Maps original input URL to its *true_base* canonical URL
    input_to_canonical_map: Dict[str, Optional[str]] = {}
    # canonical_site_regex_candidates_found_status: Tracks if regex found candidates for a *pathful* canonical URL
    canonical_site_regex_candidates_found_status: Dict[str, bool] = {}
    # canonical_site_llm_exception_details: Stores specific LLM exception details per *pathful* canonical URL
    canonical_site_llm_exception_details: Dict[str, str] = {}
    # canonical_site_regex_candidate_items: Stores regex candidate dicts per *pathful* canonical URL
    canonical_site_regex_candidate_items: Dict[str, List[Dict[str, Any]]] = {}
    # canonical_site_llm_context_paths: Stores context artifact paths per *pathful* canonical URL
    canonical_site_llm_context_paths: Dict[str, List[str]] = {}
    # optional domain-level phone-results cache hits (keyed by true_base)
    cached_domain_bundles_by_true_base: Dict[str, DomainExtractionBundle] = {}

    # Variables for homepage context
    generated_homepage_context_for_domain: Optional[HomepageContextOutput] = None # For the current true_base_domain being processed in the loop
    summarized_true_base_domains: Set[str] = set() # Tracks true_base_domains already summarized
    true_base_to_homepage_context: Dict[str, HomepageContextOutput] = {} # Stores generated context per true_base_domain

    attrition_data_list: List[Dict[str, Any]] = []
    row_level_failure_counts: Dict[str, int] = {}
    
    # --- Data structures for Canonical Domain Journey Report ---
    canonical_domain_journey_data: Dict[str, Dict[str, Any]] = {}
    # Helper dicts (some might be redundant if canonical_domain_journey_data is primary)
    true_base_to_input_row_ids: Dict[str, Set[Any]] = {} # Used to link input rows to a true_base
    # true_base_to_input_company_names: Dict[str, Set[str]] = {} # Populated during global consolidation
    # true_base_to_input_given_urls: Dict[str, Set[str]] = {} # Populated during global consolidation
    true_base_to_pathful_urls_attempted: Dict[str, Set[str]] = {} # Tracks pathful URLs under a true_base

    pass1_loop_start_time = time.time()
    rows_processed_in_pass1 = 0
    rows_failed_in_pass1 = 0

    # Pre-fetch company name and URL column names from AppConfig or use defaults
    # This assumes these columns are present after load_and_preprocess_data
    active_profile = app_config.INPUT_COLUMN_PROFILES.get(app_config.input_file_profile_name, app_config.INPUT_COLUMN_PROFILES['default'])
    company_name_col_key = active_profile.get('CompanyName', 'CompanyName')
    url_col_key = active_profile.get('GivenURL', 'GivenURL')


    for i, (index, row_series) in enumerate(df.iterrows()):
        rows_processed_in_pass1 += 1
        row: pd.Series = row_series
        company_name: str = str(row.get(company_name_col_key, f"Row_{index}"))
        given_url_original: Optional[str] = row.get(url_col_key)
        current_row_number_for_log: int = i + 1
        
        log_identifier = f"[RowID: {index}, Company: {company_name}]"
        logger.info(f"{log_identifier} --- Processing row {current_row_number_for_log}/{len(df)}: Original URL '{given_url_original}' ---")

        current_row_scraper_status: str = "Not_Run"
        final_canonical_entry_url: Optional[str] = None # This will be the *pathful* canonical URL from scraper
        
        # --- URL Processing ---
        processed_url, url_status = process_input_url(
            given_url_original,
            app_config.url_probing_tlds,
            log_identifier
        )

        if url_status == "InvalidURL":
            df.at[index, 'ScrapingStatus'] = 'InvalidURL'
            current_row_scraper_status = 'InvalidURL'
            # df.at[index, 'VerificationStatus'] = 'Skipped_InvalidURL' # This column might be determined later
            run_metrics["scraping_stats"]["scraping_failure_invalid_url"] += 1
            log_row_failure(
                failure_log_writer=failure_writer,
                input_row_identifier=index,
                company_name=company_name,
                given_url=given_url_original,
                stage_of_failure="URL_Validation_InvalidOrMissing",
                error_reason=f"Invalid or missing URL after processing: {processed_url}",
                log_timestamp=datetime.now().isoformat(),
                error_details=json.dumps({"original_url": given_url_original, "processed_url": processed_url}),
            )
            stage_key = "URL_Validation_InvalidOrMissing"
            row_level_failure_counts[stage_key] = row_level_failure_counts.get(stage_key, 0) + 1
            rows_failed_in_pass1 +=1
            continue
        # --- End URL Processing ---

        try:
            # Ensure processed_url is not None here, due to the 'continue' above
            assert processed_url is not None 
            
            run_metrics["scraping_stats"]["urls_processed_for_scraping"] += 1
            scrape_task_start_time = time.time()
            
            scraped_pages_details: List[Tuple[str, str, str]]
            scraper_status: str
            
            # scrape_website returns:
            # (List[Tuple[str, str, str]], str, Optional[str], Dict[str, Any])
            # scraped_pages_details, scraper_status, final_canonical_entry_url_from_scraper, httpx_fallback_meta
            scraped_pages_details, scraper_status, final_canonical_entry_url, httpx_fallback_meta = asyncio.run(
                scrape_website(processed_url, run_output_dir, company_name, globally_processed_urls, index) # Pass app_config
            )
            run_metrics["tasks"].setdefault("scrape_website_total_duration_seconds", 0)
            run_metrics["tasks"]["scrape_website_total_duration_seconds"] += (time.time() - scrape_task_start_time)

            df.at[index, 'ScrapingStatus'] = scraper_status
            true_base_domain_for_row = get_canonical_base_url(final_canonical_entry_url) if final_canonical_entry_url else None
            df.at[index, 'CanonicalEntryURL'] = true_base_domain_for_row # This is the true_base
            current_row_scraper_status = scraper_status

            # --- Optional: reuse cached consolidated phone results by canonical base ---
            if true_base_domain_for_row and getattr(app_config, "reuse_phone_results_if_available", False):
                if true_base_domain_for_row not in cached_domain_bundles_by_true_base:
                    cached_bundle = _load_phone_results_from_cache(true_base_domain_for_row, app_config)
                    if cached_bundle and cached_bundle.company_contact_details and (cached_bundle.company_contact_details.consolidated_numbers or []):
                        cached_domain_bundles_by_true_base[true_base_domain_for_row] = cached_bundle
                        # Mark this pathful as already "handled" so we don't do regex/LLM for it.
                        if final_canonical_entry_url and final_canonical_entry_url not in canonical_site_raw_llm_outputs:
                            canonical_site_raw_llm_outputs[final_canonical_entry_url] = []
                            canonical_site_regex_candidates_found_status[final_canonical_entry_url] = False
                            canonical_site_pathful_scraper_status[final_canonical_entry_url] = "Success_PhoneResultsCacheHit"
                        logger.info(f"{log_identifier} Phone-results cache hit for {true_base_domain_for_row}; skipping regex/LLM extraction.")

            # --- Instrumentation: httpx fallback attempted/used/result for this row ---
            try:
                if "HttpFallbackAttempted" not in df.columns:
                    df["HttpFallbackAttempted"] = None
                if "HttpFallbackUsed" not in df.columns:
                    df["HttpFallbackUsed"] = None
                if "HttpFallbackResult" not in df.columns:
                    df["HttpFallbackResult"] = None

                attempted_httpx = bool(httpx_fallback_meta.get("attempted")) if isinstance(httpx_fallback_meta, dict) else False
                used_httpx = bool(httpx_fallback_meta.get("used")) if isinstance(httpx_fallback_meta, dict) else False
                result_httpx = None
                if isinstance(httpx_fallback_meta, dict) and httpx_fallback_meta.get("result") is not None:
                    result_httpx = str(httpx_fallback_meta.get("result"))

                df.at[index, "HttpFallbackAttempted"] = "Yes" if attempted_httpx else "No"
                df.at[index, "HttpFallbackUsed"] = "Yes" if used_httpx else "No"
                df.at[index, "HttpFallbackResult"] = result_httpx or ("success" if used_httpx else ("not_attempted" if not attempted_httpx else "failed_unknown"))
            except Exception:
                pass

            # --- Homepage Summarization Logic ---
            if true_base_domain_for_row and \
               (app_config.extraction_profile in ["minimal_plus_summary", "enriched_direct"]) and \
               (true_base_domain_for_row not in summarized_true_base_domains) and \
               final_canonical_entry_url and scraped_pages_details:

                homepage_content_file_path: Optional[str] = None
                for page_file, source_url, page_type in scraped_pages_details:
                    if source_url == final_canonical_entry_url: # final_canonical_entry_url is the pathful homepage URL
                        homepage_content_file_path = page_file
                        break
                
                if homepage_content_file_path and os.path.exists(homepage_content_file_path):
                    try:
                        with open(homepage_content_file_path, 'r', encoding='utf-8') as hf:
                            homepage_content_text = hf.read()
                        
                        safe_true_base_name = "".join(c if c.isalnum() else "_" for c in true_base_domain_for_row.replace("http://","").replace("https://",""))[:100]
                        logger.info(f"{log_identifier} Generating homepage context for {true_base_domain_for_row} (safe name: {safe_true_base_name}).")
                        
                        # Reset for current domain before potential generation
                        # The variable 'generated_homepage_context_for_domain' is no longer used for the tuple.
                        # Instead, we'll use homepage_context_tuple_result and then extract the first element.
                        
                        homepage_context_tuple_result: Optional[Tuple[Optional[HomepageContextOutput], Optional[str], Optional[Dict[str, int]]]] = llm_extractor.generate_homepage_context(
                            homepage_content=homepage_content_text,
                            prompt_template_path=app_config.prompt_path_homepage_context,
                            llm_context_dir=llm_context_dir, # Ensure llm_context_dir is appropriate
                            file_identifier_prefix=f"HOMEPAGE_CTX_{safe_true_base_name}",
                            triggering_input_row_id=index, # type: ignore
                            triggering_company_name=company_name
                        )
                        
                        actual_homepage_context: Optional[HomepageContextOutput] = None
                        raw_response_str_hp_ctx: Optional[str] = None
                        token_stats_hp_ctx: Optional[Dict[str, int]] = None

                        if homepage_context_tuple_result:
                            actual_homepage_context = homepage_context_tuple_result[0]
                            raw_response_str_hp_ctx = homepage_context_tuple_result[1]
                            token_stats_hp_ctx = homepage_context_tuple_result[2]
                            
                            # Log raw_response_str and token_stats if they exist and logging is desired
                            if raw_response_str_hp_ctx:
                                logger.debug(f"{log_identifier} Homepage context raw response for {true_base_domain_for_row} (first 100 chars): {raw_response_str_hp_ctx[:100]}")
                            if token_stats_hp_ctx:
                                logger.debug(f"{log_identifier} Homepage context token stats for {true_base_domain_for_row}: {token_stats_hp_ctx}")
                        
                        if actual_homepage_context:
                            true_base_to_homepage_context[true_base_domain_for_row] = actual_homepage_context
                            summarized_true_base_domains.add(true_base_domain_for_row)
                            logger.info(f"{log_identifier} Successfully generated and stored homepage context object for {true_base_domain_for_row}.")
                            # Optionally log details of the generated context if needed (e.g., actual_homepage_context.company_name)
                        else:
                            logger.warning(f"{log_identifier} Homepage context generation resulted in None (actual_homepage_context is None) for {true_base_domain_for_row}.")
                    except Exception as e_summary:
                        logger.error(f"{log_identifier} Error during homepage summarization for {true_base_domain_for_row}: {e_summary}", exc_info=True)
                elif homepage_content_file_path:
                     logger.warning(f"{log_identifier} Homepage content file not found for summarization: {homepage_content_file_path} for {true_base_domain_for_row}")
                else:
                    logger.info(f"{log_identifier} No matching homepage content file found in scraped_pages_details for {final_canonical_entry_url} (true base: {true_base_domain_for_row}) for summarization.")
            elif true_base_domain_for_row and true_base_domain_for_row in summarized_true_base_domains:
                 logger.info(f"{log_identifier} Homepage context for {true_base_domain_for_row} already generated in this run.")
            # --- End Homepage Summarization Logic ---
            
            given_url_original_str_key = str(given_url_original) if given_url_original is not None else "None_GivenURL_Input"
            input_to_canonical_map[given_url_original_str_key] = true_base_domain_for_row

            # --- Populate/Initialize structures for Canonical Domain Journey Report ---
            if true_base_domain_for_row:
                if true_base_domain_for_row not in canonical_domain_journey_data:
                    canonical_domain_journey_data[true_base_domain_for_row] = {
                        "Input_Row_IDs": set(), "Input_CompanyNames": set(), "Input_GivenURLs": set(),
                        "Pathful_URLs_Attempted_List": set(), "Overall_Scraper_Status_For_Domain": "Unknown",
                        "Scraped_Pages_Details_Aggregated": Counter(), "Total_Pages_Scraped_For_Domain": 0,
                        "Regex_Candidates_Found_For_Any_Pathful": False, "LLM_Calls_Made_For_Domain": False,
                        "LLM_Total_Raw_Numbers_Extracted": 0, "LLM_Total_Consolidated_Numbers_Found": 0,
                        "LLM_Consolidated_Number_Types_Summary": Counter(),
                        "LLM_Processing_Error_Encountered_For_Domain": False, "LLM_Error_Messages_Aggregated": [],
                        "Final_Domain_Outcome_Reason": "Unknown", "Primary_Fault_Category_For_Domain": "Unknown"
                    }
                canonical_domain_journey_data[true_base_domain_for_row]["Input_Row_IDs"].add(index)
                canonical_domain_journey_data[true_base_domain_for_row]["Input_CompanyNames"].add(company_name)
                if given_url_original:
                     canonical_domain_journey_data[true_base_domain_for_row]["Input_GivenURLs"].add(given_url_original)
                if final_canonical_entry_url: # This is pathful
                    canonical_domain_journey_data[true_base_domain_for_row]["Pathful_URLs_Attempted_List"].add(final_canonical_entry_url)
                
                true_base_to_input_row_ids.setdefault(true_base_domain_for_row, set()).add(index)
                if final_canonical_entry_url: # This is pathful
                    true_base_to_pathful_urls_attempted.setdefault(true_base_domain_for_row, set()).add(final_canonical_entry_url)

            logger.info(f"{log_identifier} Row {current_row_number_for_log}: Scraper status: {current_row_scraper_status}, Pathful Canonical URL: {final_canonical_entry_url}, True Base Domain: {true_base_domain_for_row}")

            if current_row_scraper_status == "Success" and final_canonical_entry_url:
                # final_canonical_entry_url is the *pathful* URL that was successfully scraped
                if final_canonical_entry_url not in canonical_site_raw_llm_outputs: # Process only if new pathful canonical
                    run_metrics["scraping_stats"]["new_canonical_sites_scraped"] += 1 # This counts new pathfuls
                    run_metrics["regex_extraction_stats"]["sites_processed_for_regex"] += 1 # For this pathful
                    regex_extraction_task_start_time = time.time()

                    logger.info(f"{log_identifier} Processing new pathful canonical URL for LLM: {final_canonical_entry_url}")
                    all_candidate_items_for_llm: List[Dict[str, str]] = []
                    if scraped_pages_details:
                        run_metrics["scraping_stats"]["total_pages_scraped_overall"] += len(scraped_pages_details)
                        # total_successful_canonical_scrapes should count unique *true_base* domains successfully scraped.
                        # This is better handled after the loop when true_base_scraper_status is finalized.

                        target_codes_raw: Any = row.get('TargetCountryCodes', [])
                        target_codes_list_for_regex: List[str] = []
                        if isinstance(target_codes_raw, str) and target_codes_raw.startswith('[') and target_codes_raw.endswith(']'):
                            try: import ast; parsed_eval = ast.literal_eval(target_codes_raw)
                            except (ValueError, SyntaxError): logger.warning(f"{log_identifier} Could not parse TargetCountryCodes string: {target_codes_raw}.")
                            else:
                                if isinstance(parsed_eval, list): target_codes_list_for_regex = [str(item) for item in parsed_eval if isinstance(item, (str, int))]
                        elif isinstance(target_codes_raw, list):
                            target_codes_list_for_regex = [str(item) for item in target_codes_raw if isinstance(item, (str, int))]

                        for page_content_file, source_page_url, page_type in scraped_pages_details:
                            run_metrics["scraping_stats"]["pages_scraped_by_type"][page_type] = \
                                run_metrics["scraping_stats"]["pages_scraped_by_type"].get(page_type, 0) + 1
                            if true_base_domain_for_row and true_base_domain_for_row in canonical_domain_journey_data:
                                canonical_domain_journey_data[true_base_domain_for_row]["Scraped_Pages_Details_Aggregated"][page_type] += 1
                                canonical_domain_journey_data[true_base_domain_for_row]["Total_Pages_Scraped_For_Domain"] += 1
                            
                            if os.path.exists(page_content_file):
                                try:
                                    with open(page_content_file, 'r', encoding='utf-8') as f_content: text_content = f_content.read()
                                    page_candidate_items = extract_numbers_with_snippets_from_text(
                                        text_content=text_content, source_url=source_page_url,
                                        original_input_company_name=company_name, target_country_codes=target_codes_list_for_regex,
                                        snippet_window_chars=app_config.snippet_window_chars
                                    )
                                    # Filter duplicates per page
                                    filtered_page_candidates: List[Dict[str, str]] = []
                                    number_counts_on_page: Dict[str, int] = Counter()
                                    for candidate in page_candidate_items:
                                        number_str = candidate.get('number') or candidate.get('candidate_number')
                                        if number_str:
                                            if number_counts_on_page[number_str] < app_config.max_identical_numbers_per_page_to_llm:
                                                filtered_page_candidates.append(candidate)
                                                number_counts_on_page[number_str] += 1
                                        else: filtered_page_candidates.append(candidate)
                                    all_candidate_items_for_llm.extend(filtered_page_candidates)
                                except Exception as file_read_exc:
                                    logger.error(f"{log_identifier} Error reading {page_content_file}: {file_read_exc}", exc_info=True)
                                    # Log failure specific to this stage
                            else: logger.warning(f"{log_identifier} Scraped page content file not found: {page_content_file}")
                        
                        run_metrics["tasks"].setdefault("regex_extraction_total_duration_seconds", 0)
                        run_metrics["tasks"]["regex_extraction_total_duration_seconds"] += (time.time() - regex_extraction_task_start_time)
                        
                        if all_candidate_items_for_llm:
                            run_metrics["regex_extraction_stats"]["sites_with_regex_candidates"] += 1 # Pathful site
                            run_metrics["regex_extraction_stats"]["total_regex_candidates_found"] += len(all_candidate_items_for_llm)
                            canonical_site_regex_candidates_found_status[final_canonical_entry_url] = True
                            canonical_site_regex_candidate_items[final_canonical_entry_url] = list(all_candidate_items_for_llm)
                            if true_base_domain_for_row and true_base_domain_for_row in canonical_domain_journey_data:
                                canonical_domain_journey_data[true_base_domain_for_row]["Regex_Candidates_Found_For_Any_Pathful"] = True
                        else:
                            canonical_site_regex_candidates_found_status[final_canonical_entry_url] = False
                            canonical_site_regex_candidate_items[final_canonical_entry_url] = []
                        logger.info(f"{log_identifier} Found {len(all_candidate_items_for_llm)} regex candidates for LLM for {final_canonical_entry_url}.")

                    # LLM Processing for the current *pathful* canonical URL
                    if canonical_site_regex_candidates_found_status.get(final_canonical_entry_url, False):
                        run_metrics["llm_processing_stats"]["sites_processed_for_llm"] += 1 # Pathful site
                        if true_base_domain_for_row and true_base_domain_for_row in canonical_domain_journey_data:
                            canonical_domain_journey_data[true_base_domain_for_row]["LLM_Calls_Made_For_Domain"] = True
                        llm_task_start_time = time.time()
                        try:
                            prompt_template_abs_path = app_config.prompt_path_minimal_classification # Corrected path
                            if not os.path.isabs(prompt_template_abs_path):
                                project_root_dir_local = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up two levels from processing to project root
                                prompt_template_abs_path = os.path.join(project_root_dir_local, prompt_template_abs_path)
                            
                            if not os.path.exists(prompt_template_abs_path):
                                logger.error(f"{log_identifier} LLM prompt template missing: {prompt_template_abs_path}")
                                canonical_site_raw_llm_outputs[final_canonical_entry_url] = []
                                canonical_site_pathful_scraper_status[final_canonical_entry_url] = "Error_LLM_PromptMissing"
                                run_metrics["llm_processing_stats"]["llm_calls_failure_prompt_missing"] += 1
                                # Log failure
                            else:
                                safe_canonical_name = "".join(c if c.isalnum() else "_" for c in final_canonical_entry_url.replace("http://","").replace("https://",""))[:100]

                                # Rank/dedupe/cap candidates before LLM to reduce cost and speed up runs.
                                all_candidate_items_for_llm = _rank_and_cap_llm_candidates(all_candidate_items_for_llm, app_config)
                                canonical_site_regex_candidate_items[final_canonical_entry_url] = list(all_candidate_items_for_llm)
                                run_metrics["llm_processing_stats"]["total_llm_candidates_sent"] = run_metrics["llm_processing_stats"].get("total_llm_candidates_sent", 0) + len(all_candidate_items_for_llm)

                                llm_input_filename = f"PATHFUL_{safe_canonical_name}_llm_input.json" # Clarify pathful
                                llm_input_filepath = os.path.join(llm_context_dir, llm_input_filename)
                                canonical_site_llm_context_paths.setdefault(final_canonical_entry_url, []).append(llm_input_filepath)
                                try:
                                    with open(llm_input_filepath, 'w', encoding='utf-8') as f_in: json.dump(all_candidate_items_for_llm, f_in, indent=2)
                                except IOError as e: logger.error(f"{log_identifier} IOError saving LLM input: {e}")

                                current_homepage_context_input: Optional[HomepageContextOutput] = None
                                if app_config.extraction_profile in ["minimal_plus_summary", "enriched_direct"] and true_base_domain_for_row:
                                    current_homepage_context_input = true_base_to_homepage_context.get(true_base_domain_for_row)
                                    if current_homepage_context_input:
                                        logger.info(f"{log_identifier} Passing homepage context for {true_base_domain_for_row} to extract_phone_numbers.")
                                    else:
                                        logger.info(f"{log_identifier} Homepage context for {true_base_domain_for_row} not found in cache for extract_phone_numbers, passing None.")
                                elif app_config.extraction_profile == "minimal":
                                     logger.info(f"{log_identifier} Extraction profile is 'minimal', passing None for homepage_context_input.")


                                llm_classified_outputs, llm_raw_response, token_stats = llm_extractor.extract_phone_numbers(
                                    candidate_items=all_candidate_items_for_llm,
                                    homepage_context_input=current_homepage_context_input, # Pass the homepage context
                                    llm_context_dir=llm_context_dir,
                                    file_identifier_prefix=f"PATHFUL_{safe_canonical_name}",
                                    triggering_input_row_id=index, # type: ignore
                                    triggering_company_name=company_name
                                )
                                canonical_site_raw_llm_outputs[final_canonical_entry_url] = llm_classified_outputs
                                canonical_site_pathful_scraper_status[final_canonical_entry_url] = current_row_scraper_status # Store success or specific scrape status
                                run_metrics["llm_processing_stats"]["llm_calls_success"] += 1
                                run_metrics["llm_processing_stats"]["total_llm_extracted_numbers_raw"] += len(llm_classified_outputs)
                                if true_base_domain_for_row and true_base_domain_for_row in canonical_domain_journey_data:
                                    canonical_domain_journey_data[true_base_domain_for_row]["LLM_Total_Raw_Numbers_Extracted"] += len(llm_classified_outputs)
                                if token_stats: # Update token stats in run_metrics
                                    run_metrics["llm_processing_stats"]["llm_successful_calls_with_token_data"] += 1
                                    run_metrics["llm_processing_stats"]["total_llm_prompt_tokens"] += token_stats.get("prompt_tokens", 0)
                                    run_metrics["llm_processing_stats"]["total_llm_completion_tokens"] += token_stats.get("completion_tokens", 0)
                                    run_metrics["llm_processing_stats"]["total_llm_tokens_overall"] += token_stats.get("total_tokens", 0)

                                llm_raw_output_filename = f"PATHFUL_{safe_canonical_name}_llm_raw_output.json"
                                llm_raw_output_filepath = os.path.join(llm_context_dir, llm_raw_output_filename)
                                canonical_site_llm_context_paths.setdefault(final_canonical_entry_url, []).append(llm_raw_output_filepath)
                                try:
                                    with open(llm_raw_output_filepath, 'w', encoding='utf-8') as f_llm_out:
                                        f_llm_out.write(llm_raw_response if isinstance(llm_raw_response, str) else json.dumps(llm_raw_response or {}, indent=2))
                                except IOError as e: logger.error(f"{log_identifier} IOError saving LLM raw output: {e}")
                        except Exception as llm_exc:
                            logger.error(f"{log_identifier} Error during LLM processing for {final_canonical_entry_url}: {llm_exc}", exc_info=True)
                            canonical_site_raw_llm_outputs[final_canonical_entry_url] = []
                            canonical_site_pathful_scraper_status[final_canonical_entry_url] = "Error_LLM_Processing"
                            exc_type_name = type(llm_exc).__name__; exc_msg = str(llm_exc)
                            canonical_site_llm_exception_details[final_canonical_entry_url] = f"{exc_type_name}: {exc_msg}"
                            if true_base_domain_for_row and true_base_domain_for_row in canonical_domain_journey_data:
                                canonical_domain_journey_data[true_base_domain_for_row]["LLM_Processing_Error_Encountered_For_Domain"] = True
                                canonical_domain_journey_data[true_base_domain_for_row]["LLM_Error_Messages_Aggregated"].append(f"PathfulURL ({final_canonical_entry_url}): {exc_type_name}: {exc_msg}")
                            run_metrics["llm_processing_stats"]["llm_calls_failure_processing_error"] += 1
                            # Log failure
                        run_metrics["tasks"].setdefault("llm_extraction_total_duration_seconds", 0)
                        run_metrics["tasks"]["llm_extraction_total_duration_seconds"] += (time.time() - llm_task_start_time)
                    else: # No regex candidates for this pathful URL
                        logger.info(f"{log_identifier} No regex candidates for LLM from {final_canonical_entry_url}. LLM not called.")
                        canonical_site_raw_llm_outputs[final_canonical_entry_url] = []
                        canonical_site_pathful_scraper_status[final_canonical_entry_url] = current_row_scraper_status # Preserve scraper status
                        if final_canonical_entry_url not in run_metrics["llm_processing_stats"].get("sites_already_attempted_llm_or_skipped", set()):
                             run_metrics["llm_processing_stats"]["llm_no_candidates_to_process"] += 1
                             run_metrics["llm_processing_stats"].setdefault("sites_already_attempted_llm_or_skipped", set()).add(final_canonical_entry_url)
                else: # Data for this pathful canonical URL already cached
                    logger.info(f"{log_identifier} Raw LLM data for pathful canonical {final_canonical_entry_url} already cached.")
            
            elif current_row_scraper_status != "Success": # Scraper failed for this input's URL
                logger.info(f"{log_identifier} Row {current_row_number_for_log}: Scraper status '{current_row_scraper_status}'. No LLM processing.")
                if "Already_Processed" in current_row_scraper_status: run_metrics["scraping_stats"]["scraping_failure_already_processed"] += 1
                elif "InvalidURL" not in current_row_scraper_status : run_metrics["scraping_stats"]["scraping_failure_error"] += 1
                
                if final_canonical_entry_url and final_canonical_entry_url not in canonical_site_pathful_scraper_status:
                    canonical_site_pathful_scraper_status[final_canonical_entry_url] = current_row_scraper_status
                # Log failure for this row based on scraper status
                log_row_failure(failure_writer, index, company_name, given_url_original, f"Scraping_{current_row_scraper_status}",
                                f"Scraper status: {current_row_scraper_status}", datetime.now().isoformat(),
                                json.dumps({"pathful_canonical_url": final_canonical_entry_url, "true_base_domain": true_base_domain_for_row}),
                                associated_pathful_canonical_url=final_canonical_entry_url)
                stage_key = f"Scraping_{current_row_scraper_status}"
                row_level_failure_counts[stage_key] = row_level_failure_counts.get(stage_key, 0) + 1
                rows_failed_in_pass1 +=1
            
            if current_row_scraper_status == "Success": run_metrics["scraping_stats"]["scraping_success"] += 1
            logger.info(f"{log_identifier} Row {current_row_number_for_log}: Pass 1 processing complete.")

        except Exception as e:
            logger.error(f"{log_identifier} Error during Pass 1 for row {current_row_number_for_log}: {e}", exc_info=True)
            # Update df status, run_metrics, log failure
            df.at[index, 'ScrapingStatus'] = df.at[index, 'ScrapingStatus'] if df.at[index, 'ScrapingStatus'] else f'PipelineError_{type(e).__name__}'
            run_metrics["errors_encountered"].append(f"Pass 1 error for {company_name} (URL: {given_url_original}): {str(e)}")
            log_row_failure(failure_writer, index, company_name, given_url_original, "RowProcessing_Pass1_UnhandledException",
                            "Unhandled exception in Pass 1", datetime.now().isoformat(),
                            json.dumps({"exception_type": type(e).__name__, "exception_message": str(e)}),
                            associated_pathful_canonical_url=final_canonical_entry_url) # final_canonical_entry_url might be None
            stage_key = "RowProcessing_Pass1_UnhandledException"
            row_level_failure_counts[stage_key] = row_level_failure_counts.get(stage_key, 0) + 1
            rows_failed_in_pass1 +=1

    run_metrics["tasks"]["pass1_main_loop_duration_seconds"] = time.time() - pass1_loop_start_time
    run_metrics["data_processing_stats"]["rows_successfully_processed_pass1"] = rows_processed_in_pass1 - rows_failed_in_pass1
    run_metrics["data_processing_stats"]["rows_failed_pass1"] = rows_failed_in_pass1
    run_metrics["data_processing_stats"]["row_level_failure_summary"] = row_level_failure_counts
    logger.info(f"Pass 1 (Scraping & Raw LLM Data Collection) complete. Processed {rows_processed_in_pass1} rows.")

    # --- Global Consolidation of LLM data by True Base Domain ---
    global_consolidation_start_time = time.time()
    logger.info("Starting Global Consolidation of LLM data by True Base Domain...")
    final_consolidated_data_by_true_base: Dict[str, DomainExtractionBundle] = {} # Changed to DomainExtractionBundle
    true_base_to_pathful_map: Dict[str, List[str]] = {} # Maps true_base to its list of pathful URLs
    true_base_scraper_status: Dict[str, str] = {} # Overall scraper status for a true_base
    true_base_to_input_company_names_agg: Dict[str, Set[str]] = {} # Aggregated company names for a true_base

    # Populate true_base_to_pathful_map and determine initial true_base_scraper_status
    for pathful_url_key, _ in canonical_site_raw_llm_outputs.items(): # Iterate over pathfuls that had LLM data (even if empty)
        true_base = get_canonical_base_url(pathful_url_key)
        if not true_base: continue
        
        true_base_to_pathful_map.setdefault(true_base, []).append(pathful_url_key)
        current_pathful_status = canonical_site_pathful_scraper_status.get(pathful_url_key, "Unknown")

        # Determine overall scraper status for the true_base (prefer Success, then non-Error, then Error)
        if true_base not in true_base_scraper_status or \
           (current_pathful_status == "Success" and true_base_scraper_status[true_base] != "Success") or \
           (true_base_scraper_status[true_base] != "Success" and "Error" not in current_pathful_status and "Error" in true_base_scraper_status.get(true_base, "Unknown")):
            true_base_scraper_status[true_base] = current_pathful_status
        
        # Update canonical_domain_journey_data's Overall_Scraper_Status_For_Domain
        if true_base in canonical_domain_journey_data:
            current_domain_overall_status = canonical_domain_journey_data[true_base].get("Overall_Scraper_Status_For_Domain", "Unknown")
            if current_domain_overall_status == "Unknown" or \
               (current_pathful_status == "Success" and current_domain_overall_status != "Success") or \
               (current_domain_overall_status != "Success" and "Error" not in current_pathful_status and "Error" in current_domain_overall_status) : # Check if current_domain_overall_status is an error status
                canonical_domain_journey_data[true_base]["Overall_Scraper_Status_For_Domain"] = current_pathful_status


    # Aggregate input company names for each true_base_domain
    if 'CanonicalEntryURL' in df.columns and company_name_col_key in df.columns:
        for true_b_domain_key in true_base_to_pathful_map.keys(): # Iterate over true_bases that had processed pathfuls
            mask = (df['CanonicalEntryURL'].notna()) & (df['CanonicalEntryURL'] == true_b_domain_key)
            # Select column first to get a Series, then apply mask. This is clearer for Pylance.
            company_series: pd.Series = df[company_name_col_key]
            matching_companies_series = company_series[mask]
            cleaned_series = matching_companies_series.dropna()
            if not cleaned_series.empty:
                unique_names = cleaned_series.astype(str).unique()
                true_base_to_input_company_names_agg[true_b_domain_key] = set(unique_names)


    for true_base_domain, list_of_pathful_urls in true_base_to_pathful_map.items():
        all_llm_results_for_this_true_base: List[PhoneNumberLLMOutput] = []
        for pathful_url_item in list_of_pathful_urls:
            all_llm_results_for_this_true_base.extend(canonical_site_raw_llm_outputs.get(pathful_url_item, []))
        
        representative_company_name = "Unknown"
        if true_base_to_input_company_names_agg.get(true_base_domain):
            representative_company_name = sorted(list(true_base_to_input_company_names_agg[true_base_domain]))[0]
        
        # Initialize DomainExtractionBundle for the true_base_domain
        current_bundle = DomainExtractionBundle(
            company_contact_details=None, # Will be populated next
            homepage_context=None # Will be populated from true_base_to_homepage_context
        )

        # Populate company_contact_details
        current_bundle.company_contact_details = process_and_consolidate_contact_data(
            llm_results=all_llm_results_for_this_true_base,
            company_name_from_input=representative_company_name,
            initial_given_url=true_base_domain
        )
        
        # Populate homepage_context if available
        if true_base_domain in true_base_to_homepage_context:
            current_bundle.homepage_context = true_base_to_homepage_context[true_base_domain]
            logger.info(f"Assigned cached homepage context to DomainExtractionBundle for {true_base_domain}.")
        else:
            logger.info(f"No cached homepage context found for {true_base_domain} during global consolidation.")

        final_consolidated_data_by_true_base[true_base_domain] = current_bundle
        
        # Update journey data based on the company_contact_details within the bundle
        if true_base_domain in canonical_domain_journey_data and current_bundle.company_contact_details:
            consolidated_details = current_bundle.company_contact_details
            if consolidated_details.consolidated_numbers:
                canonical_domain_journey_data[true_base_domain]["LLM_Total_Consolidated_Numbers_Found"] = len(consolidated_details.consolidated_numbers)
                for cn_item in consolidated_details.consolidated_numbers:
                    for source_detail in cn_item.sources:
                        if source_detail.type:
                             canonical_domain_journey_data[true_base_domain]["LLM_Consolidated_Number_Types_Summary"][source_detail.type] += 1
    
    logger.info(f"Global Consolidation complete. {len(final_consolidated_data_by_true_base)} true base domains processed.")
    run_metrics["tasks"]["global_consolidation_duration_seconds"] = time.time() - global_consolidation_start_time
    run_metrics["data_processing_stats"]["unique_true_base_domains_consolidated"] = len(final_consolidated_data_by_true_base)

    # Merge in any domain-level cache hits (skip regex/LLM runs entirely for those).
    if cached_domain_bundles_by_true_base:
        for tb, bundle in cached_domain_bundles_by_true_base.items():
            if tb and tb not in final_consolidated_data_by_true_base:
                final_consolidated_data_by_true_base[tb] = bundle
                true_base_scraper_status.setdefault(tb, "Success_PhoneResultsCacheHit")
                true_base_to_pathful_map.setdefault(tb, [])
    
    # Update total_successful_canonical_scrapes metric based on true_base_scraper_status
    successful_true_base_scrapes = sum(1 for status in true_base_scraper_status.values() if status == "Success")
    run_metrics["scraping_stats"]["total_successful_canonical_scrapes"] = successful_true_base_scrapes


    # --- Determine Final Outcome and Fault Category for each Canonical Domain ---
    logger.info("Determining final outcomes for each canonical domain...")
    for domain_key, journey_data_entry in canonical_domain_journey_data.items():
        # For canonical_site_regex_candidates_found_status, use the aggregated value from journey_data
        # For canonical_site_llm_exception_details, this needs to be aggregated or passed carefully if needed by domain outcome
        final_domain_reason, final_domain_fault = determine_final_domain_outcome_and_fault(
            true_base_domain=domain_key,
            domain_journey_entry=journey_data_entry,
            true_base_scraper_status_map=true_base_scraper_status,
            true_base_to_pathful_map=true_base_to_pathful_map,
            canonical_site_pathful_scraper_status=canonical_site_pathful_scraper_status,
            final_consolidated_data=(
                bundle.company_contact_details
                if (bundle := final_consolidated_data_by_true_base.get(domain_key))
                else None
            )
        )
        canonical_domain_journey_data[domain_key]["Final_Domain_Outcome_Reason"] = final_domain_reason
        canonical_domain_journey_data[domain_key]["Primary_Fault_Category_For_Domain"] = final_domain_fault
    logger.info("Final domain outcome determination complete.")


    # --- Determine Final Row Outcome Reason and Fault Category for each input row ---
    logger.info("Determining final outcomes for each input row...")
    # Pre-calculate input duplicate counts (simplified version, assuming main_pipeline did the full one)
    # These are needed for the attrition report if generated from here.
    # For now, assume these counts (company_name_counts, input_canonical_url_counts) are passed or handled by main.
    # If not, they would need to be calculated here or passed as arguments.
    # For simplicity, this example omits their direct calculation here.
    # We will need `company_name_col_key` and `url_col_key` from app_config.
    
    # Placeholder for pre-computed duplicate counts if needed by attrition logic here
    company_name_counts_placeholder = Counter(df[company_name_col_key].astype(str).str.strip())
    df['temp_derived_input_canonical'] = df[url_col_key].apply(lambda x: get_canonical_base_url(x) or "MISSING_OR_INVALID_URL_INPUT")
    input_canonical_url_counts_placeholder = Counter(df['temp_derived_input_canonical'])


    for index, row_summary in df.iterrows():
        company_name_for_attrition = str(row_summary.get(company_name_col_key, f"Row_{index}"))
        given_url_original_for_attrition = str(row_summary.get(url_col_key)) if pd.notna(row_summary.get(url_col_key)) else ""
        canonical_url_summary = row_summary.get('CanonicalEntryURL') # This is true_base

        df_status_snapshot_for_helper = {'ScrapingStatus': row_summary.get('ScrapingStatus')} # Initial scrape status for input URL

        domain_bundle_summary = final_consolidated_data_by_true_base.get(str(canonical_url_summary)) if canonical_url_summary else None
        company_contact_details_summary = domain_bundle_summary.company_contact_details if domain_bundle_summary else None
        unique_sorted_consolidated_numbers = company_contact_details_summary.consolidated_numbers if company_contact_details_summary else []
        
        # For canonical_site_regex_candidates_found_status and llm_exception_details,
        # these are per-pathful. The row outcome needs to consider the true_base.
        # This might require aggregating these from pathful to true_base if not already done,
        # or adjusting how determine_final_row_outcome_and_fault uses them.
        # For now, we pass the pathful-level dicts. The function needs to handle this.
        # A better approach might be to aggregate these to true_base level first.
        
        # Aggregate regex_candidates_found_status for the true_base of this row
        true_base_regex_found = False
        if canonical_url_summary and canonical_url_summary in true_base_to_pathful_map:
            for pathful_u in true_base_to_pathful_map[canonical_url_summary]:
                if canonical_site_regex_candidates_found_status.get(pathful_u, False):
                    true_base_regex_found = True
                    break
        
        # Aggregate llm_exception_details for the true_base (e.g., take first or concatenate)
        # This is tricky. For now, passing the raw dict.
        # The outcome_analyzer function should be robust or this needs refinement.

        final_reason, fault_category = determine_final_row_outcome_and_fault(
            index=index, row_summary=row_summary, df_status_snapshot=df_status_snapshot_for_helper,
            company_contact_details_summary=company_contact_details_summary,
            unique_sorted_consolidated_numbers=unique_sorted_consolidated_numbers,
            canonical_url_summary=canonical_url_summary, # true_base
            true_base_scraper_status_map=true_base_scraper_status, # status for true_base
            true_base_to_pathful_map=true_base_to_pathful_map, # pathfuls for this true_base
            canonical_site_pathful_scraper_status=canonical_site_pathful_scraper_status, # status for each pathful
            canonical_site_raw_llm_outputs=canonical_site_raw_llm_outputs, # LLM output per pathful
            # Pass the aggregated regex status for the true_base
            canonical_site_regex_candidates_found_status={str(canonical_url_summary): true_base_regex_found} if canonical_url_summary else {},
            canonical_site_llm_exception_details=canonical_site_llm_exception_details # Pass the pathful-level dict
        )
        df.at[index, 'Final_Row_Outcome_Reason'] = final_reason
        df.at[index, 'Determined_Fault_Category'] = fault_category

        # Populate Top Numbers (operational call list)
        ranked_for_outreach = _rank_consolidated_numbers_for_outreach(unique_sorted_consolidated_numbers)
        if ranked_for_outreach:
            for i, top_item in enumerate(ranked_for_outreach[:3]):
                df.at[index, f'Top_Number_{i+1}'] = top_item.number
                df.at[index, f'Top_Type_{i+1}'] = ", ".join(sorted(list(set(s.type for s in top_item.sources if s.type))))
                df.at[index, f'Top_SourceURL_{i+1}'] = ", ".join(sorted(list(set(s.original_full_url for s in top_item.sources))))

        # Populate Primary_/Secondary_ legacy columns from consolidated numbers.
        primary_items = [n for n in unique_sorted_consolidated_numbers if (n.classification or "").strip() == "Primary"]
        secondary_items = [n for n in unique_sorted_consolidated_numbers if (n.classification or "").strip() == "Secondary"]
        if primary_items:
            types_csv, urls_csv = _summarize_sources_for_number(primary_items[0])
            df.at[index, "Primary_Number_1"] = primary_items[0].number
            df.at[index, "Primary_Type_1"] = types_csv
            df.at[index, "Primary_SourceURL_1"] = urls_csv
        if secondary_items:
            types_csv, urls_csv = _summarize_sources_for_number(secondary_items[0])
            df.at[index, "Secondary_Number_1"] = secondary_items[0].number
            df.at[index, "Secondary_Type_1"] = types_csv
            df.at[index, "Secondary_SourceURL_1"] = urls_csv
        if len(secondary_items) > 1:
            types_csv, urls_csv = _summarize_sources_for_number(secondary_items[1])
            df.at[index, "Secondary_Number_2"] = secondary_items[1].number
            df.at[index, "Secondary_Type_2"] = types_csv
            df.at[index, "Secondary_SourceURL_2"] = urls_csv

        # Populate debug/trace fields for this row by aggregating across its pathful URLs.
        pathfuls = true_base_to_pathful_map.get(str(canonical_url_summary), []) if canonical_url_summary else []
        all_regex_items: List[Dict[str, Any]] = []
        for p in pathfuls:
            all_regex_items.extend(canonical_site_regex_candidate_items.get(p, []) or [])
        df.at[index, "RegexCandidateSnippets"] = all_regex_items

        # Collect LLM extracted outputs (raw) across pathfuls.
        llm_items: List[Dict[str, Any]] = []
        for p in pathfuls:
            outs = canonical_site_raw_llm_outputs.get(p, []) or []
            for o in outs:
                try:
                    llm_items.append(o.model_dump() if hasattr(o, "model_dump") else dict(o))  # pydantic v2 / dict-like
                except Exception:
                    try:
                        llm_items.append({"number": getattr(o, "number", None), "type": getattr(o, "type", None),
                                          "classification": getattr(o, "classification", None), "is_valid": getattr(o, "is_valid", None),
                                          "source_url": getattr(o, "source_url", None)})
                    except Exception:
                        pass
        df.at[index, "LLMExtractedNumbers"] = llm_items

        best_numbers = [str(df.at[index, f"Top_Number_{i}"]).strip() for i in (1, 2, 3)
                        if f"Top_Number_{i}" in df.columns and df.at[index, f"Top_Number_{i}"] not in (None, "", "nan", "NaN")]
        df.at[index, "BestMatchedPhoneNumbers"] = best_numbers
        other_nums = [n.number for n in unique_sorted_consolidated_numbers if n.number not in set(best_numbers)]
        df.at[index, "OtherRelevantNumbers"] = other_nums

        # Attach context artifact paths where available.
        ctx_paths: List[str] = []
        for p in pathfuls:
            ctx_paths.extend(canonical_site_llm_context_paths.get(p, []) or [])
        # De-dupe while preserving order
        seen = set()
        ctx_paths = [x for x in ctx_paths if x and not (x in seen or seen.add(x))]
        df.at[index, "LLMContextPath"] = "; ".join(ctx_paths)

        # Simple confidence heuristic based on the best consolidated number classification.
        best_obj = unique_sorted_consolidated_numbers[0] if unique_sorted_consolidated_numbers else None
        df.at[index, "ConfidenceScore"] = _confidence_for_best(best_obj)

        # Person-associated contacts (optional)
        contacts = _extract_person_contacts(unique_sorted_consolidated_numbers)
        df.at[index, "PersonContacts"] = contacts
        best_contact = _pick_best_person_contact(contacts)
        if best_contact:
            df.at[index, "BestPersonContactName"] = best_contact.get("name")
            df.at[index, "BestPersonContactRole"] = best_contact.get("role")
            df.at[index, "BestPersonContactDepartment"] = best_contact.get("department")
            df.at[index, "BestPersonContactNumber"] = best_contact.get("number")

        # --- Live streaming to master (optional) ---
        if row_queue is not None:
            try:
                payload = df.loc[index].to_dict()
                if row_queue_meta:
                    payload.update({f"__meta_{k}": v for k, v in row_queue_meta.items()})
                row_queue.put(
                    {
                        "type": "row",
                        "run_id": run_id,
                        "ts": datetime.now().isoformat(),
                        "data": payload,
                    }
                )
            except Exception:
                # Best effort only; do not kill the worker.
                pass
        
        # Attrition data population (simplified, assumes duplicate counts are handled if needed by report)
        if final_reason != "Contact_Successfully_Extracted":
            current_input_company_name_for_counts = str(row_summary.get(company_name_col_key, "MISSING_COMPANY_NAME_INPUT")).strip()
            current_derived_input_canonical_for_counts = row_summary.get('temp_derived_input_canonical', "MISSING_OR_INVALID_URL_INPUT")

            input_company_total_count = company_name_counts_placeholder.get(current_input_company_name_for_counts, 0)
            input_url_total_count = input_canonical_url_counts_placeholder.get(current_derived_input_canonical_for_counts, 0)
            is_dup_company_name = input_company_total_count > 1 and current_input_company_name_for_counts != "MISSING_COMPANY_NAME_INPUT"
            is_dup_input_url = input_url_total_count > 1 and current_derived_input_canonical_for_counts != "MISSING_OR_INVALID_URL_INPUT"
            
            attrition_data_list.append({
                "InputRowID": index, "CompanyName": company_name_for_attrition,
                "GivenURL": given_url_original_for_attrition,
                "Final_Row_Outcome_Reason": final_reason, "Determined_Fault_Category": fault_category,
                "Relevant_Canonical_URLs": canonical_url_summary if canonical_url_summary else "N/A",
                # "LLM_Error_Detail_Summary": ..., # This needs careful handling of pathful vs true_base
                "Timestamp_Of_Determination": datetime.now().isoformat(),
                "Input_CompanyName_Total_Count": input_company_total_count,
                "Input_CanonicalURL_Total_Count": input_url_total_count,
                "Is_Input_CompanyName_Duplicate": "Yes" if is_dup_company_name else "No",
                "Is_Input_CanonicalURL_Duplicate": "Yes" if is_dup_input_url else "No",
                "Is_Input_Row_Considered_Duplicate": "Yes" if (is_dup_company_name or is_dup_input_url) else "No"
            })
    
    if 'temp_derived_input_canonical' in df.columns:
        df.drop(columns=['temp_derived_input_canonical'], inplace=True)

    logger.info("Final row outcome determination complete.")

    # Persist consolidated phone results to cache (best-effort).
    try:
        cache_dir = getattr(app_config, "phone_results_cache_dir", None)
        if cache_dir:
            os.makedirs(os.path.normpath(cache_dir), exist_ok=True)
            for tb, bundle in final_consolidated_data_by_true_base.items():
                if not tb or not bundle or not bundle.company_contact_details:
                    continue
                if not (bundle.company_contact_details.consolidated_numbers or []):
                    continue
                _save_phone_results_to_cache(tb, bundle, app_config)
    except Exception:
        pass
    
    return (
        df, 
        attrition_data_list, 
        canonical_domain_journey_data,
        final_consolidated_data_by_true_base,
        true_base_scraper_status, # Status per true_base
        true_base_to_pathful_map, # Map from true_base to its pathful URLs
        input_to_canonical_map, # Map from input URL to true_base
        row_level_failure_counts
    )