"""
Configuration module for the Intelligent Prospect Analyzer & Sales Insights Generator.

This module defines the `AppConfig` class, which centralizes all application
configurations. Settings are loaded from environment variables (typically via a
`.env` file) with sensible defaults provided for most parameters. This approach
allows for easy customization of the application's behavior without modifying
the codebase.

The configuration covers various aspects of the pipeline, including:
- Web scraping parameters (user agents, timeouts, retry logic, link prioritization).
- Output directory structures and filename conventions.
- Large Language Model (LLM) settings (API keys, model names, generation parameters).
- Paths to prompt files for different LLM tasks.
- Input data handling (file paths, column mapping profiles, row processing ranges).
- Logging levels for file and console outputs.
- Keywords for page type classification.
"""
import os
import json
from dotenv import load_dotenv
from typing import List, Optional, Dict

# Load environment variables from a .env file.
# Attempts to load from paths relative to this file's location to support
# execution from different project directory levels (e.g., src/core, src, project root).
dotenv_path_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.env')  # Project_Root/.env
dotenv_path_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env') # Workspace_Root/.env (if project is nested)
dotenv_path_project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env') # Fallback for main_pipeline.py at project root

loaded_env = False
if os.path.exists(dotenv_path_1):
    load_dotenv(dotenv_path_1)
    loaded_env = True
elif os.path.exists(dotenv_path_2):
    load_dotenv(dotenv_path_2)
    loaded_env = True
elif os.path.exists(dotenv_path_project_root):
    load_dotenv(dotenv_path_project_root)
    loaded_env = True
else:
    print(
        f"Warning: .env file not found at {dotenv_path_1}, {dotenv_path_2}, "
        f"or {dotenv_path_project_root}. Using default configurations or "
        "expecting environment variables to be set externally."
    )


class AppConfig:
    """
    Manages application configurations, loading settings primarily from environment
    variables defined in a .env file.

    This class centralizes all configurable parameters for the prospect analysis
    pipeline, including scraper settings, output directories, LLM parameters,
    prompt paths, data handling specifics, and logging levels. It provides
    default values for most settings if they are not specified in the environment.

    Attributes:
        user_agents (List[str]): A list of user-agent strings for web scraping.
        scraper_default_headers (Dict[str, str]): Default headers for scraper requests.
        default_page_timeout (int): Default timeout for page operations in milliseconds.
        default_navigation_timeout (int): Default timeout for navigation actions in milliseconds.
        scrape_max_retries (int): Maximum retries for a failed scrape attempt.
        scrape_retry_delay_seconds (int): Delay in seconds between scrape retries.
        
        target_link_keywords (List[str]): Keywords to identify relevant internal links.
        scraper_critical_priority_keywords (List[str]): Keywords for top-priority pages.
        scraper_high_priority_keywords (List[str]): Keywords for high-priority pages.
        scraper_max_keyword_path_segments (int): Max path segments for priority keywords.
        scraper_exclude_link_path_patterns (List[str]): URL path patterns to exclude.
        scraper_max_pages_per_domain (int): Max pages to scrape per domain (0 for no limit).
        scraper_min_score_to_queue (int): Minimum score for a link to be queued.
        scraper_score_threshold_for_limit_bypass (int): Score to bypass page limit.
        scraper_max_high_priority_pages_after_limit (int): Max high-priority pages after limit.
        scraper_pages_for_summary_count (int): Number of top pages for summary text.
        
        max_depth_internal_links (int): Maximum depth for following internal links.
        scraper_networkidle_timeout_ms (int): Playwright networkidle timeout (ms).
        
        output_base_dir (str): Base directory for output files.
        scraped_content_subdir (str): Subdirectory for scraped content.
        llm_context_subdir (str): Subdirectory for LLM context/raw responses.
        filename_company_name_max_len (int): Max length for company name in filenames.
        filename_url_domain_max_len (int): Max length for domain in filenames.
        filename_url_hash_max_len (int): Max length for URL hash in filenames.
        
        respect_robots_txt (bool): Whether to respect robots.txt.
        robots_txt_user_agent (str): User-agent for checking robots.txt.
        
        gemini_api_key (Optional[str]): API key for Google Gemini.
        llm_model_name (str): Google Gemini model to use.
        llm_temperature_default (float): Default LLM temperature for response generation.
        llm_temperature_sales_insights (float): LLM temperature for sales insights generation.
        llm_max_tokens (int): Maximum tokens for LLM response.
        llm_chunk_processor_max_tokens (int): Max tokens for LLM chunk processor.
        llm_max_chunks_per_url (int): Maximum number of chunks to process per URL.
        llm_top_k (Optional[int]): LLM top_k sampling parameter.
        llm_top_p (Optional[float]): LLM top_p (nucleus) sampling parameter.
        LLM_MAX_INPUT_CHARS_FOR_SUMMARY (int): Max input characters for summary LLM call.
        llm_max_tokens_summary (Optional[int]): Max tokens for summary generation.
        llm_temperature_summary (Optional[float]): Temperature for summary generation.
        
        PROMPT_PATH_WEBSITE_SUMMARIZER (str): Path to website summarizer prompt.
        prompt_path_summarization (str): Path to the (old) summarization prompt.
        prompt_path_homepage_context (str): Path to homepage context prompt.
        PROMPT_PATH_ATTRIBUTE_EXTRACTOR (str): Path to attribute extractor prompt.
        MAX_GOLDEN_PARTNERS_IN_PROMPT (int): Max golden partners to include in prompts.
        extraction_profile (str): Current extraction profile to use (e.g., "minimal").

        url_probing_tlds (List[str]): TLDs for domain-like input probing.
        enable_dns_error_fallbacks (bool): Enable DNS error fallback strategies.
        
        input_excel_file_path (str): Path to the input data file.
        input_file_profile_name (str): Name of the input column mapping profile.
        INPUT_COLUMN_PROFILES (dict): Available input column mapping profiles.
        output_excel_file_name_template (str): Template for the main summary report Excel file.
        tertiary_report_file_name_template (str): Template for the new tertiary report Excel file name.
        skip_rows_config (Optional[int]): Rows to skip from input file start (0-indexed).
        nrows_config (Optional[int]): Rows to read after skipping (None for all).
        consecutive_empty_rows_to_stop (int): Consecutive empty rows to stop processing.
        PATH_TO_GOLDEN_PARTNERS_DATA (str): Path to the Golden Partners data file (CSV or Excel).
        
        log_level (str): Logging level for the file log (e.g., INFO, DEBUG).
        console_log_level (str): Logging level for console output.
 
        page_type_keywords_about (List[str]): Keywords for 'about' pages.
        page_type_keywords_product_service (List[str]): Keywords for 'product/service' pages.
        enable_slack_notifications (bool): Enable/disable Slack notifications.
        slack_bot_token (Optional[str]): Slack Bot User OAuth Token.
        slack_channel_id (Optional[str]): Slack channel ID for notifications.
        slack_test_channel_id (Optional[str]): Slack channel ID for test notifications.
        test_mode (bool): Flag to indicate if the application is in test mode.
 
    Methods:
        __init__(): Initializes AppConfig by loading values from environment
                    variables or using defaults.
    """

    INPUT_COLUMN_PROFILES = {
        "default": {
            "Unternehmen": "CompanyName",
            "Webseite": "GivenURL",
            "Beschreibung": "Description",
            "Telefonnummer": "GivenPhoneNumber",
            "_original_phone_column_name": "Telefonnummer"
        },
        "prospect_analyzer_input": {
            "Firmennummer": "CompanyID",
            "Firma Kurzname": "CompanyNameShort",
            "Firma Vollname": "CompanyName",
            "Homepage": "GivenURL",
            "Beschreibung": "Description",
            "Kategorie": "Industry",
            "Number": "PhoneNumber"
        },
        "ManauvKlaus": {
            "firma": "CompanyName",
            "url": "GivenURL",
            "Telefonnummer": "GivenPhoneNumber",
            "_original_phone_column_name": "Telefonnummer"
        },
        "manuav_contacts_2025": {
            "firma": "CompanyName",
            "url": "GivenURL",
            "Telefonnummer": "PhoneNumber",
            "Beschreibung": "Description",
            "beschreibung": "Description",
            "Kategorie": "Industry",
            "_original_phone_column_name": "Telefonnummer"
        },
        "lean_formatted": {
            "Company Name": "CompanyName",
            "URL": "GivenURL",
            "Number": "GivenPhoneNumber",
            "_original_phone_column_name": "Number"
        },
        "template": {
            "Company Name": "CompanyName",
            "URL": "GivenURL",
            "Contact Person": "ContactPerson",
            "Email": "Email",
            "Industry": "Industry",
            "Company Description": "Description"
        },
        "final_80k": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Combined_Description": "Combined_Description",
            "Industry_Category_Standardized": "Industry",
            "Company Phone": "PhoneNumber"
        },
        "final_80k_force_extract": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Combined_Description": "Combined_Description",
            "Industry_Category_Standardized": "Industry"
        },
        "german_standard": {
            "firma": "CompanyName",
            "url": "GivenURL",
            "beschreibung": "Description",
            "kategorie": "Industry"
        },
        "new_import_profile": {
            "Company Name": "CompanyName",
            "URL": "GivenURL"
        },
        "hochbau_realview": {
            "Name": "CompanyName",
            "Website": "GivenURL",
            "Branche (WZ)": "Description",
            "_original_phone_column_name": None,
            "_new_phone_column_name": "PhoneNumber_Found",
            "_new_split_number_col": "PhoneNumber",
            "_new_split_details_col": "PhoneDetails"
        }
        ,
        "company_semicolon": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Short Description": "Description",
            "Industry": "Industry",
            "Company Phone": "PhoneNumber",
            "_original_phone_column_name": "Company Phone"
        }
        ,
        # For PHONE-ONLY pipeline (phone_extract.py): keep existing phones in GivenPhoneNumber so the
        # phone retrieval flow can decide whether to re-check/override them.
        "company_semicolon_phone_extract": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Short Description": "Description",
            "Industry": "Industry",
            "Company Phone": "GivenPhoneNumber",
            "_original_phone_column_name": "Company Phone"
        }
        ,
        # For running the full pipeline AFTER phone_extract has produced PhoneNumber_Found.
        # This maps the enriched phone column into the canonical PhoneNumber field so the full pipeline
        # can generate pitches without re-running phone retrieval for every row.
        "company_semicolon_phone_found": {
            "Company": "CompanyName",
            "Website": "GivenURL",
            "Short Description": "Description",
            "Industry": "Industry",
            "PhoneNumber_Found": "PhoneNumber",
            "_original_phone_column_name": "PhoneNumber_Found"
        }
        ,
        "pflege_plus_exhibitors": {
            "company_name": "CompanyName",
            "website": "GivenURL",
            "phone": "PhoneNumber",
            "email": "Email",
            "description": "Description",
            "categories": "categories",
            "products": "products",
            "_original_phone_column_name": "phone"
        }
        ,
        "pflege_procare_exhibitors": {
            "company_name": "CompanyName",
            "website": "GivenURL",
            "phone": "PhoneNumber",
            "email": "Email",
            "description": "Description",
            "categories": "categories",
            "products": "products",
            "_original_phone_column_name": "phone"
        }
        ,
        "emo_hannover_2025": {
            "Firma": "CompanyName",
            "Aussteller": "CompanyName",
            "Website": "GivenURL",
            "Website Unternehmen": "GivenURL",
            "_original_phone_column_name": None
        }
        ,
        "expopharm_exhibitors": {
            "company_name": "CompanyName",
            "website": "GivenURL",
            "phone": "PhoneNumber",
            "email": "Email",
            "description": "Description",
            "pdf_text": "pdf_text",
            "_original_phone_column_name": "phone"
        }
        ,
        "altenpflege_products": {
            "company_name": "CompanyName",
            "website": "GivenURL",
            "address": "Address",
            "phone": "PhoneNumber",
            "email": "Email",
            "contacts": "contacts",
            "description": "Description",
            "pdf_text": "pdf_text",
            "_original_phone_column_name": "phone"
        }
        ,
        # Apollo export: data/Apollo Shopware Partner Germany 1(Sheet1).csv
        # Intended for running the phone retrieval pipeline in isolation.
        "apollo_shopware_partner_germany": {
            "Company Name": "CompanyName",
            "Website": "GivenURL",
            # Keep the original phone as "GivenPhoneNumber" for normalization/verification flows.
            "Company Phone": "GivenPhoneNumber",
            "Industry": "Industry",
            "_original_phone_column_name": "Company Phone"
        }
    }

    def __init__(self,
                 input_file_override: Optional[str] = None,
                 row_range_override: Optional[str] = None,
                 run_id_suffix_override: Optional[str] = None,
                 test_mode: bool = False):
        """
        Initializes the AppConfig instance.

        Loads configuration from environment variables, but allows for specific
        settings to be overridden by arguments passed during instantiation.

        Args:
            input_file_override (Optional[str]): If provided, this path will be used
                for the input file, overriding the .env setting.
            row_range_override (Optional[str]): If provided, this range string
                (e.g., "1-1000") will be used, overriding the .env setting.
            run_id_suffix_override (Optional[str]): If provided, this string will be
                appended to the run ID, overriding the .env setting.
            test_mode (bool): If True, the application will run in test mode.
        """
        self.test_mode = test_mode

        # --- Scraper Configuration ---
        user_agents_str = os.getenv('SCRAPER_USER_AGENTS', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36')
        self.user_agents: List[str] = [ua.strip() for ua in user_agents_str.split(',') if ua.strip()]

        default_headers_json = os.getenv('SCRAPER_DEFAULT_HEADERS', '{}')
        try:
            self.scraper_default_headers: Dict[str, str] = json.loads(default_headers_json)
        except json.JSONDecodeError:
            self.scraper_default_headers = {}
            print(f"Warning: Invalid JSON in SCRAPER_DEFAULT_HEADERS. Using empty dictionary. Value was: {default_headers_json}")

        self.default_page_timeout: int = int(os.getenv('SCRAPER_PAGE_TIMEOUT_MS', '30000'))
        self.default_navigation_timeout: int = int(os.getenv('SCRAPER_NAVIGATION_TIMEOUT_MS', '60000'))
        self.scrape_max_retries: int = int(os.getenv('SCRAPER_MAX_RETRIES', '2'))
        self.scrape_retry_delay_seconds: int = int(os.getenv('SCRAPER_RETRY_DELAY_SECONDS', '5'))
        
        # Link Prioritization and Control Settings
        target_link_keywords_str: str = os.getenv('TARGET_LINK_KEYWORDS', 'about,company,services,products,solutions,team,mission,projekte,produkte,leistungen,lösungen,unternehmen,über-uns,ueber-uns')
        self.target_link_keywords: List[str] = [kw.strip().lower() for kw in target_link_keywords_str.split(',') if kw.strip()]
        
        critical_priority_keywords_str: str = os.getenv('SCRAPER_CRITICAL_PRIORITY_KEYWORDS', 'about-us,company-profile')
        self.scraper_critical_priority_keywords: List[str] = [kw.strip().lower() for kw in critical_priority_keywords_str.split(',') if kw.strip()]
        
        high_priority_keywords_str: str = os.getenv('SCRAPER_HIGH_PRIORITY_KEYWORDS', 'services,products,solutions,leistungen,produkte,lösungen')
        self.scraper_high_priority_keywords: List[str] = [kw.strip().lower() for kw in high_priority_keywords_str.split(',') if kw.strip()]
        
        self.scraper_max_keyword_path_segments: int = int(os.getenv('SCRAPER_MAX_KEYWORD_PATH_SEGMENTS', '3'))
        
        exclude_link_patterns_str: str = os.getenv('SCRAPER_EXCLUDE_LINK_PATH_PATTERNS', '/media/,/blog/,/wp-content/,/video/,/hilfe-video/')
        self.scraper_exclude_link_path_patterns: List[str] = [p.strip().lower() for p in exclude_link_patterns_str.split(',') if p.strip()]
        
        self.scraper_max_pages_per_domain: int = int(os.getenv('SCRAPER_MAX_PAGES_PER_DOMAIN', '20'))  # Default 20, 0 for no limit
        self.scraper_min_score_to_queue: int = int(os.getenv('SCRAPER_MIN_SCORE_TO_QUEUE', '40'))
        self.scraper_score_threshold_for_limit_bypass: int = int(os.getenv('SCRAPER_SCORE_THRESHOLD_FOR_LIMIT_BYPASS', '80'))
        self.scraper_max_high_priority_pages_after_limit: int = int(os.getenv('SCRAPER_MAX_HIGH_PRIORITY_PAGES_AFTER_LIMIT', '5'))  # Default to 5

        self.scraper_pages_for_summary_count: int = int(os.getenv('SCRAPER_PAGES_FOR_SUMMARY_COUNT', '3'))
 
        # Existing Scraper Settings
        self.max_depth_internal_links: int = int(os.getenv('MAX_DEPTH_INTERNAL_LINKS', '1'))
        scraper_timeout_str = os.getenv('SCRAPER_NETWORKIDLE_TIMEOUT_MS', '3000').split('#')[0].strip().strip('\'"')
        self.scraper_networkidle_timeout_ms: int = int(scraper_timeout_str)
        self.snippet_window_chars: int = int(os.getenv('SNIPPET_WINDOW_CHARS', '300'))

        # --- HTTP fallback scraping (used by phone_retrieval scraper) ---
        # If Playwright fails to fetch content, try a lightweight httpx GET and extract text from returned HTML.
        # This often helps with simple/static contact/imprint pages.
        self.scraper_http_fallback_enabled: bool = os.getenv('SCRAPER_HTTP_FALLBACK_ENABLED', 'True').lower() == 'true'
        self.scraper_http_fallback_timeout_seconds: int = int(os.getenv('SCRAPER_HTTP_FALLBACK_TIMEOUT_SECONDS', '15'))
        self.scraper_http_fallback_max_bytes: int = int(os.getenv('SCRAPER_HTTP_FALLBACK_MAX_BYTES', '2000000'))  # ~2MB
        self.scraper_http_fallback_min_text_chars: int = int(os.getenv('SCRAPER_HTTP_FALLBACK_MIN_TEXT_CHARS', '200'))
        block_kw_raw = os.getenv(
            'SCRAPER_HTTP_FALLBACK_BLOCK_KEYWORDS',
            'access denied,captcha,cloudflare,bot detection,verify you are human,enable javascript,forbidden,service unavailable,temporarily unavailable'
        )
        self.scraper_http_fallback_block_keywords: List[str] = [k.strip().lower() for k in block_kw_raw.split(',') if k.strip()]

        # --- Caching ---
        self.caching_enabled: bool = os.getenv('CACHING_ENABLED', 'True').lower() == 'true'
        self.cache_dir: str = os.getenv('CACHE_DIR', 'cache')

        # --- Proxy Management ---
        self.proxy_enabled: bool = os.getenv('PROXY_ENABLED', 'False').lower() == 'true'
        self.proxy_list: List[str] = [p.strip() for p in os.getenv('PROXY_LIST', '').split(',') if p.strip()]
        self.proxy_rotation_strategy: str = os.getenv('PROXY_ROTATION_STRATEGY', 'random') # 'random', 'sequential', 'rotate_on_failure'
        self.proxy_health_check_enabled: bool = os.getenv('PROXY_HEALTH_CHECK_ENABLED', 'True').lower() == 'true'
        self.proxy_cooldown_seconds: int = int(os.getenv('PROXY_COOLDOWN_SECONDS', '300'))

        # --- Interaction Handling ---
        self.interaction_handler_enabled: bool = os.getenv('INTERACTION_HANDLER_ENABLED', 'True').lower() == 'true'
        interaction_selectors_str: str = os.getenv('INTERACTION_SELECTORS', 'button[id*="accept"],button[id*="agree"],button[id*="consent"],button[id*="cookie"],button[class*="accept"],button[class*="close"],[aria-label*="close"]')
        self.interaction_selectors: List[str] = [s.strip() for s in interaction_selectors_str.split(',') if s.strip()]
        interaction_text_queries_str: str = os.getenv('INTERACTION_TEXT_QUERIES', 'Accept all,Agree,Consent,I agree,Alle akzeptieren,accept all cookies,Accept,a *c *c *e *p *t,Ich akzeptiere alle')
        self.interaction_text_queries: List[str] = [q.strip() for q in interaction_text_queries_str.split(',') if q.strip()]
        self.interaction_handler_max_passes: int = int(os.getenv('INTERACTION_HANDLER_MAX_PASSES', '2'))
        self.interaction_handler_visibility_timeout_ms: int = int(os.getenv('INTERACTION_HANDLER_VISIBILITY_TIMEOUT_MS', '200'))
 
        # --- Output Configuration ---
        self.output_base_dir: str = os.getenv('OUTPUT_BASE_DIR', 'output_data')  # Relative to project root
        self.scraped_content_subdir: str = 'scraped_content'
        self.llm_context_subdir: str = 'llm_context'  # Subdirectory for LLM raw responses
        self.filename_company_name_max_len: int = int(os.getenv('FILENAME_COMPANY_NAME_MAX_LEN', '25'))  # Default to 25
        self.filename_url_domain_max_len: int = int(os.getenv('FILENAME_URL_DOMAIN_MAX_LEN', '8'))    # Default to 8
        self.filename_url_hash_max_len: int = int(os.getenv('FILENAME_URL_HASH_MAX_LEN', '8'))        # Default to 8

        # --- Robots.txt Handling ---
        self.respect_robots_txt: bool = os.getenv('RESPECT_ROBOTS_TXT', 'True').lower() == 'true'
        self.robots_txt_user_agent: str = os.getenv('ROBOTS_TXT_USER_AGENT', '*')
        self.robots_txt_timeout_seconds: int = int(os.getenv('ROBOTS_TXT_TIMEOUT_SECONDS', '3'))

        # --- LLM Configuration ---
        self.gemini_api_key: Optional[str] = os.getenv('GEMINI_API_KEY')
        self.llm_model_name: str = os.getenv('LLM_MODEL_NAME', 'gemini-1.5-pro-latest')  # Default to a capable model
        self.llm_model_name_sales_insights: str = os.getenv('LLM_MODEL_NAME_SALES_INSIGHTS', 'gemini-1.5-pro-preview-06-05')
        self.llm_temperature_default: float = float(os.getenv('LLM_TEMPERATURE_DEFAULT', '0.3'))
        self.llm_temperature: float = float(os.getenv('LLM_TEMPERATURE', self.llm_temperature_default))
        self.llm_temperature_extraction: float = float(os.getenv('LLM_TEMPERATURE_EXTRACTION', '0.2'))
        self.llm_temperature_creative: float = float(os.getenv('LLM_TEMPERATURE_CREATIVE', '0.5'))
        self.llm_max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '3000'))  # Updated default
        self.llm_chunk_processor_max_tokens: int = int(os.getenv('LLM_CHUNK_PROCESSOR_MAX_TOKENS', '4096'))
        self.llm_max_chunks_per_url: int = int(os.getenv('LLM_MAX_CHUNKS_PER_URL', '10'))
        self.llm_max_retries_on_number_mismatch: int = int(os.getenv('LLM_MAX_RETRIES_ON_NUMBER_MISMATCH', '1'))
        self.llm_candidate_chunk_size: int = int(os.getenv('LLM_CANDIDATE_CHUNK_SIZE', '10'))

        llm_top_k_str = os.getenv('LLM_TOP_K')
        self.llm_top_k: Optional[int] = int(llm_top_k_str) if llm_top_k_str and llm_top_k_str.isdigit() and int(llm_top_k_str) > 0 else None
        
        llm_top_p_str = os.getenv('LLM_TOP_P')
        self.llm_top_p: Optional[float] = float(llm_top_p_str) if llm_top_p_str else None
        try:
            if self.llm_top_p is not None and not (0.0 <= self.llm_top_p <= 1.0):
                print(f"Warning: LLM_TOP_P value '{self.llm_top_p}' is outside the valid range [0.0, 1.0]. It will be ignored.")
                self.llm_top_p = None
        except ValueError:
            print(f"Warning: Invalid LLM_TOP_P value '{llm_top_p_str}'. It will be ignored.")
            self.llm_top_p = None

        # Specific LLM settings for summary, overriding general ones if provided
        llm_max_tokens_summary_str = os.getenv('LLM_MAX_TOKENS_SUMMARY')
        self.llm_max_tokens_summary: Optional[int] = int(llm_max_tokens_summary_str) if llm_max_tokens_summary_str and llm_max_tokens_summary_str.isdigit() else None
        
        llm_temperature_summary_str = os.getenv('LLM_TEMPERATURE_SUMMARY')
        self.llm_temperature_summary: Optional[float] = None
        if llm_temperature_summary_str:
            try:
                temp_val = float(llm_temperature_summary_str)
                if 0.0 <= temp_val <= 2.0:  # Common range for temperature
                    self.llm_temperature_summary = temp_val
                else:
                    print(f"Warning: LLM_TEMPERATURE_SUMMARY value '{temp_val}' is outside the typical range [0.0, 2.0]. It will be ignored.")
            except ValueError:
                print(f"Warning: Invalid LLM_TEMPERATURE_SUMMARY value '{llm_temperature_summary_str}'. It will be ignored.")

        # --- Extraction Profiles and Prompt Paths ---
        # --- Extraction Profiles and Prompt Paths ---
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        
        def get_clean_path(env_var: str, default_path: str) -> str:
            raw_path = os.getenv(env_var, default_path)
            # Normalize path separators for consistent checking
            raw_path = raw_path.replace('\\', '/')
            project_root_name = os.path.basename(project_root)
            
            # Check if the raw_path starts with the project root directory name and a slash
            if raw_path.startswith(f"{project_root_name}/"):
                # Strip the redundant prefix
                raw_path = raw_path[len(project_root_name) + 1:]
            
            # Normalize the path to be safe for the current OS
            return os.path.normpath(os.path.join(project_root, raw_path))

        self.prompt_path_summarization: str = get_clean_path('PROMPT_PATH_SUMMARIZATION', 'prompts/summarization_prompt.txt')
        self.extraction_profile: str = os.getenv('EXTRACTION_PROFILE', "minimal")
        self.prompt_path_homepage_context: str = get_clean_path('PROMPT_PATH_HOMEPAGE_CONTEXT', 'prompts/homepage_context_prompt.txt')
        self.PROMPT_PATH_WEBSITE_SUMMARIZER: str = get_clean_path('PROMPT_PATH_WEBSITE_SUMMARIZER', 'prompts/website_summarizer_prompt.txt')
        self.PROMPT_PATH_ATTRIBUTE_EXTRACTOR: str = get_clean_path('PROMPT_PATH_ATTRIBUTE_EXTRACTOR', 'prompts/attribute_extractor_prompt.txt')
        self.prompt_path_minimal_classification: str = get_clean_path('PROMPT_PATH_MINIMAL_CLASSIFICATION', 'prompts/b2b_capacity_check_prompt.txt')
        self.prompt_path_enriched_extraction: str = get_clean_path('PROMPT_PATH_ENRICHED_EXTRACTION', 'prompts/profile_2.txt')
        self.LLM_MAX_INPUT_CHARS_FOR_SUMMARY: int = int(os.getenv('LLM_MAX_INPUT_CHARS_FOR_SUMMARY', '40000'))
        
        # --- Language-Specific Prompt Configuration ---
        self.sales_prompt_language: str = os.getenv('SALES_PROMPT_LANGUAGE', 'en').lower()
        # Resolve once to avoid redeclaration warnings; defaults currently point to German prompts
        partner_prompt_default = 'prompts/german_partner_matching_prompt.txt'
        sales_pitch_prompt_default = 'prompts/german_sales_pitch_generation_prompt.txt'
        self.PROMPT_PATH_GERMAN_PARTNER_MATCHING: str = get_clean_path('PROMPT_PATH_GERMAN_PARTNER_MATCHING', partner_prompt_default)
        self.PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION: str = get_clean_path('PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION', sales_pitch_prompt_default)
        self.MAX_GOLDEN_PARTNERS_IN_PROMPT: int = int(os.getenv('MAX_GOLDEN_PARTNERS_IN_PROMPT', '10'))
 
        # --- URL Probing Configuration ---
        url_probing_tlds_str: str = os.getenv('URL_PROBING_TLDS', 'de,com,at,ch')
        self.url_probing_tlds: List[str] = [tld.strip().lower() for tld in url_probing_tlds_str.split(',') if tld.strip()]
        self.enable_dns_error_fallbacks: bool = os.getenv('ENABLE_DNS_ERROR_FALLBACKS', 'True').lower() == 'true'

        # --- Phone Number Normalization Configuration ---
        target_country_codes_str: str = os.getenv('TARGET_COUNTRY_CODES', 'DE,CH,AT') # Germany, Switzerland, Austria
        self.target_country_codes: List[str] = [code.strip().upper() for code in target_country_codes_str.split(',') if code.strip()]
        self.default_region_code: Optional[str] = os.getenv('DEFAULT_REGION_CODE', 'DE') # Default region for parsing if others fail
        self.force_phone_extraction: bool = os.getenv('FORCE_PHONE_EXTRACTION', 'False').lower() == 'true'
        # When True, the full pipeline may run phone retrieval (scrape/regex/LLM) if input phone is missing.
        # For two-stage workflows (phone_extract first, then main_pipeline), you typically want this False.
        self.enable_phone_retrieval_in_full_pipeline: bool = os.getenv('ENABLE_PHONE_RETRIEVAL_IN_FULL_PIPELINE', 'True').lower() == 'true'

        # --- Scrape Reuse / Cache (for full pipeline scraping) ---
        # Allows the full pipeline to reuse existing cleaned scrape artifacts (e.g. from phone_extract)
        # instead of re-scraping websites. This can dramatically speed up "resume" workflows.
        self.reuse_scraped_content_if_available: bool = os.getenv('REUSE_SCRAPED_CONTENT_IF_AVAILABLE', 'False').lower() == 'true'
        raw_cache_dirs = os.getenv('SCRAPED_CONTENT_CACHE_DIRS', '') or ''
        # Accept both comma and semicolon as separators (Windows users often paste ';' lists).
        raw_cache_dirs = raw_cache_dirs.replace(';', ',')
        self.scraped_content_cache_dirs: List[str] = [p.strip() for p in raw_cache_dirs.split(',') if p.strip()]
        self.scraped_content_cache_min_chars: int = int(os.getenv('SCRAPED_CONTENT_CACHE_MIN_CHARS', '500'))

        # --- Phone Extraction Quality/Cost Controls ---
        # Limit how many regex candidates we send into the phone-extraction LLM per canonical (pathful) URL.
        self.phone_llm_max_candidates_total: int = int(os.getenv("PHONE_LLM_MAX_CANDIDATES_TOTAL", "120"))
        prefer_paths = os.getenv("PHONE_LLM_PREFER_URL_PATH_KEYWORDS", "kontakt,contact,impressum,legal,about,ueber,über")
        self.phone_llm_prefer_url_path_keywords: List[str] = [p.strip().lower() for p in prefer_paths.replace(";", ",").split(",") if p.strip()]
        prefer_snip = os.getenv("PHONE_LLM_PREFER_SNIPPET_KEYWORDS", "tel,telefon,phone,zentrale,hotline,call,service,vertrieb,sales")
        self.phone_llm_prefer_snippet_keywords: List[str] = [p.strip().lower() for p in prefer_snip.replace(";", ",").split(",") if p.strip()]

        # Deterministic phone-results caching (optional): reuse consolidated phone results per canonical base domain
        # to avoid re-scrape/re-LLM across resume runs.
        self.reuse_phone_results_if_available: bool = os.getenv("REUSE_PHONE_RESULTS_IF_AVAILABLE", "False").lower() == "true"
        # Use an env override when provided; otherwise default to a stable local folder (not dependent on output_base_dir init order).
        self.phone_results_cache_dir: str = os.path.normpath(os.getenv("PHONE_RESULTS_CACHE_DIR", "cache/phone_results_cache"))

        # --- Data Handling & Input Profiling ---
        self.input_excel_file_path: str = input_file_override or os.getenv('INPUT_EXCEL_FILE_PATH', 'data_to_be_inputed.xlsx')  # Relative to project root
        self.input_file_profile_name: str = os.getenv("INPUT_FILE_PROFILE_NAME", "default")
        self.output_excel_file_name_template: str = os.getenv('OUTPUT_EXCEL_FILE_NAME_TEMPLATE', 'Pipeline_Summary_Report_{run_id}.xlsx')
        self.tertiary_report_file_name_template: str = os.getenv('TERTIARY_REPORT_FILE_NAME_TEMPLATE', 'Final Contacts.xlsx')
        self.processed_contacts_report_file_name_template: str = os.getenv('PROCESSED_CONTACTS_REPORT_FILE_NAME_TEMPLATE', 'Final_Processed_Contacts.xlsx')
        self.PATH_TO_GOLDEN_PARTNERS_DATA: str = get_clean_path('PATH_TO_GOLDEN_PARTNERS_DATA', 'data/kgs_001_ER47_20250626.xlsx')

        # --- Row Processing Range Configuration ---
        self.skip_rows_config: Optional[int] = None
        self.nrows_config: Optional[int] = None
        raw_row_range: Optional[str] = row_range_override or os.getenv('ROW_PROCESSING_RANGE', "")

        if raw_row_range:
            raw_row_range = raw_row_range.strip()
            if not raw_row_range or raw_row_range == "0":
                pass  # Process all rows, skip_rows_config and nrows_config remain None
            elif '-' in raw_row_range:
                parts = raw_row_range.split('-', 1)
                start_str, end_str = parts[0].strip(), parts[1].strip()

                start_val: Optional[int] = None
                end_val: Optional[int] = None

                if start_str and start_str.isdigit():
                    start_val = int(start_str)
                
                if end_str and end_str.isdigit():
                    end_val = int(end_str)

                if start_val is not None and start_val > 0:
                    self.skip_rows_config = start_val - 1  # 0-indexed skip
                    if end_val is not None and end_val >= start_val:
                        self.nrows_config = end_val - start_val + 1
                    elif end_str == "":  # Format "N-" (from N to end)
                        self.nrows_config = None  # Read all after skipping
                    elif end_val is not None and end_val < start_val:
                        print(f"Warning: Invalid ROW_PROCESSING_RANGE '{raw_row_range}'. End value < Start value. Processing all rows.")
                        self.skip_rows_config = None
                        self.nrows_config = None
                elif start_str == "" and end_val is not None and end_val > 0:  # Format "-M" (first M rows)
                    self.skip_rows_config = None  # Or 0, effectively the same for pandas
                    self.nrows_config = end_val
                else:
                    print(f"Warning: Invalid ROW_PROCESSING_RANGE format '{raw_row_range}'. Expected N-M, N-, -M, or N. Processing all rows.")
            elif raw_row_range.isdigit() and int(raw_row_range) > 0:  # Single number "N"
                self.skip_rows_config = None  # Or 0
                self.nrows_config = int(raw_row_range)
            else:
                if raw_row_range != "0":  # "0" is a valid way to say "all rows"
                    print(f"Warning: Invalid ROW_PROCESSING_RANGE value '{raw_row_range}'. Processing all rows.")
        
        # --- Data Handling Enhancements ---
        self.consecutive_empty_rows_to_stop: int = int(os.getenv('CONSECUTIVE_EMPTY_ROWS_TO_STOP', '3'))

        # --- Regex Candidate Filtering ---
        self.max_identical_numbers_per_page_to_llm: int = int(os.getenv('MAX_IDENTICAL_NUMBERS_PER_PAGE_TO_LLM', '3'))

        # --- Logging Configuration ---
        self.log_level: str = os.getenv('LOG_LEVEL', 'INFO').upper()
        self.console_log_level: str = os.getenv('CONSOLE_LOG_LEVEL', 'WARNING').upper()

        # --- Run Identifier ---
        self.run_id_suffix: Optional[str] = run_id_suffix_override or os.getenv('RUN_ID_SUFFIX')

        # --- Slack Configuration ---
        self.enable_slack_notifications: bool = os.getenv('ENABLE_SLACK_NOTIFICATIONS', 'False').lower() == 'true'
        self.slack_bot_token: Optional[str] = os.getenv('SLACK_BOT_TOKEN')
        self.slack_channel_id: Optional[str] = os.getenv('SLACK_CHANNEL_ID')
        self.slack_test_channel_id: Optional[str] = os.getenv('SLACK_TEST_CHANNEL_ID')

        # --- Page Type Classification Keywords ---
        page_type_about_str: str = os.getenv('PAGE_TYPE_KEYWORDS_ABOUT', 'about,about-us,company,profile,mission,vision,team,unternehmen,profil,ueber-uns,uber-uns')
        self.page_type_keywords_about: List[str] = [kw.strip().lower() for kw in page_type_about_str.split(',') if kw.strip()]

        page_type_product_service_str: str = os.getenv('PAGE_TYPE_KEYWORDS_PRODUCT_SERVICE', 'products,services,solutions,offerings,platform,features,produkte,leistungen,lösungen,angebot')
        self.page_type_keywords_product_service: List[str] = [kw.strip().lower() for kw in page_type_product_service_str.split(',') if kw.strip()]

        page_type_contact_str: str = os.getenv('PAGE_TYPE_KEYWORDS_CONTACT', 'contact,kontakt,ansprechpartner')
        self.page_type_keywords_contact: List[str] = [kw.strip().lower() for kw in page_type_contact_str.split(',') if kw.strip()]

        page_type_imprint_str: str = os.getenv('PAGE_TYPE_KEYWORDS_IMPRINT', 'imprint,impressum,legal-notice,legalnotice')
        self.page_type_keywords_imprint: List[str] = [kw.strip().lower() for kw in page_type_imprint_str.split(',') if kw.strip()]

        page_type_legal_str: str = os.getenv('PAGE_TYPE_KEYWORDS_LEGAL', 'privacy,datenschutz,terms,agb,legal')
        self.page_type_keywords_legal: List[str] = [kw.strip().lower() for kw in page_type_legal_str.split(',') if kw.strip()]