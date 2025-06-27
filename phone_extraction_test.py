import asyncio
import logging
import pytest
from src.core.config import AppConfig
from src.scraper.scraper_logic import scrape_website
from src.phone_retrieval.extractors.regex_extractor import extract_numbers_with_snippets_from_text
from src.phone_retrieval.extractors.llm_chunk_processor import LLMChunkProcessor
from src.phone_retrieval.llm_clients.gemini_client import GeminiClient
from src.phone_retrieval.data_handling.consolidator import process_and_consolidate_contact_data

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.mark.anyio
async def test_phone_extraction():
    """
    Test the full phone extraction pipeline for a single URL.
    """
    config = AppConfig()
    gemini_client = GeminiClient(config)

    # --- Scrape the Exxomove website ---
    logger.info("--- Scraping Exxomove website ---")
    test_url = "https://www.exxomove.de/"
    scraped_data, _, _, _ = await scrape_website(test_url, "output_data", "Exxomove", set(), "test_row")

    # --- Extract phone number candidates ---
    logger.info("--- Extracting phone number candidates ---")
    candidates = []
    if scraped_data:
        text_content = scraped_data[0].get("filepath")
        if text_content:
            with open(text_content, 'r', encoding='utf-8') as f:
                content = f.read()
            candidates = extract_numbers_with_snippets_from_text(content, test_url, "Exxomove", ["DE"])

    # --- Process candidates with LLM ---
    logger.info("--- Processing candidates with LLM ---")
    if candidates:
        chunk_processor = LLMChunkProcessor(config, gemini_client, "prompts/phone_extraction_prompt.txt")
        processed_numbers, _, _ = chunk_processor.process_candidates(candidates, "output_data", "Exxomove", "test_row", "Exxomove")

        # --- Consolidate and assert ---
        logger.info("--- Consolidating and asserting results ---")
        if processed_numbers:
            final_numbers = process_and_consolidate_contact_data(processed_numbers, "Exxomove", test_url)
            if final_numbers:
                assert any(p.classification in ["Primary", "Secondary"] for p in final_numbers.consolidated_numbers), "No primary or secondary phone number found for Exxomove"
                logger.info("✅ Exxomove has a found phone number")
            else:
                logger.error("🔥 Phone number consolidation failed.")
        else:
            logger.error("🔥 No numbers processed by LLM")
    else:
        logger.error("🔥 No candidates found")
