import asyncio
import time
import logging
from src.core.config import AppConfig
from src.scraper.scraper_logic import scrape_website
from src.phone_retrieval.extractors.regex_extractor import extract_numbers_with_snippets_from_text
from src.phone_retrieval.extractors.llm_chunk_processor import LLMChunkProcessor
from src.phone_retrieval.llm_clients.gemini_client import GeminiClient

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """
    Main function to run performance tests.
    """
    config = AppConfig()
    gemini_client = GeminiClient(config)
    candidates = []

    # --- Test Scraper Performance ---
    logger.info("--- Testing Scraper Performance ---")
    test_url = "https://www.exxomove.de/"
    start_time = time.time()
    scraped_data, _, _, _ = await scrape_website(test_url, "output_data", "Exxomove", set(), "test_row")
    end_time = time.time()
    logger.info(f"Scraping took {end_time - start_time:.2f} seconds.")

    # --- Test Regex Extractor Performance ---
    logger.info("--- Testing Regex Extractor Performance ---")
    if scraped_data:
        text_content = scraped_data[0].get("filepath")
        if text_content:
            with open(text_content, 'r', encoding='utf-8') as f:
                content = f.read()
            start_time = time.time()
            candidates = extract_numbers_with_snippets_from_text(content, test_url, "Exxomove", ["DE"])
            end_time = time.time()
            logger.info(f"Regex extraction took {end_time - start_time:.2f} seconds and found {len(candidates)} candidates.")

    # --- Test LLM Chunk Processor Performance ---
    logger.info("--- Testing LLM Chunk Processor Performance ---")
    if 'candidates' in locals() and candidates:
        chunk_processor = LLMChunkProcessor(config, gemini_client, "prompts/phone_extraction_prompt.txt")
        start_time = time.time()
        chunk_processor.process_candidates(candidates, "output_data", "Exxomove", "test_row", "Exxomove")
        end_time = time.time()
        logger.info(f"LLM chunk processing took {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())