import os
import json
import hashlib
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

def generate_cache_key(url: str) -> str:
    """
    Generates a safe and unique filename key for a given URL.
    """
    # Normalize URL to ensure consistency and hash it for a safe filename
    normalized_url = url.strip().lower()
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()

def load_from_cache(key: str, cache_dir: str) -> Optional[List[Dict[str, Any]]]:
    """
    Loads scraping results from a cache file if it exists.
    """
    cache_path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                logger.info(f"Cache hit. Loading results from {cache_path}")
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading from cache file {cache_path}: {e}")
            return None
    return None

def save_to_cache(key: str, data: List[Dict[str, Any]], cache_dir: str):
    """
    Saves scraping results to a cache file.
    """
    if not data:
        return # Do not save empty results

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved results to cache: {cache_path}")
    except IOError as e:
        logger.error(f"Error saving to cache file {cache_path}: {e}")