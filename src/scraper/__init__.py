"""
Initializes the scraper package, making its components available for import.

This package contains modules related to web scraping functionalities.
"""
from .scraper_logic import scrape_website

__all__ = ["scrape_website"]