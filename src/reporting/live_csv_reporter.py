import csv
import logging
import os
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LiveCsvReporter:
    """
    Handles the creation and incremental updating of a live CSV report during a pipeline run.
    """
    def __init__(self, filepath: str, header: List[str]):
        """
        Initializes the reporter, creates the CSV file, and writes the header.

        Args:
            filepath (str): The full path to the CSV file to be created.
            header (List[str]): A list of strings representing the CSV header.
        """
        self.filepath = filepath
        self.header = header
        self._initialize_file()

    def _initialize_file(self):
        """Creates the file and writes the header. Overwrites if the file exists."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(self.header)
            logger.info(f"Live report initialized at: {self.filepath}")
        except IOError as e:
            logger.error(f"Failed to initialize live report file at {self.filepath}: {e}")
            raise

    def append_row(self, row_data: Dict[str, Any]):
        """
        Appends a single row of data to the CSV file.

        Args:
            row_data (Dict[str, Any]): A dictionary where keys match the header.
        """
        try:
            # Create a list of values in the same order as the header
            row_values = [row_data.get(h, '') for h in self.header]
            with open(self.filepath, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(row_values)
        except IOError as e:
            logger.error(f"Failed to append row to live report file {self.filepath}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while appending a row: {e}")
