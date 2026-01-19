import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class LiveJsonlReporter:
    """
    Append-only JSONL writer (one JSON object per line) for incremental pipeline progress.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._initialize_file()

    def _initialize_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                # Create/overwrite empty file
                pass
            logger.info(f"Live JSONL initialized at: {self.filepath}")
        except IOError as e:
            logger.error(f"Failed to initialize live JSONL file at {self.filepath}: {e}")
            raise

    def append_obj(self, obj: Dict[str, Any]) -> None:
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except IOError as e:
            logger.error(f"Failed to append JSONL object to {self.filepath}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error appending JSONL object: {e}")

