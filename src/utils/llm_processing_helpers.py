import logging
import json
import re
import os
from typing import Dict, Any, List, Optional, Type

import phonenumbers
from phonenumbers import PhoneNumberFormat
from pydantic import BaseModel

# Relative imports for modules within the project
# from ..core.config import AppConfig # AppConfig might not be needed if all configs are passed as args
from .helpers import sanitize_filename_component

logger = logging.getLogger(__name__)

def load_prompt_template(prompt_file_path: str) -> str:
    """
    Loads a prompt template from the specified file path.

    Args:
        prompt_file_path (str): The absolute or relative path to the prompt
                                template file.

    Returns:
        str: The content of the prompt template file as a string.

    Raises:
        FileNotFoundError: If the prompt template file cannot be found.
        Exception: For other errors encountered during file reading.
    """
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template file not found: {prompt_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt template file {prompt_file_path}: {e}")
        raise

def extract_json_from_text(text_output: Optional[str]) -> Optional[str]:
    """
    Extracts a JSON string from a larger text block, potentially cleaning
    markdown code fences.

    Args:
        text_output (Optional[str]): The raw text output from the LLM.

    Returns:
        Optional[str]: The extracted JSON string, or None if not found or input is invalid.
    """
    if not text_output:
        return None

    def _extract_balanced_json(candidate_text: str) -> Optional[str]:
        start_positions = [
            idx for idx, char in enumerate(candidate_text)
            if char in "{["
        ]
        for start_idx in start_positions:
            opening = candidate_text[start_idx]
            closing = "}" if opening == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for end_idx in range(start_idx, len(candidate_text)):
                char = candidate_text[end_idx]
                if escape:
                    escape = False
                    continue
                if char == "\\" and in_string:
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        return candidate_text[start_idx:end_idx + 1].strip()
        return None

    stripped = text_output.strip()
    fence_matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    for fenced_block in fence_matches:
        balanced_json = _extract_balanced_json(fenced_block.strip())
        if balanced_json:
            return balanced_json

    balanced_json = _extract_balanced_json(stripped)
    if balanced_json:
        return balanced_json

    logger.debug(f"No clear JSON block found in LLM text output: {text_output[:200]}...")
    return None

def normalize_phone_number(
    number_str: str, 
    country_codes: List[str], 
    default_region_code: Optional[str]
) -> Optional[str]:
    """
    Normalizes a given phone number string to E.164 format.

    It attempts to parse the number using each of the provided `country_codes`
    as a region hint. If unsuccessful and `default_region_code` is provided,
    it falls back to using that.

    Args:
        number_str (str): The phone number string to normalize.
        country_codes (List[str]): A list of ISO 3166-1 alpha-2 country codes
                                   (e.g., ["US", "DE"]) to use as hints for parsing.
        default_region_code (Optional[str]): A default ISO 3166-1 alpha-2 country code
                                             to use as a fallback if parsing with `country_codes` fails.

    Returns:
        Optional[str]: The normalized phone number in E.164 format if successful,
                       otherwise None.
    """
    if not number_str or not isinstance(number_str, str):
        logger.debug(f"Invalid input for phone normalization: {number_str}")
        return None

    for country_code in country_codes:
        try:
            parsed_num = phonenumbers.parse(number_str, region=country_code.upper())
            if phonenumbers.is_valid_number(parsed_num):
                normalized = phonenumbers.format_number(parsed_num, PhoneNumberFormat.E164)
                logger.debug(f"Normalized '{number_str}' to '{normalized}' using region '{country_code}'.")
                return normalized
        except phonenumbers.NumberParseException:
            logger.debug(f"Could not parse '{number_str}' with region '{country_code}'.")
            continue
    
    if default_region_code:
        try:
            parsed_num = phonenumbers.parse(number_str, region=default_region_code.upper())
            if phonenumbers.is_valid_number(parsed_num):
                normalized = phonenumbers.format_number(parsed_num, PhoneNumberFormat.E164)
                logger.debug(f"Normalized '{number_str}' to '{normalized}' using default region '{default_region_code}'.")
                return normalized
        except phonenumbers.NumberParseException:
            logger.info(f"Could not parse phone number '{number_str}' even with default region '{default_region_code}'.")
            
    logger.info(f"Could not normalize phone number '{number_str}' to E.164 with hints {country_codes} or default region '{default_region_code}'.")
    return None

def save_llm_artifact(content: str, directory: str, filename: str, log_prefix: str) -> None:
    """
    Saves text content (like prompts or responses) to a file, ensuring the
    directory exists and sanitizing the filename.

    Args:
        content (str): The string content to save.
        directory (str): The directory path to save the file in.
        filename (str): The name of the file (will be sanitized).
        log_prefix (str): A string prefix for log messages (e.g., from the calling function).
    """
    try:
        os.makedirs(directory, exist_ok=True)
        # Assuming a max_len for the filename component if desired, otherwise sanitize_filename_component default.
        # For consistency with llm_extractor, let's use a placeholder for max_len or remove if not strictly needed here.
        # For now, let's assume sanitize_filename_component handles it well without max_len or use a sensible default.
        sanitized_filename = sanitize_filename_component(filename) 
        filepath = os.path.join(directory, sanitized_filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"{log_prefix} Successfully saved artifact to {filepath}")
    except OSError as e:
        logger.error(f"{log_prefix} OSError creating directory {directory} or saving artifact: {e}")
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error saving artifact {os.path.join(directory, filename)}: {e}")


def adapt_schema_for_gemini(pydantic_model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """
    Adapts a Pydantic model's JSON schema for compatibility with the Gemini API's
    `response_schema` parameter. This involves:
    - Generating the JSON schema from the Pydantic model.
    - Removing "default" keys from property definitions.
    - Simplifying "anyOf" structures typically used for Optional fields
      (e.g., Optional[str]) to the non-null type.
    - Removing the top-level "title" from the schema.
    - Ensuring the top-level schema has "type": "object".
    - Ensuring the top-level schema has a "properties" key (even if empty).

    Args:
        pydantic_model_cls (Type[BaseModel]): The Pydantic model class.

    Returns:
        Dict[str, Any]: The modified schema dictionary.
    """
    model_schema = pydantic_model_cls.model_json_schema()
    defs = model_schema.get("$defs", {}) if isinstance(model_schema, dict) else {}

    def _resolve_ref(ref: str) -> Optional[Any]:
        if not ref.startswith("#/$defs/"):
            return None
        ref_name = ref.split("/")[-1]
        return defs.get(ref_name)

    def _transform(node: Any) -> Any:
        if isinstance(node, list):
            return [_transform(item) for item in node]

        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str):
            resolved = _resolve_ref(ref)
            if resolved is None:
                return {}
            merged = dict(resolved)
            for key, value in node.items():
                if key == "$ref":
                    continue
                merged[key] = value
            return _transform(merged)

        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            if key in {"title", "default", "$defs"}:
                continue
            cleaned[key] = _transform(value)

        any_of = cleaned.get("anyOf")
        if isinstance(any_of, list):
            non_null = [
                item for item in any_of
                if not (isinstance(item, dict) and item.get("type") == "null")
            ]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                replacement = dict(non_null[0])
                if "description" in cleaned and "description" not in replacement:
                    replacement["description"] = cleaned["description"]
                return _transform(replacement)
            cleaned["anyOf"] = non_null

        if cleaned.get("type") == "object":
            cleaned.setdefault("properties", {})
            cleaned.pop("additionalProperties", None)

        return cleaned

    adapted = _transform(model_schema)
    if isinstance(adapted, dict):
        adapted.pop("$defs", None)
        adapted.pop("title", None)
        adapted.pop("default", None)
        adapted["type"] = "object"
        adapted.setdefault("properties", {})

    logger.debug(f"Adapted schema for {pydantic_model_cls.__name__}: {json.dumps(adapted, indent=2)}")
    return adapted