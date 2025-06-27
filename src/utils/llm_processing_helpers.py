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

    # Regex to find content within ```json ... ``` or ``` ... ```,
    # or a standalone JSON object/array.
    # It tries to capture the content inside the innermost curly braces or square brackets.
    match = re.search(
        r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```|(\{.*\}|\[.*\])",
        text_output,
        re.DOTALL  # DOTALL allows . to match newlines
    )

    if match:
        # Prioritize the content within backticks if both groups match
        json_str = match.group(1) or match.group(2)
        if json_str:
            return json_str.strip()
    
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
    
    # Process properties
    if "properties" in model_schema:
        for prop_name, prop_details in list(model_schema["properties"].items()):
            if isinstance(prop_details, dict): # Ensure prop_details is a dict
                # Remove "default" from properties
                if "default" in prop_details:
                    del prop_details["default"]
                
                # Remove "title" from individual properties
                if "title" in prop_details:
                    del prop_details["title"]
                
                # Simplify "anyOf" for Optional fields
                if "anyOf" in prop_details and isinstance(prop_details["anyOf"], list):
                    non_null_schemas = [
                        s for s in prop_details["anyOf"] 
                        if isinstance(s, dict) and s.get("type") != "null"
                    ]
                    if len(non_null_schemas) == 1:
                        # Replace the property's schema with the single non-null schema
                        # This preserves other keys at the property level if any (e.g. description)
                        # by updating the current prop_details with the non_null_schema
                        
                        # Create a new dict for the simplified property, preserving original keys not in anyOf
                        simplified_prop = {k: v for k, v in prop_details.items() if k != "anyOf"}
                        simplified_prop.update(non_null_schemas[0])
                        model_schema["properties"][prop_name] = simplified_prop
                        
                    elif len(non_null_schemas) > 1:
                        # If multiple non-null types in anyOf (e.g. Union[str, int]),
                        # keep the anyOf but remove the null type if present.
                        # This case might need more specific handling based on Gemini's exact requirements for Unions.
                        # For now, we'll just filter out the null type.
                        prop_details["anyOf"] = non_null_schemas
                        if not prop_details["anyOf"]: # Should not happen if there were non_null_schemas
                             del model_schema["properties"][prop_name] # Or handle as error
                        elif len(prop_details["anyOf"]) == 1: # If only one non-null type remains
                             simplified_prop = {k: v for k, v in prop_details.items() if k != "anyOf"}
                             simplified_prop.update(prop_details["anyOf"][0])
                             model_schema["properties"][prop_name] = simplified_prop

                    # If only null was in anyOf or anyOf was malformed, it might become empty or invalid.
                    # Pydantic usually ensures Optional[X] has X and null.
            else:
                logger.warning(f"Property '{prop_name}' in schema for {pydantic_model_cls.__name__} is not a dictionary. Skipping adaptation for this property.")


    # Remove top-level "title"
    if "title" in model_schema:
        del model_schema["title"]
    
    # Remove top-level "default" if present
    if "default" in model_schema:
        del model_schema["default"]

    # Ensure top-level "type": "object"
    model_schema["type"] = "object"
    
    # Ensure "properties" key exists at top-level
    if "properties" not in model_schema:
        model_schema["properties"] = {}
        
    logger.debug(f"Adapted schema for {pydantic_model_cls.__name__}: {json.dumps(model_schema, indent=2)}")
    return model_schema