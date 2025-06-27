"""
Handles the LLM task of extracting detailed company attributes from a summary.
"""
import logging
import json
from typing import Dict, Any, List, Tuple, Optional

import google.generativeai.types as genai_types
from google.api_core import exceptions as google_exceptions
from pydantic import ValidationError as PydanticValidationError

from ...core.config import AppConfig
from ...core.schemas import WebsiteTextSummary, DetailedCompanyAttributes
from ...utils.helpers import sanitize_filename_component
from ...llm_clients.gemini_client import GeminiClient
from ...utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
    extract_json_from_text,
)

logger = logging.getLogger(__name__)

def extract_detailed_attributes(
    gemini_client: GeminiClient,
    config: AppConfig,
    summary_obj: WebsiteTextSummary,
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str
) -> Tuple[Optional[DetailedCompanyAttributes], Optional[str], Optional[Dict[str, int]]]:
    """
    Extracts detailed company attributes using an LLM, based on a previously generated website summary.

    Args:
        gemini_client: The Gemini client for API interactions.
        config: The application configuration object (`AppConfig`).
        summary_obj: The `WebsiteTextSummary` object containing the summary
                     and original URL to be used as input for attribute extraction.
        llm_context_dir: Directory to save LLM interaction artifacts.
        llm_requests_dir: Directory to save LLM request payloads.
        file_identifier_prefix: Prefix for naming saved artifact files.
        triggering_input_row_id: Identifier of the original input data row.
        triggering_company_name: The name of the company.

    Returns:
        A tuple containing:
        - `parsed_output`: An instance of `DetailedCompanyAttributes` if successful,
          otherwise `None`. The `input_summary_url` from `summary_obj` is added.
        - `raw_llm_response_str`: The raw text response from the LLM or an
          error message.
        - `token_stats`: A dictionary with token usage statistics.
    """
    log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: DetailedAttributes]"
    logger.info(f"{log_prefix} Starting detailed attribute extraction.")

    raw_llm_response_str: Optional[str] = None
    token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    parsed_output: Optional[DetailedCompanyAttributes] = None
    prompt_template_path: str = "Path not initialized"
    try:
        if not hasattr(config, 'PROMPT_PATH_ATTRIBUTE_EXTRACTOR') or not config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR:
            logger.error(f"{log_prefix} AppConfig.PROMPT_PATH_ATTRIBUTE_EXTRACTOR is not set.")
            return None, "Error: PROMPT_PATH_ATTRIBUTE_EXTRACTOR not configured.", token_stats
        prompt_template_path = config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR
        prompt_template = load_prompt_template(prompt_template_path)
        website_summary_text = summary_obj.summary
        if not website_summary_text:
             logger.warning(f"{log_prefix} Website summary text from summary_obj.summary is empty. Proceeding, but LLM might not perform well.")
             website_summary_text = ""
        formatted_prompt = prompt_template.replace("{{WEBSITE_SUMMARY_TEXT_PLACEHOLDER}}", website_summary_text)
    except FileNotFoundError:
        ptp_for_log = prompt_template_path if prompt_template_path != "Path not initialized" else "Unknown path (config missing or error before assignment)"
        logger.error(f"{log_prefix} Prompt template file not found: {ptp_for_log}")
        return None, f"Error: Prompt template file not found: {ptp_for_log}", token_stats
    except AttributeError as e_attr:
        logger.error(f"{log_prefix} Configuration error: {e_attr}")
        return None, f"Error: Configuration error - {str(e_attr)}", token_stats
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format attribute extractor prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", token_stats
    s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15)
    s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)
    s_comp_name = sanitize_filename_component(triggering_company_name, max_len=config.filename_company_name_max_len if hasattr(config, 'filename_company_name_max_len') and config.filename_company_name_max_len is not None and config.filename_company_name_max_len <= 20 else 20)
    prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
    prompt_filename_with_suffix = f"{prompt_filename_base}_attribute_extractor_prompt.txt"
    try:
        save_llm_artifact(
            content=formatted_prompt,
            directory=llm_context_dir,
            filename=prompt_filename_with_suffix,
            log_prefix=log_prefix
        )
    except Exception as e_save_prompt:
         logger.error(f"{log_prefix} Failed to save formatted prompt artifact '{prompt_filename_with_suffix}': {e_save_prompt}", exc_info=True)
    try:
        max_tokens_val = config.llm_max_tokens
        temperature_val = config.llm_temperature_extraction
        
        generation_config_dict = {
            "response_mime_type": "text/plain",
            "candidate_count": 1,
            "max_output_tokens": max_tokens_val,
            "temperature": temperature_val,
        }
        if hasattr(config, 'llm_top_k') and config.llm_top_k is not None:
            generation_config_dict["top_k"] = config.llm_top_k
        if hasattr(config, 'llm_top_p') and config.llm_top_p is not None:
            generation_config_dict["top_p"] = config.llm_top_p
        
        generation_config = genai_types.GenerationConfig(**generation_config_dict)
    except AttributeError as e_attr_config:
        logger.error(f"{log_prefix} Configuration error for generation_config: {e_attr_config}")
        return None, f"Error: Configuration error for generation_config - {str(e_attr_config)}", token_stats
    except Exception as e_gen_config:
        logger.error(f"{log_prefix} Error creating generation_config: {e_gen_config}", exc_info=True)
        return None, f"Error: Creating generation_config - {str(e_gen_config)}", token_stats
    system_instruction_text = (
        "You are a data extraction assistant. Your entire response MUST be a single, "
        "valid JSON formatted string. Do NOT include any explanations, markdown formatting (like ```json), "
        "or any other text outside of this JSON string. The JSON object must strictly conform to the "
        "DetailedCompanyAttributes schema. Use `null` for optional fields if the information is not present or cannot be determined. "
        "Ensure all fields of the DetailedCompanyAttributes schema are considered."
    )
    
    contents_for_api: List[genai_types.ContentDict] = [
        {"role": "user", "parts": [{"text": formatted_prompt}]}
    ]
    serializable_contents = []
    try:
        for content_item in contents_for_api:
            serializable_contents.append(content_item)
    except Exception as e_serialize_contents:
        logger.error(f"{log_prefix} Error serializing contents_for_api for logging: {e_serialize_contents}")
        serializable_contents = [{"error": "failed to serialize contents"}]
    request_payload_to_log = {
        "model_name": config.llm_model_name,
        "system_instruction": system_instruction_text,
        "user_contents": serializable_contents,
        "generation_config": generation_config_dict
    }
    request_payload_filename = f"{prompt_filename_base}_attribute_extractor_request_payload.json"
    try:
        save_llm_artifact(
            content=json.dumps(request_payload_to_log, indent=2),
            directory=llm_requests_dir,
            filename=request_payload_filename,
            log_prefix=log_prefix
        )
    except Exception as e_save_payload:
        logger.error(f"{log_prefix} Failed to save request payload artifact: {e_save_payload}", exc_info=True)
    raw_llm_response_str_current_call: Optional[str] = None
    try:
        response = gemini_client.generate_content_with_retry(
            contents=contents_for_api,
            generation_config=generation_config,
            system_instruction=system_instruction_text,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name
        )
        if response:
            try:
                raw_llm_response_str_current_call = response.text
            except Exception as e_text_access:
                logger.error(f"{log_prefix} Error accessing response.text: {e_text_access}", exc_info=True)
                raw_llm_response_str_current_call = f"Error accessing response text: {str(e_text_access)}"
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_stats["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                token_stats["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                token_stats["total_tokens"] = response.usage_metadata.total_token_count or 0
            else:
                logger.warning(f"{log_prefix} LLM usage metadata not found or incomplete in response.")
            
            logger.info(f"{log_prefix} LLM usage: {token_stats}")
            if raw_llm_response_str_current_call:
                response_filename = f"{prompt_filename_base}_attribute_extractor_response.txt"
                try:
                    save_llm_artifact(
                        content=raw_llm_response_str_current_call,
                        directory=llm_context_dir,
                        filename=response_filename,
                        log_prefix=log_prefix
                    )
                except Exception as e_save_resp:
                    logger.error(f"{log_prefix} Failed to save raw LLM response artifact: {e_save_resp}", exc_info=True)
            
            json_string_from_text: Optional[str] = None
            if response.candidates and raw_llm_response_str_current_call and raw_llm_response_str_current_call.strip():
                try:
                    json_string_from_text = extract_json_from_text(raw_llm_response_str_current_call)
                    if json_string_from_text:
                        parsed_json_object = json.loads(json_string_from_text)
                        parsed_output = DetailedCompanyAttributes(**parsed_json_object)
                        
                        if summary_obj and hasattr(summary_obj, 'original_url') and summary_obj.original_url is not None:
                            parsed_output.input_summary_url = summary_obj.original_url
                        else:
                            logger.warning(f"{log_prefix} summary_obj.original_url is missing or None. Cannot set input_summary_url on DetailedCompanyAttributes.")
                        logger.info(f"{log_prefix} Successfully extracted, parsed, validated DetailedCompanyAttributes, and attempted to set input_summary_url.")
                    else:
                        logger.error(f"{log_prefix} Failed to extract JSON string from LLM's plain text response for detailed attributes. Raw: '{raw_llm_response_str_current_call[:500]}'")
                        
                except json.JSONDecodeError as e_json:
                    logger.error(f"{log_prefix} Failed to parse extracted JSON for detailed attributes: {e_json}. Extracted: '{json_string_from_text[:500] if json_string_from_text else 'N/A'}'. Raw: '{raw_llm_response_str_current_call[:200]}'")
                except PydanticValidationError as e_pydantic:
                    logger.error(f"{log_prefix} Pydantic validation failed for DetailedCompanyAttributes: {e_pydantic}. Data: '{json_string_from_text[:500] if json_string_from_text else 'N/A'}'")
            elif not response.candidates:
                 logger.warning(f"{log_prefix} No candidates in Gemini response for detailed attributes. Raw: '{raw_llm_response_str_current_call[:200] if raw_llm_response_str_current_call else 'N/A'}'")
            elif not raw_llm_response_str_current_call or not raw_llm_response_str_current_call.strip():
                logger.warning(f"{log_prefix} LLM response text is empty or whitespace only for detailed attributes.")
        else:
            logger.error(f"{log_prefix} No response object returned from GeminiClient for detailed attributes.")
            raw_llm_response_str_current_call = "Error: No response object from GeminiClient."
        
        raw_llm_response_str = raw_llm_response_str_current_call
        return parsed_output, raw_llm_response_str, token_stats
    except google_exceptions.GoogleAPIError as e_api:
        logger.error(f"{log_prefix} Gemini API error during detailed attribute extraction: {e_api}", exc_info=True)
        error_msg = getattr(e_api, 'message', str(e_api))
        raw_llm_response_str = json.dumps({"error": f"Gemini API error: {error_msg}", "type": type(e_api).__name__})
        return None, raw_llm_response_str, token_stats
    except PydanticValidationError as e_pydantic_outer:
        logger.error(f"{log_prefix} Outer Pydantic validation error for detailed attributes (should not happen if inner parsing is robust): {e_pydantic_outer}", exc_info=True)
        raw_llm_response_str = raw_llm_response_str_current_call if raw_llm_response_str_current_call else f"Pydantic validation error: {str(e_pydantic_outer)}"
        return None, raw_llm_response_str, token_stats
    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during detailed attribute extraction: {e_gen}", exc_info=True)
        if raw_llm_response_str_current_call:
             raw_llm_response_str = raw_llm_response_str_current_call
        else:
             raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
        return None, raw_llm_response_str, token_stats