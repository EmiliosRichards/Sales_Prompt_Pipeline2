"""Handles the LLM task of extracting detailed company attributes from a summary."""
import json
import logging
from typing import Dict, Any, Tuple, Optional

from ...core.config import AppConfig
from ...core.schemas import WebsiteTextSummary, DetailedCompanyAttributes, DetailedCompanyAttributesLLM
from ...utils.helpers import sanitize_filename_component
from ...utils.llm_processing_helpers import load_prompt_template, save_llm_artifact

logger = logging.getLogger(__name__)

def extract_detailed_attributes(
    gemini_client: Any,
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

    try:
        if not hasattr(config, "PROMPT_PATH_ATTRIBUTE_EXTRACTOR") or not config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR:
            logger.error(f"{log_prefix} AppConfig.PROMPT_PATH_ATTRIBUTE_EXTRACTOR is not set.")
            return None, "Error: PROMPT_PATH_ATTRIBUTE_EXTRACTOR not configured.", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt_template_path = config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR
        prompt_template = load_prompt_template(prompt_template_path)
        website_summary_text = summary_obj.summary
        if not website_summary_text:
             logger.warning(f"{log_prefix} Website summary text from summary_obj.summary is empty. Proceeding, but LLM might not perform well.")
             website_summary_text = ""
        formatted_prompt = prompt_template.replace("{{WEBSITE_SUMMARY_TEXT_PLACEHOLDER}}", website_summary_text)
    except FileNotFoundError:
        logger.error(f"{log_prefix} Prompt template file not found: {config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR}")
        return None, f"Error: Prompt template file not found: {config.PROMPT_PATH_ATTRIBUTE_EXTRACTOR}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except AttributeError as e_attr:
        logger.error(f"{log_prefix} Configuration error: {e_attr}")
        return None, f"Error: Configuration error - {str(e_attr)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format attribute extractor prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
    max_tokens_val = config.llm_max_tokens
    temperature_val = config.llm_temperature_extraction
    system_instruction_text = (
        "You extract detailed company attributes from a structured website summary. "
        "Return only valid JSON matching the requested schema."
    )
    request_payload_to_log = {
        "model_name": getattr(config, "llm_model_name", ""),
        "system_instruction": system_instruction_text,
        "schema_name": "detailed_company_attributes",
        "response_model": "DetailedCompanyAttributesLLM",
        "max_output_tokens": max_tokens_val,
        "temperature": temperature_val,
        "user_prompt": formatted_prompt,
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
    try:
        llm_result = gemini_client.generate_structured_output_with_retry(
            user_prompt=formatted_prompt,
            response_model=DetailedCompanyAttributesLLM,
            schema_name="detailed_company_attributes",
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_prompt=system_instruction_text,
            temperature=temperature_val,
            max_output_tokens=max_tokens_val,
        )
        token_stats = llm_result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        logger.info(f"{log_prefix} LLM usage: {token_stats}")
        if llm_result.raw_text:
            response_filename = f"{prompt_filename_base}_attribute_extractor_response.txt"
            try:
                save_llm_artifact(
                    content=llm_result.raw_text,
                    directory=llm_context_dir,
                    filename=response_filename,
                    log_prefix=log_prefix,
                )
            except Exception as e_save_resp:
                logger.error(f"{log_prefix} Failed to save raw LLM response artifact: {e_save_resp}", exc_info=True)

        if llm_result.parsed_output:
            parsed_output = DetailedCompanyAttributes(
                input_summary_url=summary_obj.original_url,
                **llm_result.parsed_output.model_dump(),
            )
            logger.info(f"{log_prefix} Successfully parsed DetailedCompanyAttributesLLM and enriched input_summary_url.")
            return parsed_output, llm_result.raw_text, token_stats

        error_message = llm_result.provider_error or (
            f"Model refusal: {llm_result.refusal}" if llm_result.refusal else "Failed to parse detailed attributes structured output."
        )
        logger.error(f"{log_prefix} {error_message}")
        return None, llm_result.raw_text or error_message, token_stats
    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during detailed attribute extraction: {e_gen}", exc_info=True)
        raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
        return None, raw_llm_response_str, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}