"""
Generate a short German summary from an input-provided description blob.

Used primarily when running the sales pipeline with --pitch-from-description
(skip scraping, but still create a human-readable German summary).
"""
import json
import logging
from typing import Any, Dict, Optional, Tuple

from ...core.config import AppConfig
from ...core.schemas import GermanShortSummaryOutput
from ...utils.helpers import sanitize_filename_component
from ...utils.llm_processing_helpers import load_prompt_template, save_llm_artifact

logger = logging.getLogger(__name__)


def _truncate_to_100_words(text: str) -> str:
    words = [w for w in (text or "").strip().split() if w]
    if len(words) <= 100:
        return (text or "").strip()
    return " ".join(words[:100]).strip()


def generate_german_short_summary_from_description(
    gemini_client: Any,
    config: AppConfig,
    description_text: str,
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, int]]]:
    """
    Returns (german_summary, raw_response_or_error, token_stats).
    """
    log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: GermanShortSummary]"
    token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    raw_llm_response_str: Optional[str] = None
    try:
        prompt_template_path = getattr(config, "PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION", "")
        if not prompt_template_path:
            logger.error(f"{log_prefix} AppConfig.PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION is not set.")
            return None, "Error: PROMPT_PATH_GERMAN_SHORT_SUMMARY_FROM_DESCRIPTION not configured.", token_stats

        prompt_template = load_prompt_template(prompt_template_path)
        max_chars = int(getattr(config, "LLM_MAX_INPUT_CHARS_FOR_DESCRIPTION_DE_SUMMARY", 12000) or 12000)
        text_for_prompt = (description_text or "").strip()
        if max_chars > 0 and len(text_for_prompt) > max_chars:
            logger.warning(f"{log_prefix} Truncating description_text from {len(text_for_prompt)} to {max_chars} chars.")
            text_for_prompt = text_for_prompt[:max_chars]

        formatted_prompt = prompt_template.replace("{{INPUT_DESCRIPTION_PLACEHOLDER}}", text_for_prompt)

        # Save prompt artifact
        s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15)
        s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)
        s_comp_name = sanitize_filename_component(
            triggering_company_name,
            max_len=config.filename_company_name_max_len
            if hasattr(config, "filename_company_name_max_len")
            and config.filename_company_name_max_len is not None
            and config.filename_company_name_max_len <= 20
            else 20,
        )
        prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
        prompt_filename = f"{prompt_filename_base}_german_short_summary_prompt.txt"
        try:
            save_llm_artifact(formatted_prompt, llm_context_dir, prompt_filename, log_prefix=log_prefix)
        except Exception:
            pass

        max_out = int(getattr(config, "llm_max_tokens_description_de_summary", 256) or 256)
        system_instruction_text = (
            "You translate and condense the provided company description into a short German summary. "
            "Return only valid JSON matching the requested schema."
        )
        request_payload = {
            "model_name": getattr(config, "llm_model_name", ""),
            "system_instruction": system_instruction_text,
            "schema_name": "german_short_summary",
            "response_model": "GermanShortSummaryOutput",
            "max_output_tokens": max_out,
            "temperature": getattr(config, "llm_temperature_extraction", 0.2),
            "user_prompt": formatted_prompt,
        }
        request_payload_filename = f"{prompt_filename_base}_german_short_summary_request_payload.json"
        try:
            save_llm_artifact(json.dumps(request_payload, indent=2), llm_requests_dir, request_payload_filename, log_prefix=log_prefix)
        except Exception:
            pass

        llm_result = gemini_client.generate_structured_output_with_retry(
            user_prompt=formatted_prompt,
            response_model=GermanShortSummaryOutput,
            schema_name="german_short_summary",
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_prompt=system_instruction_text,
            temperature=getattr(config, "llm_temperature_extraction", 0.2),
            max_output_tokens=max_out,
        )
        raw_llm_response_str = llm_result.raw_text
        token_stats = llm_result.usage or token_stats

        # Save raw response artifact
        try:
            if raw_llm_response_str:
                resp_filename = f"{prompt_filename_base}_german_short_summary_response.txt"
                save_llm_artifact(raw_llm_response_str, llm_context_dir, resp_filename, log_prefix=log_prefix)
        except Exception:
            pass

        if not raw_llm_response_str or not str(raw_llm_response_str).strip():
            return None, "Error: Empty LLM response.", token_stats

        if not llm_result.parsed_output:
            error_message = llm_result.provider_error or (
                f"Model refusal: {llm_result.refusal}" if llm_result.refusal else "Error: Failed to parse structured German short summary."
            )
            return None, error_message, token_stats
        parsed = llm_result.parsed_output
        summary = _truncate_to_100_words(parsed.german_summary)
        return summary, raw_llm_response_str, token_stats

    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during German short summary generation: {e}", exc_info=True)
        return None, f"Error: Unexpected error: {str(e)}", token_stats

