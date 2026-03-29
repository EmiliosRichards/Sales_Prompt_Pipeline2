import json
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

from ...core.config import AppConfig
from ...core.schemas import DetailedCompanyAttributes, PartnerMatchOnlyOutput
from ...utils.helpers import sanitize_filename_component
from ...utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
)

logger = logging.getLogger(__name__)


def _salvage_partial_partner_match(raw_text: str) -> Optional[PartnerMatchOnlyOutput]:
    if not raw_text:
        return None
    match_score_match = re.search(r'"match_score"\s*:\s*"([^"]+)"', raw_text)
    partner_name_match = re.search(r'"matched_partner_name"\s*:\s*"([^"]+)"', raw_text)
    if not match_score_match or not partner_name_match:
        return None
    return PartnerMatchOnlyOutput(
        match_score=match_score_match.group(1).strip(),
        matched_partner_name=partner_name_match.group(1).strip(),
        match_rationale_features=[],
    )

def match_partner(
    gemini_client: Any,
    config: AppConfig,
    target_attributes: DetailedCompanyAttributes,
    golden_partner_summaries: List[Dict[str, Any]],
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str
) -> Tuple[Optional[PartnerMatchOnlyOutput], Optional[str], Optional[Dict[str, int]]]:
    """
    Identifies the best golden partner match for a target company using an LLM.

    Args:
        gemini_client: The Gemini client for API interactions.
        config: The application configuration object (`AppConfig`).
        target_attributes: The `DetailedCompanyAttributes` object for the company being analyzed.
        golden_partner_summaries: A list of dictionaries with golden partner summaries.
        llm_context_dir: Directory to save LLM interaction artifacts.
        llm_requests_dir: Directory to save LLM request payloads.
        file_identifier_prefix: Prefix for naming saved artifact files.
        triggering_input_row_id: Identifier of the original input data row.
        triggering_company_name: The name of the company being analyzed.

    Returns:
        A tuple containing:
        - `parsed_output`: An instance of `PartnerMatchOnlyOutput` if successful, otherwise `None`.
        - `raw_llm_response_str`: The raw text response from the LLM or an error message.
        - `token_stats`: A dictionary with token usage statistics.
    """
    log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: PartnerMatch]"
    logger.info(f"{log_prefix} Starting partner matching.")

    prompt_template_path: str = "Path not initialized"

    try:
        prompt_template_path = config.PROMPT_PATH_GERMAN_PARTNER_MATCHING
        prompt_template = load_prompt_template(prompt_template_path)
        target_attributes_json = target_attributes.model_dump_json(indent=2)
        partner_summaries_str = "\n".join([f"{i+1}. {json.dumps(summary)}" for i, summary in enumerate(golden_partner_summaries)])
        formatted_prompt = prompt_template.replace("{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}", target_attributes_json)
        formatted_prompt = formatted_prompt.replace("{{GOLDEN_PARTNER_SUMMARIES_PLACEHOLDER}}", partner_summaries_str)
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format partner matching prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15)
    s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)
    s_comp_name = sanitize_filename_component(triggering_company_name, max_len=config.filename_company_name_max_len if hasattr(config, 'filename_company_name_max_len') and config.filename_company_name_max_len is not None and config.filename_company_name_max_len <= 20 else 20)

    prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
    prompt_filename_with_suffix = f"{prompt_filename_base}_partner_match_prompt.txt"
    try:
        save_llm_artifact(
            content=formatted_prompt,
            directory=llm_requests_dir,
            filename=prompt_filename_with_suffix,
            log_prefix=log_prefix
        )
    except Exception as e_save_prompt:
         logger.error(f"{log_prefix} Failed to save formatted prompt artifact '{prompt_filename_with_suffix}': {e_save_prompt}", exc_info=True)

    system_instruction_text = (
        "You match the target company to the best golden partner from the provided list. "
        "Return only valid JSON matching the requested schema."
    )
    request_payload_to_log = {
        "model_name": getattr(config, "llm_model_name", ""),
        "system_instruction": system_instruction_text,
        "schema_name": "partner_match_only",
        "response_model": "PartnerMatchOnlyOutput",
        "max_output_tokens": config.llm_max_tokens,
        "temperature": config.llm_temperature_extraction,
        "user_prompt": formatted_prompt,
    }
    request_payload_filename = f"{prompt_filename_base}_partner_match_request_payload.json"
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
            response_model=PartnerMatchOnlyOutput,
            schema_name="partner_match_only",
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_prompt=system_instruction_text,
            temperature=config.llm_temperature_extraction,
            max_output_tokens=config.llm_max_tokens,
        )
        token_stats = llm_result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if llm_result.raw_text:
            response_filename = f"{prompt_filename_base}_partner_match_response.txt"
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
            logger.info(f"{log_prefix} Successfully parsed PartnerMatchOnlyOutput.")
            return llm_result.parsed_output, llm_result.raw_text, token_stats
        salvaged_output = _salvage_partial_partner_match(llm_result.raw_text or "")
        if salvaged_output is not None:
            logger.warning(
                f"{log_prefix} Salvaged partial PartnerMatchOnlyOutput from malformed structured response."
            )
            return salvaged_output, llm_result.raw_text, token_stats
        error_message = llm_result.provider_error or (
            f"Model refusal: {llm_result.refusal}" if llm_result.refusal else "Failed to parse partner matching structured output."
        )
        logger.error(f"{log_prefix} {error_message}")
        return None, llm_result.raw_text or error_message, token_stats
    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during partner matching: {e_gen}", exc_info=True)
        raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
        return None, raw_llm_response_str, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}