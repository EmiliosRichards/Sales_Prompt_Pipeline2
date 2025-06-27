"""
Handles the LLM task of generating a sales pitch for a target company,
given a matched golden partner.
"""
import logging
import json
from typing import Dict, Any, List, Tuple, Optional

import google.generativeai.types as genai_types
from google.api_core import exceptions as google_exceptions
from pydantic import ValidationError as PydanticValidationError

from ...core.config import AppConfig
from ...core.schemas import DetailedCompanyAttributes, GoldenPartnerMatchOutput, WebsiteTextSummary
from ...utils.helpers import sanitize_filename_component
from ...llm_clients.gemini_client import GeminiClient
from ...utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
    extract_json_from_text,
)

logger = logging.getLogger(__name__)

def generate_sales_pitch(
    gemini_client: GeminiClient,
    config: AppConfig,
    target_attributes: DetailedCompanyAttributes,
    matched_partner: Dict[str, Any],
    website_summary_obj: WebsiteTextSummary,
    previous_match_rationale: List[str],
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str
) -> Tuple[Optional[GoldenPartnerMatchOutput], Optional[str], Optional[Dict[str, int]]]:
    """
    Generates a sales pitch for a target company based on a matched golden partner.

    Args:
        gemini_client: The Gemini client for API interactions.
        config: The application configuration object (`AppConfig`).
        target_attributes: The `DetailedCompanyAttributes` object for the company being analyzed.
        matched_partner: A dictionary containing the full data of the matched golden partner.
        website_summary_obj: The `WebsiteTextSummary` object for the company being analyzed.
        previous_match_rationale: A list of strings representing the rationale from the partner matching step.
        llm_context_dir: Directory to save LLM interaction artifacts.
        llm_requests_dir: Directory to save LLM request payloads.
        file_identifier_prefix: Prefix for naming saved artifact files.
        triggering_input_row_id: Identifier of the original input data row.
        triggering_company_name: The name of the company being analyzed.

    Returns:
        A tuple containing:
        - `parsed_output`: An instance of `GoldenPartnerMatchOutput` if successful, otherwise `None`.
        - `raw_llm_response_str`: The raw text response from the LLM or an error message.
        - `token_stats`: A dictionary with token usage statistics.
    """
    log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: SalesPitch]"
    logger.info(f"{log_prefix} Starting sales pitch generation.")

    raw_llm_response_str: Optional[str] = None
    token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    parsed_output: Optional[GoldenPartnerMatchOutput] = None
    prompt_template_path: str = "Path not initialized"

    try:
        prompt_template_path = config.PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION
        prompt_template = load_prompt_template(prompt_template_path)
        target_attributes_json = target_attributes.model_dump_json(indent=2)
        matched_partner_json = json.dumps(matched_partner, indent=2)
        formatted_prompt = prompt_template.replace("{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}", target_attributes_json)
        formatted_prompt = formatted_prompt.replace("{{MATCHED_GOLDEN_PARTNER_JSON_PLACEHOLDER}}", matched_partner_json)
        previous_rationale_str = "\n".join([f"- {item}" for item in previous_match_rationale])
        formatted_prompt = formatted_prompt.replace("{{PREVIOUS_MATCH_RATIONALE_PLACEHOLDER}}", previous_rationale_str)
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format sales pitch prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", token_stats

    s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15)
    s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)
    s_comp_name = sanitize_filename_component(triggering_company_name, max_len=config.filename_company_name_max_len if hasattr(config, 'filename_company_name_max_len') and config.filename_company_name_max_len is not None and config.filename_company_name_max_len <= 20 else 20)

    prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
    prompt_filename_with_suffix = f"{prompt_filename_base}_sales_pitch_prompt.txt"
    try:
        save_llm_artifact(
            content=formatted_prompt,
            directory=llm_requests_dir,
            filename=prompt_filename_with_suffix,
            log_prefix=log_prefix
        )
    except Exception as e_save_prompt:
         logger.error(f"{log_prefix} Failed to save formatted prompt artifact '{prompt_filename_with_suffix}': {e_save_prompt}", exc_info=True)

    try:
        generation_config_dict = {
            "response_mime_type": "text/plain",
            "candidate_count": 1,
            "max_output_tokens": config.llm_max_tokens,
            "temperature": config.llm_temperature_creative,
        }
        if hasattr(config, 'llm_top_k') and config.llm_top_k is not None:
            generation_config_dict["top_k"] = config.llm_top_k
        if hasattr(config, 'llm_top_p') and config.llm_top_p is not None:
            generation_config_dict["top_p"] = config.llm_top_p
        
        generation_config = genai_types.GenerationConfig(**generation_config_dict)
    except Exception as e_gen_config:
        logger.error(f"{log_prefix} Error creating generation_config: {e_gen_config}", exc_info=True)
        return None, f"Error: Creating generation_config - {str(e_gen_config)}", token_stats

    system_instruction_text = (
        "You are a sales pitch generation assistant. Your entire response MUST be a single, "
        "valid JSON formatted string. Do NOT include any explanations, markdown formatting (like ```json), "
        "or any other text outside of this JSON string. The JSON object must strictly conform to the "
        "GoldenPartnerMatchOutput schema."
    )
    
    contents_for_api: List[genai_types.ContentDict] = [
        {"role": "user", "parts": [{"text": formatted_prompt}]}
    ]

    request_payload_to_log = {
        "model_name": config.llm_model_name_sales_insights,
        "system_instruction": system_instruction_text,
        "user_contents": contents_for_api,
        "generation_config": generation_config_dict
    }
    request_payload_filename = f"{prompt_filename_base}_sales_pitch_request_payload.json"
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
        response = gemini_client.generate_content_with_retry(
            contents=contents_for_api,
            generation_config=generation_config,
            system_instruction=system_instruction_text,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name
        )

        if response:
            raw_llm_response_str = response.text
            if raw_llm_response_str:
                response_filename = f"{prompt_filename_base}_sales_pitch_response.txt"
                try:
                    save_llm_artifact(
                        content=raw_llm_response_str,
                        directory=llm_context_dir,
                        filename=response_filename,
                        log_prefix=log_prefix
                    )
                except Exception as e_save_resp:
                    logger.error(f"{log_prefix} Failed to save raw LLM response artifact: {e_save_resp}", exc_info=True)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                token_stats["prompt_tokens"] = response.usage_metadata.prompt_token_count or 0
                token_stats["completion_tokens"] = response.usage_metadata.candidates_token_count or 0
                token_stats["total_tokens"] = response.usage_metadata.total_token_count or 0
            
            json_string_from_text = extract_json_from_text(raw_llm_response_str)
            if json_string_from_text:
                parsed_json_object = json.loads(json_string_from_text)
                parsed_json_object['analyzed_company_url'] = target_attributes.input_summary_url
                parsed_json_object['analyzed_company_attributes'] = target_attributes.model_dump()
                if website_summary_obj and website_summary_obj.summary:
                    parsed_json_object['summary'] = website_summary_obj.summary
                parsed_json_object['matched_partner_name'] = matched_partner.get('name')
                parsed_json_object['matched_partner_description'] = matched_partner.get('summary')
                parsed_json_object['avg_leads_per_day'] = matched_partner.get('avg_leads_per_day')
                parsed_json_object['rank'] = matched_partner.get('rank')
                parsed_output = GoldenPartnerMatchOutput(**parsed_json_object)
                logger.info(f"{log_prefix} Successfully parsed and enriched GoldenPartnerMatchOutput.")
            else:
                logger.error(f"{log_prefix} Failed to extract JSON from LLM response for sales pitch generation.")
        else:
            logger.error(f"{log_prefix} No response from GeminiClient for sales pitch generation.")
            raw_llm_response_str = "Error: No response object from GeminiClient."

    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during sales pitch generation: {e_gen}", exc_info=True)
        raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})

    return parsed_output, raw_llm_response_str, token_stats