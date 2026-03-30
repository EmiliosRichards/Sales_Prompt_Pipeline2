import json
import logging
from typing import Dict, Any, List, Tuple, Optional

from ...core.config import AppConfig
from ...core.schemas import (
    DetailedCompanyAttributes,
    GoldenPartnerMatchOutput,
    PartnerMatchOnlyOutput,
    SalesPitchLLMOutput,
    WebsiteTextSummary,
)
from ...utils.helpers import sanitize_filename_component
from ...utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
)

logger = logging.getLogger(__name__)


def _clean_phrase(value: Any) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text or text.lower() in {"none", "null", "nan", "unknown", "not specified"}:
        return ""
    return text


def _first_meaningful_item(values: Any) -> str:
    if isinstance(values, list):
        for item in values:
            cleaned = _clean_phrase(item)
            if cleaned:
                return cleaned
        return ""
    return _clean_phrase(values)


def _looks_english_heavy(text: str) -> bool:
    low = _clean_phrase(text).lower()
    if not low:
        return False
    markers = (
        " and ",
        " for ",
        " with ",
        "software",
        "solutions",
        "management",
        "analytics",
        "cloud",
        "production",
        "mobility",
        "fleet",
        "digital shelf",
        "ai-powered",
    )
    hits = sum(1 for marker in markers if marker in f" {low} ")
    return hits >= 2 or len(low) > 90


def _apply_match_metadata(
    output: GoldenPartnerMatchOutput,
    partner_match_output: Optional[PartnerMatchOnlyOutput],
    acceptance_reason: Optional[str] = None,
) -> GoldenPartnerMatchOutput:
    if partner_match_output:
        output.match_score = partner_match_output.match_score or output.match_score
        output.match_confidence = partner_match_output.match_confidence or output.match_confidence
        output.overlap_type = partner_match_output.overlap_type or output.overlap_type
        output.matched_partner_id = partner_match_output.matched_partner_id or output.matched_partner_id
        output.runner_up_partner_id = partner_match_output.runner_up_partner_id or output.runner_up_partner_id
        output.runner_up_partner_name = partner_match_output.runner_up_partner_name or output.runner_up_partner_name
        if partner_match_output.shortlisted_partner_ids:
            output.shortlisted_partner_ids = list(partner_match_output.shortlisted_partner_ids)
        if partner_match_output.shortlisted_partner_names:
            output.shortlisted_partner_names = list(partner_match_output.shortlisted_partner_names)
        if partner_match_output.target_evidence:
            output.target_evidence = list(partner_match_output.target_evidence)
        if partner_match_output.partner_evidence:
            output.partner_evidence = list(partner_match_output.partner_evidence)
        if partner_match_output.match_rationale_features and not output.match_rationale_features:
            output.match_rationale_features = list(partner_match_output.match_rationale_features)
    if acceptance_reason:
        output.acceptance_reason = acceptance_reason
    return output


def build_no_match_sales_pitch_output(
    *,
    target_attributes: DetailedCompanyAttributes,
    website_summary_obj: Optional[WebsiteTextSummary],
    rejection_reason: Optional[str] = None,
    company_name: Optional[str] = None,
    partner_match_output: Optional[PartnerMatchOnlyOutput] = None,
) -> GoldenPartnerMatchOutput:
    summary = _clean_phrase(getattr(website_summary_obj, "summary", None) if website_summary_obj else None)
    company_ref = _clean_phrase(company_name) or "Ihr Unternehmen"
    if _looks_english_heavy(summary):
        summary = ""

    phone_sales_line = (
        f"Ich rufe Sie an, weil ich gesehen habe, dass {company_ref} ein klares B2B-Angebot mit erklärungsbedürftigem "
        "Leistungsprofil hat. Ich würde gerne kurz mit Ihnen besprechen, wie Sie neue relevante Gespräche "
        "mit passenden Ansprechpartnern noch planbarer aufbauen können."
    )
    if summary:
        phone_sales_line = (
            f"Ich rufe Sie an, weil ich gesehen habe, dass {company_ref} ein klares Leistungsprofil hat. "
            "Ich würde gerne kurz mit Ihnen besprechen, wie Sie neue relevante Gespräche mit passenden "
            "Ansprechpartnern noch planbarer aufbauen können."
        )

    rationale = ["Kein ausreichend belastbarer Golden-Partner-Match; neutraler, personalisierter Pitch verwendet."]
    reason_clean = _clean_phrase(rejection_reason)
    if reason_clean:
        rationale.append(f"Match verworfen: {reason_clean}")

    output = GoldenPartnerMatchOutput(
        analyzed_company_url=target_attributes.input_summary_url,
        analyzed_company_attributes=target_attributes,
        summary=summary or None,
        match_score="No Match",
        match_confidence="no_match",
        overlap_type=getattr(partner_match_output, "overlap_type", None) if partner_match_output else None,
        match_rationale_features=rationale,
        target_evidence=[],
        partner_evidence=[],
        phone_sales_line=phone_sales_line,
        matched_partner_id=None,
        matched_partner_name=None,
        matched_partner_description=None,
        runner_up_partner_id=getattr(partner_match_output, "runner_up_partner_id", None) if partner_match_output else None,
        runner_up_partner_name=getattr(partner_match_output, "runner_up_partner_name", None) if partner_match_output else None,
        acceptance_reason=reason_clean or None,
        avg_leads_per_day=None,
        rank=None,
    )
    output = _apply_match_metadata(output, partner_match_output, acceptance_reason=reason_clean or None)
    output.match_score = "No Match"
    output.match_confidence = "no_match"
    output.matched_partner_id = None
    output.matched_partner_name = None
    output.matched_partner_description = None
    return output


def build_weak_match_sales_pitch_output(
    *,
    target_attributes: DetailedCompanyAttributes,
    matched_partner: Dict[str, Any],
    website_summary_obj: Optional[WebsiteTextSummary],
    partner_match_output: Optional[PartnerMatchOnlyOutput],
    rejection_reason: Optional[str] = None,
    company_name: Optional[str] = None,
) -> GoldenPartnerMatchOutput:
    summary = _clean_phrase(getattr(website_summary_obj, "summary", None) if website_summary_obj else None)
    company_ref = _clean_phrase(company_name) or "Ihr Unternehmen"
    overlap_type = _clean_phrase(getattr(partner_match_output, "overlap_type", None))
    target_evidence = list(getattr(partner_match_output, "target_evidence", None) or [])
    partner_evidence = list(getattr(partner_match_output, "partner_evidence", None) or [])
    overlap_hint = target_evidence[0] if target_evidence else partner_evidence[0] if partner_evidence else overlap_type

    if overlap_hint:
        phone_sales_line = (
            f"Ich rufe Sie an, weil ich gesehen habe, dass {company_ref} bei {overlap_hint} erkennbare Anknüpfungspunkte hat. "
            "Ich würde gerne kurz mit Ihnen besprechen, wie Sie daraus neue relevante Gespräche mit passenden Ansprechpartnern planbarer aufbauen können."
        )
    else:
        phone_sales_line = (
            f"Ich rufe Sie an, weil ich gesehen habe, dass {company_ref} ein klares B2B-Angebot mit interessanten Anknüpfungspunkten hat. "
            "Ich würde gerne kurz mit Ihnen besprechen, wie Sie daraus neue relevante Gespräche mit passenden Ansprechpartnern planbarer aufbauen können."
        )
    if summary and not _looks_english_heavy(summary):
        phone_sales_line = (
            f"Ich rufe Sie an, weil ich gesehen habe, dass {company_ref} ein klares Leistungsprofil hat und wir in ähnlichen Konstellationen relevante Anknüpfungspunkte sehen. "
            "Ich würde gerne kurz mit Ihnen besprechen, wie Sie daraus neue relevante Gespräche planbarer aufbauen können."
        )

    rationale = ["Teilweise belastbarer Golden-Partner-Match; softer Pitch ohne harte Case-Study-Behauptung verwendet."]
    reason_clean = _clean_phrase(rejection_reason)
    if reason_clean:
        rationale.append(f"Schwacher Matchpfad: {reason_clean}")
    rationale.extend([item for item in (getattr(partner_match_output, "match_rationale_features", None) or []) if _clean_phrase(item)])

    output = GoldenPartnerMatchOutput(
        analyzed_company_url=target_attributes.input_summary_url,
        analyzed_company_attributes=target_attributes,
        summary=summary or None,
        match_score=getattr(partner_match_output, "match_score", None) or "Medium",
        match_confidence="weak",
        overlap_type=getattr(partner_match_output, "overlap_type", None) if partner_match_output else None,
        match_rationale_features=rationale,
        target_evidence=target_evidence,
        partner_evidence=partner_evidence,
        phone_sales_line=phone_sales_line,
        matched_partner_id=matched_partner.get("partner_id"),
        matched_partner_name=matched_partner.get("name"),
        matched_partner_description=matched_partner.get("summary"),
        runner_up_partner_id=getattr(partner_match_output, "runner_up_partner_id", None) if partner_match_output else None,
        runner_up_partner_name=getattr(partner_match_output, "runner_up_partner_name", None) if partner_match_output else None,
        acceptance_reason=reason_clean or "accepted_weak_match",
        avg_leads_per_day=None,
        rank=matched_partner.get("rank"),
    )
    output = _apply_match_metadata(output, partner_match_output, acceptance_reason=reason_clean or "accepted_weak_match")
    output.match_confidence = "weak"
    return output

def generate_sales_pitch(
    gemini_client: Any,
    config: AppConfig,
    target_attributes: DetailedCompanyAttributes,
    matched_partner: Dict[str, Any],
    website_summary_obj: WebsiteTextSummary,
    previous_match_rationale: List[str],
    llm_context_dir: str,
    llm_requests_dir: str,
    file_identifier_prefix: str,
    triggering_input_row_id: Any,
    triggering_company_name: str,
    partner_match_output: Optional[PartnerMatchOnlyOutput] = None,
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

    prompt_template_path: str = "Path not initialized"

    try:
        prompt_template_path = config.PROMPT_PATH_GERMAN_SALES_PITCH_GENERATION
        prompt_template = load_prompt_template(prompt_template_path)
        target_attributes_json = target_attributes.model_dump_json(indent=2)
        matched_partner_json = json.dumps(matched_partner, indent=2)
        formatted_prompt = prompt_template.replace("{{TARGET_COMPANY_ATTRIBUTES_JSON_PLACEHOLDER}}", target_attributes_json)
        formatted_prompt = formatted_prompt.replace("{{MATCHED_GOLDEN_PARTNER_JSON_PLACEHOLDER}}", matched_partner_json)
        previous_rationale_str = "\n".join([f"- {item}" for item in (previous_match_rationale or [])])
        formatted_prompt = formatted_prompt.replace("{{PREVIOUS_MATCH_RATIONALE_PLACEHOLDER}}", previous_rationale_str)
    except Exception as e:
        logger.error(f"{log_prefix} Failed to load/format sales pitch prompt: {e}", exc_info=True)
        return None, f"Error: Failed to load/format prompt - {str(e)}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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

    system_instruction_text = (
        "You generate the final outreach pitch for the already selected golden partner. "
        "Return only valid JSON matching the requested schema."
    )
    request_payload_to_log = {
        "model_name": getattr(config, "llm_model_name_sales_insights", "") or getattr(config, "llm_model_name", ""),
        "system_instruction": system_instruction_text,
        "schema_name": "sales_pitch_output",
        "response_model": "SalesPitchLLMOutput",
        "max_output_tokens": config.llm_max_tokens,
        "temperature": config.llm_temperature_creative,
        "user_prompt": formatted_prompt,
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
        sales_model_override = getattr(config, "llm_model_name_sales_insights", None) or getattr(config, "openai_model_name_sales_insights", None)
        llm_result = gemini_client.generate_structured_output_with_retry(
            user_prompt=formatted_prompt,
            response_model=SalesPitchLLMOutput,
            schema_name="sales_pitch_output",
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_prompt=system_instruction_text,
            model_name_override=sales_model_override,
        )

        token_stats = llm_result.usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if llm_result.raw_text:
            response_filename = f"{prompt_filename_base}_sales_pitch_response.txt"
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
            parsed_output = GoldenPartnerMatchOutput(
                analyzed_company_url=target_attributes.input_summary_url,
                analyzed_company_attributes=target_attributes,
                summary=website_summary_obj.summary if website_summary_obj else None,
                match_score=llm_result.parsed_output.match_score,
                match_rationale_features=llm_result.parsed_output.match_rationale_features or [],
                target_evidence=[],
                partner_evidence=[],
                phone_sales_line=llm_result.parsed_output.phone_sales_line,
                matched_partner_id=matched_partner.get("partner_id"),
                matched_partner_name=matched_partner.get("name"),
                matched_partner_description=matched_partner.get("summary"),
                avg_leads_per_day=matched_partner.get("avg_leads_per_day"),
                rank=matched_partner.get("rank"),
            )
            parsed_output = _apply_match_metadata(
                parsed_output,
                partner_match_output,
                acceptance_reason="accepted_strong_match",
            )
            logger.info(f"{log_prefix} Successfully parsed SalesPitchLLMOutput and enriched GoldenPartnerMatchOutput.")
            return parsed_output, llm_result.raw_text, token_stats

        error_message = llm_result.provider_error or (
            f"Model refusal: {llm_result.refusal}" if llm_result.refusal else "Failed to parse sales pitch structured output."
        )
        logger.error(f"{log_prefix} {error_message}")
        return None, llm_result.raw_text or error_message, token_stats
    except Exception as e_gen:
        logger.error(f"{log_prefix} Unexpected error during sales pitch generation: {e_gen}", exc_info=True)
        raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
        return None, raw_llm_response_str, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}