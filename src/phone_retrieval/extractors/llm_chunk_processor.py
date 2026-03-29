import logging
import json
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from google.genai import types as genai_types

from src.core.config import AppConfig
from src.core.schemas import PhoneNumberLLMOutput, MinimalExtractionOutput, HomepageContextOutput
from src.phone_retrieval.utils.llm_processing_helpers import (
    load_prompt_template,
    extract_json_from_text,
    normalize_phone_number,
    process_successful_llm_item,
    create_error_llm_item,
    save_llm_artifact
)

logger = logging.getLogger(__name__)

def _recursively_remove_key(obj: Any, key_to_remove: str) -> Any:
    """
    Recursively removes a specific key from a nested dictionary or list of dictionaries.
    """
    if isinstance(obj, dict):
        # Create a new dictionary, excluding the key_to_remove
        new_dict = {}
        for k, v in obj.items():
            if k == key_to_remove:
                continue
            new_dict[k] = _recursively_remove_key(v, key_to_remove)
        return new_dict
    elif isinstance(obj, list):
        return [_recursively_remove_key(item, key_to_remove) for item in obj]
    else:
        return obj

class LLMChunkProcessor:
    """
    Manages the chunked processing of candidate items for phone number extraction
    using an LLM.
    """

    def __init__(
        self,
        config: AppConfig,
        llm_client: Any,
        llm_provider: str,
        prompt_template_path: str,
    ):
        """
        Initializes the LLMChunkProcessor.

        Args:
            config: The application configuration.
            llm_client: The client for interacting with the configured LLM provider.
            llm_provider: The active provider name (e.g. gemini, openai).
            prompt_template_path: Path to the base prompt template.
        """
        self.config = config
        self.llm_client = llm_client
        self.llm_provider = (llm_provider or "gemini").strip().lower()
        self.prompt_template_path = prompt_template_path
        try:
            self.base_prompt_template = load_prompt_template(self.prompt_template_path)
            logger.info(f"LLMChunkProcessor initialized with prompt template: {self.prompt_template_path}")
        except Exception as e:
            logger.error(f"Failed to load base prompt template from {self.prompt_template_path}: {e}")
            raise

    def _call_llm_for_chunk(
        self,
        *,
        prompt_text: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
    ) -> Tuple[Optional[str], Optional[MinimalExtractionOutput], Dict[str, int]]:
        if self.llm_provider == "openai":
            result = self.llm_client.generate_structured_output_with_retry(
                user_prompt=prompt_text,
                response_model=MinimalExtractionOutput,
                schema_name="phone_number_extraction",
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name,
                system_prompt=(
                    "You classify phone number candidates for a company's website. "
                    "Return only valid JSON matching the schema."
                ),
            )
            return result.raw_text, result.parsed_output, result.usage

        generation_config = genai_types.GenerateContentConfig(
            max_output_tokens=self.config.llm_chunk_processor_max_tokens,
            temperature=self.config.llm_temperature,
            response_mime_type="application/json",
        )
        llm_response_obj = self.llm_client.generate_content_with_retry(
            contents=prompt_text,
            generation_config=generation_config,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            system_instruction=(
                "Return only valid JSON that matches the provided schema exactly. "
                "Do not wrap the JSON in markdown fences. "
                "Do not add commentary before or after the JSON."
            ),
        )
        raw_llm_response_text_chunk = getattr(llm_response_obj, 'text', None) if llm_response_obj else None
        token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if llm_response_obj and hasattr(llm_response_obj, 'usage_metadata') and llm_response_obj.usage_metadata:
            token_stats = {
                "prompt_tokens": llm_response_obj.usage_metadata.prompt_token_count or 0,
                "completion_tokens": llm_response_obj.usage_metadata.candidates_token_count or 0,
                "total_tokens": llm_response_obj.usage_metadata.total_token_count or 0,
            }
        return raw_llm_response_text_chunk, None, token_stats

    def _prepare_prompt_for_chunk(
        self,
        current_chunk_candidate_items: List[Dict[str, str]],
        homepage_context_input: Optional[HomepageContextOutput]
    ) -> str:
        """
        Formats the prompt for a given chunk of candidate items.
        """
        prompt_with_context = self.base_prompt_template

        if homepage_context_input:
            company_name_ctx = homepage_context_input.company_name if homepage_context_input.company_name else "N/A"
            summary_ctx = homepage_context_input.summary_description if homepage_context_input.summary_description else "N/A"
            industry_ctx = homepage_context_input.industry if homepage_context_input.industry else "N/A"
            
            prompt_with_context = prompt_with_context.replace(
                "[Insert Company Name from Summary Here or \"N/A\"]", company_name_ctx
            ).replace(
                "[Insert Website Summary Here or \"N/A\"]", summary_ctx
            ).replace(
                "[Insert Industry from Summary Here or \"N/A\"]", industry_ctx
            )
        else:
            prompt_with_context = prompt_with_context.replace(
                "[Insert Company Name from Summary Here or \"N/A\"]", "N/A"
            ).replace(
                "[Insert Website Summary Here or \"N/A\"]", "N/A"
            ).replace(
                "[Insert Industry from Summary Here or \"N/A\"]", "N/A"
            )

        candidate_items_json_str_chunk = json.dumps(current_chunk_candidate_items, indent=2)
        formatted_prompt_chunk = prompt_with_context.replace(
            "{{PHONE_CANDIDATES_JSON_PLACEHOLDER}}",
            candidate_items_json_str_chunk
        )
        logger.debug(f"Formatted prompt for chunk: {formatted_prompt_chunk}")
        return formatted_prompt_chunk

    def _numbers_match(self, llm_number: Optional[str], input_item_detail: Dict[str, Any]) -> bool:
        llm_val = (llm_number or "").strip()
        input_val = str(input_item_detail.get("number") or input_item_detail.get("candidate_number") or "").strip()
        if not llm_val or not input_val:
            return False
        if llm_val == input_val:
            return True

        llm_norm = normalize_phone_number(
            llm_val,
            country_codes=self.config.target_country_codes,
            default_region_code=self.config.default_region_code,
        )
        input_norm = normalize_phone_number(
            input_val,
            country_codes=self.config.target_country_codes,
            default_region_code=self.config.default_region_code,
        )
        return bool(llm_norm and input_norm and llm_norm == input_norm)

    def _process_llm_response_for_chunk(
        self,
        llm_response_text: Optional[str],
        parsed_llm_result: Optional[MinimalExtractionOutput],
        current_chunk_candidate_items: List[Dict[str, str]],
        final_processed_outputs_for_chunk: List[Optional[PhoneNumberLLMOutput]],
        items_needing_retry_for_chunk: List[Tuple[int, Dict[str, Any]]],
        extra_processed_outputs_for_chunk: List[PhoneNumberLLMOutput],
        chunk_file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str
    ) -> None:
        """
        Processes the LLM response for a single chunk, populating output lists.
        """
        if not llm_response_text and parsed_llm_result is None:
            logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM response text is empty for chunk.")
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None: # Only fill if not already processed (e.g. by retry)
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_ChunkEmptyResponse", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
            return

        logger.debug(f"Raw LLM response for chunk: {llm_response_text}")

        try:
            if parsed_llm_result is not None:
                llm_result_chunk = parsed_llm_result
            else:
                parsed_json_object_chunk = None
                json_candidate_str_chunk = None
                if llm_response_text:
                    try:
                        parsed_json_object_chunk = json.loads(llm_response_text)
                        json_candidate_str_chunk = llm_response_text
                    except json.JSONDecodeError:
                        json_candidate_str_chunk = extract_json_from_text(llm_response_text)
                        if json_candidate_str_chunk:
                            parsed_json_object_chunk = json.loads(json_candidate_str_chunk)

                if parsed_json_object_chunk is None:
                    logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Could not extract JSON block from LLM response for chunk.")
                    for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                        if final_processed_outputs_for_chunk[k_err] is None:
                            final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_ChunkNoJsonBlock", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                    return

                if (
                    isinstance(parsed_json_object_chunk, dict)
                    and "extracted_numbers" not in parsed_json_object_chunk
                    and {"number", "type", "classification", "is_valid", "source_url"}.issubset(parsed_json_object_chunk.keys())
                ):
                    parsed_json_object_chunk = {"extracted_numbers": [parsed_json_object_chunk]}
                logger.debug(f"Parsed JSON object for chunk: {parsed_json_object_chunk}")
                llm_result_chunk = MinimalExtractionOutput(**parsed_json_object_chunk)
            validated_numbers_chunk = llm_result_chunk.extracted_numbers

            if len(validated_numbers_chunk) != len(current_chunk_candidate_items):
                # Best-effort salvage: models sometimes emit multiple objects per input snippet
                # (e.g., same number + source_url but multiple associated people).
                # We align by (number, source_url) and keep one "best" item per input; extras are preserved.
                logger.error(
                    f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                    f"LLM chunk call: Mismatch in item count. Input: {len(current_chunk_candidate_items)}, Output: {len(validated_numbers_chunk)}. "
                    f"Attempting best-effort alignment by (number, source_url)."
                )

                def _score_out(o: PhoneNumberLLMOutput) -> int:
                    # Higher is better.
                    score = 0
                    try:
                        if bool(getattr(o, "is_valid", False)):
                            score += 100
                    except Exception:
                        pass
                    cls = (getattr(o, "classification", None) or "").strip()
                    if cls == "Primary":
                        score += 30
                    elif cls == "Secondary":
                        score += 10
                    elif cls == "Non-Business":
                        score -= 50
                    t = (getattr(o, "type", None) or "").strip()
                    type_bonus = {
                        "Direct Dial": 25,
                        "Sales": 20,
                        "Main Office": 15,
                        "Support": 10,
                        "Mobile": 5,
                        "Hotline": 3,
                        "HR": -5,
                        "Billing": -5,
                        "Other Department": 0,
                        "Fax": -100,
                        "Unknown": -10,
                    }
                    score += type_bonus.get(t, 0)
                    # Prefer outputs that actually captured a person association when present
                    if (getattr(o, "associated_person_name", None) or "").strip():
                        score += 5
                    if (getattr(o, "associated_person_role", None) or "").strip():
                        score += 3
                    return score

                key_to_input_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
                num_to_input_indices: Dict[str, List[int]] = defaultdict(list)
                for idx, inp in enumerate(current_chunk_candidate_items):
                    in_num = str(inp.get("candidate_number") or inp.get("number") or "").strip()
                    in_num_norm = normalize_phone_number(
                        in_num,
                        country_codes=self.config.target_country_codes,
                        default_region_code=self.config.default_region_code,
                    ) if in_num else None
                    in_url = str(inp.get("source_url") or "").strip().lower()
                    key_to_input_indices[(in_num, in_url)].append(idx)
                    if in_num_norm:
                        key_to_input_indices[(in_num_norm, in_url)].append(idx)
                    if in_num:
                        num_to_input_indices[in_num].append(idx)
                    if in_num_norm:
                        num_to_input_indices[in_num_norm].append(idx)

                idx_to_outputs: Dict[int, List[PhoneNumberLLMOutput]] = defaultdict(list)
                for out in validated_numbers_chunk:
                    out_num = str(getattr(out, "number", "") or "").strip()
                    out_num_norm = normalize_phone_number(
                        out_num,
                        country_codes=self.config.target_country_codes,
                        default_region_code=self.config.default_region_code,
                    ) if out_num else None
                    out_url = str(getattr(out, "source_url", "") or "").strip().lower()
                    match_indices = key_to_input_indices.get((out_num, out_url))
                    if not match_indices and out_num_norm:
                        match_indices = key_to_input_indices.get((out_num_norm, out_url))
                    if not match_indices:
                        # Fallback: match on number only (less safe)
                        match_indices = num_to_input_indices.get(out_num)
                    if not match_indices and out_num_norm:
                        match_indices = num_to_input_indices.get(out_num_norm)
                    if not match_indices:
                        continue
                    idx_to_outputs[match_indices[0]].append(out)

                for k, input_item_detail_chunk in enumerate(current_chunk_candidate_items):
                    if final_processed_outputs_for_chunk[k] is not None:
                        continue
                    outs = idx_to_outputs.get(k, [])
                    if not outs:
                        items_needing_retry_for_chunk.append((k, input_item_detail_chunk))
                        continue
                    best = max(outs, key=_score_out)
                    try:
                        final_processed_outputs_for_chunk[k] = process_successful_llm_item(
                            best,
                            input_item_detail_chunk,
                            self.config.target_country_codes,
                            self.config.default_region_code,
                        )
                    except Exception:
                        items_needing_retry_for_chunk.append((k, input_item_detail_chunk))
                        continue

                    # Preserve additional outputs that map to this same input item.
                    for extra in outs:
                        if extra is best:
                            continue
                        try:
                            extra_processed_outputs_for_chunk.append(
                                process_successful_llm_item(
                                    extra,
                                    input_item_detail_chunk,
                                    self.config.target_country_codes,
                                    self.config.default_region_code,
                                )
                            )
                        except Exception:
                            continue

            else:
                for k, input_item_detail_chunk in enumerate(current_chunk_candidate_items):
                    if final_processed_outputs_for_chunk[k] is not None: # Already processed (e.g. by a successful retry pass)
                        continue
                    llm_output_item_chunk = validated_numbers_chunk[k]
                    # Compare based on 'number' field as per original logic
                    if self._numbers_match(llm_output_item_chunk.number, input_item_detail_chunk):
                        final_processed_outputs_for_chunk[k] = process_successful_llm_item(
                            llm_output_item_chunk,
                            input_item_detail_chunk,
                            self.config.target_country_codes, # Assuming AppConfig has target_country_codes
                            self.config.default_region_code   # Assuming AppConfig has default_region_code
                        )
                    else:
                        logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Mismatch for item {k}: Input '{input_item_detail_chunk.get('number') or input_item_detail_chunk.get('candidate_number')}', LLM output '{llm_output_item_chunk.number}'. Adding to retry queue.")
                        items_needing_retry_for_chunk.append((k, input_item_detail_chunk))

        except json.JSONDecodeError as e_parse_validate:
            logger.error(
                f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                f"Failed to parse/validate JSON for chunk: JSONDecodeError - {e_parse_validate}. Error at char {e_parse_validate.pos}. "
                f"Attempted to parse (json_candidate_str_chunk, first 1000 chars): '{json_candidate_str_chunk[:1000] if 'json_candidate_str_chunk' in locals() and json_candidate_str_chunk else 'N/A'}'. "
                f"Original LLM response text (llm_response_text, first 1000 chars): '{llm_response_text[:1000] if llm_response_text else 'N/A'}'"
            )
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None:
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, f"Error_ChunkJsonParseValidate_JSONDecodeError", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
        except Exception as e_parse_validate: # Broader exception for Pydantic if direct parsing
            logger.error(
                f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                f"Failed to parse/validate JSON for chunk: {type(e_parse_validate).__name__} - {e_parse_validate}. "
                f"Attempted to parse (json_candidate_str_chunk): '{json_candidate_str_chunk if 'json_candidate_str_chunk' in locals() else 'N/A'}'. "
                f"Original LLM response text (llm_response_text, first 1000 chars): '{llm_response_text[:1000] if llm_response_text else 'N/A'}...'"
            )
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None:
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, f"Error_ChunkJsonParseValidate_{type(e_parse_validate).__name__}", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)


    def process_candidates(
        self,
        candidate_items: List[Dict[str, str]],
        llm_context_dir: str, # For potential artifact logging
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        homepage_context_input: Optional[HomepageContextOutput] = None,
    ) -> Tuple[List[PhoneNumberLLMOutput], Optional[str], Optional[Dict[str, int]]]:
        """
        Processes candidate items in chunks, calls the LLM, handles retries, and aggregates results.

        Args:
            candidate_items: List of candidate items to process.
            llm_context_dir: Directory for saving LLM artifacts.
            file_identifier_prefix: Prefix for artifact filenames.
            triggering_input_row_id: Identifier for the triggering input row.
            triggering_company_name: Company name for context.
            homepage_context_input: Optional homepage context.

        Returns:
            A tuple containing:
                - List of processed PhoneNumberLLMOutput objects.
                - Combined raw LLM responses string (optional).
                - Accumulated token usage statistics (optional).
        """
        overall_processed_outputs: List[PhoneNumberLLMOutput] = []
        overall_raw_responses_list: List[str] = []
        accumulated_token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        chunk_size = self.config.llm_candidate_chunk_size
        if self.llm_provider == "gemini":
            # Gemini is more likely to under-return items on large candidate batches.
            chunk_size = max(1, min(chunk_size, 5))
        max_chunks = self.config.llm_max_chunks_per_url
        chunks_processed_count = 0

        for i in range(0, len(candidate_items), chunk_size):
            if max_chunks > 0 and chunks_processed_count >= max_chunks:
                logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Reached max_chunks limit ({max_chunks}). Processed {chunks_processed_count * chunk_size} candidates out of {len(candidate_items)}.")
                break
            
            current_chunk_candidate_items = candidate_items[i : i + chunk_size]
            if not current_chunk_candidate_items:
                break

            chunks_processed_count += 1
            chunk_log_prefix = f"{file_identifier_prefix}_chunk_{chunks_processed_count}"
            
            logger.info(f"[{chunk_log_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Processing chunk {chunks_processed_count} with {len(current_chunk_candidate_items)} items.")

            final_processed_outputs_for_chunk: List[Optional[PhoneNumberLLMOutput]] = [None] * len(current_chunk_candidate_items)
            # Some models may emit more output objects than inputs (e.g., multiple people for one snippet).
            # We preserve these as additional processed outputs instead of failing the whole chunk.
            extra_outputs_for_chunk: List[PhoneNumberLLMOutput] = []
            
            # Initial LLM call for the chunk
            items_needing_retry_this_pass: List[Tuple[int, Dict[str, Any]]] = []
            
            try:
                formatted_prompt_chunk = self._prepare_prompt_for_chunk(current_chunk_candidate_items, homepage_context_input)
                raw_llm_response_text_chunk, parsed_llm_result_chunk, token_stats_chunk = self._call_llm_for_chunk(
                    prompt_text=formatted_prompt_chunk,
                    file_identifier_prefix=chunk_log_prefix,
                    triggering_input_row_id=triggering_input_row_id,
                    triggering_company_name=triggering_company_name,
                )
                for key_token in accumulated_token_stats:
                    accumulated_token_stats[key_token] += token_stats_chunk.get(key_token, 0)
                logger.info(f"[{chunk_log_prefix}] LLM (initial chunk) usage: {token_stats_chunk}")

                self._process_llm_response_for_chunk(
                    raw_llm_response_text_chunk,
                    parsed_llm_result_chunk,
                    current_chunk_candidate_items,
                    final_processed_outputs_for_chunk,
                    items_needing_retry_this_pass,
                    extra_outputs_for_chunk,
                    chunk_log_prefix,
                    triggering_input_row_id,
                    triggering_company_name
                )
                if raw_llm_response_text_chunk:
                    overall_raw_responses_list.append(raw_llm_response_text_chunk)


            except Exception as e_initial_call:
                logger.error(f"[{chunk_log_prefix}] Error during initial LLM call for chunk: {e_initial_call}", exc_info=True)
                for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, f"Error_ChunkInitialLLMCall_{type(e_initial_call).__name__}", file_identifier_prefix=chunk_log_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                overall_raw_responses_list.append(json.dumps({"error": f"Chunk initial LLM call error: {str(e_initial_call)}", "type": type(e_initial_call).__name__}))
            
            # Mismatch Retry Loop for the Current Chunk
            current_chunk_mismatch_retry_attempt = 0
            while items_needing_retry_this_pass and current_chunk_mismatch_retry_attempt < self.config.llm_max_retries_on_number_mismatch:
                current_chunk_mismatch_retry_attempt += 1
                retry_log_prefix = f"{chunk_log_prefix}_mismatch_retry_{current_chunk_mismatch_retry_attempt}"
                
                inputs_for_this_retry_pass_details = [item_tuple[1] for item_tuple in items_needing_retry_this_pass]
                original_indices_for_this_retry_pass = [item_tuple[0] for item_tuple in items_needing_retry_this_pass]
                
                logger.info(f"[{retry_log_prefix}] Attempting mismatch retry pass #{current_chunk_mismatch_retry_attempt} for {len(inputs_for_this_retry_pass_details)} items.")

                items_still_needing_retry_after_this_pass: List[Tuple[int, Dict[str, Any]]] = []

                try:
                    formatted_prompt_retry_chunk = self._prepare_prompt_for_chunk(inputs_for_this_retry_pass_details, homepage_context_input)
                    raw_llm_response_text_retry_chunk, parsed_retry_result, token_stats_retry_chunk = self._call_llm_for_chunk(
                        prompt_text=formatted_prompt_retry_chunk,
                        file_identifier_prefix=retry_log_prefix,
                        triggering_input_row_id=triggering_input_row_id,
                        triggering_company_name=triggering_company_name,
                    )
                    for key_token_r in accumulated_token_stats:
                        accumulated_token_stats[key_token_r] += token_stats_retry_chunk.get(key_token_r, 0)
                    logger.info(f"[{retry_log_prefix}] LLM (mismatch retry) usage: {token_stats_retry_chunk}")

                    temp_processed_outputs_for_retry_pass: List[Optional[PhoneNumberLLMOutput]] = [None] * len(inputs_for_this_retry_pass_details)
                    temp_items_needing_further_retry: List[Tuple[int, Dict[str, Any]]] = []
                    temp_extra_outputs_for_retry_pass: List[PhoneNumberLLMOutput] = []

                    self._process_llm_response_for_chunk(
                        raw_llm_response_text_retry_chunk,
                        parsed_retry_result,
                        inputs_for_this_retry_pass_details,
                        temp_processed_outputs_for_retry_pass,
                        temp_items_needing_further_retry,
                        temp_extra_outputs_for_retry_pass,
                        retry_log_prefix,
                        triggering_input_row_id,
                        triggering_company_name
                    )
                    if raw_llm_response_text_retry_chunk:
                        overall_raw_responses_list.append(raw_llm_response_text_retry_chunk)

                    for j_retry, processed_item_from_retry in enumerate(temp_processed_outputs_for_retry_pass):
                        original_chunk_index = original_indices_for_this_retry_pass[j_retry]
                        if processed_item_from_retry is not None:
                            final_processed_outputs_for_chunk[original_chunk_index] = processed_item_from_retry
                    if temp_extra_outputs_for_retry_pass:
                        extra_outputs_for_chunk.extend(temp_extra_outputs_for_retry_pass)

                    for k_further_retry, (idx_in_retry_pass, item_detail) in enumerate(temp_items_needing_further_retry):
                        original_chunk_idx_for_further_retry = original_indices_for_this_retry_pass[idx_in_retry_pass]
                        items_still_needing_retry_after_this_pass.append((original_chunk_idx_for_further_retry, item_detail))


                except Exception as e_retry_call:
                    logger.error(f"[{retry_log_prefix}] Error during mismatch retry LLM call: {e_retry_call}", exc_info=True)
                    # All items in this retry pass remain needing retry if call fails
                    items_still_needing_retry_after_this_pass.extend(items_needing_retry_this_pass)
                    overall_raw_responses_list.append(json.dumps({"error": f"Chunk mismatch retry call error: {str(e_retry_call)}", "type": type(e_retry_call).__name__}))

                items_needing_retry_this_pass = items_still_needing_retry_after_this_pass
            # End of mismatch retry loop for the chunk

            # Handle items persistently mismatched after all retries for this chunk
            if items_needing_retry_this_pass:
                # Gemini sometimes collapses multi-item chunks into a single output object.
                # Fall back to one-item salvage retries before giving up on those candidates.
                if self.llm_provider == "gemini":
                    still_unresolved_after_singletons: List[Tuple[int, Dict[str, Any]]] = []
                    for original_idx_persist, item_detail_persist_error in items_needing_retry_this_pass:
                        singleton_log_prefix = f"{chunk_log_prefix}_single_retry_{original_idx_persist}"
                        try:
                            formatted_singleton_prompt = self._prepare_prompt_for_chunk(
                                [item_detail_persist_error],
                                homepage_context_input,
                            )
                            raw_singleton_text, parsed_singleton_result, token_stats_singleton = self._call_llm_for_chunk(
                                prompt_text=formatted_singleton_prompt,
                                file_identifier_prefix=singleton_log_prefix,
                                triggering_input_row_id=triggering_input_row_id,
                                triggering_company_name=triggering_company_name,
                            )
                            for key_token_s in accumulated_token_stats:
                                accumulated_token_stats[key_token_s] += token_stats_singleton.get(key_token_s, 0)

                            singleton_outputs: List[Optional[PhoneNumberLLMOutput]] = [None]
                            singleton_retry_queue: List[Tuple[int, Dict[str, Any]]] = []
                            singleton_extra_outputs: List[PhoneNumberLLMOutput] = []
                            self._process_llm_response_for_chunk(
                                raw_singleton_text,
                                parsed_singleton_result,
                                [item_detail_persist_error],
                                singleton_outputs,
                                singleton_retry_queue,
                                singleton_extra_outputs,
                                singleton_log_prefix,
                                triggering_input_row_id,
                                triggering_company_name,
                            )
                            if raw_singleton_text:
                                overall_raw_responses_list.append(raw_singleton_text)

                            if singleton_outputs[0] is not None and not getattr(singleton_outputs[0], "type", "").startswith("Error_"):
                                final_processed_outputs_for_chunk[original_idx_persist] = singleton_outputs[0]
                                if singleton_extra_outputs:
                                    extra_outputs_for_chunk.extend(singleton_extra_outputs)
                            else:
                                still_unresolved_after_singletons.append((original_idx_persist, item_detail_persist_error))
                        except Exception as e_singleton:
                            logger.warning(
                                f"[{singleton_log_prefix}] Single-item salvage retry failed: {e_singleton}",
                                exc_info=True,
                            )
                            still_unresolved_after_singletons.append((original_idx_persist, item_detail_persist_error))
                    items_needing_retry_this_pass = still_unresolved_after_singletons

            if items_needing_retry_this_pass:
                logger.warning(f"[{chunk_log_prefix}] {len(items_needing_retry_this_pass)} items remain mismatched after all retries for this chunk.")
                for original_idx_persist, item_detail_persist_error in items_needing_retry_this_pass:
                    if final_processed_outputs_for_chunk[original_idx_persist] is None: # Only if not already set
                        final_processed_outputs_for_chunk[original_idx_persist] = create_error_llm_item(
                            item_detail_persist_error,
                            "Error_PersistentMismatchAfterRetries",
                            file_identifier_prefix=chunk_log_prefix,
                            triggering_input_row_id=triggering_input_row_id,
                            triggering_company_name=triggering_company_name
                        )
            
            # Fill any remaining None slots in this chunk's outputs with a generic error
            for k_final_check, output_item_chunk_final in enumerate(final_processed_outputs_for_chunk):
                if output_item_chunk_final is None:
                    final_processed_outputs_for_chunk[k_final_check] = create_error_llm_item(
                        current_chunk_candidate_items[k_final_check],
                        "Error_NotProcessedInChunk",
                        file_identifier_prefix=chunk_log_prefix,
                        triggering_input_row_id=triggering_input_row_id,
                        triggering_company_name=triggering_company_name
                    )
            
            overall_processed_outputs.extend([item for item in final_processed_outputs_for_chunk if item is not None])
            if extra_outputs_for_chunk:
                overall_processed_outputs.extend(extra_outputs_for_chunk)
        # End of chunk loop

        final_combined_raw_response_str = "\n\n---CHUNK_SEPARATOR---\n\n".join(overall_raw_responses_list) if overall_raw_responses_list else None
        
        successful_items_count = sum(1 for item in overall_processed_outputs if item and not item.type.startswith("Error_"))
        error_items_count = len(overall_processed_outputs) - successful_items_count
        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Overall LLM chunk processing summary: {successful_items_count} successful, {error_items_count} errors out of {len(candidate_items)} candidates processed over {chunks_processed_count} chunks.")

        return overall_processed_outputs, final_combined_raw_response_str, accumulated_token_stats