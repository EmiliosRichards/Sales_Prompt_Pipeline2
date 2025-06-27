import logging
import json
from typing import List, Dict, Any, Tuple, Optional

import google.generativeai.types as genai_types # Standardized to google.generativeai
# Assuming google.genai.types is the correct path for GenerationConfig based on llm_extractor
# If it's google.generativeai.types, it will be adjusted. The user provided google.genai.types

from src.core.config import AppConfig
from src.core.schemas import PhoneNumberLLMOutput, MinimalExtractionOutput, HomepageContextOutput
from src.phone_retrieval.llm_clients.gemini_client import GeminiClient
from src.phone_retrieval.utils.llm_processing_helpers import (
    load_prompt_template,
    extract_json_from_text,
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
        gemini_client: GeminiClient,
        prompt_template_path: str,
    ):
        """
        Initializes the LLMChunkProcessor.

        Args:
            config: The application configuration.
            gemini_client: The client for interacting with the Gemini LLM.
            prompt_template_path: Path to the base prompt template.
        """
        self.config = config
        self.gemini_client = gemini_client
        self.prompt_template_path = prompt_template_path
        try:
            self.base_prompt_template = load_prompt_template(self.prompt_template_path)
            logger.info(f"LLMChunkProcessor initialized with prompt template: {self.prompt_template_path}")
        except Exception as e:
            logger.error(f"Failed to load base prompt template from {self.prompt_template_path}: {e}")
            raise

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
            "[Insert JSON list of (candidate_number, source_url, snippet) objects here]",
            candidate_items_json_str_chunk
        )
        return formatted_prompt_chunk

    def _process_llm_response_for_chunk(
        self,
        llm_response_text: Optional[str],
        current_chunk_candidate_items: List[Dict[str, str]],
        final_processed_outputs_for_chunk: List[Optional[PhoneNumberLLMOutput]],
        items_needing_retry_for_chunk: List[Tuple[int, Dict[str, Any]]],
        chunk_file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str
    ) -> None:
        """
        Processes the LLM response for a single chunk, populating output lists.
        """
        if not llm_response_text:
            logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM response text is empty for chunk.")
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None: # Only fill if not already processed (e.g. by retry)
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_ChunkEmptyResponse", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
            return

        # The new SDK might provide parsed Pydantic objects directly if response_schema is used effectively.
        # For now, assuming we might still need to parse from text as a fallback or primary method.
        # This part needs to align with how gemini_client.generate_content_with_retry actually returns data.
        # If it returns a Pydantic object directly, this parsing logic changes.
        # The prompt implies `MinimalExtractionOutput` is the response_schema.

        # Attempt to parse with Pydantic if direct object is not available
        # This assumes gemini_client might return a structure that includes the parsed object or raw text
        # For now, let's assume we get raw text and need to parse it.

        json_candidate_str_chunk = extract_json_from_text(llm_response_text)
        if not json_candidate_str_chunk:
            logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Could not extract JSON block from LLM response for chunk.")
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None:
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_ChunkNoJsonBlock", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
            return

        try:
            # Gemini client with response_schema should ideally return a parsed object or allow easy parsing
            # For now, we parse the extracted JSON string.
            parsed_json_object_chunk = json.loads(json_candidate_str_chunk)
            llm_result_chunk = MinimalExtractionOutput(**parsed_json_object_chunk)
            validated_numbers_chunk = llm_result_chunk.extracted_numbers

            if len(validated_numbers_chunk) != len(current_chunk_candidate_items):
                logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Mismatch in item count. Input: {len(current_chunk_candidate_items)}, Output: {len(validated_numbers_chunk)}. Marking all in chunk as error or for retry.")
                # Decide if all become errors or add all to retry
                for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                    if final_processed_outputs_for_chunk[k_err] is None:
                         # Add to retry, or mark as error if retries exhausted / not applicable
                        items_needing_retry_for_chunk.append((k_err, item_detail_chunk_err))
                        # As a fallback if retry doesn't resolve:
                        # final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_LLMItemCountMismatch", ...)
            else:
                for k, input_item_detail_chunk in enumerate(current_chunk_candidate_items):
                    if final_processed_outputs_for_chunk[k] is not None: # Already processed (e.g. by a successful retry pass)
                        continue
                    llm_output_item_chunk = validated_numbers_chunk[k]
                    # Compare based on 'number' field as per original logic
                    if llm_output_item_chunk.number == input_item_detail_chunk.get('number') or \
                       llm_output_item_chunk.number == input_item_detail_chunk.get('candidate_number'): # Accommodate both key names
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
                f"Attempted to parse (json_candidate_str_chunk, first 1000 chars): '{json_candidate_str_chunk[:1000]}'. "
                f"Original LLM response text (llm_response_text, first 1000 chars): '{llm_response_text[:1000]}'"
            )
            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                if final_processed_outputs_for_chunk[k_err] is None:
                    final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, f"Error_ChunkJsonParseValidate_JSONDecodeError", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
        except Exception as e_parse_validate: # Broader exception for Pydantic if direct parsing
            logger.error(
                f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                f"Failed to parse/validate JSON for chunk: {type(e_parse_validate).__name__} - {e_parse_validate}. "
                f"Attempted to parse (json_candidate_str_chunk): '{json_candidate_str_chunk}'. "
                f"Original LLM response text (llm_response_text, first 1000 chars): '{llm_response_text[:1000]}...'"
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
            
            # Initial LLM call for the chunk
            items_needing_retry_this_pass: List[Tuple[int, Dict[str, Any]]] = []
            
            try:
                formatted_prompt_chunk = self._prepare_prompt_for_chunk(current_chunk_candidate_items, homepage_context_input)
                
                # Artifact logging for prompt (optional, can be done by caller)
                # save_llm_artifact(llm_context_dir, f"{chunk_log_prefix}_prompt.txt", formatted_prompt_chunk, logger)

                generation_config_dict = {
                    "max_output_tokens": self.config.llm_chunk_processor_max_tokens,
                    "temperature": self.config.llm_temperature,
                    # response_mime_type is not directly part of GenerationConfig dict for the client method,
                    # but often handled by the client or model endpoint.
                    # If it needs to be in a specific format for the client, adjust here.
                    # For now, assuming the client handles mime type or it's set globally.
                }
                # The gemini_client.generate_content_with_retry expects a GenerationConfig object.
                # We need to ensure that if we are constructing it from a dict, it's done correctly.
                # The example shows direct creation: genai_types.GenerationConfig(...)
                # Let's stick to direct creation for clarity and to match existing patterns.

                generation_config = genai_types.GenerationConfig(
                    max_output_tokens=self.config.llm_chunk_processor_max_tokens,
                    temperature=self.config.llm_temperature,
                    # response_mime_type="text/plain" # This is often default or handled by client
                    # If "text/plain" is critical and supported, keep it. Otherwise, it might be implicit.
                    # For Gemini, response_mime_type is valid.
                    response_mime_type="text/plain"
                )

                # Use gemini_client.generate_content_with_retry()
                # This client method should handle the actual API call and retries for network issues.
                # It should return a structure that includes the response text and token counts.
                llm_response_obj = self.gemini_client.generate_content_with_retry(
                    contents=formatted_prompt_chunk,
                    generation_config=generation_config,
                    file_identifier_prefix=chunk_log_prefix,
                    triggering_input_row_id=triggering_input_row_id,
                    triggering_company_name=triggering_company_name
                )
                
                raw_llm_response_text_chunk = None
                if llm_response_obj:
                    # Adapt based on actual structure of llm_response_obj from GeminiClient
                    raw_llm_response_text_chunk = getattr(llm_response_obj, 'text', None) # Or equivalent
                    if hasattr(llm_response_obj, 'usage_metadata') and llm_response_obj.usage_metadata:
                        prompt_tokens_val = llm_response_obj.usage_metadata.prompt_token_count
                        candidates_tokens_val = llm_response_obj.usage_metadata.candidates_token_count
                        total_tokens_val = llm_response_obj.usage_metadata.total_token_count
                        
                        token_stats_chunk = {
                            "prompt_tokens": prompt_tokens_val if prompt_tokens_val is not None else 0,
                            "completion_tokens": candidates_tokens_val if candidates_tokens_val is not None else 0, # or completion_token_count
                            "total_tokens": total_tokens_val if total_tokens_val is not None else 0
                        }
                        for key_token in accumulated_token_stats:
                            accumulated_token_stats[key_token] += token_stats_chunk.get(key_token, 0)
                        logger.info(f"[{chunk_log_prefix}] LLM (initial chunk) usage: {token_stats_chunk}")
                    
                    # Artifact logging for response
                    # if raw_llm_response_text_chunk:
                    #    save_llm_artifact(llm_context_dir, f"{chunk_log_prefix}_response_initial.txt", raw_llm_response_text_chunk, logger)

                    self._process_llm_response_for_chunk(
                        raw_llm_response_text_chunk,
                        current_chunk_candidate_items,
                        final_processed_outputs_for_chunk,
                        items_needing_retry_this_pass, # Populated by _process_llm_response_for_chunk
                        chunk_log_prefix,
                        triggering_input_row_id,
                        triggering_company_name
                    )
                    if raw_llm_response_text_chunk:
                        overall_raw_responses_list.append(raw_llm_response_text_chunk)

                else: # No response object from LLM call
                    logger.error(f"[{chunk_log_prefix}] No response object from LLM client for initial chunk call.")
                    for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                        final_processed_outputs_for_chunk[k_err] = create_error_llm_item(item_detail_chunk_err, "Error_ChunkNoLLMResponseObject", file_identifier_prefix=chunk_log_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)


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
                    # save_llm_artifact(llm_context_dir, f"{retry_log_prefix}_prompt.txt", formatted_prompt_retry_chunk, logger)

                    generation_config_retry = genai_types.GenerationConfig(
                        max_output_tokens=self.config.llm_chunk_processor_max_tokens,
                        temperature=self.config.llm_temperature,
                        response_mime_type="text/plain"
                    )

                    llm_response_obj_retry = self.gemini_client.generate_content_with_retry(
                        contents=formatted_prompt_retry_chunk,
                        generation_config=generation_config_retry,
                        file_identifier_prefix=retry_log_prefix,
                        triggering_input_row_id=triggering_input_row_id,
                        triggering_company_name=triggering_company_name
                    )

                    raw_llm_response_text_retry_chunk = None
                    if llm_response_obj_retry:
                        raw_llm_response_text_retry_chunk = getattr(llm_response_obj_retry, 'text', None)
                        if hasattr(llm_response_obj_retry, 'usage_metadata') and llm_response_obj_retry.usage_metadata:
                            prompt_tokens_retry_val = llm_response_obj_retry.usage_metadata.prompt_token_count
                            candidates_tokens_retry_val = llm_response_obj_retry.usage_metadata.candidates_token_count
                            total_tokens_retry_val = llm_response_obj_retry.usage_metadata.total_token_count

                            token_stats_retry_chunk = {
                                "prompt_tokens": prompt_tokens_retry_val if prompt_tokens_retry_val is not None else 0,
                                "completion_tokens": candidates_tokens_retry_val if candidates_tokens_retry_val is not None else 0,
                                "total_tokens": total_tokens_retry_val if total_tokens_retry_val is not None else 0
                            }
                            for key_token_r in accumulated_token_stats: accumulated_token_stats[key_token_r] += token_stats_retry_chunk.get(key_token_r, 0)
                            logger.info(f"[{retry_log_prefix}] LLM (mismatch retry) usage: {token_stats_retry_chunk}")
                        
                        # if raw_llm_response_text_retry_chunk:
                        #    save_llm_artifact(llm_context_dir, f"{retry_log_prefix}_response.txt", raw_llm_response_text_retry_chunk, logger)
                        
                        # Create temporary output list for this retry pass
                        temp_processed_outputs_for_retry_pass: List[Optional[PhoneNumberLLMOutput]] = [None] * len(inputs_for_this_retry_pass_details)
                        temp_items_needing_further_retry: List[Tuple[int, Dict[str, Any]]] = [] # Indices here are relative to inputs_for_this_retry_pass_details

                        self._process_llm_response_for_chunk(
                            raw_llm_response_text_retry_chunk,
                            inputs_for_this_retry_pass_details, # Current items being retried
                            temp_processed_outputs_for_retry_pass, # Temp list for this pass's results
                            temp_items_needing_further_retry,    # Items that *still* mismatch after this retry
                            retry_log_prefix,
                            triggering_input_row_id,
                            triggering_company_name
                        )
                        if raw_llm_response_text_retry_chunk:
                             overall_raw_responses_list.append(raw_llm_response_text_retry_chunk)


                        # Update final_processed_outputs_for_chunk with results from this retry pass
                        for j_retry, processed_item_from_retry in enumerate(temp_processed_outputs_for_retry_pass):
                            original_chunk_index = original_indices_for_this_retry_pass[j_retry]
                            if processed_item_from_retry is not None: # If successfully processed or became an error
                                final_processed_outputs_for_chunk[original_chunk_index] = processed_item_from_retry
                        
                        # Map items from temp_items_needing_further_retry back to original chunk indices
                        for k_further_retry, (idx_in_retry_pass, item_detail) in enumerate(temp_items_needing_further_retry):
                            original_chunk_idx_for_further_retry = original_indices_for_this_retry_pass[idx_in_retry_pass]
                            items_still_needing_retry_after_this_pass.append((original_chunk_idx_for_further_retry, item_detail))
                    
                    else: # No response object from LLM retry call
                        logger.error(f"[{retry_log_prefix}] No response object from LLM client for mismatch retry.")
                        # All items in this retry pass remain needing retry
                        items_still_needing_retry_after_this_pass.extend(items_needing_retry_this_pass)


                except Exception as e_retry_call:
                    logger.error(f"[{retry_log_prefix}] Error during mismatch retry LLM call: {e_retry_call}", exc_info=True)
                    # All items in this retry pass remain needing retry if call fails
                    items_still_needing_retry_after_this_pass.extend(items_needing_retry_this_pass)
                    overall_raw_responses_list.append(json.dumps({"error": f"Chunk mismatch retry call error: {str(e_retry_call)}", "type": type(e_retry_call).__name__}))

                items_needing_retry_this_pass = items_still_needing_retry_after_this_pass
            # End of mismatch retry loop for the chunk

            # Handle items persistently mismatched after all retries for this chunk
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
        # End of chunk loop

        final_combined_raw_response_str = "\n\n---CHUNK_SEPARATOR---\n\n".join(overall_raw_responses_list) if overall_raw_responses_list else None
        
        successful_items_count = sum(1 for item in overall_processed_outputs if item and not item.type.startswith("Error_"))
        error_items_count = len(overall_processed_outputs) - successful_items_count
        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Overall LLM chunk processing summary: {successful_items_count} successful, {error_items_count} errors out of {len(candidate_items)} candidates processed over {chunks_processed_count} chunks.")

        return overall_processed_outputs, final_combined_raw_response_str, accumulated_token_stats