import logging
import json
import os
from typing import Dict, Any, List, Tuple, Optional, Union

# New SDK imports
import google.generativeai.types as genai_types # Standardized to google.generativeai
from google.api_core import exceptions as google_exceptions # Keep for specific error handling
from pydantic import ValidationError as PydanticValidationError

# Project-specific imports
from src.core.config import AppConfig
from src.core.schemas import PhoneNumberLLMOutput, HomepageContextOutput, PhoneRankingOutput
from src.phone_retrieval.utils.helpers import sanitize_filename_component
from src.phone_retrieval.llm_clients.gemini_client import GeminiClient
from src.phone_retrieval.extractors.llm_chunk_processor import LLMChunkProcessor
from src.phone_retrieval.utils.llm_processing_helpers import (
    load_prompt_template,
    save_llm_artifact,
    adapt_schema_for_gemini,
    extract_json_from_text,
)


logger = logging.getLogger(__name__)

# RETRYABLE_GEMINI_EXCEPTIONS is now handled by GeminiClient

class GeminiLLMExtractor:
    """
    Orchestrates LLM-based extraction tasks, utilizing GeminiClient for API interactions
    and LLMChunkProcessor for handling chunked processing of candidate items.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the GeminiLLMExtractor.

        Args:
            config (AppConfig): Application configuration.
        """
        self.config = config
        self.gemini_client = GeminiClient(config)
        
        # Use prompt_path_minimal_classification for phone number extraction prompt
        self.phone_extraction_prompt_path = "prompts/phone_extraction_prompt.txt"
        logger.info(f"Using phone extraction prompt: {self.phone_extraction_prompt_path}")

        # Second-stage phone ranking prompt (LLM ranks callable business numbers for outreach)
        self.phone_ranking_prompt_path = "prompts/phone_ranking_prompt.txt"
        logger.info(f"Using phone ranking prompt: {self.phone_ranking_prompt_path}")


        self.chunk_processor = LLMChunkProcessor(
            config=self.config,
            gemini_client=self.gemini_client,
            prompt_template_path=self.phone_extraction_prompt_path # Corrected parameter name
        )
        logger.info(f"GeminiLLMExtractor initialized, using GeminiClient and LLMChunkProcessor for phone extraction.")

    def rank_phone_numbers_for_outreach(
        self,
        *,
        company_name: str,
        canonical_base_url: str,
        candidates: List[Dict[str, Any]],
        llm_context_dir: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
    ) -> Tuple[Optional[PhoneRankingOutput], Optional[str], Optional[str]]:
        """
        Second-stage LLM call: rank callable business numbers for outreach (top 3)
        and optionally identify a main-office backup number.

        Returns:
            (parsed_output, raw_text, error_message)
        """
        log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {company_name}, Type: PhoneRanking]"
        try:
            if not candidates:
                return None, None, None

            prompt_template = load_prompt_template(self.phone_ranking_prompt_path)
            payload = {
                "company_name": company_name,
                "canonical_base_url": canonical_base_url,
                "candidates": candidates,
            }
            prompt = prompt_template.replace(
                "{{PHONE_RANKING_INPUT_JSON_PLACEHOLDER}}",
                json.dumps(payload, ensure_ascii=False),
            )

            # NOTE: The legacy `google.generativeai` SDK rejects JSON schemas containing `$defs`.
            # To keep this robust, we rely on prompt-enforced JSON and parse manually (no response_schema).
            gen_cfg = genai_types.GenerationConfig(
                temperature=0.1,
                # Some sites generate a lot of candidates; give the model enough room to finish valid JSON.
                max_output_tokens=4096,
                response_mime_type="application/json",
            )
            resp = self.gemini_client.generate_content_with_retry(
                contents=prompt,
                generation_config=gen_cfg,
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=company_name,
            )
            raw_text = getattr(resp, "text", None)

            # Persist artifacts (best-effort)
            try:
                save_llm_artifact(
                    content=prompt,
                    directory=llm_context_dir,
                    filename=f"{sanitize_filename_component(file_identifier_prefix)}__phone_ranking_prompt.txt",
                    log_prefix=log_prefix,
                )
                if raw_text:
                    save_llm_artifact(
                        content=raw_text,
                        directory=llm_context_dir,
                        filename=f"{sanitize_filename_component(file_identifier_prefix)}__phone_ranking_raw_output.txt",
                        log_prefix=log_prefix,
                    )
            except Exception:
                pass

            json_str = extract_json_from_text(raw_text) if raw_text else None
            if not json_str:
                err = "Could not extract JSON from phone ranking response."
                logger.warning(f"{log_prefix} {err}")
                # Persist error artifact (best-effort)
                try:
                    save_llm_artifact(
                        content=err,
                        directory=llm_context_dir,
                        filename=f"{sanitize_filename_component(file_identifier_prefix)}__phone_ranking_error.txt",
                        log_prefix=log_prefix,
                    )
                except Exception:
                    pass
                return None, raw_text, err
            try:
                obj = json.loads(json_str)
                parsed = PhoneRankingOutput(**obj)
                return parsed, raw_text, None
            except Exception as e:
                err = f"Failed to parse phone ranking JSON: {e}"
                logger.warning(f"{log_prefix} {err}")
                # Persist error artifact (best-effort)
                try:
                    save_llm_artifact(
                        content=err,
                        directory=llm_context_dir,
                        filename=f"{sanitize_filename_component(file_identifier_prefix)}__phone_ranking_error.txt",
                        log_prefix=log_prefix,
                    )
                except Exception:
                    pass
                return None, raw_text, err
        except Exception as e:
            err = f"Unexpected error during phone ranking: {e}"
            logger.error(f"{log_prefix} {err}", exc_info=True)
            # Persist error artifact (best-effort)
            try:
                save_llm_artifact(
                    content=err,
                    directory=llm_context_dir,
                    filename=f"{sanitize_filename_component(file_identifier_prefix)}__phone_ranking_error.txt",
                    log_prefix=log_prefix,
                )
            except Exception:
                pass
            return None, None, err

    # Redundant private methods (_load_prompt_template, _normalize_phone_number,
    # _extract_json_from_text, _generate_content_with_retry,
    # _process_successful_llm_item, _create_error_llm_item) are removed.
    # Their functionalities are now in llm_processing_helpers or GeminiClient or LLMChunkProcessor.

    def generate_homepage_context(
        self,
        homepage_content: str,
        prompt_template_path: str,
        llm_context_dir: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str
    ) -> Tuple[Optional[HomepageContextOutput], Optional[str], Optional[Dict[str, int]]]:
        """
        Generates homepage context (summary, company name, industry) using the LLM.

        Args:
            homepage_content (str): The full text content of the homepage.
            prompt_template_path (str): Path to the prompt template for homepage context generation.
            llm_context_dir (str): Directory to save LLM interaction files (prompt, response).
            file_identifier_prefix (str): Prefix for naming context files.
            triggering_input_row_id (Any): ID of the input row triggering this generation.
            triggering_company_name (str): Company name associated with the input row.

        Returns:
            Tuple[Optional[HomepageContextOutput], Optional[str], Optional[Dict[str, int]]]:
            A tuple containing the parsed HomepageContextOutput, the raw LLM response string,
            and token usage statistics. Returns (None, raw_response, None) on error.
        """
        log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: HomepageContext]"
        logger.info(f"{log_prefix} Starting homepage context generation.")
        
        raw_llm_response_str: Optional[str] = None
        token_stats: Optional[Dict[str, int]] = None # Handled by GeminiClient now
        parsed_output: Optional[HomepageContextOutput] = None

        try:
            # Log and truncate homepage_content before using it in the prompt
            logger.info(f"{log_prefix} Original homepage_content length: {len(homepage_content)}")
            MAX_HOMEPAGE_CONTENT_CHARS = self.config.scraper_max_pages_per_domain * 1000 # Example: 20 * 1000 = 20000, make this configurable if needed
            # A more robust approach might be to use a dedicated config for this char limit.
            # For now, using a value derived from an existing config or a sensible default.
            # Let's use a fixed value for now, can be made configurable later.
            # MAX_HOMEPAGE_CONTENT_CHARS = 20000 # Example fixed value
            # Using a potentially more relevant config: llm_max_tokens for input, assuming 1 token ~ 4 chars
            # This is a rough estimate. A dedicated config is better.
            # Let's use a simpler fixed limit for now.
            MAX_HOMEPAGE_CONTENT_CHARS = 30000 # Increased slightly

            if len(homepage_content) > MAX_HOMEPAGE_CONTENT_CHARS:
                logger.warning(f"{log_prefix} Truncating homepage_content from {len(homepage_content)} to {MAX_HOMEPAGE_CONTENT_CHARS} chars.")
                homepage_content_for_prompt = homepage_content[:MAX_HOMEPAGE_CONTENT_CHARS]
            else:
                homepage_content_for_prompt = homepage_content

            prompt_template = load_prompt_template(prompt_template_path)
            formatted_prompt = prompt_template.replace("[TEXT Content]", homepage_content_for_prompt)
        except FileNotFoundError:
            logger.error(f"{log_prefix} Prompt template file not found: {prompt_template_path}")
            return None, f"Error: Prompt template file not found: {prompt_template_path}", None
        except Exception as e:
            logger.error(f"{log_prefix} Failed to load/format homepage context prompt: {e}")
            return None, f"Error: Failed to load/format prompt: {str(e)}", None

        # Sanitize filename components - using shorter max_len to avoid truncation of suffixes by save_llm_artifact
        s_file_id_prefix = sanitize_filename_component(file_identifier_prefix, max_len=15) # Shorter
        s_row_id = sanitize_filename_component(str(triggering_input_row_id), max_len=8)    # Shorter
        s_comp_name = sanitize_filename_component(triggering_company_name, max_len=self.config.filename_company_name_max_len if self.config.filename_company_name_max_len <= 20 else 20) # Shorter, capped at 20

        # --- Diagnostic logging for directory (moved before any save attempt) ---
        logger.info(f"{log_prefix} Attempting to save artifacts to directory: {llm_context_dir}")
        if not os.path.exists(llm_context_dir):
            logger.warning(f"{log_prefix} LLM context directory does NOT exist: {llm_context_dir} - it should be created by setup_output_directories or save_llm_artifact.")
            # save_llm_artifact will attempt os.makedirs, so this warning is informational.
        else:
            logger.info(f"{log_prefix} LLM context directory confirmed to exist: {llm_context_dir}")
            if not os.access(llm_context_dir, os.W_OK): # Check writability if it exists
                logger.error(f"{log_prefix} LLM context directory is NOT writable: {llm_context_dir}")
            else:
                logger.info(f"{log_prefix} LLM context directory IS writable: {llm_context_dir}")
        # --- End diagnostic logging ---

        # Save formatted prompt using helper
        # Ensure the full intended filename is passed to sanitize_filename_component if necessary,
        # or construct the base and let save_llm_artifact sanitize only the 'filename' part.
        # Current approach: construct filename, then save_llm_artifact sanitizes it.
        prompt_filename_base = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}"
        prompt_filename_with_suffix = f"{prompt_filename_base}_homepage_context_prompt.txt"
        
        try:
            save_llm_artifact(
                content=formatted_prompt,
                directory=llm_context_dir,
                filename=prompt_filename_with_suffix, # Use the full name
                log_prefix=log_prefix
            )
        except Exception as e_save_prompt:
             logger.error(f"{log_prefix} Failed to save formatted prompt artifact '{prompt_filename_with_suffix}': {e_save_prompt}")
             # Decide if we should return or continue if prompt saving fails
        
        # System prompt for structured JSON output
        # For now, keeping it separate as per original logic structure.
        # The GeminiClient might also handle system prompts in a specific way.
        # Let's assume the `formatted_prompt` is the primary user input to the LLM.
        # If `GeminiClient` expects a list of `ContentDict`, this needs adjustment.
        # Based on `GeminiClient` structure, it takes `prompt_parts` which can be a list.
        # For simplicity here, let's assume `formatted_prompt` is the main content.
        # The `GeminiClient` will construct the `contents` list.

        # Prepare schema for Gemini
        gemini_schema = adapt_schema_for_gemini(HomepageContextOutput)

        generation_config_dict = {
            "response_mime_type": "text/plain",
            "candidate_count": 1, # Default, can be configured
            "max_output_tokens": self.config.llm_max_tokens, # Revert to AppConfig value
            "temperature": self.config.llm_temperature,
            # Add top_k and top_p if they are set in config
        }
        if self.config.llm_top_k is not None:
            generation_config_dict["top_k"] = self.config.llm_top_k
        if self.config.llm_top_p is not None:
            generation_config_dict["top_p"] = self.config.llm_top_p
            
        generation_config = genai_types.GenerationConfig(**generation_config_dict) # Corrected: use genai_types

        system_instruction_text = ("You are a data extraction assistant. Your entire response MUST be a single, valid JSON formatted string. Do NOT include any explanations, markdown formatting (like ```json), or any other text outside of this JSON string. Within the JSON string, for the 'summary_description' field, adhere STRICTLY to a maximum of 250 characters and 2-3 short sentences. If a value is not clearly present, set it to null within the JSON.")

        # system_instruction_text = (
        #     "You are a structured data extraction assistant. Respond only with a valid JSON "
        #     "object that matches the provided schema. Do not include explanations or any extra text. "
        #     "If a value is not clearly present, set it to null."
        # )
        # Combine system instruction with the user-facing prompt for the `contents` parameter.
        # The `GeminiClient` expects `contents` to be `Union[str, Iterable[genai_types.ContentDict]]`.
        # For a single turn with instructions, a combined string is simplest.
        # Or, a single ContentDict with one Part.
        # `Iterable[ContentDict]` is for multi-turn.
        # Let's use a single string for `contents` as `GeminiClient` can handle it.
        
        # The client's `generate_content_with_retry` takes `contents`.
        # If `contents` is a string, it's treated as a single user message.
        # If it's `Iterable[ContentDict]`, it's for multi-turn.
        # For this case, we want to pass the system instruction and the formatted prompt.
        # The most robust way if the client doesn't have a separate system_instruction param
        # is to structure `contents` as `[ContentDict(role='user', parts=[Part(text=combined_prompt)])]`.
        # Or, if the model supports it, `[ContentDict(role='system', ...), ContentDict(role='user', ...)]`.
        # Given `GeminiClient`'s signature, let's construct `Iterable[ContentDict]`.
        # The simplest for a single query with system instructions is often to prepend them to the user prompt.
        
        # Construct `contents` as `Iterable[genai_types.ContentDict]`
        # For a single turn, this is typically one ContentDict with role 'user'.
        # The `system_instruction_text` will be passed separately to the client.
        contents_for_api: List[genai_types.ContentDict] = [
            genai_types.ContentDict(
                role="user",
                parts=[{'text': formatted_prompt}] # Use only the formatted_prompt
            )
        ]

        # Log the request payload for debugging
        # The `contents_for_api` is already a list of TypedDicts, which is serializable.
        request_payload_to_log = {
            "model_name": self.config.llm_model_name,
            "system_instruction": system_instruction_text,
            "user_contents": contents_for_api, # This is directly serializable
            "generation_config": generation_config_dict
        }
        request_payload_filename = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}_homepage_context_request_payload.json"
        
        try:
            save_llm_artifact(
                content=json.dumps(request_payload_to_log, indent=2), # Serialize the dict to a JSON string
                directory=llm_context_dir,
                filename=request_payload_filename,
                log_prefix=log_prefix
                # Removed is_json=True as it's not a valid parameter for save_llm_artifact
            )
        except Exception as e_save_payload:
            logger.error(f"{log_prefix} Failed to save request payload artifact: {e_save_payload}")

        try:
            # Call GeminiClient with corrected parameters, including system_instruction
            response = self.gemini_client.generate_content_with_retry(
                contents=contents_for_api, # Original contents_for_api is fine for the client
                generation_config=generation_config,
                system_instruction=system_instruction_text, # Pass system instruction separately
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name
            )

            raw_llm_response_str = None
            token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            parsed_output = None

            if response:
                raw_llm_response_str = response.text

                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    prompt_tokens_val = response.usage_metadata.prompt_token_count
                    candidates_tokens_val = response.usage_metadata.candidates_token_count
                    total_tokens_val = response.usage_metadata.total_token_count

                    token_stats["prompt_tokens"] = prompt_tokens_val if prompt_tokens_val is not None else 0
                    token_stats["completion_tokens"] = candidates_tokens_val if candidates_tokens_val is not None else 0
                    token_stats["total_tokens"] = total_tokens_val if total_tokens_val is not None else 0
                else:
                    logger.warning(f"{log_prefix} LLM usage metadata not found in response.")
                
                logger.info(f"{log_prefix} LLM usage: {token_stats}")

                if raw_llm_response_str:
                    response_filename = f"{s_file_id_prefix}_rid{s_row_id}_comp{s_comp_name}_homepage_context_response.txt"
                    save_llm_artifact(
                        content=raw_llm_response_str,
                        directory=llm_context_dir,
                        filename=response_filename,
                        log_prefix=log_prefix
                    )
                
                # Parsing logic moved inside the `if response:` block and correctly indented
                if response.candidates and raw_llm_response_str:
                    json_string_from_text: Optional[str] = None # Initialize to ensure it's always defined
                    try:
                        json_string_from_text = extract_json_from_text(raw_llm_response_str) # Ensure extract_json_from_text is imported
                        if json_string_from_text:
                            parsed_json_object = json.loads(json_string_from_text)
                            parsed_output = HomepageContextOutput(**parsed_json_object)
                            logger.info(f"{log_prefix} Successfully extracted, parsed, and validated homepage context from text response.")
                        else:
                            logger.error(f"{log_prefix} Failed to extract a JSON string from LLM's plain text response. Raw: '{raw_llm_response_str[:500]}'")
                            # parsed_output will remain None due to its initialization
                            
                    except json.JSONDecodeError as e_json:
                        logger.error(f"{log_prefix} Failed to parse extracted JSON string: {e_json}. Extracted string: '{json_string_from_text[:500] if json_string_from_text else 'N/A'}'. Raw LLM response: '{raw_llm_response_str[:200]}'")
                        # parsed_output will remain None
                    except PydanticValidationError as e_pydantic:
                        logger.error(f"{log_prefix} Pydantic validation failed for homepage context: {e_pydantic}. Data: '{json_string_from_text[:500] if json_string_from_text else 'N/A'}'")
                        # parsed_output will remain None
                elif not response.candidates: # Handles case where response exists but no candidates
                     logger.error(f"{log_prefix} No candidates in Gemini response for homepage context. Raw: '{raw_llm_response_str[:200] if raw_llm_response_str else 'N/A'}'")

            else: # No response object from GeminiClient
                logger.error(f"{log_prefix} No response object returned from GeminiClient for homepage context.")
            
            return parsed_output, raw_llm_response_str, token_stats

        except google_exceptions.GoogleAPIError as e_api:
            logger.error(f"{log_prefix} Gemini API error during homepage context generation (extractor level): {e_api}")
            error_msg = getattr(e_api, 'message', str(e_api))
            current_raw_response = json.dumps({"error": f"Gemini API error: {error_msg}", "type": type(e_api).__name__})
            # Ensure token_stats is returned, even if it's the default initialized one
            return None, current_raw_response, token_stats if token_stats else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except PydanticValidationError as e_pydantic:
            logger.error(f"""{log_prefix} Pydantic validation failed for homepage context (extractor level): {e_pydantic}. Data: '{raw_llm_response_str[:200] if raw_llm_response_str else "N/A"}...'""")
            return None, raw_llm_response_str, token_stats if token_stats else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        except Exception as e_gen:
            logger.error(f"{log_prefix} Unexpected error during homepage context generation (extractor level): {e_gen}", exc_info=True)
            current_raw_response = raw_llm_response_str if raw_llm_response_str else json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
            return None, current_raw_response, token_stats if token_stats else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
# The mis-indented block from line 256 to 278 in the original file is now correctly part of the try-except structure.
# The following lines were part of that mis-indented block and are removed here as their logic is now integrated above.
#                 # The new SDK with response_schema might provide parsed data directly.
#                 # Check `response_obj.candidates[0].content.parts[0].json_data` (or similar)
#                 # or if GeminiClient already parsed it into Pydantic.
#                 # Assuming GeminiClient returns the Pydantic object if parsing is successful.
#                 if isinstance(response_obj, HomepageContextOutput):
#                     parsed_output = response_obj
#                     logger.info(f"{log_prefix} Successfully generated and parsed homepage context via GeminiClient.")
#                 else: # Fallback if GeminiClient didn't return Pydantic (e.g. error or unexpected structure)
#                       # This case should ideally be handled within GeminiClient or be an error state.
#                       # For now, if it's not the Pydantic object, we assume parsing failed upstream.
#                     logger.warning(f"{log_prefix} GeminiClient did not return a parsed HomepageContextOutput. Raw response: {raw_llm_response_str[:200] if raw_llm_response_str else 'N/A'}")
#                     # Attempt manual parsing as a last resort if raw_llm_response_str is available
#                     if raw_llm_response_str:
#                         try:
#                             parsed_json_object = json.loads(raw_llm_response_str)
#                             parsed_output = HomepageContextOutput(**parsed_json_object)
#                             logger.info(f"{log_prefix} Successfully parsed homepage context from raw response as fallback.")
#                         except (json.JSONDecodeError, PydanticValidationError) as e_fallback_parse:
#                             logger.error(f"{log_prefix} Fallback parsing of raw response failed: {e_fallback_parse}")
#                             # raw_llm_response_str is already set
#             else: # response_obj is None, indicating an error handled by GeminiClient
#                 logger.error(f"{log_prefix} No valid response object from GeminiClient for homepage context. Raw: {raw_llm_response_str[:200] if raw_llm_response_str else 'N/A'}")
#                 # raw_llm_response_str and token_stats should be set by GeminiClient even on error
#
#             return parsed_output, raw_llm_response_str, token_stats
#
#         except google_exceptions.GoogleAPIError as e_api: # Should be caught by GeminiClient, but as a safeguard
#             logger.error(f"{log_prefix} Gemini API error during homepage context generation (extractor level): {e_api}")
#             error_msg = getattr(e_api, 'message', str(e_api))
#             if raw_llm_response_str is None: raw_llm_response_str = json.dumps({"error": f"Gemini API error: {error_msg}", "type": type(e_api).__name__})
#             return None, raw_llm_response_str, token_stats
#         except PydanticValidationError as e_pydantic: # If parsing happens here (fallback)
#             logger.error(f"""{log_prefix} Pydantic validation failed for homepage context (extractor level): {e_pydantic}. Data: '{raw_llm_response_str[:200] if raw_llm_response_str else "N/A"}...'""")
#             return None, raw_llm_response_str, token_stats
#         except Exception as e_gen:
#             logger.error(f"{log_prefix} Unexpected error during homepage context generation (extractor level): {e_gen}", exc_info=True)
#             if raw_llm_response_str is None: raw_llm_response_str = json.dumps({"error": f"Unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
#             return None, raw_llm_response_str, token_stats


    def extract_phone_numbers(
        self,
        candidate_items: List[Dict[str, Any]], # Type changed to Any for flexibility with LLMChunkProcessor
        # prompt_template_path: str, # This is now handled by LLMChunkProcessor's init or process_candidates
        llm_context_dir: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        homepage_context_input: Optional[HomepageContextOutput] = None
    ) -> Tuple[List[PhoneNumberLLMOutput], Optional[str], Optional[Dict[str, int]]]:
        """
        Extracts and classifies phone numbers from candidate items using LLMChunkProcessor.

        Args:
            candidate_items (List[Dict[str, Any]]): List of candidate items to process.
            llm_context_dir (str): Directory to save LLM interaction files.
            file_identifier_prefix (str): Prefix for naming context files.
            triggering_input_row_id (Any): ID of the input row.
            triggering_company_name (str): Company name.
            homepage_context_input (Optional[HomepageContextOutput]): Context from homepage analysis.

        Returns:
            Tuple[List[PhoneNumberLLMOutput], Optional[str], Optional[Dict[str, int]]]:
            Processed phone numbers, combined raw LLM responses, and accumulated token stats.
        """
        log_prefix = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Type: PhoneExtraction]"
        logger.info(f"{log_prefix} Starting phone number extraction using LLMChunkProcessor.")

        if not self.phone_extraction_prompt_path:
            logger.error(f"{log_prefix} Phone extraction prompt path not configured. Cannot proceed.")
            return [], json.dumps({"error": "Phone extraction prompt path not configured."}), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Save the base prompt template once if not already done by LLMChunkProcessor or elsewhere
        # LLMChunkProcessor might handle this internally, or we can do it here.
        # For now, let's assume LLMChunkProcessor might need the prompt path.
        # The original logic saved "llm_prompt_template.txt".
        # This can be simplified if LLMChunkProcessor handles its own prompt loading and artifact saving.
        # For now, we'll rely on LLMChunkProcessor to manage its prompt.

        # Delegate to LLMChunkProcessor
        # Ensure all necessary arguments are passed.
        # LLMChunkProcessor's process_candidates will handle chunking, API calls, retries, parsing, etc.
        
        processed_outputs, combined_raw_responses_str, accumulated_token_stats = \
            self.chunk_processor.process_candidates(
                candidate_items=candidate_items,
                # prompt_template_path is part of chunk_processor's init or passed if dynamic
                llm_context_dir=llm_context_dir,
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name,
                homepage_context_input=homepage_context_input,
                # output_schema=PhoneNumberLLMOutput, # LLMChunkProcessor should know this from its config
                # item_schema_for_prompt=... # LLMChunkProcessor should know this
            )

        successful_items_count = sum(1 for item in processed_outputs if item and not getattr(item, 'type', '').startswith("Error_"))
        error_items_count = len(processed_outputs) - successful_items_count
        logger.info(f"{log_prefix} LLMChunkProcessor completed. Summary: {successful_items_count} successful, {error_items_count} errors out of {len(candidate_items)} candidates.")

        return processed_outputs, combined_raw_responses_str, accumulated_token_stats