import logging
import json
import re
import os
from typing import Dict, Any, List, Tuple, Optional

from google import genai # New SDK
# from google.generativeai.generative_models import GenerativeModel # Removed
from google.genai import types # New SDK
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import ValidationError as PydanticValidationError

import phonenumbers
from phonenumbers import PhoneNumberFormat

# Assuming schemas are in core.schemas and config in core.config
from .core.schemas import PhoneNumberLLMOutput, MinimalExtractionOutput
from .core.config import AppConfig

logger = logging.getLogger(__name__)

RETRYABLE_GEMINI_EXCEPTIONS = (
    google_exceptions.DeadlineExceeded,
    google_exceptions.ServiceUnavailable,
    google_exceptions.ResourceExhausted,  # For rate limits
    google_exceptions.InternalServerError, # 500 errors
    google_exceptions.Aborted
    # google_exceptions.Unavailable # Removed as it's not a known attribute, ServiceUnavailable covers the intent
)

class GeminiLLMExtractor:
    """
    A component responsible for extracting phone numbers from text using the
    Google Gemini Large Language Model (LLM).

    This class handles loading prompt templates, interacting with the Gemini API
    to get structured JSON output (conforming to `PhoneNumberLLMOutput` schema),
    and normalizing the extracted phone numbers.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the GeminiLLMExtractor with necessary configurations.

        Args:
            config (AppConfig): An instance of `AppConfig` containing settings
                                such as the Gemini API key, model name, temperature,
                                max tokens, and paths for prompt templates.

        Raises:
            ValueError: If `GEMINI_API_KEY` is not found in the provided configuration.
        """
        self.config = config
        if not self.config.gemini_api_key:
            logger.error("GEMINI_API_KEY not provided in configuration.")
            raise ValueError("GEMINI_API_KEY not found in configuration.")
        
        # configure(api_key=self.config.gemini_api_key) # Removed
        
        self.client = genai.Client(api_key=self.config.gemini_api_key) # New SDK client
        # self.model = GenerativeModel( # Removed
        #     self.config.llm_model_name,
        #     # generation_config is set per-request to include response_schema
        # ) # Removed
        logger.info(f"GeminiLLMExtractor initialized to use model: {self.config.llm_model_name} via google-genai SDK.")

    def _load_prompt_template(self, prompt_file_path: str) -> str:
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
            # Ensure path is absolute or correctly relative to where config expects it
            # AppConfig.llm_prompt_template_path is relative to project root.
            # If prompt_file_path is passed directly, ensure it's resolvable.
            # For now, assume prompt_file_path is correctly resolved by the caller.
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {prompt_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading prompt template file {prompt_file_path}: {e}")
            raise

    def _normalize_phone_number(self, number_str: str, country_codes: List[str]) -> Optional[str]:
        """
        Normalizes a given phone number string to E.164 format.

        It attempts to parse the number using each of the provided `country_codes`
        as a region hint. If unsuccessful, it falls back to using the
        `self.config.default_region_code`.

        Args:
            number_str (str): The phone number string to normalize.
            country_codes (List[str]): A list of ISO 3166-1 alpha-2 country codes
                                       (e.g., ["US", "DE"]) to use as hints for parsing.

        Returns:
            Optional[str]: The normalized phone number in E.164 format if successful,
                           otherwise None.
        """
        if not number_str or not isinstance(number_str, str):
            return None

        for country_code in country_codes: # Iterate through preferred country codes
            try:
                parsed_num = phonenumbers.parse(number_str, region=country_code.upper())
                if phonenumbers.is_valid_number(parsed_num):
                    return phonenumbers.format_number(parsed_num, PhoneNumberFormat.E164)
            except phonenumbers.NumberParseException:
                # Log lightly, as this is expected for some numbers/regions
                logger.debug(f"Could not parse '{number_str}' with region '{country_code}'.")
                continue # Try next country code
        
        # Fallback if not parsed with specific country codes, try with default_region_code
        if self.config.default_region_code:
            try:
                parsed_num = phonenumbers.parse(number_str, region=self.config.default_region_code.upper())
                if phonenumbers.is_valid_number(parsed_num):
                    logger.debug(f"Normalized '{number_str}' to E.164 using default region '{self.config.default_region_code}'.")
                    return phonenumbers.format_number(parsed_num, PhoneNumberFormat.E164)
            except phonenumbers.NumberParseException:
                logger.info(f"Could not parse phone number '{number_str}' even with default region '{self.config.default_region_code}'.")
        
        logger.info(f"Could not normalize phone number '{number_str}' to E.164 with hints {country_codes} or default region.")
        return None
    def _extract_json_from_text(self, text_output: Optional[str]) -> Optional[str]:
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
        # This is a best-effort extraction.
        match = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```|(\{.*\}|\[.*\])",
            text_output,
            re.DOTALL # DOTALL allows . to match newlines
        )

        if match:
            # Prioritize the content within backticks if both groups match
            # (e.g. ```json { "key": "value" } ```)
            # Group 1 is for content within ```json ... ``` or ``` ... ```
            # Group 2 is for standalone JSON object/array
            json_str = match.group(1) or match.group(2)
            if json_str:
                return json_str.strip()
        
        logger.debug(f"No clear JSON block found in LLM text output: {text_output[:200]}...")
        return None

    @retry(
        stop=stop_after_attempt(3),  # Try 3 times in total (1 initial + 2 retries)
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Wait 2s, then 4s (max 10s)
        retry=retry_if_exception_type(RETRYABLE_GEMINI_EXCEPTIONS),
        reraise=True  # Reraise the exception if all retries fail
    )
    def _generate_content_with_retry(self, formatted_prompt: str, generation_config: types.GenerateContentConfig, file_identifier_prefix: str, triggering_input_row_id: Any, triggering_company_name: str):
        """
        Internal method to call Gemini API with retry logic.
        """
        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Attempting to generate content with Gemini API...")
        response = self.client.models.generate_content(
            model=self.config.llm_model_name,
            contents=formatted_prompt,
            config=generation_config
        )
        # Basic check for safety, though specific non-retriable content blocks
        # would ideally be handled by the caller if they are not exceptions.
        # Accessing prompt_feedback might differ in the new SDK.
        # Example: response.candidates[0].finish_reason (if it's 'SAFETY')
        # or a dedicated safety_ratings attribute. This needs verification.
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
            logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Gemini content generation blocked. Reason: {response.prompt_feedback.block_reason.name}. This might not be retriable by network retries.")
        elif hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason.name == 'SAFETY': # Example, actual enum might differ
                     logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Gemini content generation blocked due to safety settings. Finish Reason: SAFETY.")
                     # Potentially inspect candidate.safety_ratings here
                     break 

        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Successfully generated content from Gemini API attempt.")
        return response

    def _process_successful_llm_item(
        self,
        llm_output: PhoneNumberLLMOutput,
        input_item_details: Dict[str, Any]
    ) -> PhoneNumberLLMOutput:
        """Enriches and normalizes a successfully matched LLM output item."""
        llm_output.source_url = input_item_details.get('source_url')
        llm_output.original_input_company_name = input_item_details.get('original_input_company_name')

        if llm_output.number:
            normalized_num = self._normalize_phone_number(llm_output.number, self.config.target_country_codes)
            if normalized_num:
                llm_output.number = normalized_num
            else:
                logger.warning(f"LLM output number '{llm_output.number}' (from input '{input_item_details.get('number')}') could not be normalized. Keeping as is.")
        return llm_output

    def _create_error_llm_item(
        self,
        input_item_details: Dict[str, Any],
        error_type_str: str = "Error_ProcessingFailed",
        classification_str: str = "Non-Business",
        file_identifier_prefix: Optional[str] = "N/A", # Added
        triggering_input_row_id: Optional[Any] = "N/A", # Added
        triggering_company_name: Optional[str] = "N/A" # Added
    ) -> PhoneNumberLLMOutput:
        """Creates a PhoneNumberLLMOutput for an item that failed processing."""
        logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Creating error item for input number '{input_item_details.get('number')}' from source '{input_item_details.get('source_url', 'Unknown Source')}' due to: {error_type_str}")
        return PhoneNumberLLMOutput(
            number=str(input_item_details.get('number')), # Use the original input number
            type=error_type_str,
            classification=classification_str,
            source_url=input_item_details.get('source_url'),
            original_input_company_name=input_item_details.get('original_input_company_name')
            # snippet is not part of PhoneNumberLLMOutput, it's input context
            # confidence and other fields will use Pydantic defaults (e.g., None)
        )

    def extract_phone_numbers(
        self,
        candidate_items: List[Dict[str, str]], # Changed input
        prompt_template_path: str,
        llm_context_dir: str,  # New parameter
        file_identifier_prefix: str,  # New parameter
        triggering_input_row_id: Any,
        triggering_company_name: str
    ) -> Tuple[List[PhoneNumberLLMOutput], Optional[str], Optional[Dict[str, int]]]:
        """
        Classifies candidate phone numbers based on their snippets and source URLs using the Gemini API.

        The method loads a prompt template, formats it with the list of candidate items (each
        containing a number, its snippet, and source URL), and sends it to the Gemini model.
        It expects a JSON response conforming to LLMExtractionResult, which contains a list
        of PhoneNumberLLMOutput objects (now with a 'classification' field).

        Args:
            candidate_items (List[Dict[str, str]]): A list of dictionaries, where each
                                                   dictionary contains "candidate_number",
                                                   "snippet", "source_url", and "original_input_company_name".
           prompt_template_path (str): The file path to the prompt template. The template
                                       should expect a JSON list of these candidate items.
            llm_context_dir (str): The directory path to save LLM context files.
            file_identifier_prefix (str): A prefix for naming LLM context files (e.g., "CANONICAL_domain_com").

        Returns:
            Tuple[List[PhoneNumberLLMOutput], Optional[str]]:
            A tuple where the first element is a list of `PhoneNumberLLMOutput`
            objects (each potentially containing a normalized phone number and
            its context/confidence). The second element is a string containing
            the raw JSON response from the LLM, or an error message string if
            an error occurred. The list of `PhoneNumberLLMOutput` objects will
            be empty if no numbers are found or if an error prevents extraction.

        Raises:
            Catches and logs various exceptions including `FileNotFoundError` for
            the prompt, `google_exceptions.GoogleAPIError` for API issues,
            `json.JSONDecodeError`, and `PydanticValidationError`.
        """
        overall_processed_outputs: List[PhoneNumberLLMOutput] = []
        overall_raw_responses: List[str] = []
        accumulated_token_stats: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        chunk_size = self.config.llm_candidate_chunk_size
        max_chunks = self.config.llm_max_chunks_per_url
        chunks_processed_count = 0

        # --- BEGIN LOGIC FOR SAVING TEMPLATE ONCE (moved outside chunk loop) ---
        try:
            run_output_dir = os.path.dirname(llm_context_dir)
            if not run_output_dir:
                logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Could not determine run_output_dir from llm_context_dir: '{llm_context_dir}'. Cannot save prompt template.")
            else:
                os.makedirs(run_output_dir, exist_ok=True)
                template_output_filename = "llm_prompt_template.txt"
                template_output_filepath = os.path.join(run_output_dir, template_output_filename)
                if not os.path.exists(template_output_filepath):
                    logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Attempting to save base LLM prompt template to {template_output_filepath}")
                    try:
                        base_prompt_content = self._load_prompt_template(prompt_template_path)
                        with open(template_output_filepath, 'w', encoding='utf-8') as f_template:
                            f_template.write(base_prompt_content)
                        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Successfully saved base LLM prompt template to {template_output_filepath}")
                    except Exception as e_template_save:
                        logger.error(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Error saving base LLM prompt template: {e_template_save}")
                else:
                    logger.debug(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Base LLM prompt template '{template_output_filepath}' already exists.")
        except Exception as e_path_setup:
            logger.error(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Error in pre-processing for saving prompt template: {e_path_setup}")
        # --- END LOGIC FOR SAVING TEMPLATE ONCE ---

        for i in range(0, len(candidate_items), chunk_size):
            if chunks_processed_count >= max_chunks:
                logger.warning(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Reached max_chunks limit ({max_chunks}). Processing {chunks_processed_count * chunk_size} candidates out of {len(candidate_items)}.")
                break
            
            current_chunk_candidate_items = candidate_items[i : i + chunk_size]
            if not current_chunk_candidate_items:
                break # Should not happen if loop condition is correct

            chunks_processed_count += 1
            chunk_file_identifier_prefix = f"{file_identifier_prefix}_chunk_{chunks_processed_count}"
            
            logger.info(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Processing chunk {chunks_processed_count}/{max_chunks if max_chunks > 0 else 'unlimited'} with {len(current_chunk_candidate_items)} items.")

            # --- Per-Chunk LLM Call Logic (adapted from original single call logic) ---
            final_processed_outputs_for_chunk: List[Optional[PhoneNumberLLMOutput]] = [None] * len(current_chunk_candidate_items)
            items_needing_retry_for_chunk: List[Tuple[int, Dict[str, Any]]] = []
            raw_llm_response_str_initial_for_chunk: Optional[str] = None
            
            try:
                prompt_template_chunk = self._load_prompt_template(prompt_template_path)
                candidate_items_json_str_chunk = json.dumps(current_chunk_candidate_items, indent=2)
                formatted_prompt_chunk = prompt_template_chunk.replace(
                    "[Insert JSON list of (candidate_number, source_url, snippet) objects here]",
                    candidate_items_json_str_chunk
                )
            except Exception as e:
                logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Failed to load/format prompt for chunk: {e}")
                for k, item_detail_chunk in enumerate(current_chunk_candidate_items):
                    final_processed_outputs_for_chunk[k] = self._create_error_llm_item(item_detail_chunk, "Error_PromptLoading", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                overall_processed_outputs.extend([item for item in final_processed_outputs_for_chunk if item is not None])
                overall_raw_responses.append(json.dumps({"error": f"Error loading prompt for chunk: {str(e)}"}))
                continue # Next chunk

            generation_config_chunk = types.GenerateContentConfig(
                # candidate_count=1, # Often default in new SDK
                max_output_tokens=self.config.llm_max_tokens, 
                temperature=self.config.llm_temperature,
                response_mime_type="application/json", # Forcing JSON output
                response_schema=MinimalExtractionOutput # Pydantic schema for parsing
            )

            try:
                response_chunk = self._generate_content_with_retry(formatted_prompt_chunk, generation_config_chunk, chunk_file_identifier_prefix, triggering_input_row_id, triggering_company_name)
                raw_llm_response_str_initial_for_chunk = response_chunk.text # Still useful for logging raw output

                # Accessing token usage - this needs verification with the new SDK's response structure
                token_stats_chunk: Dict[str, int] = {} # Initialize
                token_stats_found = False
                if hasattr(response_chunk, 'usage_metadata') and response_chunk.usage_metadata: # Old SDK style
                    token_stats_chunk = {
                        "prompt_tokens": response_chunk.usage_metadata.prompt_token_count,
                        "completion_tokens": response_chunk.usage_metadata.candidates_token_count, # or equivalent
                        "total_tokens": response_chunk.usage_metadata.total_token_count
                    }
                    token_stats_found = True
                # elif hasattr(response_chunk, 'token_count_data'): # Hypothetical new SDK structure
                    # token_stats_chunk = {
                    #     "prompt_tokens": response_chunk.token_count_data.prompt,
                    #     "completion_tokens": response_chunk.token_count_data.completion,
                    #     "total_tokens": response_chunk.token_count_data.total
                    # }
                    # token_stats_found = True
                
                if token_stats_found:
                    for key_token in accumulated_token_stats: accumulated_token_stats[key_token] += token_stats_chunk.get(key_token, 0)
                    logger.info(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call usage: {token_stats_chunk}")
                else:
                    logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Gemini API usage metadata not found or structure unknown.")

                # Attempt to use SDK-parsed JSON if response_schema was provided
                parsed_llm_result = None
                if hasattr(response_chunk, 'candidates') and response_chunk.candidates:
                    # The new SDK might place parsed content within candidates.
                    # Example from migration guide: response.candidates[0].parts[0].function_call
                    # For JSON schema, it might be response.candidates[0].content.parsed (if using pydantic schema)
                    # This needs verification.
                    # For now, let's assume if response_schema is used, the SDK might provide a top-level 'parsed' attribute
                    # or it's within the first candidate.
                    if hasattr(response_chunk, 'parsed') and isinstance(response_chunk.parsed, MinimalExtractionOutput):
                         parsed_llm_result = response_chunk.parsed
                    elif hasattr(response_chunk.candidates[0], 'content') and \
                         hasattr(response_chunk.candidates[0].content, 'parts') and \
                         response_chunk.candidates[0].content.parts and \
                         hasattr(response_chunk.candidates[0].content.parts[0], 'json') and \
                         isinstance(response_chunk.candidates[0].content.parts[0].json, MinimalExtractionOutput) : # Hypothetical access
                         parsed_llm_result = response_chunk.candidates[0].content.parts[0].json


                if parsed_llm_result:
                    llm_result_chunk = parsed_llm_result
                    validated_numbers_chunk = llm_result_chunk.extracted_numbers

                    if len(validated_numbers_chunk) != len(current_chunk_candidate_items):
                        logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Mismatch in item count (SDK parsed). Input: {len(current_chunk_candidate_items)}, Output: {len(validated_numbers_chunk)}. Marking all as error.")
                        for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                            final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, "Error_LLMItemCountMismatch_SDKParsed", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                    else:
                        for k, input_item_detail_chunk in enumerate(current_chunk_candidate_items):
                            llm_output_item_chunk = validated_numbers_chunk[k]
                            if llm_output_item_chunk.number == input_item_detail_chunk['number']:
                                final_processed_outputs_for_chunk[k] = self._process_successful_llm_item(llm_output_item_chunk, input_item_detail_chunk)
                            else:
                                items_needing_retry_for_chunk.append((k, input_item_detail_chunk))
                elif not hasattr(response_chunk, 'candidates') or not response_chunk.candidates:
                     logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] No candidates in Gemini response for chunk.")
                     for k, item_detail_chunk in enumerate(current_chunk_candidate_items):
                         final_processed_outputs_for_chunk[k] = self._create_error_llm_item(item_detail_chunk, "Error_NoLLMCandidates", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                elif raw_llm_response_str_initial_for_chunk: # Fallback to manual JSON extraction
                    json_candidate_str_chunk = self._extract_json_from_text(raw_llm_response_str_initial_for_chunk)
                    if json_candidate_str_chunk:
                        try:
                            parsed_json_object_chunk = json.loads(json_candidate_str_chunk)
                            llm_result_chunk = MinimalExtractionOutput(**parsed_json_object_chunk)
                            validated_numbers_chunk = llm_result_chunk.extracted_numbers

                            if len(validated_numbers_chunk) != len(current_chunk_candidate_items):
                                logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Mismatch in item count (manual parse). Input: {len(current_chunk_candidate_items)}, Output: {len(validated_numbers_chunk)}. Marking all as error.")
                                for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                                    final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, "Error_LLMItemCountMismatch_ManualParse", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                            else:
                                for k, input_item_detail_chunk in enumerate(current_chunk_candidate_items):
                                    llm_output_item_chunk = validated_numbers_chunk[k]
                                    if llm_output_item_chunk.number == input_item_detail_chunk['number']:
                                        final_processed_outputs_for_chunk[k] = self._process_successful_llm_item(llm_output_item_chunk, input_item_detail_chunk)
                                    else:
                                        items_needing_retry_for_chunk.append((k, input_item_detail_chunk))
                        except (json.JSONDecodeError, PydanticValidationError) as e_parse_validate:
                            logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Failed to parse/validate JSON (manual): {e_parse_validate}. Raw: '{raw_llm_response_str_initial_for_chunk[:200]}...'")
                            for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                                final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, f"Error_ChunkJsonParseValidate_{type(e_parse_validate).__name__}", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                    else: # No JSON block in chunk
                        logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Could not extract JSON block from raw text.")
                        for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                             final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, "Error_ChunkNoJsonBlock", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                else: # Empty response for chunk
                    logger.warning(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Response text is empty and no candidates/parsed data.")
                    for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                        final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, "Error_ChunkEmptyResponseOrNoCandidates", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)

            except google_exceptions.GoogleAPIError as e_api: # Assuming exceptions remain the same
                logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Gemini API error: {e_api}")
                for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                    final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, f"Error_ChunkApiError_{type(e_api).__name__}", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                raw_llm_response_str_initial_for_chunk = json.dumps({"error": f"Chunk Gemini API error: {str(e_api)}", "type": type(e_api).__name__})
            except Exception as e_gen:
                logger.error(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] LLM chunk call: Unexpected error: {e_gen}", exc_info=True)
                for k_err, item_detail_chunk_err in enumerate(current_chunk_candidate_items):
                    final_processed_outputs_for_chunk[k_err] = self._create_error_llm_item(item_detail_chunk_err, f"Error_ChunkUnexpected_{type(e_gen).__name__}", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                raw_llm_response_str_initial_for_chunk = json.dumps({"error": f"Chunk unexpected error: {str(e_gen)}", "type": type(e_gen).__name__})
            
            # --- Mismatch Retry Loop for the Current Chunk ---
            current_chunk_retry_attempt = 0
            raw_llm_response_str_retry_for_chunk: Optional[str] = None

            while items_needing_retry_for_chunk and current_chunk_retry_attempt < self.config.llm_max_retries_on_number_mismatch:
                current_chunk_retry_attempt += 1
                logger.info(f"[{chunk_file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Attempting LLM chunk retry pass #{current_chunk_retry_attempt} for {len(items_needing_retry_for_chunk)} items.")
                
                inputs_for_this_chunk_retry_pass = [item_tuple[1] for item_tuple in items_needing_retry_for_chunk]
                original_indices_within_chunk_for_this_pass = [item_tuple[0] for item_tuple in items_needing_retry_for_chunk]

                try:
                    prompt_template_chunk_retry = self._load_prompt_template(prompt_template_path)
                    candidate_items_json_str_chunk_retry = json.dumps(inputs_for_this_chunk_retry_pass, indent=2)
                    formatted_prompt_chunk_retry = prompt_template_chunk_retry.replace(
                        "[Insert JSON list of (candidate_number, source_url, snippet) objects here]",
                        candidate_items_json_str_chunk_retry
                    )
                except Exception as e_prompt_retry:
                    logger.error(f"[{chunk_file_identifier_prefix}] Failed to load/format prompt for chunk retry #{current_chunk_retry_attempt}: {e_prompt_retry}")
                    for original_idx_in_chunk, item_detail_retry_err in items_needing_retry_for_chunk:
                        if final_processed_outputs_for_chunk[original_idx_in_chunk] is None:
                            final_processed_outputs_for_chunk[original_idx_in_chunk] = self._create_error_llm_item(item_detail_retry_err, f"Error_ChunkRetryPromptLoading_Pass{current_chunk_retry_attempt}", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)
                    items_needing_retry_for_chunk.clear()
                    break 

                generation_config_chunk_retry = types.GenerateContentConfig(
                    # candidate_count=1,
                    max_output_tokens=self.config.llm_max_tokens, 
                    temperature=self.config.llm_temperature,
                    response_mime_type="application/json",
                    response_schema=MinimalExtractionOutput
                )
                
                try:
                    response_chunk_retry = self._generate_content_with_retry(formatted_prompt_chunk_retry, generation_config_chunk_retry, f"{chunk_file_identifier_prefix}_retry{current_chunk_retry_attempt}", triggering_input_row_id, triggering_company_name)
                    raw_llm_response_str_retry_for_chunk = response_chunk_retry.text
                    
                    # Token counting for retry - needs verification
                    token_stats_chunk_retry: Dict[str, int] = {} # Initialize
                    token_stats_found_retry = False
                    if hasattr(response_chunk_retry, 'usage_metadata') and response_chunk_retry.usage_metadata:
                        token_stats_chunk_retry = { "prompt_tokens": response_chunk_retry.usage_metadata.prompt_token_count, "completion_tokens": response_chunk_retry.usage_metadata.candidates_token_count, "total_tokens": response_chunk_retry.usage_metadata.total_token_count }
                        token_stats_found_retry = True
                    
                    if token_stats_found_retry:
                        for key_token_r in accumulated_token_stats: accumulated_token_stats[key_token_r] += token_stats_chunk_retry.get(key_token_r, 0)
                        logger.info(f"[{chunk_file_identifier_prefix}] LLM chunk retry #{current_chunk_retry_attempt} usage: {token_stats_chunk_retry}")
                    else:
                        logger.warning(f"[{chunk_file_identifier_prefix}] LLM chunk retry #{current_chunk_retry_attempt} usage metadata not found.")


                    still_mismatched_after_this_chunk_retry: List[Tuple[int, Dict[str, Any]]] = []
                    parsed_llm_result_retry = None
                    if hasattr(response_chunk_retry, 'parsed') and isinstance(response_chunk_retry.parsed, MinimalExtractionOutput):
                        parsed_llm_result_retry = response_chunk_retry.parsed
                    elif hasattr(response_chunk_retry, 'candidates') and response_chunk_retry.candidates and \
                         hasattr(response_chunk_retry.candidates[0], 'content') and \
                         hasattr(response_chunk_retry.candidates[0].content, 'parts') and \
                         response_chunk_retry.candidates[0].content.parts and \
                         hasattr(response_chunk_retry.candidates[0].content.parts[0], 'json') and \
                         isinstance(response_chunk_retry.candidates[0].content.parts[0].json, MinimalExtractionOutput):
                         parsed_llm_result_retry = response_chunk_retry.candidates[0].content.parts[0].json

                    if parsed_llm_result_retry:
                        llm_result_chunk_retry = parsed_llm_result_retry
                        validated_numbers_chunk_retry = llm_result_chunk_retry.extracted_numbers

                        if len(validated_numbers_chunk_retry) != len(inputs_for_this_chunk_retry_pass):
                            still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                        else:
                            for j_retry, retried_input_item_detail_chunk in enumerate(inputs_for_this_chunk_retry_pass):
                                original_idx_within_chunk = original_indices_within_chunk_for_this_pass[j_retry]
                                retried_llm_output_item_chunk = validated_numbers_chunk_retry[j_retry]
                                if retried_llm_output_item_chunk.number == retried_input_item_detail_chunk['number']:
                                    final_processed_outputs_for_chunk[original_idx_within_chunk] = self._process_successful_llm_item(retried_llm_output_item_chunk, retried_input_item_detail_chunk)
                                else:
                                    still_mismatched_after_this_chunk_retry.append((original_idx_within_chunk, retried_input_item_detail_chunk))
                    elif not hasattr(response_chunk_retry, 'candidates') or not response_chunk_retry.candidates:
                        still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                    elif raw_llm_response_str_retry_for_chunk: # Fallback to manual parse for retry
                        json_candidate_str_chunk_retry = self._extract_json_from_text(raw_llm_response_str_retry_for_chunk)
                        if json_candidate_str_chunk_retry:
                            try:
                                parsed_json_object_chunk_retry = json.loads(json_candidate_str_chunk_retry)
                                llm_result_chunk_retry = MinimalExtractionOutput(**parsed_json_object_chunk_retry)
                                validated_numbers_chunk_retry = llm_result_chunk_retry.extracted_numbers

                                if len(validated_numbers_chunk_retry) != len(inputs_for_this_chunk_retry_pass):
                                    still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                                else:
                                    for j_retry, retried_input_item_detail_chunk in enumerate(inputs_for_this_chunk_retry_pass):
                                        original_idx_within_chunk = original_indices_within_chunk_for_this_pass[j_retry]
                                        retried_llm_output_item_chunk = validated_numbers_chunk_retry[j_retry]
                                        if retried_llm_output_item_chunk.number == retried_input_item_detail_chunk['number']:
                                            final_processed_outputs_for_chunk[original_idx_within_chunk] = self._process_successful_llm_item(retried_llm_output_item_chunk, retried_input_item_detail_chunk)
                                        else:
                                            still_mismatched_after_this_chunk_retry.append((original_idx_within_chunk, retried_input_item_detail_chunk))
                            except (json.JSONDecodeError, PydanticValidationError):
                                still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                        else: 
                            still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                    else: 
                        still_mismatched_after_this_chunk_retry.extend(items_needing_retry_for_chunk)
                    items_needing_retry_for_chunk = still_mismatched_after_this_chunk_retry
                except google_exceptions.GoogleAPIError as e_api_retry:
                     logger.error(f"[{chunk_file_identifier_prefix}] Retry #{current_chunk_retry_attempt}: Gemini API error: {e_api_retry}")
                except Exception as e_gen_retry:
                     logger.error(f"[{chunk_file_identifier_prefix}] Retry #{current_chunk_retry_attempt}: Unexpected error: {e_gen_retry}", exc_info=True)
            
            if items_needing_retry_for_chunk: # Persistently mismatched in chunk
                for original_idx_in_chunk_persist, item_detail_persist_error_chunk in items_needing_retry_for_chunk:
                    if final_processed_outputs_for_chunk[original_idx_in_chunk_persist] is None:
                        final_processed_outputs_for_chunk[original_idx_in_chunk_persist] = self._create_error_llm_item(item_detail_persist_error_chunk, "Error_PersistentMismatchAfterRetries", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)

            # Fill any remaining None slots in this chunk's outputs with errors
            for k_final_check, output_item_chunk_final in enumerate(final_processed_outputs_for_chunk):
                if output_item_chunk_final is None:
                    final_processed_outputs_for_chunk[k_final_check] = self._create_error_llm_item(current_chunk_candidate_items[k_final_check], "Error_NotProcessedInChunk", file_identifier_prefix=chunk_file_identifier_prefix, triggering_input_row_id=triggering_input_row_id, triggering_company_name=triggering_company_name)

            overall_processed_outputs.extend([item for item in final_processed_outputs_for_chunk if item is not None])
            if raw_llm_response_str_initial_for_chunk: 
                 overall_raw_responses.append(raw_llm_response_str_initial_for_chunk)
            elif raw_llm_response_str_retry_for_chunk: 
                 overall_raw_responses.append(raw_llm_response_str_retry_for_chunk)
            else: 
                 overall_raw_responses.append(json.dumps({"error": f"LLM response for chunk {chunks_processed_count} not captured."}))


        final_combined_raw_response_str = "\n\n---CHUNK_SEPARATOR---\n\n".join(overall_raw_responses) if overall_raw_responses else json.dumps({"error": "No LLM responses captured."})
        
        successful_items_count = sum(1 for item in overall_processed_outputs if item and not item.type.startswith("Error_"))
        error_items_count = len(overall_processed_outputs) - successful_items_count
        logger.info(f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] Overall LLM extraction summary: {successful_items_count} successful, {error_items_count} errors out of {len(candidate_items)} candidates processed over {chunks_processed_count} chunks.")

        return overall_processed_outputs, final_combined_raw_response_str, accumulated_token_stats