"""
Client for interacting with the Google Gemini API.

This module provides the `GeminiClient` class, which encapsulates the logic for
making calls to the Google Gemini large language models (LLMs). It handles
API key configuration, model instantiation, and robust content generation
with built-in retry mechanisms for common transient API errors.

Key functionalities include:
- Initialization with an API key and application configuration.
- Generation of content using specified models, prompts, and generation parameters.
- Automatic retries for retryable API exceptions.
- Support for overriding default model names and providing system instructions.
"""
import logging
from typing import Optional, Any, Union, Iterable

from google.generativeai.client import configure # Specific import for configure
from google.generativeai.generative_models import GenerativeModel # Specific import for GenerativeModel
from google.generativeai import types as genai_types # Explicitly alias types
from google.api_core import exceptions as google_api_core_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# AppConfig is located in src/core/config.py, so use a relative import
from ..core.config import AppConfig

logger = logging.getLogger(__name__)

# Define retryable exceptions, consistent with other Gemini interactions in the project
RETRYABLE_GEMINI_EXCEPTIONS = (
    google_api_core_exceptions.DeadlineExceeded,    # e.g., 504 Gateway Timeout
    google_api_core_exceptions.ServiceUnavailable,  # e.g., 503 Service Unavailable
    google_api_core_exceptions.ResourceExhausted,   # e.g., 429 Too Many Requests (Rate Limits)
    google_api_core_exceptions.InternalServerError, # e.g., 500 Internal Server Error
    google_api_core_exceptions.Aborted,             # Often due to concurrency or transient issues
    # google_api_core_exceptions.Unavailable was previously commented out, maintaining that.
)


class GeminiClient:
    """
    Client for direct interactions with the Google Gemini API using the google-genai SDK.
    Handles API key configuration, client initialization, and content generation with retries.
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the GeminiClient.

        Args:
            config (AppConfig): The application configuration object, which includes
                                the Gemini API key and default model name.

        Raises:
            ValueError: If the Gemini API key is not found in the configuration.
            RuntimeError: If configuration or client initialization fails.
        """
        self.config = config
        if not self.config.gemini_api_key:
            logger.error("GeminiClient: GEMINI_API_KEY not provided in AppConfig.")
            raise ValueError("GEMINI_API_KEY not found in AppConfig for GeminiClient.")

        # Configure the google-genai SDK with the API key.
        # This is typically done once per application lifecycle.
        try:
            configure(api_key=self.config.gemini_api_key) # Direct call
            logger.info(f"GeminiClient: configure called successfully. Default model from config: {self.config.llm_model_name}")
        except Exception as e: # Catch a broader exception if configure itself fails
            logger.error(f"GeminiClient: Failed during configure: {e}", exc_info=True)
            raise RuntimeError(f"Failed to configure Gemini client with API key: {e}") from e

    @retry(
        stop=stop_after_attempt(3),  # Total 3 attempts: 1 initial + 2 retries
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Waits 2s, then 4s (max wait 10s between retries)
        retry=retry_if_exception_type(RETRYABLE_GEMINI_EXCEPTIONS),
        reraise=True  # If all retries fail, the last exception is reraised.
    )
    def generate_content_with_retry(
        self,
        contents: Union[str, Iterable[genai_types.ContentDict]], # Use aliased types
        generation_config: genai_types.GenerationConfig, # Use aliased types
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        model_name_override: Optional[str] = None,
        system_instruction: Optional[str] = None  # New parameter
    ) -> genai_types.GenerateContentResponse: # Use aliased types
        """
        Generates content using the Gemini API with retry logic.

        This method calls the Gemini API's `generate_content` endpoint, handling
        model instantiation and incorporating retry mechanisms for transient errors
        as defined in `RETRYABLE_GEMINI_EXCEPTIONS`.

        Args:
            contents (Union[str, Iterable[genai_types.ContentDict]]): The prompt or content to send to the model.
                This can be a simple string for a basic prompt, or an iterable of
                `genai_types.ContentDict` objects for more complex inputs, such as
                multi-turn conversations or multi-modal content.
                Example of `ContentDict`: `{"parts": [{"text": "Your prompt here"}], "role": "user"}`
            generation_config (genai_types.GenerationConfig): Configuration for the generation request.
                Key parameters typically include:
                - `temperature` (Optional[float]): Controls randomness (e.g., 0.0 to 1.0).
                - `max_output_tokens` (Optional[int]): Maximum number of tokens to generate.
                - `top_p` (Optional[float]): Nucleus sampling parameter.
                - `top_k` (Optional[int]): Top-k sampling parameter.
                - `candidate_count` (Optional[int]): Number of generated response candidates to return.
                - `stop_sequences` (Optional[Iterable[str]]): Sequences where the API will stop generating.
                - `response_mime_type` (Optional[str]): Output format (e.g., "text/plain", "application/json").
                - `response_schema` (Optional[genai_types.Schema]): Schema for structured JSON output if
                                                                  `response_mime_type` is "application/json".
            file_identifier_prefix (str): A string prefix for contextual logging (e.g., URL or process name).
            triggering_input_row_id (Any): An identifier for the input data row that triggered this call, for logging.
            triggering_company_name (str): The company name associated with the input, for logging.
            model_name_override (Optional[str]): The name of the Gemini model to use (e.g., "gemini-1.5-pro-latest",
                                               "gemini-1.0-pro"). If provided, this overrides the default
                                               model from `AppConfig`. The "models/" prefix will be
                                               added automatically if not present.
            system_instruction (Optional[str]): A system-level instruction for the model, guiding its behavior.
                                                This is supported by newer Gemini models.

        Returns:
            genai_types.GenerateContentResponse: The raw response object from the Gemini API,
                containing the generated content, prompt feedback, and other metadata.

        Raises:
            google_api_core_exceptions.GoogleAPIError: If the API call fails after all retries
                for a retryable error, or immediately for a non-retryable API error.
            ValueError: If no LLM model name is available (neither in `AppConfig` nor as an override).
            Exception: For other unexpected errors during the API call.
        """
        effective_model_name = model_name_override if model_name_override else self.config.llm_model_name
        if not effective_model_name:
            log_err_context = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}]"
            logger.error(f"{log_err_context} No LLM model name configured in AppConfig or provided via model_name_override.")
            raise ValueError("LLM model name must be configured or provided as an override.")

        # The client.models.generate_content() expects model names like "models/gemini-pro".
        # Ensure the "models/" prefix is present.
        if not effective_model_name.startswith("models/"):
            qualified_model_name = f"models/{effective_model_name}"
        else:
            qualified_model_name = effective_model_name

        log_context = f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Model: {qualified_model_name}]"

        logger.info(f"{log_context} Attempting Gemini API call with GenerativeModel('{qualified_model_name}').generate_content")
        
        try:
            # Instantiate the model directly
            model = GenerativeModel(
                model_name=qualified_model_name,
                system_instruction=system_instruction # Pass system_instruction
            )
            response = model.generate_content(
                contents=contents,
                generation_config=generation_config
                # safety_settings can be added here if needed, e.g., from AppConfig
            )
            
            # Log if the content was blocked or if there are no candidates.
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                logger.warning(f"{log_context} Gemini content generation was blocked. Reason: {response.prompt_feedback.block_reason.name}. This is typically not retried by network-level retries.")
            
            if not response.candidates:
                 logger.warning(f"{log_context} Gemini API call returned no candidates. This might be due to safety filters or other reasons. Review response.prompt_feedback if available. Full response parts: {len(response.parts) if response.parts else 'N/A'}")

            logger.info(f"{log_context} Gemini API call successful.")
            return response
        except google_api_core_exceptions.GoogleAPIError as api_error:
            logger.error(f"{log_context} Gemini API error: {api_error}", exc_info=True)
            raise # Handled by tenacity for retries or reraised if non-retryable/exhausted.
        except Exception as e:
            logger.error(f"{log_context} Unexpected error during Gemini API call: {e}", exc_info=True)
            raise