"""
Client for interacting with the Google Gemini API via the GA `google-genai` SDK.
"""
import json
import logging
from typing import Optional, Any, Union, Iterable, Type

from google import genai
from google.genai import types as genai_types
from google.api_core import exceptions as google_api_core_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from pydantic import BaseModel

from .backpressure import record_provider_error, record_provider_success, sleep_if_provider_cooling_down
from .common import CompatLLMResponse, StructuredLLMResult, UsageMetadata
from ..core.config import AppConfig
from ..utils.llm_processing_helpers import adapt_schema_for_gemini, extract_json_from_text

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
    Client for direct interactions with the Gemini API using `google-genai`.
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

        try:
            self.client = genai.Client(api_key=self.config.gemini_api_key)
            logger.info("GeminiClient initialized successfully with google-genai.")
        except Exception as e:
            logger.error(f"GeminiClient: Failed during client initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Gemini client with API key: {e}") from e

    def _usage_metadata(self, response: Any) -> UsageMetadata:
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usageMetadata", None)
        if usage is None:
            return UsageMetadata()
        return UsageMetadata(
            prompt_token_count=int(getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0) or 0),
            candidates_token_count=int(getattr(usage, "candidates_token_count", 0) or getattr(usage, "candidatesTokenCount", 0) or 0),
            total_token_count=int(getattr(usage, "total_token_count", 0) or getattr(usage, "totalTokenCount", 0) or 0),
        )

    def _config_to_sdk_config(
        self,
        generation_config: Optional[Union[genai_types.GenerateContentConfig, dict]],
        *,
        system_instruction: Optional[str] = None,
    ) -> genai_types.GenerateContentConfig:
        if generation_config is None:
            config_dict = {}
        elif isinstance(generation_config, dict):
            config_dict = {k: v for k, v in generation_config.items() if v is not None}
        elif hasattr(generation_config, "model_dump"):
            config_dict = {
                k: v
                for k, v in generation_config.model_dump(exclude_none=True).items()
                if v is not None
            }
        else:
            config_dict = {
                key: value
                for key, value in vars(generation_config).items()
                if not key.startswith("_") and value is not None
            }
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        return genai_types.GenerateContentConfig(**config_dict)

    @retry(
        stop=stop_after_attempt(3),  # Total 3 attempts: 1 initial + 2 retries
        wait=wait_exponential(multiplier=1, min=2, max=10),  # Waits 2s, then 4s (max wait 10s between retries)
        retry=retry_if_exception_type(RETRYABLE_GEMINI_EXCEPTIONS),
        reraise=True  # If all retries fail, the last exception is reraised.
    )
    def generate_content_with_retry(
        self,
        contents: Union[str, Iterable[Any]],
        generation_config: Any,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        model_name_override: Optional[str] = None,
        system_instruction: Optional[str] = None
    ) -> CompatLLMResponse:
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

        logger.info(f"{log_context} Attempting Gemini API call with client.models.generate_content")
        
        try:
            sleep_if_provider_cooling_down(self.config, "gemini", log_context)
            sdk_config = self._config_to_sdk_config(generation_config, system_instruction=system_instruction)
            response = self.client.models.generate_content(
                model=qualified_model_name,
                contents=contents,
                config=sdk_config,
            )
            raw_text = getattr(response, "text", None) or ""
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(prompt_feedback, "block_reason", None) if prompt_feedback else None
            refusal = getattr(block_reason, "name", None) if block_reason is not None else None
            usage_metadata = self._usage_metadata(response)
            compat_response = CompatLLMResponse(
                text=raw_text,
                usage_metadata=usage_metadata,
                candidates=[{"text": raw_text}] if raw_text else [],
                parsed=getattr(response, "parsed", None),
                prompt_feedback=prompt_feedback,
                status="blocked" if refusal else ("empty" if not raw_text else "success"),
                refusal=refusal,
                model_name=qualified_model_name,
            )
            if compat_response.refusal:
                logger.warning(f"{log_context} Gemini content generation was blocked. Reason: {compat_response.refusal}.")
            if not compat_response.candidates:
                logger.warning(f"{log_context} Gemini API call returned no candidates/text.")
            record_provider_success("gemini")
            logger.info(f"{log_context} Gemini API call successful.")
            return compat_response
        except google_api_core_exceptions.GoogleAPIError as api_error:
            record_provider_error(self.config, "gemini", api_error, log_context)
            logger.error(f"{log_context} Gemini API error: {api_error}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"{log_context} Unexpected error during Gemini API call: {e}", exc_info=True)
            raise

    def generate_structured_output_with_retry(
        self,
        *,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        system_prompt: Optional[str] = None,
        model_name_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> StructuredLLMResult:
        del schema_name  # Gemini does not require a separate schema name string.

        generation_config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=adapt_schema_for_gemini(response_model),
            temperature=self.config.llm_temperature_extraction if temperature is None else temperature,
            max_output_tokens=self.config.llm_max_tokens if max_output_tokens is None else max_output_tokens,
        )
        retry_count = 0

        def _parse_structured_response(resp: CompatLLMResponse) -> tuple[Optional[Any], str, str, Optional[str]]:
            parsed_output_local = None
            provider_error_local = resp.provider_error
            raw_text_local = resp.text or ""
            if isinstance(resp.parsed, response_model):
                parsed_output_local = resp.parsed
            elif raw_text_local:
                try:
                    parsed_output_local = response_model.model_validate_json(raw_text_local)
                except Exception:
                    json_str = extract_json_from_text(raw_text_local)
                    if json_str:
                        try:
                            parsed_output_local = response_model.model_validate(json.loads(json_str))
                        except Exception:
                            parsed_output_local = None
            status_local = resp.status if parsed_output_local is not None else ("parse_error" if raw_text_local else resp.status)
            if parsed_output_local is None and not provider_error_local and raw_text_local:
                provider_error_local = f"Failed to parse Gemini structured output for {response_model.__name__}."
            return parsed_output_local, raw_text_local, status_local, provider_error_local

        response = self.generate_content_with_retry(
            contents=user_prompt,
            generation_config=generation_config,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            model_name_override=model_name_override,
            system_instruction=system_prompt,
        )
        parsed_output, raw_text, status, provider_error = _parse_structured_response(response)

        if parsed_output is None and raw_text and not response.refusal:
            retry_count = 1
            repair_instruction = (
                f"{(system_prompt or '').strip()} "
                "Return one complete JSON object only. Ensure the object is fully closed and every array/string is complete."
            ).strip()
            repair_response = self.generate_content_with_retry(
                contents=user_prompt,
                generation_config=generation_config,
                file_identifier_prefix=f"{file_identifier_prefix}_repair",
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name,
                model_name_override=model_name_override,
                system_instruction=repair_instruction,
            )
            repaired_output, repaired_text, repaired_status, repaired_error = _parse_structured_response(repair_response)
            if repaired_text:
                raw_text = repaired_text
            if repaired_output is not None:
                parsed_output = repaired_output
                status = repaired_status
                provider_error = repaired_error
            elif repaired_error:
                status = repaired_status
                provider_error = repaired_error

        return StructuredLLMResult(
            parsed_output=parsed_output,
            raw_text=raw_text,
            usage=response.usage_metadata.as_dict(),
            status=status,
            refusal=response.refusal,
            provider_error=provider_error,
            retry_count=response.retry_count + retry_count,
            model_name=response.model_name,
        )