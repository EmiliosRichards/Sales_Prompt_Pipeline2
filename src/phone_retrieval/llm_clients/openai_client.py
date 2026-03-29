import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from src.core.config import AppConfig
from src.llm_clients.backpressure import compute_retry_sleep, record_provider_error, record_provider_success, sleep_if_provider_cooling_down
from src.phone_retrieval.utils.llm_processing_helpers import adapt_schema_for_openai

logger = logging.getLogger(__name__)


RETRYABLE_OPENAI_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    APIError,
)


@dataclass
class OpenAIStructuredResult:
    parsed_output: Optional[BaseModel]
    raw_text: str
    usage: Dict[str, int]
    status: str = "success"
    refusal: Optional[str] = None
    provider_error: Optional[str] = None


class OpenAIClient:
    """
    Responses API client for structured phone-pipeline LLM tasks.

    This client intentionally avoids web tools because the phone pipeline already
    scrapes websites and only needs classification/ranking over scraped content.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        if not self.config.openai_api_key:
            logger.error("OpenAIClient: OPENAI_API_KEY not provided in AppConfig.")
            raise ValueError("OPENAI_API_KEY not found in AppConfig for OpenAIClient.")

        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            timeout=float(self.config.openai_timeout_seconds),
        )

    def _normalize_service_tier(self, value: Optional[str]) -> Optional[str]:
        v = (value or "").strip().lower()
        return v or None

    def _usage_stats(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {
            "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _prompt_cache_key(self, *, model_name: str, schema_name: str, system_prompt: str) -> str:
        """
        OpenAI currently enforces a 64-char limit for prompt_cache_key.
        Use a short, stable digest-based key so longer model names do not break calls.
        """
        raw = f"{model_name}|{schema_name}|{system_prompt}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return f"phone:v1:{digest}"

    def _create_once(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        service_tier: Optional[str],
        include_prompt_cache_retention: bool,
        include_prompt_cache: bool,
    ) -> OpenAIStructuredResult:
        schema = adapt_schema_for_openai(response_model)
        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }

        normalized_tier = self._normalize_service_tier(service_tier)
        if normalized_tier is not None:
            create_kwargs["service_tier"] = normalized_tier

        if self.config.openai_reasoning_effort:
            create_kwargs["reasoning"] = {"effort": self.config.openai_reasoning_effort}

        if include_prompt_cache and self.config.openai_prompt_cache:
            create_kwargs["prompt_cache_key"] = self._prompt_cache_key(
                model_name=model_name,
                schema_name=schema_name,
                system_prompt=system_prompt,
            )
            if include_prompt_cache_retention and self.config.openai_prompt_cache_retention:
                create_kwargs["prompt_cache_retention"] = self.config.openai_prompt_cache_retention

        log_context = (
            f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, "
            f"Company: {triggering_company_name}, Model: {model_name}, Tier: {normalized_tier or 'default'}]"
        )
        logger.info(f"{log_context} Attempting OpenAI Responses API call.")

        response = self.client.responses.create(**create_kwargs)
        raw_text = getattr(response, "output_text", None) or ""
        usage = self._usage_stats(response)
        response_status = getattr(response, "status", None) or "completed"
        refusal = None
        provider_error = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "refusal":
                refusal = getattr(item, "refusal", None) or getattr(item, "content", None) or "refusal"
                break
        parsed_output = None
        if raw_text:
            parsed_output = response_model.model_validate_json(raw_text)
        elif refusal:
            provider_error = f"Model refusal: {refusal}"
        else:
            provider_error = f"OpenAI response did not include output_text (status={response_status})."
        logger.info(f"{log_context} OpenAI call successful. Usage: {usage}")
        return OpenAIStructuredResult(
            parsed_output=parsed_output,
            raw_text=raw_text,
            usage=usage,
            status="refusal" if refusal else response_status,
            refusal=refusal,
            provider_error=provider_error,
        )

    def _run_with_retries(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        service_tier: Optional[str],
    ) -> OpenAIStructuredResult:
        max_attempts = max(1, int(self.config.openai_flex_max_retries or 1))
        include_retention = True
        include_prompt_cache = True
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                sleep_if_provider_cooling_down(
                    self.config,
                    "openai",
                    f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Model: {model_name}]",
                )
                result = self._create_once(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    schema_name=schema_name,
                    file_identifier_prefix=file_identifier_prefix,
                    triggering_input_row_id=triggering_input_row_id,
                    triggering_company_name=triggering_company_name,
                    service_tier=service_tier,
                    include_prompt_cache_retention=include_retention,
                    include_prompt_cache=include_prompt_cache,
                )
                record_provider_success("openai")
                return result
            except BadRequestError as exc:
                msg = str(exc)
                if include_retention and "prompt_cache_retention" in msg:
                    logger.warning("OpenAI prompt_cache_retention unsupported; retrying without it.")
                    include_retention = False
                    continue
                if include_prompt_cache and "prompt_cache_key" in msg:
                    logger.warning("OpenAI prompt_cache_key rejected; retrying without prompt caching.")
                    include_prompt_cache = False
                    include_retention = False
                    continue
                raise
            except RETRYABLE_OPENAI_EXCEPTIONS as exc:
                last_exc = exc
                record_provider_error(
                    self.config,
                    "openai",
                    exc,
                    f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Model: {model_name}]",
                )
                if attempt >= max_attempts:
                    break
                sleep_seconds = compute_retry_sleep(attempt)
                logger.warning(
                    f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                    f"OpenAI retryable error on attempt {attempt}/{max_attempts}: {exc}. Sleeping {sleep_seconds}s."
                )
                time.sleep(sleep_seconds)

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("OpenAI call failed without a captured exception.")

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
    ) -> OpenAIStructuredResult:
        del temperature
        del max_output_tokens
        model_name = (model_name_override or self.config.openai_model_name or "").strip()
        if not model_name:
            raise ValueError("OpenAI model name must be configured.")

        system_prompt = (
            system_prompt
            or "You are a structured data extraction assistant. Return only valid JSON matching the requested schema."
        )
        preferred_tier = self._normalize_service_tier(self.config.openai_service_tier)

        try:
            return self._run_with_retries(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=response_model,
                schema_name=schema_name,
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name,
                service_tier=preferred_tier,
            )
        except RETRYABLE_OPENAI_EXCEPTIONS:
            if preferred_tier == "flex" and self.config.openai_flex_fallback_to_auto:
                logger.warning(
                    f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                    "OpenAI flex tier exhausted; retrying with auto tier."
                )
                return self._run_with_retries(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    schema_name=schema_name,
                    file_identifier_prefix=file_identifier_prefix,
                    triggering_input_row_id=triggering_input_row_id,
                    triggering_company_name=triggering_company_name,
                    service_tier="auto",
                )
            raise
