"""
Compatibility-focused OpenAI client for the full pipeline.

Design goals:
- preserve the existing `generate_content_with_retry(...)` shape used by the
  Gemini-backed task helpers
- use OpenAI Responses API under the hood
- use explicit strict JSON-schema mode for all structured task calls
"""
import hashlib
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Type, Union

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

from .backpressure import compute_retry_sleep, record_provider_error, record_provider_success, sleep_if_provider_cooling_down
from .common import CompatLLMResponse, StructuredLLMResult, UsageMetadata
from ..core.config import AppConfig

logger = logging.getLogger(__name__)


RETRYABLE_OPENAI_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    APIError,
)

def _adapt_schema_for_openai(pydantic_model_cls: Type[BaseModel]) -> Dict[str, Any]:
    model_schema = pydantic_model_cls.model_json_schema()
    defs = model_schema.get("$defs", {}) if isinstance(model_schema, dict) else {}

    def _resolve_ref(ref: str) -> Any:
        if not ref.startswith("#/$defs/"):
            return None
        ref_name = ref.split("/")[-1]
        return defs.get(ref_name)

    def _transform(node: Any) -> Any:
        if isinstance(node, list):
            return [_transform(item) for item in node]

        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str):
            resolved = _resolve_ref(ref)
            if resolved is not None:
                return _transform(resolved)
            return {}

        cleaned: Dict[str, Any] = {}
        for key, value in node.items():
            if key in {"title", "default", "$defs"}:
                continue
            cleaned[key] = _transform(value)

        any_of = cleaned.get("anyOf")
        if isinstance(any_of, list):
            non_null = [item for item in any_of if not (isinstance(item, dict) and item.get("type") == "null")]
            has_null = len(non_null) != len(any_of)
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                replacement = dict(non_null[0])
                if "description" in cleaned and "description" not in replacement:
                    replacement["description"] = cleaned["description"]
                replacement = _transform(replacement)
                replacement_type = replacement.get("type")
                if has_null:
                    if isinstance(replacement_type, str):
                        replacement["type"] = [replacement_type, "null"]
                    elif isinstance(replacement_type, list) and "null" not in replacement_type:
                        replacement["type"] = [*replacement_type, "null"]
                return replacement
            cleaned["anyOf"] = non_null if not has_null else [*_transform(non_null), {"type": "null"}]

        if cleaned.get("type") == "object":
            cleaned.setdefault("properties", {})
            cleaned["required"] = list(cleaned["properties"].keys())
            cleaned.setdefault("additionalProperties", False)

        return cleaned

    adapted = _transform(model_schema)
    if isinstance(adapted, dict):
        adapted.pop("$defs", None)
    return adapted


class OpenAIClient:
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

    def _usage_metadata(self, response: Any) -> UsageMetadata:
        usage = getattr(response, "usage", None)
        if usage is None:
            return UsageMetadata()
        return UsageMetadata(
            prompt_token_count=int(getattr(usage, "input_tokens", 0) or 0),
            candidates_token_count=int(getattr(usage, "output_tokens", 0) or 0),
            total_token_count=int(getattr(usage, "total_tokens", 0) or 0),
        )

    def _prompt_cache_key(self, *, model_name: str, schema_name: str, system_instruction: str) -> str:
        raw = f"{model_name}|{schema_name}|{system_instruction}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return f"full:v1:{digest}"

    def _extract_text_from_contents(self, contents: Union[str, Iterable[Any]]) -> str:
        if isinstance(contents, str):
            return contents

        text_parts: List[str] = []
        for item in contents or []:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                text_parts.append(str(item))
                continue
            parts = item.get("parts") or []
            for part in parts:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict):
                    text_val = part.get("text")
                    if text_val:
                        text_parts.append(str(text_val))
        return "\n\n".join([p for p in text_parts if p]).strip()

    def _resolve_model_name(
        self,
        *,
        response_model: Optional[Type[BaseModel]],
        model_name_override: Optional[str],
    ) -> str:
        raw_override = (model_name_override or "").strip()
        normalized_override = raw_override.removeprefix("models/").strip()
        if normalized_override and "gemini" not in normalized_override.lower():
            return normalized_override

        if response_model and response_model.__name__ == "GoldenPartnerMatchOutput":
            return (self.config.openai_model_name_sales_insights or self.config.openai_model_name or "").strip()

        return (self.config.openai_model_name or "").strip()

    def _create_once(
        self,
        *,
        model_name: str,
        user_text: str,
        system_instruction: Optional[str],
        response_model: Optional[Type[BaseModel]],
        schema_name: str,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        include_prompt_cache: bool,
        include_prompt_cache_retention: bool,
        service_tier: Optional[str],
    ) -> CompatLLMResponse:
        system_text = (system_instruction or "").strip()

        create_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": [
                {"role": "system", "content": system_text or "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ],
        }

        normalized_tier = self._normalize_service_tier(service_tier)
        if normalized_tier is not None:
            create_kwargs["service_tier"] = normalized_tier

        if self.config.openai_reasoning_effort:
            create_kwargs["reasoning"] = {"effort": self.config.openai_reasoning_effort}

        if response_model is not None:
            create_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": _adapt_schema_for_openai(response_model),
                    "strict": True,
                }
            }

        if include_prompt_cache and self.config.openai_prompt_cache:
            create_kwargs["prompt_cache_key"] = self._prompt_cache_key(
                model_name=model_name,
                schema_name=schema_name,
                system_instruction=system_text,
            )
            if include_prompt_cache_retention and self.config.openai_prompt_cache_retention:
                create_kwargs["prompt_cache_retention"] = self.config.openai_prompt_cache_retention

        log_context = (
            f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, "
            f"Company: {triggering_company_name}, Model: {model_name}, Tier: {normalized_tier or 'default'}]"
        )
        logger.info(f"{log_context} Attempting OpenAI Responses API call.")

        response = self.client.responses.create(**create_kwargs)
        response_status = getattr(response, "status", None) or "completed"
        raw_text = getattr(response, "output_text", None) or ""
        refusal = None
        provider_error = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "refusal":
                refusal = getattr(item, "refusal", None) or getattr(item, "content", None) or "refusal"
                break
        if refusal and not raw_text:
            provider_error = f"Model refusal: {refusal}"
        elif not raw_text:
            provider_error = f"OpenAI response did not include output_text (status={response_status})."

        usage_metadata = self._usage_metadata(response)
        logger.info(
            f"{log_context} OpenAI call successful. Usage: "
            f"prompt={usage_metadata.prompt_token_count}, "
            f"completion={usage_metadata.candidates_token_count}, "
            f"total={usage_metadata.total_token_count}"
        )
        return CompatLLMResponse(
            text=raw_text,
            usage_metadata=usage_metadata,
            candidates=[{"text": raw_text}] if raw_text else [],
            status="refusal" if refusal else response_status,
            refusal=refusal,
            provider_error=provider_error,
            model_name=model_name,
        )

    def generate_content_with_retry(
        self,
        contents: Union[str, Iterable[Any]],
        generation_config: Any,
        file_identifier_prefix: str,
        triggering_input_row_id: Any,
        triggering_company_name: str,
        model_name_override: Optional[str] = None,
        system_instruction: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        schema_name: Optional[str] = None,
    ) -> CompatLLMResponse:
        del generation_config  # Existing task code still builds this for Gemini; OpenAI path does not use it directly.

        user_text = self._extract_text_from_contents(contents)
        model_name = self._resolve_model_name(
            response_model=response_model,
            model_name_override=model_name_override,
        )
        if not model_name:
            raise ValueError("OpenAI model name must be configured.")
        effective_schema_name = schema_name or (response_model.__name__ if response_model else "text_response")

        preferred_tier = self._normalize_service_tier(self.config.openai_service_tier)
        max_attempts = max(1, int(self.config.openai_flex_max_retries or 1))
        include_prompt_cache = True
        include_retention = True
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                sleep_if_provider_cooling_down(
                    self.config,
                    "openai",
                    f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Model: {model_name}]",
                )
                response = self._create_once(
                    model_name=model_name,
                    user_text=user_text,
                    system_instruction=system_instruction,
                    response_model=response_model,
                    schema_name=effective_schema_name,
                    file_identifier_prefix=file_identifier_prefix,
                    triggering_input_row_id=triggering_input_row_id,
                    triggering_company_name=triggering_company_name,
                    include_prompt_cache=include_prompt_cache,
                    include_prompt_cache_retention=include_retention,
                    service_tier=preferred_tier,
                )
                record_provider_success("openai")
                response.retry_count = attempt - 1
                return response
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

        if last_exc is not None and preferred_tier == "flex" and self.config.openai_flex_fallback_to_auto:
            logger.warning(
                f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}] "
                "OpenAI flex tier exhausted; retrying once with auto tier."
            )
            sleep_if_provider_cooling_down(
                self.config,
                "openai",
                f"[{file_identifier_prefix}, RowID: {triggering_input_row_id}, Company: {triggering_company_name}, Model: {model_name}]",
            )
            response = self._create_once(
                model_name=model_name,
                user_text=user_text,
                system_instruction=system_instruction,
                response_model=response_model,
                schema_name=effective_schema_name,
                file_identifier_prefix=file_identifier_prefix,
                triggering_input_row_id=triggering_input_row_id,
                triggering_company_name=triggering_company_name,
                include_prompt_cache=include_prompt_cache,
                include_prompt_cache_retention=include_retention,
                service_tier="auto",
            )
            record_provider_success("openai")
            response.retry_count = max_attempts
            return response

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
    ) -> StructuredLLMResult:
        del temperature
        del max_output_tokens
        response = self.generate_content_with_retry(
            contents=user_prompt,
            generation_config=None,
            file_identifier_prefix=file_identifier_prefix,
            triggering_input_row_id=triggering_input_row_id,
            triggering_company_name=triggering_company_name,
            model_name_override=model_name_override,
            system_instruction=system_prompt,
            response_model=response_model,
            schema_name=schema_name,
        )
        parsed_output = None
        provider_error = response.provider_error
        if response.text:
            try:
                parsed_output = response_model.model_validate_json(response.text)
            except Exception as exc:
                provider_error = response.provider_error or f"Failed to parse OpenAI structured output for {response_model.__name__}: {exc}"
        if parsed_output is None and not provider_error:
            provider_error = f"Failed to parse OpenAI structured output for {response_model.__name__}."
        return StructuredLLMResult(
            parsed_output=parsed_output,
            raw_text=response.text,
            usage=response.usage_metadata.as_dict(),
            status=response.status,
            refusal=response.refusal,
            provider_error=provider_error,
            retry_count=response.retry_count,
            model_name=response.model_name,
        )
