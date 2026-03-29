import json
import logging
import os
import random
import tempfile
import time
from typing import Any, Dict


logger = logging.getLogger(__name__)

_STATE_DIR = os.path.join(tempfile.gettempdir(), "sales_prompt_pipeline_provider_state")


def provider_max_inflight(config: Any, provider: str) -> int:
    normalized = (provider or "").strip().lower()
    if normalized == "openai":
        return int(getattr(config, "provider_max_inflight_openai", getattr(config, "provider_max_inflight_default", 1)) or 1)
    if normalized == "gemini":
        return int(getattr(config, "provider_max_inflight_gemini", getattr(config, "provider_max_inflight_default", 1)) or 1)
    return int(getattr(config, "provider_max_inflight_default", 1) or 1)


def effective_worker_count(config: Any, provider: str, requested_workers: int) -> int:
    return max(1, min(int(requested_workers or 1), provider_max_inflight(config, provider)))


def sleep_if_provider_cooling_down(config: Any, provider: str, log_context: str) -> None:
    state = _read_state(provider)
    cooldown_until = float(state.get("cooldown_until", 0) or 0)
    remaining = cooldown_until - time.time()
    if remaining > 0:
        logger.warning(f"{log_context} Provider cooldown active for {remaining:.1f}s; delaying request.")
        time.sleep(remaining)


def record_provider_success(provider: str) -> None:
    state = _read_state(provider)
    if state.get("burst_count") or state.get("cooldown_until"):
        state["burst_count"] = 0
        state["cooldown_until"] = 0
        state["last_error_type"] = None
        _write_state(provider, state)


def record_provider_error(config: Any, provider: str, exc: Exception, log_context: str) -> None:
    if not _should_trigger_backpressure(exc):
        return
    state = _read_state(provider)
    burst_count = int(state.get("burst_count", 0) or 0) + 1
    threshold = int(getattr(config, "provider_backpressure_error_burst_threshold", 3) or 3)
    state["burst_count"] = burst_count
    state["last_error_type"] = type(exc).__name__
    if burst_count >= threshold:
        cooldown_seconds = int(getattr(config, "provider_backpressure_cooldown_seconds", 30) or 30)
        state["cooldown_until"] = time.time() + cooldown_seconds
        state["burst_count"] = 0
        logger.warning(
            f"{log_context} Triggered provider cooldown for {cooldown_seconds}s after repeated {type(exc).__name__} errors."
        )
    _write_state(provider, state)


def compute_retry_sleep(attempt: int, *, base_seconds: float = 1.0, max_seconds: float = 30.0) -> float:
    capped = min(max_seconds, base_seconds * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, min(1.0, capped * 0.25))
    return capped + jitter


def _should_trigger_backpressure(exc: Exception) -> bool:
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "ratelimit" in exc_name
        or "resourceexhausted" in exc_name
        or "serviceunavailable" in exc_name
        or "429" in message
        or "503" in message
    )


def _state_path(provider: str) -> str:
    safe_provider = (provider or "unknown").strip().lower() or "unknown"
    os.makedirs(_STATE_DIR, exist_ok=True)
    return os.path.join(_STATE_DIR, f"{safe_provider}.json")


def _read_state(provider: str) -> Dict[str, Any]:
    path = _state_path(provider)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _write_state(provider: str, state: Dict[str, Any]) -> None:
    path = _state_path(provider)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp_path, path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
