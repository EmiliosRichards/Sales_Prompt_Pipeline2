from types import SimpleNamespace

from src.llm_clients.backpressure import compute_retry_sleep, effective_worker_count


def test_effective_worker_count_clamps_to_provider_limit():
    config = SimpleNamespace(
        provider_max_inflight_default=6,
        provider_max_inflight_openai=2,
        provider_max_inflight_gemini=3,
    )

    assert effective_worker_count(config, "openai", 10) == 2
    assert effective_worker_count(config, "gemini", 10) == 3
    assert effective_worker_count(config, "unknown", 10) == 6


def test_compute_retry_sleep_stays_positive_and_grows():
    first = compute_retry_sleep(1, base_seconds=1.0, max_seconds=30.0)
    third = compute_retry_sleep(3, base_seconds=1.0, max_seconds=30.0)

    assert first >= 1.0
    assert third > first
