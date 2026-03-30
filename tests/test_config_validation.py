from src.core.config import AppConfig


def test_invalid_full_provider_raises(monkeypatch):
    monkeypatch.setenv("FULL_LLM_PROVIDER", "bogus")
    monkeypatch.setenv("PHONE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    try:
        AppConfig()
        assert False, "Expected invalid FULL_LLM_PROVIDER to raise ValueError."
    except ValueError as exc:
        assert "FULL_LLM_PROVIDER" in str(exc)


def test_invalid_openai_service_tier_raises(monkeypatch):
    monkeypatch.setenv("FULL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("PHONE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_SERVICE_TIER", "bad-tier")

    try:
        AppConfig()
        assert False, "Expected invalid OPENAI_SERVICE_TIER to raise ValueError."
    except ValueError as exc:
        assert "OPENAI_SERVICE_TIER" in str(exc)


def test_legacy_english_summary_prompt_env_is_upgraded_to_german(monkeypatch):
    monkeypatch.setenv("FULL_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("PHONE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("PROMPT_PATH_WEBSITE_SUMMARIZER", "prompts/website_summarizer_prompt.txt")

    config = AppConfig()

    assert config.PROMPT_PATH_WEBSITE_SUMMARIZER.endswith("prompts\\website_summarizer_prompt_de.txt")
