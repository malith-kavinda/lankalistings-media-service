"""Choosing an LLM provider, and refusing to choose a fake one in production.

The same builder-table shape as the OCR registry, and for the same reason: function-local imports so
selecting one provider never imports another's dependencies.

The part that is not shared is `guard_offline_provider`. PRD 11.6 forbids falling back to heuristic
advertisement creation in production and says it must be enforced **by configuration rather than
convention, failing at startup rather than at extraction time**. That second half is the important
one. A check at extraction time fails once a real page has already been uploaded, on a worker
thread, where the operator sees an item in `needs_attention` and no explanation. A startup check
fails on deploy, with the reason in the logs, before anything has been promised.

So `fake` and `rule_based` are constructible only when the environment is local or test, or when
`MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS` is set deliberately.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from media_service.config import LLM_PROVIDERS, Settings
from media_service.llm.prompts.registry import PromptTemplate, load_prompt
from media_service.llm.runner import RunnerSettings

if TYPE_CHECKING:
    from media_service.llm.service import LlmExtractionService

# Providers that produce advertisements without a model. Useful, and dangerous in production for
# exactly that reason.
OFFLINE_PROVIDERS: Final = frozenset({"fake", "rule_based"})

PROMPT_FAMILY: Final = "ad_extraction"
DEFAULT_PROMPT_VERSION: Final = "v1"


def guard_offline_provider(settings: Settings) -> None:
    """Refuse a heuristic provider outside local and test (PRD 11.6)."""
    if settings.llm_provider not in OFFLINE_PROVIDERS:
        return
    if settings.allow_fake_providers:
        return
    raise ValueError(
        f"LLM_PROVIDER={settings.llm_provider!r} produces advertisements without a model and is "
        f"refused when MEDIA_SERVICE_ENV={settings.environment!r}. Set a real provider, or set "
        "MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS=true deliberately."
    )


def prompt_for(settings: Settings) -> PromptTemplate:
    return load_prompt(PROMPT_FAMILY, settings.llm_prompt_version or DEFAULT_PROMPT_VERSION)


def runner_settings(settings: Settings) -> RunnerSettings:
    return RunnerSettings(
        max_attempts=settings.llm_max_attempts,
        max_repairs=settings.llm_max_repairs,
        backoff_base_seconds=settings.llm_backoff_base_seconds,
        backoff_max_seconds=settings.llm_backoff_max_seconds,
        total_deadline_seconds=settings.llm_total_deadline_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
    )


def build_openai_compatible(settings: Settings) -> Any:
    from media_service.llm.providers.openai_compatible import OpenAiCompatibleProvider

    return OpenAiCompatibleProvider(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.llm_timeout_seconds,
        structured_mode=settings.llm_structured_mode,
        auth_header=settings.openai_auth_header,
    )


def build_gemini(settings: Settings) -> Any:
    from media_service.llm.providers.gemini import GeminiProvider

    return GeminiProvider(
        base_url=settings.gemini_base_url,
        api_key=settings.gemini_api_key,
        model=settings.gemini_model,
        timeout_seconds=settings.llm_timeout_seconds,
        thinking_budget=settings.gemini_thinking_budget,
    )


def build_anthropic(settings: Settings) -> Any:
    from media_service.llm.providers.anthropic import AnthropicProvider

    return AnthropicProvider(
        base_url=settings.anthropic_base_url,
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model,
        timeout_seconds=settings.llm_timeout_seconds,
        version=settings.anthropic_version,
        tool_name=settings.anthropic_tool_name,
    )


def build_fake(settings: Settings) -> Any:
    from media_service.llm.providers.fake import FakeLlmProvider

    return FakeLlmProvider(
        mode=settings.llm_fake_mode, fixture_dir=settings.llm_fake_fixture_dir
    )


def build_rule_based(settings: Settings) -> Any:
    from media_service.llm.providers.rule_based import RuleBasedLlmProvider

    return RuleBasedLlmProvider()


BUILDERS = {
    "openai_compatible": build_openai_compatible,
    "gemini": build_gemini,
    "anthropic": build_anthropic,
    "fake": build_fake,
    "rule_based": build_rule_based,
}

assert set(BUILDERS) == set(LLM_PROVIDERS), "every configurable provider needs a builder"


def build_provider(settings: Settings) -> Any:
    guard_offline_provider(settings)
    builder = BUILDERS.get(settings.llm_provider)
    if builder is None:  # pragma: no cover - validate_startup refuses this first
        raise ValueError(
            f"LLM_PROVIDER={settings.llm_provider!r} is not one of {', '.join(sorted(BUILDERS))}."
        )
    return builder(settings)


def build_extraction_service(settings: Settings) -> LlmExtractionService:
    from media_service.llm.service import LlmExtractionService

    return LlmExtractionService(
        provider=build_provider(settings),
        prompt=prompt_for(settings),
        runner_settings=runner_settings(settings),
        max_candidates=settings.llm_max_candidates_per_image,
        max_ocr_chars=settings.llm_max_ocr_chars,
    )
