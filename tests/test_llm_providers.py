"""Adapters, the registry, and the guarantee that no test ever calls a provider.

The adapters are exercised against a stubbed transport, never a live endpoint. What is being tested
is the translation each one performs: the payload it builds, and -- more importantly -- how it
classifies its own failures, because that classification is what the retry policy acts on.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr

from media_service.config import Settings
from media_service.llm.providers.anthropic import AnthropicProvider
from media_service.llm.providers.fake import FakeLlmProvider
from media_service.llm.providers.gemini import GeminiProvider
from media_service.llm.providers.openai_compatible import OpenAiCompatibleProvider
from media_service.llm.providers.rule_based import RuleBasedLlmProvider, ocr_section
from media_service.llm.registry import build_extraction_service, build_provider
from media_service.llm.schema import json_schema
from media_service.llm.types import (
    AUTH_FAILED,
    BAD_REQUEST,
    CONTENT_BLOCKED,
    MODEL_UNAVAILABLE,
    RATE_LIMITED,
    SERVER_ERROR,
    TRUNCATED_OUTPUT,
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
)

PAYLOAD = {
    "schema_version": "1.0",
    "advertisements": [
        {
            "source_block_ids": [1],
            "language": "en",
            "title": "Toyota Prius 2016",
            "category": "vehicles",
            "confidence": {"overall": 0.9},
        }
    ],
}


def request() -> LlmRequest:
    return LlmRequest(system="system", user="user", json_schema=json_schema())


def stub(provider, *, status: int = 200, body=None, headers=None):  # type: ignore[no-untyped-def]
    """Give an adapter a transport that answers without a network."""

    def handler(request_: httpx.Request) -> httpx.Response:
        handler.seen = request_  # type: ignore[attr-defined]
        return httpx.Response(status, json=body or {}, headers=headers or {})

    provider._client = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    return handler


# -- OpenAI-compatible -------------------------------------------------------------------------


def openai(**kwargs):  # type: ignore[no-untyped-def]
    return OpenAiCompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key=SecretStr("sk-test"),
        model="gpt-4o-mini",
        **kwargs,
    )


def test_openai_sends_a_strict_schema_and_reads_the_content() -> None:
    provider = openai(structured_mode="json_schema")
    handler = stub(
        provider,
        body={
            "model": "gpt-4o-mini",
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(PAYLOAD)}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    )

    response = provider.complete(request())

    sent = json.loads(handler.seen.content)
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert response.payload == PAYLOAD
    assert response.usage.input_tokens == 10


def test_the_key_travels_in_a_header_not_a_url() -> None:
    provider = openai()
    handler = stub(
        provider,
        body={"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
    )

    provider.complete(request())

    assert handler.seen.headers["authorization"] == "Bearer sk-test"
    assert "sk-test" not in str(handler.seen.url)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.openai.com/v1", "json_schema"),
        ("https://api.groq.com/openai/v1", "json_object"),
        ("http://localhost:1234/v1", "json_object"),
    ],
)
def test_auto_mode_degrades_by_host(base_url: str, expected: str) -> None:
    """A 400 naming a parameter the operator never set is a poor way to learn this."""
    provider = OpenAiCompatibleProvider(
        base_url=base_url, api_key=SecretStr("k"), model="m", structured_mode="auto"
    )

    assert provider.structured_mode == expected


def test_a_length_finish_is_a_truncation_not_a_schema_failure() -> None:
    provider = openai()
    stub(provider, body={"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == TRUNCATED_OUTPUT
    assert caught.value.retryable is True


def test_a_content_filter_is_terminal() -> None:
    provider = openai()
    stub(
        provider,
        body={"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]},
    )

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == CONTENT_BLOCKED
    assert caught.value.retryable is False


def test_a_fenced_response_is_still_parsed() -> None:
    """Some compatible servers wrap JSON even in json_object mode."""
    provider = openai(structured_mode="json_object")
    stub(
        provider,
        body={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": f"```json\n{json.dumps(PAYLOAD)}\n```"},
                }
            ]
        },
    )

    assert provider.complete(request()).payload == PAYLOAD


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, AUTH_FAILED, False),
        (403, AUTH_FAILED, False),
        (404, MODEL_UNAVAILABLE, False),
        (400, BAD_REQUEST, False),
        (429, RATE_LIMITED, True),
        (500, SERVER_ERROR, True),
        (503, SERVER_ERROR, True),
    ],
)
def test_http_statuses_are_classified(status: int, code: str, retryable: bool) -> None:
    provider = openai()
    stub(provider, status=status, body={"error": {"message": "nope"}})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == code
    assert caught.value.retryable is retryable


def test_a_retry_after_header_is_carried_into_the_error() -> None:
    provider = openai()
    stub(provider, status=429, body={"error": {}}, headers={"retry-after": "12"})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.retry_after_seconds == 12.0


def test_an_error_body_is_excerpted_not_echoed_whole() -> None:
    """A provider error can echo the request, and the request carries OCR text (PRD 15.4)."""
    provider = openai()
    stub(provider, status=400, body={"error": {"message": "x" * 5000, "type": "invalid"}})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert len(caught.value.detail) < 1000


# -- Gemini ------------------------------------------------------------------------------------


def gemini(**kwargs):  # type: ignore[no-untyped-def]
    return GeminiProvider(
        base_url="https://generativelanguage.googleapis.com",
        api_key=SecretStr("gem-key"),
        model="gemini-2.0-flash",
        **kwargs,
    )


def test_gemini_sends_its_key_in_a_header_never_the_query_string() -> None:
    """A `?key=` lands in every proxy log along the path."""
    provider = gemini()
    handler = stub(
        provider,
        body={
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(PAYLOAD)}]}}
            ]
        },
    )

    provider.complete(request())

    assert handler.seen.headers["x-goog-api-key"] == "gem-key"
    assert "key=" not in str(handler.seen.url)
    assert "gem-key" not in str(handler.seen.url)


def test_gemini_sends_its_own_schema_dialect() -> None:
    provider = gemini()
    handler = stub(
        provider,
        body={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
    )

    provider.complete(request())

    schema = json.loads(handler.seen.content)["generationConfig"]["responseSchema"]
    assert schema["type"] == "OBJECT"
    assert "propertyOrdering" in schema


def test_gemini_reports_rate_limiting_in_its_body() -> None:
    """The kind of provider-specific knowledge that belongs in an adapter and nowhere else."""
    provider = gemini()
    stub(provider, status=400, body={"error": {"status": "RESOURCE_EXHAUSTED"}})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == RATE_LIMITED
    assert caught.value.retryable is True


def test_gemini_max_tokens_is_a_truncation() -> None:
    provider = gemini()
    stub(provider, body={"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == TRUNCATED_OUTPUT


def test_a_blocked_prompt_is_terminal() -> None:
    provider = gemini()
    stub(provider, body={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == CONTENT_BLOCKED
    assert caught.value.retryable is False


# -- Anthropic ---------------------------------------------------------------------------------


def anthropic(**kwargs):  # type: ignore[no-untyped-def]
    return AnthropicProvider(
        base_url="https://api.anthropic.com",
        api_key=SecretStr("ant-key"),
        model="claude-sonnet-4-5",
        **kwargs,
    )


def test_anthropic_forces_a_tool_call_and_reads_its_input_directly() -> None:
    """No string is parsed, so "the model wrapped its JSON in prose" cannot happen."""
    provider = anthropic()
    handler = stub(
        provider,
        body={
            "stop_reason": "tool_use",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "name": "emit_advertisements", "input": PAYLOAD}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 6},
        },
    )

    response = provider.complete(request())

    sent = json.loads(handler.seen.content)
    assert sent["tool_choice"] == {"type": "tool", "name": "emit_advertisements"}
    assert response.payload == PAYLOAD
    assert handler.seen.headers["x-api-key"] == "ant-key"


def test_anthropic_checks_truncation_before_it_looks_at_the_payload() -> None:
    """A truncated tool call looks like a schema failure and is not."""
    provider = anthropic()
    stub(
        provider,
        body={
            "stop_reason": "max_tokens",
            "content": [{"type": "tool_use", "name": "emit_advertisements", "input": {}}],
        },
    )

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == TRUNCATED_OUTPUT


def test_anthropic_overloaded_is_retryable() -> None:
    """529 is outside the 5xx range every client already retries, so it needs naming."""
    provider = anthropic()
    stub(provider, status=529, body={"error": {"type": "overloaded_error"}})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == SERVER_ERROR
    assert caught.value.retryable is True


def test_a_response_with_no_tool_call_is_unparseable() -> None:
    provider = anthropic()
    stub(provider, body={"stop_reason": "end_turn", "content": [{"type": "text", "text": "hi"}]})

    with pytest.raises(LlmProviderError) as caught:
        provider.complete(request())

    assert caught.value.code == UNPARSEABLE_RESPONSE


# -- availability ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        (
            OpenAiCompatibleProvider(base_url="u", api_key=None, model="m"),
            "OPENAI_API_KEY",
        ),
        (GeminiProvider(base_url="u", api_key=None, model="m"), "GEMINI_API_KEY"),
        (AnthropicProvider(base_url="u", api_key=None, model="m"), "ANTHROPIC_API_KEY"),
    ],
)
def test_a_missing_key_names_the_variable_never_a_value(provider, variable: str) -> None:  # type: ignore[no-untyped-def]
    """This string reaches `/health`, which is unauthenticated."""
    reason = provider.availability_reason()

    assert reason == f"{variable} is not set."
    assert provider.is_available() is False


def test_a_configured_provider_is_available() -> None:
    assert openai().is_available() is True


def test_a_secret_is_not_printed_by_repr() -> None:
    settings = Settings(openai_api_key=SecretStr("sk-very-secret"))

    assert "sk-very-secret" not in repr(settings)
    assert "sk-very-secret" not in str(settings)


# -- fake and rule-based -----------------------------------------------------------------------


def test_the_fake_provider_constructs_no_http_client() -> None:
    """A fake that built a client would be one refactor from making a call."""
    provider = FakeLlmProvider(mode="rule_based")

    assert not hasattr(provider, "_client")


def test_the_empty_mode_returns_no_advertisements() -> None:
    """AC-003."""
    provider = FakeLlmProvider(mode="empty")

    assert provider.complete(request()).payload == {
        "schema_version": "1.0",
        "advertisements": [],
    }


def test_the_always_invalid_mode_stays_invalid_on_repair() -> None:
    """AC-005 is about what happens when repair does not help."""
    provider = FakeLlmProvider(mode="always_invalid")
    first = provider.complete(request())
    repair = provider.complete(
        LlmRequest(
            system="s",
            user="u",
            json_schema=json_schema(),
            repair_instruction="fix it",
            prior_response_text="{}",
        )
    )

    assert first.payload == repair.payload


def test_an_unknown_fake_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="LLM_FAKE_MODE"):
        FakeLlmProvider(mode="imaginary")


def test_fixture_mode_replays_a_recorded_response(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "case.json").write_text(
        json.dumps({"match": "Toyota", "response": PAYLOAD}), encoding="utf-8"
    )
    provider = FakeLlmProvider(mode="fixture", fixture_dir=tmp_path)

    response = provider.complete(
        LlmRequest(system="s", user="a page about Toyota", json_schema=json_schema())
    )

    assert response.payload == PAYLOAD


def test_fixture_mode_without_a_directory_says_so() -> None:
    provider = FakeLlmProvider(mode="fixture")

    assert "LLM_FAKE_FIXTURE_DIR" in provider.availability_reason()


def test_the_rule_based_provider_reads_the_prompt_back() -> None:
    """It sees exactly what a real provider sees, rather than a privileged side channel."""
    prompt = "<ocr_blocks>\n[3] (x=0,y=0,w=1,h=1) conf=0.9\nToyota Prius Kandy\n</ocr_blocks>"

    text, block_ids = ocr_section(prompt)

    assert text == "Toyota Prius Kandy"
    assert block_ids == (3,)


def test_the_rule_based_provider_cites_the_blocks_it_was_shown() -> None:
    provider = RuleBasedLlmProvider()
    payload = provider.extract(
        "<ocr_blocks>\n[1] conf=0.9\nToyota Prius 2016 Rs. 5,750,000 Kandy 0771234567\n"
        "[2] conf=0.9\nMore text\n</ocr_blocks>"
    )

    advertisement = payload["advertisements"][0]
    assert advertisement["source_block_ids"] == [1, 2]
    assert advertisement["contacts"]["phones"] == ["0771234567"]


def test_the_rule_based_provider_does_not_file_unmatched_ads_into_a_real_category() -> None:
    """The prototype filed them as `Home`, silently mis-filing everything it could not classify."""
    payload = RuleBasedLlmProvider().extract(
        "<ocr_blocks>\n[1] conf=0.9\nAntique map collection for exhibition\n</ocr_blocks>"
    )

    assert payload["advertisements"][0]["category"] == "Other"


def test_an_empty_page_produces_no_advertisements() -> None:
    assert RuleBasedLlmProvider().extract("<ocr_blocks>\n</ocr_blocks>")["advertisements"] == []


# -- registry ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["openai_compatible", "gemini", "anthropic", "fake", "rule_based"]
)
def test_every_configurable_provider_can_be_built(name: str) -> None:
    provider = build_provider(Settings(llm_provider=name))

    assert provider.name == name


def test_an_unknown_provider_stops_the_service_starting() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER") as caught:
        Settings(llm_provider="gpt5000").validate_startup()

    assert "anthropic" in str(caught.value)


@pytest.mark.parametrize("name", ["fake", "rule_based"])
def test_a_heuristic_provider_is_refused_in_production(name: str) -> None:
    """PRD 11.6, enforced at startup rather than at extraction time.

    A check at extraction time fails once a page has already been uploaded, on a worker thread,
    where the operator sees `needs_attention` and no explanation.
    """
    with pytest.raises(ValueError, match="MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS"):
        build_provider(Settings(environment="production", llm_provider=name))


def test_a_heuristic_provider_is_allowed_when_asked_for_deliberately() -> None:
    provider = build_provider(
        Settings(
            environment="production",
            llm_provider="fake",
            allow_fake_providers_override=True,
        )
    )

    assert provider.name == "fake"


def test_a_real_provider_is_unaffected_by_the_guard() -> None:
    provider = build_provider(
        Settings(environment="production", llm_provider="anthropic")
    )

    assert provider.name == "anthropic"


def test_the_extraction_service_reports_its_prompt_provenance() -> None:
    service = build_extraction_service(Settings())

    assert service.prompt_version == "ad-extraction/v1"
    assert len(service.prompt_checksum) == 12


# -- the network guard -------------------------------------------------------------------------


def test_an_unmarked_test_cannot_reach_the_network() -> None:
    """AC-016 as a guarantee rather than a convention: a fake is a convention, this is not."""
    with pytest.raises(AssertionError, match="real HTTP request"):
        httpx.Client().get("https://example.com")
