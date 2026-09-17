"""Retry and repair: two budgets that must not spend each other."""

from __future__ import annotations

import json

import pytest

from media_service.llm.prompts.registry import load_prompt
from media_service.llm.runner import (
    SCHEMA_INVALID,
    AttemptOutcome,
    AttemptStart,
    LlmExtractionRunner,
    RunnerSettings,
    _safe_errors,
)
from media_service.llm.schema import json_schema
from media_service.llm.types import (
    AUTH_FAILED,
    RATE_LIMITED,
    TIMEOUT,
    TRUNCATED_OUTPUT,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
)

VALID = {
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
INVALID = {"schema_version": "1.0", "advertisements": [{"confidence": {"overall": 5.0}}]}


class ScriptedProvider:
    """Replays a script of outcomes, recording the requests it was given."""

    name = "scripted"
    model = "scripted/v1"

    def __init__(self, *script) -> None:  # type: ignore[no-untyped-def]
        self._script = list(script)
        self.requests: list[LlmRequest] = []

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str | None:
        return None

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        step = self._script.pop(0) if self._script else VALID
        if isinstance(step, Exception):
            raise step
        return LlmResponse(raw_text=json.dumps(step), payload=step, provider=self.name)


class RecordingRecorder:
    def __init__(self) -> None:
        self.started_attempts: list[AttemptStart] = []
        self.finished_outcomes: list[AttemptOutcome] = []

    def started(self, attempt: AttemptStart) -> str:
        self.started_attempts.append(attempt)
        return f"run-{len(self.started_attempts)}"

    def finished(self, run_id: str, outcome: AttemptOutcome) -> None:
        self.finished_outcomes.append(outcome)


@pytest.fixture
def prompt():  # type: ignore[no-untyped-def]
    return load_prompt("ad_extraction", "v1")


def run(provider, prompt, **overrides):  # type: ignore[no-untyped-def]
    settings = RunnerSettings(**{"backoff_base_seconds": 0.001, **overrides})
    runner = LlmExtractionRunner(
        provider=provider, prompt=prompt, settings=settings, sleep=lambda _: None
    )
    return runner.run(system="system", user="user", json_schema=json_schema())


# -- the happy path ----------------------------------------------------------------------------


def test_a_valid_first_response_costs_one_call(prompt) -> None:
    provider = ScriptedProvider(VALID)

    outcome = run(provider, prompt)

    assert outcome.succeeded
    assert outcome.attempts_used == 1
    assert outcome.repairs_used == 0
    assert len(provider.requests) == 1


# -- transport ---------------------------------------------------------------------------------


def test_a_retryable_failure_is_retried(prompt) -> None:
    provider = ScriptedProvider(LlmProviderError(TIMEOUT, "timed out"), VALID)

    outcome = run(provider, prompt)

    assert outcome.succeeded
    assert outcome.attempts_used == 2


def test_a_non_retryable_failure_stops_immediately(prompt) -> None:
    """Repeating an authentication failure produces the same failure and another bill."""
    provider = ScriptedProvider(LlmProviderError(AUTH_FAILED, "bad key"), VALID)

    outcome = run(provider, prompt)

    assert not outcome.succeeded
    assert outcome.failure_code == AUTH_FAILED
    assert len(provider.requests) == 1


def test_attempts_are_bounded(prompt) -> None:
    provider = ScriptedProvider(*[LlmProviderError(TIMEOUT, "timed out")] * 5)

    outcome = run(provider, prompt, max_attempts=3)

    assert not outcome.succeeded
    assert outcome.attempts_used == 3
    assert len(provider.requests) == 3


def test_a_provider_supplied_retry_after_is_honoured(prompt) -> None:
    """PRD 11.6: the provider knows when its window reopens better than our backoff does."""
    slept: list[float] = []
    provider = ScriptedProvider(
        LlmProviderError(RATE_LIMITED, "slow down", retry_after_seconds=7.0), VALID
    )
    runner = LlmExtractionRunner(
        provider=provider,
        prompt=prompt,
        settings=RunnerSettings(backoff_base_seconds=0.001),
        sleep=slept.append,
    )

    runner.run(system="s", user="u", json_schema=json_schema())

    assert slept == [7.0]


def test_backoff_grows_between_attempts(prompt) -> None:
    slept: list[float] = []
    provider = ScriptedProvider(*[LlmProviderError(TIMEOUT, "t")] * 2, VALID)
    runner = LlmExtractionRunner(
        provider=provider,
        prompt=prompt,
        settings=RunnerSettings(backoff_base_seconds=1.0, backoff_max_seconds=60.0),
        sleep=slept.append,
    )

    runner.run(system="s", user="u", json_schema=json_schema())

    assert slept == [1.0, 2.0]


# -- truncation --------------------------------------------------------------------------------


def test_truncation_raises_the_output_limit_rather_than_repairing(prompt) -> None:
    """Repair cannot lengthen a response cut off mid-object; only a larger budget can."""
    provider = ScriptedProvider(
        LlmProviderError(TRUNCATED_OUTPUT, "hit the limit"), VALID
    )

    outcome = run(provider, prompt, max_output_tokens=1000)

    assert outcome.succeeded
    assert outcome.repairs_used == 0, "truncation must not spend the repair budget"
    assert provider.requests[1].max_output_tokens > provider.requests[0].max_output_tokens


# -- repair ------------------------------------------------------------------------------------


def test_a_schema_failure_is_repaired_once(prompt) -> None:
    provider = ScriptedProvider(INVALID, VALID)

    outcome = run(provider, prompt)

    assert outcome.succeeded
    assert outcome.repairs_used == 1
    assert provider.requests[1].is_repair


def test_a_repair_replays_the_models_own_answer(prompt) -> None:
    """So it corrects that answer rather than starting again and losing what it read."""
    provider = ScriptedProvider(INVALID, VALID)

    run(provider, prompt)

    assert provider.requests[1].prior_response_text == json.dumps(INVALID)


def test_the_repair_budget_is_bounded(prompt) -> None:
    """AC-005: invalid after the repair is spent means a visible failure and no advertisement."""
    provider = ScriptedProvider(INVALID, INVALID, INVALID)

    outcome = run(provider, prompt, max_repairs=1)

    assert not outcome.succeeded
    assert outcome.failure_code == SCHEMA_INVALID
    assert outcome.repairs_used == 1
    assert len(provider.requests) == 2


def test_a_transport_retry_does_not_consume_the_repair_budget(prompt) -> None:
    """The budgets are independent, which is the whole point of counting them separately."""
    provider = ScriptedProvider(
        LlmProviderError(TIMEOUT, "t"), INVALID, VALID
    )

    outcome = run(provider, prompt, max_attempts=3, max_repairs=1)

    assert outcome.succeeded
    assert outcome.attempts_used == 2
    assert outcome.repairs_used == 1


def test_a_repair_does_not_consume_the_transport_budget(prompt) -> None:
    provider = ScriptedProvider(INVALID, LlmProviderError(TIMEOUT, "t"), VALID)

    outcome = run(provider, prompt, max_attempts=3, max_repairs=1)

    assert outcome.succeeded
    assert outcome.attempts_used <= 3


# -- the repair message ------------------------------------------------------------------------


def test_the_repair_message_carries_paths_but_never_values() -> None:
    """PRD 15.4: an OCR-derived phone number must not be echoed into a second prompt."""
    from pydantic import ValidationError

    from media_service.llm.schema import AdExtractionEnvelope

    secret = "0771234567"
    try:
        AdExtractionEnvelope.model_validate(
            {
                "schema_version": "1.0",
                "advertisements": [
                    {
                        "source_block_ids": [1],
                        "confidence": {"overall": 0.9},
                        "title": secret,
                        "location": 12345,
                    }
                ],
            }
        )
    except ValidationError as error:
        safe = _safe_errors(error)
    else:  # pragma: no cover - the payload is invalid by construction
        pytest.fail("the payload should not have validated")

    rendered = str(safe)
    assert "location" in rendered
    assert secret not in rendered
    assert "12345" not in rendered
    assert all(set(entry) == {"loc", "msg", "type"} for entry in safe)


def test_the_repair_prompt_contains_no_values(prompt) -> None:
    provider = ScriptedProvider(
        {"schema_version": "1.0", "advertisements": [{"confidence": {"overall": 9.9}}]}, VALID
    )

    run(provider, prompt)

    instruction = provider.requests[1].repair_instruction or ""
    assert "confidence.overall" in instruction
    assert "9.9" not in instruction


# -- recording ---------------------------------------------------------------------------------


def test_every_attempt_is_recorded_before_it_is_made(prompt) -> None:
    """FR-JOB-007: a hung provider must be visible in the store, not invisible."""
    recorder = RecordingRecorder()
    provider = ScriptedProvider(LlmProviderError(TIMEOUT, "t"), INVALID, VALID)
    runner = LlmExtractionRunner(
        provider=provider,
        prompt=prompt,
        settings=RunnerSettings(backoff_base_seconds=0.001),
        recorder=recorder,
        sleep=lambda _: None,
    )

    runner.run(system="s", user="u", json_schema=json_schema())

    assert len(recorder.started_attempts) == 3
    assert [outcome.status for outcome in recorder.finished_outcomes] == [
        "provider_error",
        "schema_invalid",
        "validated",
    ]
    assert [attempt.kind for attempt in recorder.started_attempts] == [
        "primary",
        "primary",
        "repair",
    ]


def test_the_validated_attempt_records_its_candidate_count(prompt) -> None:
    recorder = RecordingRecorder()
    runner = LlmExtractionRunner(
        provider=ScriptedProvider(VALID), prompt=prompt, recorder=recorder
    )

    runner.run(system="s", user="u", json_schema=json_schema())

    assert recorder.finished_outcomes[-1].candidate_count == 1


# -- deadline ----------------------------------------------------------------------------------


def test_the_total_deadline_stops_retrying(prompt) -> None:
    clock = iter([0.0, 0.0, 100.0, 100.0, 200.0, 200.0, 300.0])
    provider = ScriptedProvider(*[LlmProviderError(TIMEOUT, "t")] * 5)
    runner = LlmExtractionRunner(
        provider=provider,
        prompt=prompt,
        settings=RunnerSettings(max_attempts=10, total_deadline_seconds=150.0),
        sleep=lambda _: None,
        now=lambda: next(clock),
    )

    outcome = runner.run(system="s", user="u", json_schema=json_schema())

    assert not outcome.succeeded
    assert len(provider.requests) < 10
