"""Retry and repair policy, in one place and provider-agnostic.

The adapters classify; this decides. That split is what keeps the policy testable: every transport
failure arrives here already named, so the rules below never branch on a provider.

**Two budgets, genuinely independent** (PRD 11.6). `attempts` covers transport -- timeouts, rate
limits, a provider's 500s, and truncation. `repairs` covers one thing only: a response that parsed
but failed Tier 1. A rate-limited request must not consume the item's single repair, and a schema
failure must not eat the retries it might need afterwards, so they are counted separately and
neither can spend the other.

**Truncation raises the output limit instead of repairing.** A response cut off mid-object cannot be
repaired by asking again with the same budget; the next attempt gets a larger one. Classifying it as
a schema failure would spend the repair on something only more tokens can fix.

**The repair message carries field paths and error messages only -- never values.** Pydantic's
`errors()` includes the offending input, and an OCR-derived phone number echoed into a second prompt
is a personal-data leak into a third party's logs (PRD 15.4). `_safe_errors` strips it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from pydantic import ValidationError

from media_service.llm.prompts.registry import PromptTemplate
from media_service.llm.schema import AdExtractionEnvelope
from media_service.llm.types import (
    TRUNCATED_OUTPUT,
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
)

SCHEMA_INVALID: Final = "LLM_SCHEMA_INVALID"
BUDGET_EXHAUSTED: Final = "LLM_ATTEMPTS_EXHAUSTED"
DEADLINE_EXCEEDED: Final = "LLM_DEADLINE_EXCEEDED"

# Each retry after a truncation asks for more room. 1.5 is enough to clear a response that only just
# overran without multiplying the bill for one that was never going to fit.
TRUNCATION_GROWTH: Final = 1.5
MAX_OUTPUT_TOKEN_CEILING: Final = 32000


class AttemptRecorder(Protocol):
    """Writes one row per attempt, before the call and again after it.

    A record written *before* the request is what makes a hung provider visible in the store rather
    than invisible (FR-JOB-007): an attempt that never returns still left a row saying it started.
    """

    def started(self, attempt: AttemptStart) -> str: ...

    def finished(self, run_id: str, outcome: AttemptOutcome) -> None: ...


@dataclass(frozen=True, slots=True)
class AttemptStart:
    attempt: int
    kind: str
    provider: str
    model: str
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    status: str
    response: LlmResponse | None = None
    error_code: str | None = None
    error_detail: str | None = None
    validation_errors: tuple[dict[str, Any], ...] = ()
    candidate_count: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractionOutcome:
    envelope: AdExtractionEnvelope | None = None
    response: LlmResponse | None = None
    attempts_used: int = 0
    repairs_used: int = 0
    failure_code: str | None = None
    failure_detail: str | None = None
    validation_errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> bool:
        return self.envelope is not None


@dataclass(frozen=True, slots=True)
class RunnerSettings:
    max_attempts: int = 3
    max_repairs: int = 1
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 20.0
    total_deadline_seconds: float = 240.0
    max_output_tokens: int = 8000
    temperature: float = 0.0


class LlmExtractionRunner:
    def __init__(
        self,
        *,
        provider: Any,
        prompt: PromptTemplate,
        settings: RunnerSettings | None = None,
        recorder: AttemptRecorder | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider = provider
        self._prompt = prompt
        self._settings = settings or RunnerSettings()
        self._recorder = recorder
        self._sleep = sleep
        self._now = now

    def run(self, *, system: str, user: str, json_schema: dict[str, Any]) -> ExtractionOutcome:
        settings = self._settings
        deadline = self._now() + settings.total_deadline_seconds

        attempts = 0
        repairs = 0
        max_output_tokens = settings.max_output_tokens
        prior_text: str | None = None
        repair_instruction: str | None = None
        last_code: str | None = None
        last_detail: str | None = None
        last_errors: tuple[dict[str, Any], ...] = ()

        while True:
            if self._now() >= deadline:
                return self._failed(
                    DEADLINE_EXCEEDED,
                    "The extraction deadline elapsed before a valid response arrived.",
                    attempts,
                    repairs,
                    last_errors,
                )

            is_repair = repair_instruction is not None
            if is_repair:
                repairs += 1
            else:
                attempts += 1

            request = LlmRequest(
                system=system,
                user=user,
                json_schema=json_schema,
                max_output_tokens=max_output_tokens,
                temperature=settings.temperature,
                prior_response_text=prior_text if is_repair else None,
                repair_instruction=repair_instruction,
            )
            run_id = self._start(attempts + repairs, "repair" if is_repair else "primary",
                                 max_output_tokens)

            try:
                response = self._provider.complete(request)
            except LlmProviderError as error:
                last_code, last_detail = error.code, error.detail
                self._finish(run_id, AttemptOutcome(
                    status="provider_error", error_code=error.code, error_detail=error.detail
                ))

                if error.code == TRUNCATED_OUTPUT:
                    # More room, not a repair. This is the distinction PRD 11.6 makes explicitly.
                    max_output_tokens = min(
                        int(max_output_tokens * TRUNCATION_GROWTH), MAX_OUTPUT_TOKEN_CEILING
                    )
                if not error.retryable or attempts >= settings.max_attempts:
                    return self._failed(error.code, error.detail, attempts, repairs, last_errors)

                # A repair that failed in transport is retried as a repair, not restarted.
                if is_repair:
                    repairs -= 1
                self._wait(error, attempts, deadline)
                continue

            # -- Tier 1 --------------------------------------------------------------------
            payload = response.payload
            if payload is None:
                last_code = UNPARSEABLE_RESPONSE
                last_detail = "The response body was not a JSON object."
                last_errors = ({"loc": "", "msg": last_detail, "type": "json_invalid"},)
            else:
                try:
                    envelope = AdExtractionEnvelope.model_validate(payload)
                except ValidationError as error:
                    last_code = SCHEMA_INVALID
                    last_errors = _safe_errors(error)
                    last_detail = f"{len(last_errors)} field(s) failed schema validation."
                else:
                    self._finish(run_id, AttemptOutcome(
                        status="validated",
                        response=response,
                        candidate_count=len(envelope.advertisements),
                    ))
                    return ExtractionOutcome(
                        envelope=envelope,
                        response=response,
                        attempts_used=attempts,
                        repairs_used=repairs,
                    )

            self._finish(run_id, AttemptOutcome(
                status="schema_invalid",
                response=response,
                error_code=last_code,
                error_detail=last_detail,
                validation_errors=last_errors,
            ))

            if repairs >= settings.max_repairs:
                return self._failed(last_code, last_detail, attempts, repairs, last_errors)

            prior_text = response.raw_text
            repair_instruction = self._prompt.render_repair(
                {"validation_errors": _render_errors(last_errors)}
            )

    # -- helpers -------------------------------------------------------------------------------

    def _wait(self, error: LlmProviderError, attempts: int, deadline: float) -> None:
        """Back off, honouring the provider's own instruction when it gave one."""
        settings = self._settings
        backoff = min(
            settings.backoff_base_seconds * (2 ** max(attempts - 1, 0)),
            settings.backoff_max_seconds,
        )
        delay = error.retry_after_seconds if error.retry_after_seconds is not None else backoff
        # Never sleep past the deadline: waiting for a window that closes first wastes the wait.
        self._sleep(max(min(delay, deadline - self._now()), 0.0))

    def _start(self, sequence: int, kind: str, max_output_tokens: int) -> str:
        if self._recorder is None:
            return ""
        return self._recorder.started(
            AttemptStart(
                attempt=sequence,
                kind=kind,
                provider=getattr(self._provider, "name", "unknown"),
                model=getattr(self._provider, "model", "unknown"),
                max_output_tokens=max_output_tokens,
            )
        )

    def _finish(self, run_id: str, outcome: AttemptOutcome) -> None:
        if self._recorder is not None:
            self._recorder.finished(run_id, outcome)

    @staticmethod
    def _failed(
        code: str | None,
        detail: str | None,
        attempts: int,
        repairs: int,
        errors: tuple[dict[str, Any], ...],
    ) -> ExtractionOutcome:
        return ExtractionOutcome(
            attempts_used=attempts,
            repairs_used=repairs,
            failure_code=code or BUDGET_EXHAUSTED,
            failure_detail=detail,
            validation_errors=errors,
        )


def _safe_errors(error: ValidationError) -> tuple[dict[str, Any], ...]:
    """Field paths and messages, with the offending values removed.

    Pydantic reports `input` alongside every error. That value is model output derived from a
    newspaper scan and can be a phone number or an address; echoing it into a repair prompt would
    send personal data to a third party for no benefit, since the model already knows what it wrote.
    """
    return tuple(
        {
            "loc": ".".join(str(part) for part in entry.get("loc", ())),
            "msg": str(entry.get("msg", "")),
            "type": str(entry.get("type", "")),
        }
        for entry in error.errors(include_url=False)
    )


def _render_errors(errors: tuple[dict[str, Any], ...]) -> str:
    if not errors:
        return "- The response could not be parsed as JSON."
    return "\n".join(
        f"- {entry['loc'] or '(root)'}: {entry['msg']}" for entry in errors
    )
