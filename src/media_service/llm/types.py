"""The provider contract: what goes to a model, what comes back, and how failures are named.

One decision shapes this module: **classification is the adapter's job, policy is the runner's.**
Only the Gemini adapter knows that `RESOURCE_EXHAUSTED` means a rate limit; only the Anthropic one
knows that `529 overloaded_error` is transient. Each maps its native failure into a single
`LlmProviderError` carrying a code, whether it is worth retrying, and any `Retry-After` supplied --
and the runner decides what to do without ever seeing a provider-specific string.

Without that split, retry policy would have to grow a branch per provider, and adding a fourth
provider would mean editing the runner. With it, an adapter is self-contained and the policy is
tested once.

`LlmRequest` carries a **JSON schema, not a Pydantic model**. That is what makes the adapters
reusable: the same three of them serve advertisement extraction and, later, the `vision_llm` OCR
provider, which needs an entirely different shape out of the same wire protocols (FR-LLM-003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

# Retryable: the request was fine and the far side was not.
TIMEOUT: Final = "LLM_TIMEOUT"
RATE_LIMITED: Final = "LLM_RATE_LIMITED"
SERVER_ERROR: Final = "LLM_SERVER_ERROR"
TRANSPORT_ERROR: Final = "LLM_TRANSPORT_ERROR"
# Truncation is a *transport* failure, not a schema one. Repair cannot fix a response that was cut
# off mid-object, and treating it as repairable would spend the item's single repair on something
# only a larger output budget can solve (PRD 11.6).
TRUNCATED_OUTPUT: Final = "LLM_TRUNCATED_OUTPUT"

# Not retryable: repeating the same request produces the same failure.
AUTH_FAILED: Final = "LLM_AUTH_FAILED"
BAD_REQUEST: Final = "LLM_BAD_REQUEST"
MODEL_UNAVAILABLE: Final = "LLM_MODEL_UNAVAILABLE"
CONTENT_BLOCKED: Final = "LLM_CONTENT_BLOCKED"
UNPARSEABLE_RESPONSE: Final = "LLM_UNPARSEABLE_RESPONSE"
NOT_CONFIGURED: Final = "LLM_NOT_CONFIGURED"

RETRYABLE_CODES: Final = frozenset(
    {TIMEOUT, RATE_LIMITED, SERVER_ERROR, TRANSPORT_ERROR, TRUNCATED_OUTPUT}
)


class LlmProviderError(Exception):
    """A provider failure, already classified by the adapter that understands it."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        retryable: bool | None = None,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        # The code already implies the answer; the override exists for the rare case an adapter
        # knows better than the table, such as a 400 that names a transient condition.
        self.retryable = code in RETRYABLE_CODES if retryable is None else retryable
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LlmProviderError({self.code}, retryable={self.retryable})"


@dataclass(frozen=True, slots=True)
class LlmRequest:
    """One turn. Schema-agnostic by design -- see the module docstring."""

    system: str
    user: str
    json_schema: dict[str, Any]
    schema_name: str = "ad_extraction"
    max_output_tokens: int = 8000
    temperature: float = 0.0
    # Set on a repair turn. The prior text is replayed as the assistant's turn so the model corrects
    # its own answer rather than starting again; the instruction carries **field paths and messages
    # only, never values** (PRD 15.4).
    prior_response_text: str | None = None
    repair_instruction: str | None = None

    @property
    def is_repair(self) -> bool:
        return self.repair_instruction is not None


@dataclass(frozen=True, slots=True)
class LlmUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_micros: int | None = None


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """What an adapter returns once it has read the provider's own envelope.

    `payload` is the parsed object when the provider handed back structure -- Anthropic's
    `tool_use.input` is already a dict, so no string ever gets parsed for it. `raw_text` is kept for
    provenance and is what a repair turn replays.
    """

    raw_text: str
    payload: dict[str, Any] | None = None
    provider: str = ""
    model: str = ""
    usage: LlmUsage = field(default_factory=LlmUsage)
    latency_ms: int = 0
    finish_reason: str | None = None
    # True when the provider stopped because it hit the output limit. The adapter raises
    # `TRUNCATED_OUTPUT` for this rather than returning it, but the flag is kept for the record.
    truncated: bool = False
