"""A provider that never opens a socket.

Four modes, each existing for a specific acceptance criterion rather than for convenience:

* `rule_based` — the deterministic heuristic, so the pipeline runs end to end offline.
* `empty` — returns zero advertisements, which is AC-003: a page with no advertisement on it is a
  legitimate outcome, not a failure.
* `always_invalid` — returns JSON that parses and fails Tier 1, which is AC-005: the item must end
  visibly failed with no advertisement created, after exactly one repair.
* `fixture` — replays recorded responses keyed by a substring of the prompt, which is how the
  provider-matrix test proves every adapter produces identical candidates from identical input.

**No HTTP client is constructed here at all.** That matters more than it looks: a fake that built a
client and simply chose not to use it would still be one refactor away from making a call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from media_service.llm.providers.rule_based import RuleBasedLlmProvider
from media_service.llm.schema import SCHEMA_VERSION
from media_service.llm.types import (
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

PROVIDER_NAME: Final = "fake"

MODES: Final = ("rule_based", "empty", "always_invalid", "fixture")

# Parses as JSON, conforms to nothing: the confidence is out of range and the field is unknown.
# Chosen so the failure is a Tier 1 structural one -- which is what makes it repairable, and what
# AC-005 is about.
INVALID_PAYLOAD: Final[dict[str, Any]] = {
    "schema_version": SCHEMA_VERSION,
    "advertisements": [
        {
            "source_block_ids": [1],
            "language": "en",
            "title": "Structurally invalid",
            "category": "vehicles",
            "confidence": {"overall": 4.2},
            "invented_field": "should be rejected",
        }
    ],
}


class FakeLlmProvider:
    def __init__(
        self,
        *,
        mode: str = "rule_based",
        fixture_dir: Path | None = None,
        unmatched_category: str = "Other",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"LLM_FAKE_MODE={mode!r} is not one of {', '.join(MODES)}.")
        self._mode = mode
        self._fixture_dir = fixture_dir
        self._rule_based = RuleBasedLlmProvider(unmatched_category=unmatched_category)
        self.calls = 0

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return f"{PROVIDER_NAME}/{self._mode}"

    @property
    def mode(self) -> str:
        return self._mode

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        if self._mode == "fixture" and self._fixture_dir is None:
            return "LLM_FAKE_MODE=fixture requires LLM_FAKE_FIXTURE_DIR."
        return None

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.calls += 1
        payload = self._payload(request)
        return LlmResponse(
            raw_text=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            provider=PROVIDER_NAME,
            model=self.model,
            usage=LlmUsage(cost_micros=0),
        )

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        if self._mode == "empty":
            return {"schema_version": SCHEMA_VERSION, "advertisements": []}
        if self._mode == "always_invalid":
            # Invalid on the repair turn too. A fake that repaired itself would let a broken repair
            # path pass, and AC-005 is precisely about what happens when repair does not help.
            return dict(INVALID_PAYLOAD)
        if self._mode == "fixture":
            return self._from_fixture(request)
        return self._rule_based.extract(request.user)

    def _from_fixture(self, request: LlmRequest) -> dict[str, Any]:
        """Replay the first recording whose `match` appears in the prompt."""
        if self._fixture_dir is None:  # pragma: no cover - availability_reason refuses first
            raise LlmProviderError(UNPARSEABLE_RESPONSE, "No fixture directory is configured.")

        for path in sorted(self._fixture_dir.glob("*.json")):
            recording = json.loads(path.read_text(encoding="utf-8"))
            match = recording.get("match")
            if isinstance(match, str) and match and match in request.user:
                return recording["response"]  # type: ignore[no-any-return]

        raise LlmProviderError(
            UNPARSEABLE_RESPONSE,
            f"No recorded response in {self._fixture_dir} matches this prompt.",
            retryable=False,
        )
