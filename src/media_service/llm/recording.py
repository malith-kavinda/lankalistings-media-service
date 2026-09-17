"""Writing one `llm_extraction_runs` row per attempt.

The row is written **before** the call, with `status="running"`. That ordering is the whole point
(FR-JOB-007): an attempt that never returns -- a provider that accepts the connection and then hangs
-- still left a record saying it started. Without it, a hung call is invisible in the store, and the
only evidence is an item that has been in `llm_processing` for an hour.

Each attempt gets its own row, so a failure and the retry that fixed it are both visible afterwards
rather than the last one overwriting the story (PRD 11.6).

One row per attempt also means the partial unique index does its job: at most one row per generation
may be `validated`, so two workers racing the same item cannot both record a success, and the loser
resumes from the winner instead of paying for another call.
"""

from __future__ import annotations

from typing import Final

from media_service.db.base import utcnow
from media_service.db.tables import LlmExtractionRun
from media_service.db.uow import UnitOfWorkFactory
from media_service.domain.ids import new_llm_run_id
from media_service.llm.runner import AttemptOutcome, AttemptStart

# The status vocabulary `ocr_extractions`' sibling table already declares.
RUNNING: Final = "running"
VALIDATED: Final = "validated"
SCHEMA_INVALID: Final = "schema_invalid"
PROVIDER_ERROR: Final = "provider_error"

# Raw text is kept for provenance and capped, because it is model output derived from a scan and can
# be arbitrarily long (PRD 12.5).
MAX_RAW_TEXT_CHARS: Final = 20000


class DbAttemptRecorder:
    """Records attempts for one item generation, and remembers which one validated."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkFactory,
        *,
        item_id: str,
        generation: int,
        ocr_extraction_id: str,
        request_hash: str,
        prompt_version: str,
        prompt_checksum: str,
        schema_version: str,
        catalog_version: str,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._item_id = item_id
        self._generation = generation
        self._ocr_extraction_id = ocr_extraction_id
        self._request_hash = request_hash
        self._prompt_version = prompt_version
        self._prompt_checksum = prompt_checksum
        self._schema_version = schema_version
        self._catalog_version = catalog_version
        self.validated_run_id: str | None = None

    def started(self, attempt: AttemptStart) -> str:
        with self._unit_of_work() as unit:
            number = unit.artifacts.next_llm_attempt(
                item_id=self._item_id, generation=self._generation
            )
            run, _ = unit.artifacts.record_llm_run(
                LlmExtractionRun(
                    id=new_llm_run_id(),
                    ingestion_item_id=self._item_id,
                    ocr_extraction_id=self._ocr_extraction_id,
                    generation=self._generation,
                    attempt=number,
                    attempt_kind=attempt.kind,
                    provider=attempt.provider,
                    model=attempt.model,
                    prompt_version=self._prompt_version,
                    prompt_checksum=self._prompt_checksum,
                    schema_version=self._schema_version,
                    category_catalog_version=self._catalog_version,
                    request_hash=self._request_hash,
                    status=RUNNING,
                )
            )
            run_id = run.id
            unit.commit()
        return run_id

    def finished(self, run_id: str, outcome: AttemptOutcome) -> None:
        if not run_id:
            return
        with self._unit_of_work() as unit:
            run = unit.artifacts.get_llm_run(run_id)
            if run is None:  # pragma: no cover - written moments earlier
                return

            response = outcome.response
            run.status = _status_of(outcome)
            run.completed_at = utcnow()
            run.error_code = outcome.error_code
            run.error_message = outcome.error_detail
            run.validation_errors = list(outcome.validation_errors) or None
            run.candidate_count = outcome.candidate_count

            if response is not None:
                run.raw_response_text = response.raw_text[:MAX_RAW_TEXT_CHARS]
                run.latency_ms = response.latency_ms
                run.input_tokens = response.usage.input_tokens
                run.output_tokens = response.usage.output_tokens
                run.cost_micros = response.usage.cost_micros
                if run.status == VALIDATED:
                    run.validated_response = response.payload

            if run.status == VALIDATED:
                self.validated_run_id = run.id
            unit.commit()


def _status_of(outcome: AttemptOutcome) -> str:
    if outcome.status in (VALIDATED, SCHEMA_INVALID, PROVIDER_ERROR):
        return outcome.status
    return PROVIDER_ERROR
