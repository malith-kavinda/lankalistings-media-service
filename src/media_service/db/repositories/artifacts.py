"""OCR extractions and LLM runs -- the per-stage artifacts a resume reads.

Every row here is scoped by `(item, generation, attempt)`:

* `attempt` is one row per external call, so a failed call and its retry are both visible. PRD 11.6
  asks for exactly that.
* `generation` is the idempotency *scope*. It is bumped only when prior artifacts are declared
  invalid -- a reprocess -- which is what separates "run it again because it broke" from "run it
  again because the inputs changed".

Two partial unique indexes carry the weight: at most one *completed* OCR row and at most one
*validated* LLM run per generation. They turn a duplicate dispatch from a correctness problem into
a resume: the loser catches the IntegrityError, re-selects the winner's row, and carries on from
it. For the LLM run that is the difference between paying a provider twice and paying once.
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from media_service.db.base import utcnow
from media_service.db.engine import savepoint
from media_service.db.tables import LlmExtractionRun, OcrExtraction


class ArtifactRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    # -- OCR -----------------------------------------------------------------------------------

    def completed_ocr(self, *, item_id: str, generation: int) -> OcrExtraction | None:
        return self._session.scalar(
            select(OcrExtraction).where(
                OcrExtraction.ingestion_item_id == item_id,
                OcrExtraction.generation == generation,
                OcrExtraction.status == "completed",
            )
        )

    def get_ocr(self, extraction_id: str) -> OcrExtraction | None:
        return self._session.get(OcrExtraction, extraction_id)

    def next_ocr_attempt(self, *, item_id: str, generation: int) -> int:
        highest = self._session.scalar(
            select(func.max(OcrExtraction.attempt)).where(
                OcrExtraction.ingestion_item_id == item_id,
                OcrExtraction.generation == generation,
            )
        )
        return (highest or 0) + 1

    def record_ocr(self, extraction: OcrExtraction) -> tuple[OcrExtraction, bool]:
        """Persist an OCR result, yielding to whoever completed this generation first."""
        try:
            with savepoint(self._session):
                self._session.add(extraction)
                self._session.flush()
        except IntegrityError:
            winner = self.completed_ocr(
                item_id=extraction.ingestion_item_id, generation=extraction.generation
            )
            if winner is None:  # pragma: no cover - a different constraint fired
                raise
            return winner, False
        return extraction, True

    def abandon_running_ocr(self, *, item_id: str, generation: int) -> int:
        """Close out rows whose worker died mid-call.

        A `running` row with no live worker is not evidence of anything, and leaving it would make
        `next_ocr_attempt` and the run history lie about what actually happened.
        """
        result = self._session.execute(
            update(OcrExtraction)
            .where(
                OcrExtraction.ingestion_item_id == item_id,
                OcrExtraction.generation == generation,
                OcrExtraction.status == "running",
            )
            .values(status="abandoned", completed_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount

    # -- LLM -----------------------------------------------------------------------------------

    def validated_llm_run(self, *, item_id: str, generation: int) -> LlmExtractionRun | None:
        return self._session.scalar(
            select(LlmExtractionRun).where(
                LlmExtractionRun.ingestion_item_id == item_id,
                LlmExtractionRun.generation == generation,
                LlmExtractionRun.status == "validated",
            )
        )

    def get_llm_run(self, run_id: str) -> LlmExtractionRun | None:
        return self._session.get(LlmExtractionRun, run_id)

    def next_llm_attempt(self, *, item_id: str, generation: int) -> int:
        highest = self._session.scalar(
            select(func.max(LlmExtractionRun.attempt)).where(
                LlmExtractionRun.ingestion_item_id == item_id,
                LlmExtractionRun.generation == generation,
            )
        )
        return (highest or 0) + 1

    def record_llm_run(self, run: LlmExtractionRun) -> tuple[LlmExtractionRun, bool]:
        """Persist a run, yielding to whoever validated this generation first.

        Losing this race is the cheap outcome: the winner's validated response is already on disk,
        so the loser resumes from it instead of calling the provider again.
        """
        try:
            with savepoint(self._session):
                self._session.add(run)
                self._session.flush()
        except IntegrityError:
            winner = self.validated_llm_run(
                item_id=run.ingestion_item_id, generation=run.generation
            )
            if winner is None:  # pragma: no cover - a different constraint fired
                raise
            return winner, False
        return run, True

    def abandon_running_llm_runs(self, *, item_id: str, generation: int) -> int:
        result = self._session.execute(
            update(LlmExtractionRun)
            .where(
                LlmExtractionRun.ingestion_item_id == item_id,
                LlmExtractionRun.generation == generation,
                LlmExtractionRun.status == "running",
            )
            .values(status="abandoned", completed_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        return result.rowcount
