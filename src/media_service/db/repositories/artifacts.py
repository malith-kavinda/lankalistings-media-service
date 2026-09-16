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

# How many times an insert may step over an attempt number another writer took first. The numbers
# come from an unlocked `max(attempt) + 1`, so two workers racing on one item -- a zombie holding an
# expired lease against its replacement -- can compute the same one.
MAX_ATTEMPT_COLLISIONS = 3


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
        """Persist an OCR result, yielding to whoever completed this generation first.

        Two different constraints can reject this insert, and they mean opposite things.

        The partial unique index on `status='completed'` means someone else finished this
        generation: their row is the one of record, and this caller resumes from it.

        The full `(item, generation, attempt)` constraint means only that the attempt *number* was
        taken -- both rows may be failures, with nothing completed. Yielding there would be wrong,
        so the insert simply takes the next number and tries again.
        """
        for _ in range(MAX_ATTEMPT_COLLISIONS):
            try:
                with savepoint(self._session):
                    self._session.add(extraction)
                    self._session.flush()
            except IntegrityError:
                _detach(self._session, extraction)
                winner = self.completed_ocr(
                    item_id=extraction.ingestion_item_id, generation=extraction.generation
                )
                if winner is not None:
                    return winner, False
                if not self._ocr_attempt_taken(extraction):
                    raise
                extraction.attempt = self.next_ocr_attempt(
                    item_id=extraction.ingestion_item_id, generation=extraction.generation
                )
                continue
            return extraction, True

        raise RuntimeError(
            f"Could not allocate an OCR attempt number for item {extraction.ingestion_item_id} "
            f"after {MAX_ATTEMPT_COLLISIONS} collisions."
        )

    def _ocr_attempt_taken(self, extraction: OcrExtraction) -> bool:
        return (
            self._session.scalar(
                select(OcrExtraction.id).where(
                    OcrExtraction.ingestion_item_id == extraction.ingestion_item_id,
                    OcrExtraction.generation == extraction.generation,
                    OcrExtraction.attempt == extraction.attempt,
                )
            )
            is not None
        )

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

        Losing *that* race is the cheap outcome: the winner's validated response is already on
        disk, so the loser resumes from it instead of calling the provider again.

        A collision on the attempt number alone is a different thing and is not a reason to yield
        -- see `record_ocr`, which resolves it the same way.
        """
        for _ in range(MAX_ATTEMPT_COLLISIONS):
            try:
                with savepoint(self._session):
                    self._session.add(run)
                    self._session.flush()
            except IntegrityError:
                _detach(self._session, run)
                winner = self.validated_llm_run(
                    item_id=run.ingestion_item_id, generation=run.generation
                )
                if winner is not None:
                    return winner, False
                if not self._llm_attempt_taken(run):
                    raise
                run.attempt = self.next_llm_attempt(
                    item_id=run.ingestion_item_id, generation=run.generation
                )
                continue
            return run, True

        raise RuntimeError(
            f"Could not allocate an LLM attempt number for item {run.ingestion_item_id} "
            f"after {MAX_ATTEMPT_COLLISIONS} collisions."
        )

    def _llm_attempt_taken(self, run: LlmExtractionRun) -> bool:
        return (
            self._session.scalar(
                select(LlmExtractionRun.id).where(
                    LlmExtractionRun.ingestion_item_id == run.ingestion_item_id,
                    LlmExtractionRun.generation == run.generation,
                    LlmExtractionRun.attempt == run.attempt,
                )
            )
            is not None
        )

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


def _detach(session: Session, instance: object) -> None:
    """Take a rejected insert out of the session so it can be retried with a new attempt number.

    Rolling back the savepoint usually evicts it already, so this has to tolerate an instance the
    session no longer holds -- `expunge` raises on one it does not know.
    """
    if instance in session:
        session.expunge(instance)
