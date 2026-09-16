"""Where to restart an item whose worker died (FR-JOB-006).

One pure function over plain reads: no side effects, no writes, nothing to mock. It answers a
single question -- *what is the first stage whose output is missing?* -- by walking the artifacts in
pipeline order and stopping at the first gap.

The point is cost, not tidiness. Each surviving artifact is work that is not repeated:

* the `ocr_input` derivative means preprocessing is skipped, for free, because the file is already
  keyed by the exact parameters that produced it;
* a completed OCR row means Tesseract does not run again;
* a **validated LLM run means the provider is not called again** -- the crash that happens between
  a successful paid call and the candidate rows is the expensive one, and this is what makes it
  cost nothing.

`start_at` is the first stage that must actually execute. `DONE` means every artifact is present and
a re-run has nothing to do, which is how a retry of an item that already finished becomes a no-op
rather than a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass

from media_service.db.tables import IngestionItem
from media_service.db.uow import UnitOfWork
from media_service.domain.item_state import Stage
from media_service.storage import DerivativePurpose


@dataclass(frozen=True, slots=True)
class ResumePlan:
    start_at: Stage
    derivative_id: str | None = None
    ocr_extraction_id: str | None = None
    llm_run_id: str | None = None
    candidate_count: int = 0

    @property
    def needs_preprocessing(self) -> bool:
        return self.start_at is Stage.PREPROCESS

    @property
    def needs_ocr(self) -> bool:
        return self.start_at in (Stage.PREPROCESS, Stage.OCR)

    @property
    def needs_extraction(self) -> bool:
        return self.llm_run_id is None

    @property
    def is_complete(self) -> bool:
        return self.start_at is Stage.DONE


def plan_resume(unit: UnitOfWork, item: IngestionItem, *, params_hash: str) -> ResumePlan:
    generation = item.pipeline_generation

    derivative = unit.assets.get_derivative(
        asset_id=item.source_asset_id,
        purpose=DerivativePurpose.OCR_INPUT.value,
        params_hash=params_hash,
    )
    if derivative is None:
        return ResumePlan(start_at=Stage.PREPROCESS)

    extraction = unit.artifacts.completed_ocr(item_id=item.id, generation=generation)
    if extraction is None:
        return ResumePlan(start_at=Stage.OCR, derivative_id=derivative.id)

    run = unit.artifacts.validated_llm_run(item_id=item.id, generation=generation)
    if run is None:
        return ResumePlan(
            start_at=Stage.LLM,
            derivative_id=derivative.id,
            ocr_extraction_id=extraction.id,
        )

    # The provider call is already paid for. What may still be missing is the candidate rows, and
    # the run's own count is the record of how many there should be.
    candidates = unit.candidates.count_for(item.id, generation=generation)
    expected = run.candidate_count or 0
    return ResumePlan(
        start_at=Stage.DONE if candidates >= expected else Stage.LLM,
        derivative_id=derivative.id,
        ocr_extraction_id=extraction.id,
        llm_run_id=run.id,
        candidate_count=candidates,
    )
