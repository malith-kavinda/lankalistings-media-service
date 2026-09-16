"""Runs one claimed item through the pipeline.

The shape of this module is set by one rule: **a transaction is never open across a long call.**
Tesseract is a subprocess and a provider call is a network round trip; either can take tens of
seconds, and a transaction held across one pins a connection, holds the row locks the reaper and the
progress endpoint want, and shows up as `idle in transaction`. So each stage is read-run-write --
open a unit of work to learn what to do, close it, do the slow thing, open another to record the
result.

There is one deliberate exception, and it removes a failure mode rather than adding one: the
candidate rows, the item's status, its candidate count, and the batch projection are written in a
**single commit**. A worker cannot die between creating candidates and recording that it did
(FR-CAN-003), because there is no moment when one is durable and the other is not.

Every write carries the claim token. If the lease expired and another worker took the item, the
compare-and-swap affects no rows, `ItemClaimLostError` is raised, and this worker stops touching
work it no longer owns.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any, Final

from media_service.api.errors import ItemClaimLostError
from media_service.config import Settings
from media_service.db.base import utcnow
from media_service.db.tables import LlmExtractionRun, MediaDerivative, OcrExtraction
from media_service.db.uow import UnitOfWorkFactory
from media_service.domain.categories import CATEGORY_CATALOG_VERSION
from media_service.domain.ids import new_derivative_id, new_llm_run_id, new_ocr_extraction_id
from media_service.domain.item_state import ItemStatus, stage_of
from media_service.domain.listings import CandidateDraft, ListingGateway
from media_service.jobs.resume import ResumePlan, plan_resume
from media_service.jobs.stages import (
    ExtractionStep,
    PreprocessStep,
    StageError,
    to_stage_error,
)
from media_service.ocr import quality
from media_service.ocr.preprocess import PreprocessError
from media_service.ocr.protocol import OcrProvider
from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrLine, OcrResult
from media_service.storage import DerivativePurpose, FilesystemAssetStore, derivative_key
from media_service.storage.filesystem import params_hash as compute_params_hash

logger = logging.getLogger(__name__)

RETRY_BACKOFF_BASE_SECONDS: Final = 2.0
RETRY_BACKOFF_MAX_SECONDS: Final = 60.0

# Recorded on Phase 1's runs so they are distinguishable from model output after the fact. Phase 3
# replaces the extractor, not this bookkeeping.
RULE_BASED_PROVIDER: Final = "rule_based"
RULE_BASED_MODEL: Final = "rule_based/v0"
RULE_BASED_PROMPT_VERSION: Final = "none"


@dataclass(frozen=True, slots=True)
class ItemContext:
    """Everything the stages need, read once so the row is not held open across them."""

    item_id: str
    batch_id: str
    claim_token: str
    generation: int
    asset_id: str
    storage_key: str
    content_type: str


class ItemRunner:
    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        store: FilesystemAssetStore,
        settings: Settings,
        preprocess: PreprocessStep,
        ocr: OcrProvider,
        extraction: ExtractionStep,
        gateway: ListingGateway,
        ocr_semaphore: threading.Semaphore | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._store = store
        self._settings = settings
        # OCR concurrency is gated separately from worker concurrency, because they answer
        # different questions. `MAX_WORKER_CONCURRENCY` is how many items may be in flight;
        # `OCR_CONCURRENCY` is how many may be inside the engine at once -- which is what an
        # operator turns down to protect a CPU-bound or rate-limited recogniser without throttling
        # the rest of the pipeline.
        self._ocr_semaphore = ocr_semaphore or threading.Semaphore(settings.ocr_concurrency)
        self._preprocess = preprocess
        self._ocr = ocr
        self._extraction = extraction
        self._gateway = gateway
        # The derivative is keyed by the settings that produced it, so changing a preprocessing
        # option produces a new file rather than overwriting one an existing extraction cites.
        self._params_hash = compute_params_hash(
            DerivativePurpose.OCR_INPUT, preprocess.version, preprocess.params()
        )

    def run(self, item_id: str, *, claim_token: str) -> None:
        """Take one claimed item as far as it goes. Never raises for an item-level failure.

        That promise covers `_begin` as well. An item whose asset row is missing, or whose first
        read hits a database blip, fails *before* there is a context to fail it with -- and if that
        escaped, a pool worker would drop it silently (nothing awaits the future) while an inline
        dispatcher would abort the rest of the batch, with no poller to pick the remainder up.
        """
        try:
            context, plan = self._begin(item_id, claim_token)
        except ItemClaimLostError:
            return
        except StageError as error:
            self._fail(item_id, claim_token, error)
            return
        except Exception as error:  # noqa: BLE001 - an unknown fault must not lose the item
            logger.exception("Unhandled failure starting item %s", item_id)
            self._fail(item_id, claim_token, StageError("INTERNAL_ERROR", str(error)))
            return

        try:
            derivative_id = self._run_preprocess(context, plan)
            extraction_id, result = self._run_ocr(context, plan, derivative_id)
            run_id, drafts = self._run_extraction(context, plan, extraction_id, result)
            self._finish(context, extraction_id=extraction_id, run_id=run_id, drafts=drafts)
        except ItemClaimLostError:
            # The lease expired and someone else owns this item. Stop, silently: the new owner has
            # already resumed from whatever artifacts this attempt managed to commit.
            logger.info("Claim lost for item %s; abandoning this attempt.", item_id)
        except StageError as error:
            self._fail(item_id, claim_token, error)
        except Exception as error:  # noqa: BLE001 - an unknown fault must not kill the worker
            logger.exception("Unhandled failure processing item %s", item_id)
            self._fail(item_id, claim_token, StageError("INTERNAL_ERROR", str(error)))

    # -- stages --------------------------------------------------------------------------------

    def _begin(self, item_id: str, claim_token: str) -> tuple[ItemContext, ResumePlan]:
        with self._unit_of_work() as unit:
            item = unit.items.get(item_id)
            if item is None or item.claim_token != claim_token:
                raise ItemClaimLostError(item_id)

            asset = unit.assets.get(item.source_asset_id)
            if asset is None:  # pragma: no cover - the foreign key forbids it
                raise StageError(
                    "ASSET_MISSING", "The item's source asset is gone.", retryable=False
                )

            plan = plan_resume(unit, item, params_hash=self._params_hash)
            context = ItemContext(
                item_id=item.id,
                batch_id=item.batch_id,
                claim_token=claim_token,
                generation=item.pipeline_generation,
                asset_id=asset.id,
                storage_key=asset.storage_key,
                content_type=asset.content_type,
            )
            return context, plan

    def _run_preprocess(self, context: ItemContext, plan: ResumePlan) -> str:
        if plan.derivative_id is not None:
            # The derivative is keyed by the parameters that produced it, so an identical re-run
            # would rewrite an identical file. Skipping is free correctness, not an optimisation.
            return plan.derivative_id

        original = self._read_bytes(context.storage_key)
        try:
            result = self._preprocess.run(original)
        except PreprocessError as error:
            raise to_stage_error(error) from error
        key = derivative_key(
            context.asset_id, DerivativePurpose.OCR_INPUT, self._params_hash, result.content_type
        )
        self._store.write_bytes(result.image_bytes, key)

        with self._unit_of_work() as unit:
            derivative, _ = unit.assets.get_or_create_derivative(
                MediaDerivative(
                    id=new_derivative_id(),
                    source_asset_id=context.asset_id,
                    purpose=DerivativePurpose.OCR_INPUT.value,
                    storage_key=key,
                    content_type=result.content_type,
                    byte_size=len(result.image_bytes),
                    width=result.width,
                    height=result.height,
                    preprocessing_version=result.version,
                    params=result.params,
                    params_hash=self._params_hash,
                )
            )
            derivative_id = derivative.id
            unit.commit()
        return derivative_id

    def _run_ocr(
        self, context: ItemContext, plan: ResumePlan, derivative_id: str
    ) -> tuple[str, OcrResult | None]:
        self._advance(context, ItemStatus.PREPROCESSING, ItemStatus.OCR_PROCESSING)

        if plan.ocr_extraction_id is not None:
            return plan.ocr_extraction_id, None

        with self._unit_of_work() as unit:
            attempt = unit.artifacts.next_ocr_attempt(
                item_id=context.item_id, generation=context.generation
            )

        image_bytes = self._read_derivative(derivative_id)
        try:
            with self._ocr_semaphore:
                result = self._ocr.extract(image_bytes, content_type="image/png")
        except StageError:
            raise
        except Exception as error:  # noqa: BLE001 - translated, not swallowed
            raise to_stage_error(error) from error

        with self._unit_of_work() as unit:
            extraction, _ = unit.artifacts.record_ocr(
                OcrExtraction(
                    id=new_ocr_extraction_id(),
                    ingestion_item_id=context.item_id,
                    source_asset_id=context.asset_id,
                    input_derivative_id=derivative_id,
                    generation=context.generation,
                    attempt=attempt,
                    # `empty` is a real outcome, not a failure: the page was read and carried
                    # nothing worth extracting from.
                    status="empty" if quality.is_empty(result.text) else "completed",
                    raw_text=result.text,
                    blocks=result.block_documents(),
                    block_count=result.block_count,
                    max_block_id=result.max_block_id,
                    engine=result.engine,
                    engine_version=result.engine_version,
                    traineddata_version=result.traineddata_version,
                    languages=result.languages,
                    mean_confidence=result.mean_confidence,
                    low_confidence=result.low_confidence,
                    width=result.width,
                    height=result.height,
                    preprocessing_version=result.preprocess_version,
                    duration_ms=result.duration_ms,
                    completed_at=utcnow(),
                )
            )
            extraction_id = extraction.id
            unit.commit()
        return extraction_id, result

    def _run_extraction(
        self,
        context: ItemContext,
        plan: ResumePlan,
        extraction_id: str,
        result: OcrResult | None,
    ) -> tuple[str, list[CandidateDraft]]:
        self._advance(context, ItemStatus.OCR_PROCESSING, ItemStatus.LLM_PROCESSING)

        if plan.llm_run_id is not None:
            # The call already succeeded and its output is on disk. Rebuilding the drafts from the
            # stored response is what stops a crash after a paid call from paying for it twice.
            with self._unit_of_work() as unit:
                run = unit.artifacts.get_llm_run(plan.llm_run_id)
                return run.id, _drafts_from(run.validated_response or {})

        with self._unit_of_work() as unit:
            attempt = unit.artifacts.next_llm_attempt(
                item_id=context.item_id, generation=context.generation
            )
            if result is None:
                stored = unit.artifacts.get_ocr(extraction_id)
                result = _result_from(stored)

        drafts = self._extraction.run(result)
        response = _response_from(drafts)

        with self._unit_of_work() as unit:
            run, created = unit.artifacts.record_llm_run(
                LlmExtractionRun(
                    id=new_llm_run_id(),
                    ingestion_item_id=context.item_id,
                    ocr_extraction_id=extraction_id,
                    generation=context.generation,
                    attempt=attempt,
                    attempt_kind="primary",
                    provider=RULE_BASED_PROVIDER,
                    model=RULE_BASED_MODEL,
                    prompt_version=RULE_BASED_PROMPT_VERSION,
                    schema_version="1.0",
                    category_catalog_version=CATEGORY_CATALOG_VERSION,
                    request_hash=_request_hash(result.text),
                    status="validated",
                    validated_response=response,
                    candidate_count=len(drafts),
                    latency_ms=result.duration_ms,
                    completed_at=utcnow(),
                )
            )
            run_id = run.id
            if not created:
                # Another worker validated this generation first. Its answer is the one of record.
                drafts = _drafts_from(run.validated_response or {})
            unit.commit()

        return run_id, drafts

    def _finish(
        self,
        context: ItemContext,
        *,
        extraction_id: str,
        run_id: str,
        drafts: list[CandidateDraft],
    ) -> None:
        """Candidates and the item's own status, in one commit."""
        target = ItemStatus.AWAITING_REVIEW if drafts else ItemStatus.NO_ADS

        with self._unit_of_work() as unit:
            item = unit.items.get(context.item_id)
            if item is None or item.claim_token != context.claim_token:
                raise ItemClaimLostError(context.item_id)

            self._gateway.create_candidates(
                unit,
                item=item,
                drafts=drafts,
                idempotency_key=_candidate_key(context, run_id),
                origin="ocr_heuristic",
                ocr_extraction_id=extraction_id,
                llm_extraction_run_id=run_id,
            )
            unit.items.transition(
                item,
                target=target,
                expected=ItemStatus.LLM_PROCESSING,
                claim_token=context.claim_token,
                release_claim=True,
                candidate_count=unit.candidates.count_for(
                    context.item_id, generation=context.generation
                ),
                error_code=None,
                error_message=None,
                failed_stage=None,
            )
            unit.commit()

    # -- failure -------------------------------------------------------------------------------

    def _fail(self, item_id: str, claim_token: str, error: StageError) -> None:
        """Requeue with backoff, or park the item for a person to look at.

        Takes ids rather than a context, because the caller may not have one: a failure inside
        `_begin` happens before the context is built, and that item still has to be marked.

        This is the last handler in the chain, so it swallows its own failures too. A database
        problem here would otherwise escape `run()` and leave the item claimed and silent until its
        lease expired -- the outcome this method exists to prevent.
        """
        try:
            with self._unit_of_work() as unit:
                item = unit.items.get(item_id)
                if item is None or item.claim_token != claim_token:
                    return

                generation = item.pipeline_generation
                unit.artifacts.abandon_running_ocr(item_id=item_id, generation=generation)
                unit.artifacts.abandon_running_llm_runs(item_id=item_id, generation=generation)

                current = ItemStatus(item.status)
                exhausted = (
                    not error.retryable or item.attempt_count >= self._settings.max_item_attempts
                )
                shared = {
                    "expected": current,
                    "claim_token": claim_token,
                    "release_claim": True,
                    "error_code": error.code,
                    "error_message": error.message,
                    "failed_stage": stage_of(current).value,
                }
                if exhausted:
                    unit.items.transition(item, target=ItemStatus.NEEDS_ATTENTION, **shared)
                else:
                    unit.items.transition(
                        item,
                        target=ItemStatus.UPLOADED,
                        run_after=_backoff_from(item.attempt_count),
                        **shared,
                    )
                unit.commit()
        except ItemClaimLostError:
            logger.info("Claim lost while failing item %s.", item_id)
        except Exception:  # noqa: BLE001 - the reaper is the backstop if even this fails
            logger.exception("Could not record the failure of item %s", item_id)

    # -- helpers -------------------------------------------------------------------------------

    def _advance(self, context: ItemContext, current: ItemStatus, target: ItemStatus) -> None:
        with self._unit_of_work() as unit:
            item = unit.items.get(context.item_id)
            if item is None:  # pragma: no cover - deleted mid-run
                raise ItemClaimLostError(context.item_id)
            if ItemStatus(item.status) is target:
                return
            unit.items.transition(
                item,
                target=target,
                expected=current,
                claim_token=context.claim_token,
            )
            # The lease is extended at each boundary rather than from a background thread: a stage
            # is the unit of work that can hang, and this is the moment we know one finished.
            unit.items.heartbeat(
                context.item_id,
                claim_token=context.claim_token,
                lease_seconds=self._settings.lease_seconds,
            )
            unit.commit()

    def _read_bytes(self, storage_key: str) -> bytes:
        try:
            return self._store.read_bytes(storage_key)
        except FileNotFoundError as exc:
            raise StageError(
                "MEDIA_BYTES_UNAVAILABLE",
                f"The stored bytes for {storage_key} are missing.",
                retryable=False,
            ) from exc

    def _read_derivative(self, derivative_id: str) -> bytes:
        with self._unit_of_work() as unit:
            derivative = unit.session.get(MediaDerivative, derivative_id)
            key = derivative.storage_key if derivative else None
        if key is None:  # pragma: no cover - written moments earlier
            raise StageError("DERIVATIVE_MISSING", "The OCR input is gone.", retryable=True)
        return self._read_bytes(key)


def _backoff_from(attempt_count: int) -> datetime:
    """Exponential, capped. A failing dependency should not be hammered while it recovers."""
    delay = min(
        RETRY_BACKOFF_BASE_SECONDS * (2 ** max(attempt_count - 1, 0)), RETRY_BACKOFF_MAX_SECONDS
    )
    return utcnow() + timedelta(seconds=delay)


def _request_hash(text: str) -> str:
    material = "|".join([RULE_BASED_PROVIDER, RULE_BASED_MODEL, RULE_BASED_PROMPT_VERSION, text])
    return sha256(material.encode()).hexdigest()


def _candidate_key(context: ItemContext, run_id: str) -> str:
    """Identifies this candidate set, so a remote gateway can make it land exactly once."""
    material = f"{context.item_id}|{context.generation}|{run_id}"
    return sha256(material.encode()).hexdigest()


def _response_from(drafts: list[CandidateDraft]) -> dict[str, Any]:
    """The extractor's output, stored in the shape Phase 3's schema will produce."""
    return {
        "advertisements": [
            {
                "index": draft.index,
                "title": draft.title,
                "description": draft.description,
                "category": draft.category,
                "location": draft.location,
                "price": draft.price,
                "phones": list(draft.phones),
                "language": draft.language,
                "confidence": draft.confidence,
                "confidence_label": draft.confidence_label,
                "source_text": draft.source_text,
                "source_block_ids": list(draft.source_block_ids),
                "field_confidence": dict(draft.field_confidence),
                "warnings": list(draft.warnings),
                "warning_codes": list(draft.warning_codes),
                "extracted_values": dict(draft.extracted_values),
            }
            for draft in drafts
        ]
    }


def _drafts_from(response: dict[str, Any]) -> list[CandidateDraft]:
    return [
        CandidateDraft(
            index=int(entry.get("index", position)),
            title=entry.get("title", ""),
            description=entry.get("description", ""),
            category=entry.get("category", "other"),
            location=entry.get("location", ""),
            price=entry.get("price", ""),
            phones=tuple(entry.get("phones", ())),
            language=entry.get("language"),
            confidence=entry.get("confidence"),
            confidence_label=entry.get("confidence_label", "medium"),
            source_text=entry.get("source_text", ""),
            source_block_ids=tuple(entry.get("source_block_ids", ())),
            field_confidence=dict(entry.get("field_confidence", {})),
            warnings=tuple(entry.get("warnings", ())),
            warning_codes=tuple(entry.get("warning_codes", ())),
            extracted_values=dict(entry.get("extracted_values", {})),
        )
        for position, entry in enumerate(response.get("advertisements", []))
    ]


def _result_from(extraction: OcrExtraction | None) -> OcrResult:
    """Rebuild a result from the stored row, for a resume that skipped the OCR call.

    The blocks come back from the database, not from a re-run, because **the persisted ids are
    authoritative**. A candidate cites `source_block_ids`; recomputing them could renumber a region
    and silently point an existing citation at different text.
    """
    if extraction is None:  # pragma: no cover - the caller checked
        raise StageError("OCR_MISSING", "The OCR result is gone.", retryable=True)
    return OcrResult(
        text=extraction.raw_text,
        blocks=_blocks_from(extraction.blocks),
        provider=extraction.engine,
        engine=extraction.engine,
        engine_version=extraction.engine_version,
        traineddata_version=extraction.traineddata_version,
        languages=extraction.languages,
        mean_confidence=extraction.mean_confidence,
        low_confidence=extraction.low_confidence,
        width=extraction.width,
        height=extraction.height,
        preprocess_version=extraction.preprocessing_version,
        duration_ms=extraction.duration_ms or 0,
    )


def _blocks_from(documents: list[Any] | None) -> tuple[OcrBlock, ...]:
    """Read back the JSONB block document written by `OcrBlock.as_document`."""
    blocks: list[OcrBlock] = []
    for document in documents or []:
        box = document.get("box")
        blocks.append(
            OcrBlock(
                id=int(document.get("id", len(blocks) + 1)),
                text=str(document.get("text", "")),
                confidence=document.get("confidence"),
                box=BoundingBox.from_list(box) if box else None,
                box_source=BoxSource(document.get("box_source", BoxSource.NONE.value)),
                lines=tuple(
                    OcrLine(text=line) for line in str(document.get("text", "")).splitlines()
                ),
                source_ref=str(document.get("source_ref", "")),
                detector=document.get("detector"),
            )
        )
    return tuple(blocks)
