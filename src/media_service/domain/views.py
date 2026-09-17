"""Read models: what a caller gets back from a service.

Services return these, never ORM rows. Two reasons, and the second is the one that bites.

An ORM instance is only meaningful while its session is open. Closing a unit of work expires every
instance it loaded, so a row returned to a caller raises `DetachedInstanceError` on the first
attribute read -- at the API layer, far from the code that produced it. A frozen dataclass has no
such lifetime.

And a mapped row is writable. Handing one to a route invites an assignment that looks like it
persisted something and did not, because no session is watching any more. These are read models:
there is nothing to assign to.

`current_stage` is derived here rather than stored, for the same reason it is not a column -- two
representations of one truth drift, and the drift is undetectable after the fact (PRD 12.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from media_service.db.tables import (
    Advertisement,
    AdvertisementProvenance,
    IngestionBatch,
    IngestionItem,
    MediaAsset,
)
from media_service.domain.item_state import BatchStatus, ItemStatus, Stage, stage_of


@dataclass(frozen=True, slots=True)
class ItemView:
    id: str
    batch_id: str
    item_index: int
    status: ItemStatus
    stage: Stage
    original_filename: str
    declared_content_type: str | None
    source_asset_id: str
    pipeline_generation: int
    attempt_count: int
    candidate_count: int
    warning_codes: tuple[str, ...]
    is_duplicate_in_batch: bool
    duplicate_of_item_id: str | None
    failed_stage: str | None
    error_code: str | None
    error_message: str | None
    run_after: datetime
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, item: IngestionItem) -> ItemView:
        status = ItemStatus(item.status)
        return cls(
            id=item.id,
            batch_id=item.batch_id,
            item_index=item.item_index,
            status=status,
            stage=stage_of(status),
            original_filename=item.original_filename,
            declared_content_type=item.declared_content_type,
            source_asset_id=item.source_asset_id,
            pipeline_generation=item.pipeline_generation,
            attempt_count=item.attempt_count,
            candidate_count=item.candidate_count,
            warning_codes=tuple(item.warning_codes or ()),
            is_duplicate_in_batch=item.is_duplicate_in_batch,
            duplicate_of_item_id=item.duplicate_of_item_id,
            failed_stage=item.failed_stage,
            error_code=item.error_code,
            error_message=item.error_message,
            run_after=item.run_after,
            created_at=item.created_at,
            updated_at=item.updated_at,
            completed_at=item.completed_at,
        )


@dataclass(frozen=True, slots=True)
class BatchView:
    id: str
    kind: str
    created_by: str
    status: BatchStatus
    counts: dict[str, int]
    total_items: int
    correlation_id: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    committed_at: datetime | None
    items: tuple[ItemView, ...] = ()
    # True when this batch was not created by the request that returned it, but replayed for a
    # repeated idempotency key. The caller needs to know: it decides 202 against 200.
    replayed: bool = False

    @classmethod
    def of(
        cls,
        batch: IngestionBatch,
        *,
        status: BatchStatus,
        counts: dict[str, int],
        items: tuple[ItemView, ...] = (),
        replayed: bool = False,
    ) -> BatchView:
        return cls(
            id=batch.id,
            kind=batch.kind,
            created_by=batch.created_by,
            status=status,
            counts=counts,
            total_items=batch.total_items,
            correlation_id=batch.correlation_id,
            created_at=batch.created_at,
            updated_at=batch.updated_at,
            completed_at=batch.completed_at,
            committed_at=batch.committed_at,
            items=items,
            replayed=replayed,
        )


@dataclass(frozen=True, slots=True)
class AssetView:
    id: str
    storage_key: str
    content_type: str
    byte_size: int
    checksum_sha256: str
    width: int | None
    height: int | None
    bytes_state: str
    created_at: datetime

    @classmethod
    def of(cls, asset: MediaAsset) -> AssetView:
        return cls(
            id=asset.id,
            storage_key=asset.storage_key,
            content_type=asset.content_type,
            byte_size=asset.byte_size,
            checksum_sha256=asset.checksum_sha256,
            width=asset.width,
            height=asset.height,
            bytes_state=asset.bytes_state,
            created_at=asset.created_at,
        )


@dataclass(frozen=True, slots=True)
class RetryOutcome:
    mode: str
    items: tuple[ItemView, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class CandidateView:
    """One row of the review queue.

    Flattens the advertisement and its provenance into a single shape, because the split between
    them is an implementation detail of where the data will eventually live (PRD 7.3) and a
    moderator has no use for it.
    """

    id: str
    provenance_id: str
    status: str
    candidate_state: str
    origin: str
    title: str
    description: str
    category: str
    location: str
    price: str
    phones: tuple[str, ...]
    language: str | None
    confidence_overall: float | None
    confidence_label: str
    warning_codes: tuple[str, ...]
    version: int
    batch_id: str
    item_id: str
    source_asset_id: str
    candidate_index: int | None
    generation: int
    created_at: datetime
    updated_at: datetime
    reviewed_at: datetime | None
    reviewer_id: str | None

    @classmethod
    def of(cls, provenance: AdvertisementProvenance, advertisement: Advertisement) -> CandidateView:
        return cls(
            id=advertisement.id,
            provenance_id=provenance.id,
            status=advertisement.status,
            candidate_state=provenance.candidate_state,
            origin=advertisement.origin,
            title=advertisement.title,
            description=advertisement.description,
            category=advertisement.category,
            location=advertisement.location,
            price=advertisement.price,
            phones=tuple(str(phone) for phone in advertisement.phones or ()),
            language=advertisement.language,
            confidence_overall=advertisement.confidence_overall,
            confidence_label=advertisement.extraction_confidence,
            # Both sides' warnings, merged and deduplicated in order. A moderator asking "what is
            # wrong with this one" does not care whether the pipeline flagged the page or the
            # candidate.
            warning_codes=_merged_warnings(provenance, advertisement),
            version=advertisement.version,
            batch_id=provenance.ingestion_batch_id,
            item_id=provenance.ingestion_item_id,
            source_asset_id=provenance.source_asset_id,
            candidate_index=provenance.candidate_index,
            generation=provenance.generation,
            created_at=advertisement.created_at,
            updated_at=advertisement.updated_at,
            reviewed_at=provenance.reviewed_at,
            reviewer_id=provenance.reviewer_id,
        )


@dataclass(frozen=True, slots=True)
class EvidenceView:
    """Why the pipeline believes what it believes (FR-REV-003).

    The OCR text and block ids are what let a reviewer check an extracted field against the page
    instead of against their own guess, which is the difference between review and rubber-stamping.
    """

    source_text: str
    ocr_text: str
    ocr_extraction_id: str | None
    llm_extraction_run_id: str | None
    source_block_ids: tuple[int, ...]
    blocks: tuple[dict[str, Any], ...]
    field_confidence: dict[str, float]
    warnings: tuple[str, ...]
    extracted_values: dict[str, Any]
    accepted_values: dict[str, Any] | None
    provider: str | None
    model: str | None
    ocr_engine: str | None
    ocr_languages: str | None
    # The page size the block coordinates are expressed in. Preprocessing deskews and rescales, so
    # this is the *ocr_input* derivative's size, not the original scan's -- an overlay drawn against
    # the original with these numbers points at the wrong text.
    ocr_width: int | None = None
    ocr_height: int | None = None
    ocr_input_derivative_id: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateDetailView:
    """A candidate with everything needed to rule on it without a second request.

    `siblings` carries every candidate from the same image in order, which is what makes
    "Ad 2 of 5 from page-3.png" and the previous/next controls possible (PRD 14.3).
    """

    candidate: CandidateView
    evidence: EvidenceView
    item: ItemView
    siblings: tuple[str, ...]
    position: int
    sibling_count: int
    source_filename: str


@dataclass(frozen=True, slots=True)
class CandidatePageView:
    candidates: tuple[CandidateView, ...]
    total: int
    limit: int
    offset: int
    counts: dict[str, int]


def _merged_warnings(
    provenance: AdvertisementProvenance, advertisement: Advertisement
) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for code in list(advertisement.warning_codes or ()) + list(provenance.warning_codes or ()):
        seen.setdefault(str(code), None)
    return tuple(seen)
