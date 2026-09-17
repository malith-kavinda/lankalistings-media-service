"""Wire shapes for the review endpoints (PRD 13.2).

`status` is translated on the way out, not stored translated. The database holds the target
vocabulary (`pending`); the prototype portal reads `pending_review`, and PRD 10.3 keeps that name
alive behind `ADVERTISEMENT_STATUS_WIRE=legacy` until the portal moves. Doing the mapping here means
one function decides it, rather than every query needing to remember which vocabulary it is in.

Edit requests distinguish **absent** from **empty**. `None` means "I did not touch this field";
`""` means "I cleared it". A schema that defaulted absent fields to empty strings would let a portal
that sends only the field it changed silently blank the other five.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from media_service.domain.listings import ReviewerEdits
from media_service.domain.review import MAX_NOTE_LENGTH, REJECTION_REASONS
from media_service.domain.views import CandidateDetailView, CandidatePageView, CandidateView

LEGACY_STATUS_WIRE = {"pending": "pending_review"}


def wire_status(status: str, *, mode: str) -> str:
    if mode == "legacy":
        return LEGACY_STATUS_WIRE.get(status, status)
    return status


def stored_status(status: str, *, mode: str) -> str:
    """The inverse, for a client filtering by whatever vocabulary it was given."""
    if mode == "legacy":
        for stored, wire in LEGACY_STATUS_WIRE.items():
            if wire == status:
                return stored
    return status


class CandidateResponse(BaseModel):
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
    phones: list[str] = Field(default_factory=list)
    language: str | None
    confidence_overall: float | None
    confidence_label: str
    warning_codes: list[str] = Field(default_factory=list)
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
    def of(cls, candidate: CandidateView, *, status_wire: str) -> CandidateResponse:
        return cls(
            id=candidate.id,
            provenance_id=candidate.provenance_id,
            status=wire_status(candidate.status, mode=status_wire),
            candidate_state=candidate.candidate_state,
            origin=candidate.origin,
            title=candidate.title,
            description=candidate.description,
            category=candidate.category,
            location=candidate.location,
            price=candidate.price,
            phones=list(candidate.phones),
            language=candidate.language,
            confidence_overall=candidate.confidence_overall,
            confidence_label=candidate.confidence_label,
            warning_codes=list(candidate.warning_codes),
            version=candidate.version,
            batch_id=candidate.batch_id,
            item_id=candidate.item_id,
            source_asset_id=candidate.source_asset_id,
            candidate_index=candidate.candidate_index,
            generation=candidate.generation,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            reviewed_at=candidate.reviewed_at,
            reviewer_id=candidate.reviewer_id,
        )


class EvidenceResponse(BaseModel):
    source_text: str
    ocr_text: str
    ocr_extraction_id: str | None
    llm_extraction_run_id: str | None
    source_block_ids: list[int] = Field(default_factory=list)
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    extracted_values: dict[str, Any] = Field(default_factory=dict)
    accepted_values: dict[str, Any] | None
    provider: str | None
    model: str | None
    ocr_engine: str | None
    ocr_languages: str | None


class CandidateDetailResponse(BaseModel):
    candidate: CandidateResponse
    evidence: EvidenceResponse
    item_id: str
    batch_id: str
    source_filename: str
    source_asset_id: str
    item_status: str
    item_stage: str
    siblings: list[str] = Field(default_factory=list)
    position: int
    sibling_count: int

    @classmethod
    def of(cls, detail: CandidateDetailView, *, status_wire: str) -> CandidateDetailResponse:
        return cls(
            candidate=CandidateResponse.of(detail.candidate, status_wire=status_wire),
            evidence=EvidenceResponse(
                source_text=detail.evidence.source_text,
                ocr_text=detail.evidence.ocr_text,
                ocr_extraction_id=detail.evidence.ocr_extraction_id,
                llm_extraction_run_id=detail.evidence.llm_extraction_run_id,
                source_block_ids=list(detail.evidence.source_block_ids),
                blocks=[dict(block) for block in detail.evidence.blocks],
                field_confidence=dict(detail.evidence.field_confidence),
                warnings=list(detail.evidence.warnings),
                extracted_values=dict(detail.evidence.extracted_values),
                accepted_values=detail.evidence.accepted_values,
                provider=detail.evidence.provider,
                model=detail.evidence.model,
                ocr_engine=detail.evidence.ocr_engine,
                ocr_languages=detail.evidence.ocr_languages,
            ),
            item_id=detail.item.id,
            batch_id=detail.item.batch_id,
            source_filename=detail.source_filename,
            source_asset_id=detail.item.source_asset_id,
            item_status=detail.item.status.value,
            item_stage=detail.item.stage.value,
            siblings=list(detail.siblings),
            position=detail.position,
            sibling_count=detail.sibling_count,
        )


class CandidatePageResponse(BaseModel):
    candidates: list[CandidateResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
    counts: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def of(cls, page: CandidatePageView, *, status_wire: str) -> CandidatePageResponse:
        return cls(
            candidates=[
                CandidateResponse.of(candidate, status_wire=status_wire)
                for candidate in page.candidates
            ],
            total=page.total,
            limit=page.limit,
            offset=page.offset,
            counts={
                wire_status(status, mode=status_wire): count
                for status, count in page.counts.items()
            },
        )


class CandidateEditRequest(BaseModel):
    """Absent means untouched; present-and-empty means cleared."""

    title: str | None = Field(default=None, max_length=200)
    description: str | None = None
    category: str | None = Field(default=None, max_length=48)
    location: str | None = Field(default=None, max_length=120)
    price: str | None = Field(default=None, max_length=64)
    phones: list[str] | None = None
    version: int | None = None

    def to_edits(self) -> ReviewerEdits:
        return ReviewerEdits(
            title=self.title,
            description=self.description,
            category=self.category,
            location=self.location,
            price=self.price,
            phones=tuple(self.phones) if self.phones is not None else None,
        )

    @property
    def has_changes(self) -> bool:
        return any(
            value is not None
            for value in (
                self.title,
                self.description,
                self.category,
                self.location,
                self.price,
                self.phones,
            )
        )


class ApproveRequest(BaseModel):
    """Corrections and the decision in one request.

    Optional: approving straight from the queue with nothing to fix is the common case. Sending
    edits here rather than through a preceding PATCH is what makes the pair atomic.
    """

    edits: CandidateEditRequest | None = None
    version: int | None = None


class RejectRequest(BaseModel):
    reason_code: str
    note: str | None = Field(default=None, max_length=MAX_NOTE_LENGTH)
    version: int | None = None


class RejectionReasonResponse(BaseModel):
    code: str
    description: str

    @classmethod
    def catalog(cls) -> list[RejectionReasonResponse]:
        """Published so the portal's dropdown cannot drift from the server's closed set."""
        return [
            cls(code=code, description=description)
            for code, description in sorted(REJECTION_REASONS.items())
        ]
