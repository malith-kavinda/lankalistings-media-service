"""The seam where extraction hands candidates to whoever owns advertisements.

Today that owner is a table in this database. The target architecture gives it to a separate
listing service (PRD 7.3, invariant 14), and the whole point of this module is that the move is an
adapter swap rather than a rewrite.

Two decisions make that true.

`create_candidates` takes **the whole candidate list plus an idempotency key**, not one candidate at
a time. A remote implementation must be able to make the entire set land exactly once; a
per-candidate call cannot, because a crash halfway through leaves a partial set nobody can identify
as partial.

The provenance rows -- the ones carrying the uniqueness that makes a retry safe -- are written by
this service either way. They reference the advertisement by an opaque string with no foreign key,
so an advertisement that lives somewhere else changes nothing about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from media_service.db.tables import Advertisement, AdvertisementProvenance, IngestionItem
from media_service.db.uow import UnitOfWork
from media_service.domain.ids import new_advertisement_id, new_provenance_id


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    """One advertisement an extractor believes it found in an image.

    A draft, not a listing. It becomes publicly readable only when a person approves it
    (invariant 2), which is why nothing here can set a status.
    """

    index: int
    title: str = ""
    description: str = ""
    category: str = "other"
    location: str = ""
    price: str = ""
    phones: tuple[str, ...] = ()
    language: str | None = None
    confidence: float | None = None
    confidence_label: str = "medium"
    source_text: str = ""
    source_block_ids: tuple[int, ...] = ()
    field_confidence: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    extracted_values: dict[str, Any] = field(default_factory=dict)


class ListingGateway(Protocol):
    def create_candidates(
        self,
        unit: UnitOfWork,
        *,
        item: IngestionItem,
        drafts: list[CandidateDraft],
        idempotency_key: str,
        origin: str,
        ocr_extraction_id: str | None = None,
        llm_extraction_run_id: str | None = None,
    ) -> list[AdvertisementProvenance]:
        """Create every candidate for one item, or return the set already created for that key."""


class LocalListingGateway:
    """Writes candidates to the advertisements table in this database.

    Deliberately participates in the caller's unit of work rather than opening its own. Candidate
    rows, the item's status, its candidate count, and the batch projection have to commit together
    -- that is what makes "worker died after creating candidates but before updating the item"
    impossible rather than merely unlikely (FR-CAN-003).
    """

    def create_candidates(
        self,
        unit: UnitOfWork,
        *,
        item: IngestionItem,
        drafts: list[CandidateDraft],
        idempotency_key: str,
        origin: str,
        ocr_extraction_id: str | None = None,
        llm_extraction_run_id: str | None = None,
    ) -> list[AdvertisementProvenance]:
        created: list[AdvertisementProvenance] = []

        for draft in drafts:
            advertisement = Advertisement(
                id=new_advertisement_id(),
                title=draft.title,
                description=draft.description,
                category=draft.category,
                location=draft.location,
                price=draft.price,
                phones=list(draft.phones),
                language=draft.language,
                # Never `active`. No machine-generated advertisement becomes publicly readable
                # without a moderator approving it (invariant 2, AC-010).
                status="pending",
                origin=origin,
                source_text=draft.source_text,
                extraction_confidence=draft.confidence_label,
                confidence_overall=draft.confidence,
                warning_codes=list(draft.warning_codes),
                image_asset_id=item.source_asset_id,
                submitted_at=None,
            )
            unit.session.add(advertisement)
            unit.flush()

            provenance, was_created = unit.candidates.record(
                AdvertisementProvenance(
                    id=new_provenance_id(),
                    advertisement_id=advertisement.id,
                    ingestion_batch_id=item.batch_id,
                    ingestion_item_id=item.id,
                    source_asset_id=item.source_asset_id,
                    ocr_extraction_id=ocr_extraction_id,
                    llm_extraction_run_id=llm_extraction_run_id,
                    generation=item.pipeline_generation,
                    candidate_index=draft.index,
                    source_block_ids=list(draft.source_block_ids),
                    field_confidence=dict(draft.field_confidence),
                    warnings=list(draft.warnings),
                    warning_codes=list(draft.warning_codes),
                    extracted_values=dict(draft.extracted_values),
                    candidate_fingerprint=idempotency_key,
                )
            )
            if not was_created:
                # A concurrent worker already created this candidate. Its advertisement is the
                # real one; drop the row we speculatively added rather than leaving an orphan.
                unit.session.delete(advertisement)
                unit.flush()

            created.append(provenance)

        return created
