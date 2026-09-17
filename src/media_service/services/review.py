"""Human review: the step that turns a candidate into a listing, or throws it away.

One decision shapes everything here. **Approve applies the reviewer's edits and performs the
transition in a single transaction.** The portal previously issued a PATCH followed by a POST, and
a failure between them left the edits saved with the candidate still pending and nothing on screen
saying so -- the reviewer's correction silently became someone else's problem. A single call cannot
land halfway.

Every mutating path reads its candidate with `lock()`, which holds the row for the rest of the
transaction. The `version` column catches a reviewer working from a stale screen; the lock catches
two requests racing, which a version read outside a lock cannot (both would read the same number
and both would pass). They answer different questions and both are needed.

Validation runs against the candidate **after** the edits are applied, never against what the model
produced. The reviewer is there precisely to fill in what the pipeline could not, so checking the
pre-edit values would reject exactly the corrections that make approval possible (FR-REV-008).
"""

from __future__ import annotations

import logging
from typing import Any

from media_service.api.errors import (
    AdvertisementNotFoundError,
    InvalidReviewActionError,
    ServiceError,
    VersionConflictError,
)
from media_service.db.repositories.review import DEFAULT_LIMIT, ReviewFilters
from media_service.db.tables import Advertisement, AdvertisementProvenance
from media_service.db.uow import UnitOfWork, UnitOfWorkFactory
from media_service.domain.listings import ListingGateway, ReviewerEdits
from media_service.domain.review import publication_errors, rejection_errors
from media_service.domain.views import (
    CandidateDetailView,
    CandidatePageView,
    CandidateView,
    EvidenceView,
    ItemView,
)

logger = logging.getLogger(__name__)

# The only state in which a decision means anything. `linked` and `discarded` already carry one;
# `superseded` belongs to a generation nobody should be ruling on any more.
DECIDABLE = "pending_publish"


class ApprovalValidationFailedError(ServiceError):
    """The candidate is missing values a published advertisement must have (PRD 13.4)."""

    def __init__(self, details: list[dict[str, str | None]]) -> None:
        super().__init__(
            status_code=422,
            code="APPROVAL_VALIDATION_FAILED",
            message="This candidate cannot be published yet.",
            details=details,
        )


class ReviewService:
    def __init__(self, *, unit_of_work: UnitOfWorkFactory, gateway: ListingGateway) -> None:
        self._unit_of_work = unit_of_work
        self._gateway = gateway

    # -- reading -------------------------------------------------------------------------------

    def queue(
        self,
        filters: ReviewFilters | None = None,
        *,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> CandidatePageView:
        with self._unit_of_work() as unit:
            page = unit.review.search(filters, limit=limit, offset=offset)
            counts = unit.review.counts_by_status(batch_id=filters.batch_id if filters else None)
            return CandidatePageView(
                candidates=tuple(
                    CandidateView.of(provenance, advertisement)
                    for provenance, advertisement in page.rows
                ),
                total=page.total,
                limit=page.limit,
                offset=page.offset,
                counts=counts,
            )

    def detail(self, advertisement_id: str) -> CandidateDetailView:
        with self._unit_of_work() as unit:
            provenance, advertisement = self._require(unit, advertisement_id)
            return self._detail_of(unit, provenance, advertisement)

    # -- deciding ------------------------------------------------------------------------------

    def apply_edits(
        self,
        advertisement_id: str,
        *,
        edits: ReviewerEdits,
        expected_version: int | None,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> CandidateDetailView:
        with self._unit_of_work() as unit:
            provenance, advertisement = self._for_decision(
                unit, advertisement_id, expected_version=expected_version, action="edited"
            )

            before = _values_of(advertisement)
            changed = edits.changed_fields(advertisement)
            self._gateway.apply_edits(unit, advertisement=advertisement, edits=edits)

            if changed:
                # No event for a save that changed nothing. An audit trail full of no-ops is an
                # audit trail nobody reads.
                self._record(
                    unit,
                    action="edited",
                    advertisement=advertisement,
                    provenance=provenance,
                    actor_id=actor_id,
                    correlation_id=correlation_id,
                    before=before,
                    changed=list(changed),
                )

            detail = self._detail_of(unit, provenance, advertisement)
            unit.commit()
            return detail

    def approve(
        self,
        advertisement_id: str,
        *,
        edits: ReviewerEdits | None = None,
        expected_version: int | None,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> CandidateDetailView:
        """Save any corrections and publish, together or not at all."""
        with self._unit_of_work() as unit:
            provenance, advertisement = self._for_decision(
                unit, advertisement_id, expected_version=expected_version, action="approved"
            )

            before = _values_of(advertisement)
            changed = list(edits.changed_fields(advertisement)) if edits else []
            if edits is not None:
                self._gateway.apply_edits(unit, advertisement=advertisement, edits=edits)

            errors = publication_errors(
                title=advertisement.title,
                category=advertisement.category,
                location=advertisement.location,
                price=advertisement.price,
                description=advertisement.description,
                phones=[str(phone) for phone in advertisement.phones or ()],
            )
            if errors:
                # Raising rolls the unit back on the way out, so a refused approval saves nothing:
                # the reviewer's screen and the database still agree about what is stored.
                raise ApprovalValidationFailedError(errors)

            self._gateway.approve(
                unit, advertisement=advertisement, provenance=provenance, actor_id=actor_id
            )
            self._record(
                unit,
                action="approved",
                advertisement=advertisement,
                provenance=provenance,
                actor_id=actor_id,
                correlation_id=correlation_id,
                before=before,
                changed=changed,
            )

            detail = self._detail_of(unit, provenance, advertisement)
            unit.commit()
            logger.info(
                "candidate approved advertisement=%s item=%s actor=%s",
                advertisement.id,
                provenance.ingestion_item_id,
                actor_id,
            )
            return detail

    def reject(
        self,
        advertisement_id: str,
        *,
        reason_code: str,
        note: str | None = None,
        expected_version: int | None,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> CandidateDetailView:
        errors = rejection_errors(reason_code, note)
        if errors:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=errors,
            )

        with self._unit_of_work() as unit:
            provenance, advertisement = self._for_decision(
                unit, advertisement_id, expected_version=expected_version, action="rejected"
            )

            before = _values_of(advertisement)
            self._gateway.reject(
                unit,
                advertisement=advertisement,
                provenance=provenance,
                actor_id=actor_id,
                reason_code=reason_code,
                note=note,
            )
            self._record(
                unit,
                action="rejected",
                advertisement=advertisement,
                provenance=provenance,
                actor_id=actor_id,
                correlation_id=correlation_id,
                before=before,
                changed=[],
                reason_code=reason_code,
                note=note,
            )

            detail = self._detail_of(unit, provenance, advertisement)
            unit.commit()
            logger.info(
                "candidate rejected advertisement=%s reason=%s actor=%s",
                advertisement.id,
                reason_code,
                actor_id,
            )
            return detail

    # -- internals -----------------------------------------------------------------------------

    def _require(
        self, unit: UnitOfWork, advertisement_id: str
    ) -> tuple[AdvertisementProvenance, Advertisement]:
        found = unit.review.detail(advertisement_id)
        if found is None:
            raise AdvertisementNotFoundError(advertisement_id)
        return found

    def _for_decision(
        self,
        unit: UnitOfWork,
        advertisement_id: str,
        *,
        expected_version: int | None,
        action: str,
    ) -> tuple[AdvertisementProvenance, Advertisement]:
        found = unit.review.lock(advertisement_id)
        if found is None:
            raise AdvertisementNotFoundError(advertisement_id)
        provenance, advertisement = found

        if provenance.candidate_state != DECIDABLE:
            raise InvalidReviewActionError(
                f"This candidate is {provenance.candidate_state} and cannot be {action}. "
                "Only an undecided candidate can be changed."
            )

        # A client that sends no version is saying "I did not read one", which is legitimate for a
        # decision taken straight from the queue; one that sends a stale number is not.
        if expected_version is not None and expected_version != advertisement.version:
            raise VersionConflictError(expected=expected_version, actual=advertisement.version)

        return provenance, advertisement

    def _record(
        self,
        unit: UnitOfWork,
        *,
        action: str,
        advertisement: Advertisement,
        provenance: AdvertisementProvenance,
        actor_id: str,
        correlation_id: str | None,
        before: dict[str, Any],
        changed: list[str],
        reason_code: str | None = None,
        note: str | None = None,
    ) -> None:
        unit.events.record(
            subject_type="advertisement",
            action=action,
            actor_id=actor_id,
            advertisement_id=advertisement.id,
            ingestion_item_id=provenance.ingestion_item_id,
            ingestion_batch_id=provenance.ingestion_batch_id,
            reason_code=reason_code,
            note=note,
            before_values=before,
            after_values=_values_of(advertisement),
            changed_fields=changed,
            correlation_id=correlation_id,
        )

    def _detail_of(
        self,
        unit: UnitOfWork,
        provenance: AdvertisementProvenance,
        advertisement: Advertisement,
    ) -> CandidateDetailView:
        item = unit.review.item_of(provenance.ingestion_item_id)
        if item is None:  # pragma: no cover - the foreign key makes this unreachable
            raise AdvertisementNotFoundError(advertisement.id)

        siblings = unit.review.siblings(
            provenance.ingestion_item_id, generation=provenance.generation
        )
        position = siblings.index(advertisement.id) + 1 if advertisement.id in siblings else 1

        ocr = (
            unit.artifacts.get_ocr(provenance.ocr_extraction_id)
            if provenance.ocr_extraction_id
            else None
        )
        run = (
            unit.artifacts.get_llm_run(provenance.llm_extraction_run_id)
            if provenance.llm_extraction_run_id
            else None
        )

        return CandidateDetailView(
            candidate=CandidateView.of(provenance, advertisement),
            evidence=EvidenceView(
                source_text=advertisement.source_text,
                ocr_text=ocr.raw_text if ocr else "",
                ocr_extraction_id=provenance.ocr_extraction_id,
                llm_extraction_run_id=provenance.llm_extraction_run_id,
                source_block_ids=tuple(
                    int(block_id) for block_id in provenance.source_block_ids or ()
                ),
                blocks=tuple(dict(block) for block in (ocr.blocks if ocr else None) or ()),
                field_confidence={
                    str(name): float(value)
                    for name, value in (provenance.field_confidence or {}).items()
                },
                warnings=tuple(str(warning) for warning in provenance.warnings or ()),
                extracted_values=dict(provenance.extracted_values or {}),
                accepted_values=(
                    dict(provenance.accepted_values) if provenance.accepted_values else None
                ),
                provider=run.provider if run else None,
                model=run.model if run else None,
                ocr_engine=ocr.engine if ocr else None,
                ocr_languages=ocr.languages if ocr else None,
            ),
            item=ItemView.of(item),
            siblings=tuple(siblings),
            position=position,
            sibling_count=len(siblings),
            source_filename=item.original_filename,
        )


def _values_of(advertisement: Advertisement) -> dict[str, Any]:
    return {
        "title": advertisement.title,
        "description": advertisement.description,
        "category": advertisement.category,
        "location": advertisement.location,
        "price": advertisement.price,
        "phones": [str(phone) for phone in advertisement.phones or ()],
        "status": advertisement.status,
    }
