"""Review events: append-only audit of who did what.

No update and no delete. A corrected event is a second event, because the value of this table is
that it says what was believed at the time -- rewriting it would destroy the only record of the
decision a person actually made.

The subject is explicit rather than implied by which foreign key is set. Reprocessing is
item-scoped and may have no advertisement at all (PRD 12.7), so a table keyed only on
`advertisement_id` could not record it.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from media_service.db.tables import ReviewEvent
from media_service.domain.ids import new_review_event_id


class ReviewEventRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(
        self,
        *,
        subject_type: str,
        action: str,
        actor_id: str,
        advertisement_id: str | None = None,
        ingestion_item_id: str | None = None,
        ingestion_batch_id: str | None = None,
        actor_role: str | None = None,
        reason_code: str | None = None,
        note: str | None = None,
        before_values: dict[str, Any] | None = None,
        after_values: dict[str, Any] | None = None,
        changed_fields: list[str] | None = None,
        correlation_id: str | None = None,
    ) -> ReviewEvent:
        event = ReviewEvent(
            id=new_review_event_id(),
            subject_type=subject_type,
            action=action,
            actor_id=actor_id,
            advertisement_id=advertisement_id,
            ingestion_item_id=ingestion_item_id,
            ingestion_batch_id=ingestion_batch_id,
            actor_role=actor_role,
            reason_code=reason_code,
            note=note,
            before_values=before_values,
            after_values=after_values,
            changed_fields=changed_fields,
            correlation_id=correlation_id,
        )
        self._session.add(event)
        return event

    def list_for_item(self, item_id: str) -> list[ReviewEvent]:
        return list(
            self._session.scalars(
                select(ReviewEvent)
                .where(ReviewEvent.ingestion_item_id == item_id)
                .order_by(ReviewEvent.created_at, ReviewEvent.id)
            )
        )

    def list_for_advertisement(self, advertisement_id: str) -> list[ReviewEvent]:
        return list(
            self._session.scalars(
                select(ReviewEvent)
                .where(ReviewEvent.advertisement_id == advertisement_id)
                .order_by(ReviewEvent.created_at, ReviewEvent.id)
            )
        )
