"""Batches: creation, idempotency arbitration, and the derived progress projection."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from media_service.db.base import utcnow
from media_service.db.tables import IngestionBatch, IngestionItem
from media_service.domain.item_state import (
    BatchStatus,
    ItemStatus,
    counts_from,
    derive_batch_status,
    is_terminal_batch_status,
)


class BatchRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, batch: IngestionBatch) -> IngestionBatch:
        self._session.add(batch)
        return batch

    def get(self, batch_id: str) -> IngestionBatch | None:
        return self._session.get(IngestionBatch, batch_id)

    def find_by_idempotency_key(self, *, created_by: str, key: str) -> IngestionBatch | None:
        return self._session.scalar(
            select(IngestionBatch).where(
                IngestionBatch.created_by == created_by,
                IngestionBatch.idempotency_key == key,
            )
        )

    def list_recent(self, *, limit: int = 50, offset: int = 0) -> list[IngestionBatch]:
        # Identifiers are ULIDs, so ordering by id descending is reverse-chronological without a
        # secondary sort column and without reading `created_at`.
        return list(
            self._session.scalars(
                select(IngestionBatch)
                .order_by(IngestionBatch.id.desc())
                .limit(limit)
                .offset(offset)
            )
        )

    def mark_committed(self, batch: IngestionBatch, *, response_snapshot: dict[str, Any]) -> None:
        """Publish the batch to concurrent duplicate requests.

        Until this is set the row exists but is not yet answerable: a duplicate request that finds
        it must wait rather than replay a response that has not been assembled. That is what
        removes the need for a separate `receiving` batch state.
        """
        batch.response_snapshot = response_snapshot
        batch.committed_at = utcnow()

    def status_counts(self, batch_id: str) -> dict[ItemStatus, int]:
        rows = self._session.execute(
            select(IngestionItem.status, func.count())
            .where(IngestionItem.batch_id == batch_id)
            .group_by(IngestionItem.status)
        ).all()
        return {ItemStatus(status): count for status, count in rows}

    def refresh_projection(self, batch_id: str, *, now: datetime | None = None) -> BatchStatus:
        """Recompute the cached status and counts from the items.

        Called inside the same transaction as every item transition, so the cache cannot drift from
        the rows it summarises. Readers of a single batch recompute anyway (FR-JOB-003); the stored
        copy exists so the batch *list* endpoint can filter and sort without a correlated subquery.
        """
        batch = self.get(batch_id)
        if batch is None:
            return BatchStatus.QUEUED

        statuses = self.status_counts(batch_id)
        status = derive_batch_status(statuses)

        batch.status_counts = counts_from(statuses)
        batch.status = status.value
        # A retry reopens a finished batch, so this clears as well as sets (PRD 10.1).
        batch.completed_at = (now or utcnow()) if is_terminal_batch_status(status) else None
        return status
