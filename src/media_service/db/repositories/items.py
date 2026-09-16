"""Items: the work queue, its claims, and the one writer of `status`.

`ingestion_items` *is* the queue -- there is no second jobs table, because two rows describing the
same work drift, and the moment they disagree an item is processed twice or never.

Two mechanisms guard the status column, and they cover different things.

*Legality* is `domain.item_state.is_legal_transition`, a pure function: it knows that
`ocr_processing` may not jump to `awaiting_review`, but it cannot see concurrency.

*Concurrency* is a compare-and-swap. Every write matches on the status it expects to replace and,
for worker writes, on the claim token that must still be held; an affected-row count other than 1
means the row moved underneath the writer, and it raises instead of overwriting. That is what makes
a zombie worker -- lease expired, process still running after a VM suspend -- harmless: its first
write misses and it stops.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from media_service.api.errors import InvalidStatusTransitionError, ItemClaimLostError
from media_service.db.base import utcnow
from media_service.db.repositories.batches import BatchRepository
from media_service.db.tables import IngestionItem
from media_service.domain.ids import new_claim_token
from media_service.domain.item_state import (
    IN_FLIGHT,
    PIPELINE_ACTIVE,
    ItemStatus,
    is_legal_transition,
)

# Claiming an item is itself a status transition (uploaded -> preprocessing), so it goes through
# the same legality table as every other write.
CLAIM_TARGET = ItemStatus.PREPROCESSING


class ItemRepository:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._batches = BatchRepository(session)

    # -- reads ---------------------------------------------------------------------------------

    def add(self, item: IngestionItem) -> IngestionItem:
        self._session.add(item)
        return item

    def get(self, item_id: str) -> IngestionItem | None:
        return self._session.get(IngestionItem, item_id)

    def list_for_batch(self, batch_id: str) -> list[IngestionItem]:
        return list(
            self._session.scalars(
                select(IngestionItem)
                .where(IngestionItem.batch_id == batch_id)
                .order_by(IngestionItem.item_index)
            )
        )

    def claimable_count(self, *, now: datetime | None = None) -> int:
        return (
            self._session.scalar(
                select(func.count())
                .select_from(IngestionItem)
                .where(*self._claimable_predicates(now or utcnow()))
            )
            or 0
        )

    # -- the one writer of `status` ------------------------------------------------------------

    def transition(
        self,
        item: IngestionItem,
        *,
        target: ItemStatus,
        expected: ItemStatus | None = None,
        claim_token: str | None = None,
        release_claim: bool = False,
        now: datetime | None = None,
        **fields: Any,
    ) -> IngestionItem:
        """Move an item to `target`, or raise.

        `expected` defaults to the status currently on the instance. Pass it explicitly when the
        in-memory copy may be stale and the caller knows which status it is replacing.
        """
        current = expected or ItemStatus(item.status)
        if not is_legal_transition(current, target):
            raise InvalidStatusTransitionError(current=current.value, target=target.value)

        moment = now or utcnow()
        values: dict[str, Any] = {"status": target.value, "updated_at": moment, **fields}

        if release_claim:
            values |= {
                "claim_token": None,
                "claimed_by": None,
                "claimed_at": None,
                "lease_expires_at": None,
            }
        # `completed_at` means the *pipeline* finished with this item, so it is cleared by anything
        # that puts the item back in motion -- a retry must not leave a completion timestamp
        # behind.
        if "completed_at" not in fields:
            values["completed_at"] = None if target in PIPELINE_ACTIVE else moment

        predicates = [IngestionItem.id == item.id, IngestionItem.status == current.value]
        if claim_token is not None:
            predicates.append(IngestionItem.claim_token == claim_token)

        result = self._session.execute(
            update(IngestionItem)
            .where(*predicates)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise ItemClaimLostError(item.id)

        refreshed = self._reload(item.id)
        self._batches.refresh_projection(refreshed.batch_id, now=moment)
        return refreshed

    # -- claiming ------------------------------------------------------------------------------

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
        batch_id: str | None = None,
    ) -> IngestionItem | None:
        """Take the oldest claimable item, or return None.

        `SKIP LOCKED` is what makes this a queue rather than a contention point: a second worker
        arriving at the same instant takes the *next* row instead of blocking on this one and then
        losing a compare-and-swap. The `claim_token IS NULL` predicate is kept as a cheap invariant
        assertion on top of the lock.

        No attempt cap is applied here. Both paths that requeue an item -- a stage failure and the
        reaper -- park it in `needs_attention` once its attempts are spent, so an item sitting in
        `uploaded` always has budget left. Filtering here instead would strand exhausted rows in a
        state that looks queued forever.
        """
        moment = now or utcnow()
        predicates = list(self._claimable_predicates(moment))
        if batch_id is not None:
            predicates.append(IngestionItem.batch_id == batch_id)

        oldest = (
            select(IngestionItem.id)
            .where(*predicates)
            .order_by(IngestionItem.run_after, IngestionItem.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )

        claimed_id = self._session.scalar(
            update(IngestionItem)
            .where(IngestionItem.id == oldest)
            .values(**self._claim_values(worker_id, lease_seconds, moment))
            .returning(IngestionItem.id)
            .execution_options(synchronize_session=False)
        )
        if claimed_id is None:
            return None

        item = self._reload(claimed_id)
        self._batches.refresh_projection(item.batch_id, now=moment)
        return item

    def claim(
        self, item_id: str, *, worker_id: str, lease_seconds: int, now: datetime | None = None
    ) -> IngestionItem | None:
        """Claim one specific item, used when the dispatcher already knows which row it wants."""
        moment = now or utcnow()
        claimed_id = self._session.scalar(
            update(IngestionItem)
            .where(IngestionItem.id == item_id, *self._claimable_predicates(moment))
            .values(**self._claim_values(worker_id, lease_seconds, moment))
            .returning(IngestionItem.id)
            .execution_options(synchronize_session=False)
        )
        if claimed_id is None:
            return None

        item = self._reload(claimed_id)
        self._batches.refresh_projection(item.batch_id, now=moment)
        return item

    def heartbeat(
        self,
        item_id: str,
        *,
        claim_token: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        """Extend a lease. False means the claim is gone and the worker must stop."""
        moment = now or utcnow()
        result = self._session.execute(
            update(IngestionItem)
            .where(IngestionItem.id == item_id, IngestionItem.claim_token == claim_token)
            .values(
                heartbeat_at=moment,
                lease_expires_at=moment + timedelta(seconds=lease_seconds),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount == 1

    # -- recovery ------------------------------------------------------------------------------

    def find_expired_leases(self, *, now: datetime | None = None) -> list[IngestionItem]:
        moment = now or utcnow()
        return list(
            self._session.scalars(
                select(IngestionItem)
                .where(
                    IngestionItem.claim_token.is_not(None),
                    IngestionItem.lease_expires_at < moment,
                )
                .order_by(IngestionItem.lease_expires_at)
                .with_for_update(skip_locked=True)
            )
        )

    def find_in_flight(self) -> list[IngestionItem]:
        return list(
            self._session.scalars(
                select(IngestionItem)
                .where(IngestionItem.status.in_([status.value for status in IN_FLIGHT]))
                .order_by(IngestionItem.id)
                .with_for_update(skip_locked=True)
            )
        )

    def requeue_abandoned(
        self,
        item: IngestionItem,
        *,
        delay_seconds: float = 0.0,
        now: datetime | None = None,
    ) -> IngestionItem:
        """Return an item whose worker vanished to the queue.

        Not a retry: nobody asked for it, no review event is written, and the attempt count is left
        as the claim set it -- a process that keeps dying mid-item exhausts its budget and parks in
        `needs_attention` rather than looping forever.
        """
        moment = now or utcnow()
        return self.transition(
            item,
            target=ItemStatus.UPLOADED,
            release_claim=True,
            now=moment,
            run_after=moment + timedelta(seconds=delay_seconds),
        )

    # -- helpers -------------------------------------------------------------------------------

    def _claimable_predicates(self, now: datetime) -> tuple[Any, ...]:
        return (
            IngestionItem.status == ItemStatus.UPLOADED.value,
            IngestionItem.run_after <= now,
            IngestionItem.claim_token.is_(None),
        )

    def _claim_values(self, worker_id: str, lease_seconds: int, now: datetime) -> dict[str, Any]:
        return {
            "claim_token": new_claim_token(),
            "claimed_by": worker_id,
            "claimed_at": now,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
            "heartbeat_at": now,
            "attempt_count": IngestionItem.attempt_count + 1,
            "status": CLAIM_TARGET.value,
            "updated_at": now,
            "completed_at": None,
        }

    def _reload(self, item_id: str) -> IngestionItem:
        """Re-read a row the database changed behind the identity map.

        The writes above are Core statements, so an instance already loaded in this session still
        holds the pre-update values. `populate_existing` overwrites it rather than returning the
        stale copy.
        """
        return self._session.scalars(
            select(IngestionItem)
            .where(IngestionItem.id == item_id)
            .execution_options(populate_existing=True)
        ).one()
