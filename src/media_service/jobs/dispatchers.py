"""Dispatchers that run work on the calling thread.

Both exist so a test can assert what the pipeline *did* without a thread pool making the assertion
racy, and `inline` is a legitimate deployment mode for a single-operator install where a background
worker is more machinery than the job needs.

Neither invents a second claim path. They take the same lease the pool takes, so an item processed
inline is protected from a concurrent worker exactly as one processed by the pool is.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from media_service.db.uow import UnitOfWorkFactory
from media_service.jobs.runner import ItemRunner

logger = logging.getLogger(__name__)

# A guard, not a queue limit: it stops a mistake in the claim predicates from spinning forever
# inside one call. A batch is 25 items, so any real drain finishes far below this.
MAX_DRAIN_ITERATIONS = 1000


class InlineDispatcher:
    """Processes each item immediately, on the thread that enqueued it.

    The HTTP response therefore waits for the pipeline. That is acceptable only because the caller
    chose it: the API still answers `202`, and the batch is already durable before this runs.
    """

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        runner: ItemRunner,
        lease_seconds: int = 300,
        worker_id: str = "inline",
    ) -> None:
        self._unit_of_work = unit_of_work
        self._runner = runner
        self._lease_seconds = lease_seconds
        self._worker_id = worker_id

    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        self._process(item_id)

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        for item_id in item_ids:
            self._process(item_id)

    def start(self) -> None:
        return None

    def stop(self, *, timeout: float = 30.0) -> None:
        return None

    def _process(self, item_id: str) -> None:
        """Run one item, and never let it take the rest of the batch with it.

        The runner handles its own item-level failures; this catches what it could not. In inline
        mode there is no poller, so an exception escaping here would abort the loop and leave every
        remaining item of an already-committed batch queued with nothing to pick it up.
        """
        claim = self._claim(item_id)
        if claim is None:
            return
        try:
            self._runner.run(item_id, claim_token=claim)
        except Exception:  # noqa: BLE001 - one bad item must not strand its siblings
            logger.exception("Item %s failed outside the runner's own handling.", item_id)

    def _claim(self, item_id: str) -> str | None:
        with self._unit_of_work() as unit:
            item = unit.items.claim(
                item_id, worker_id=self._worker_id, lease_seconds=self._lease_seconds
            )
            # Read the token before the unit of work closes: an ORM row is only meaningful while
            # its session is open.
            token = item.claim_token if item is not None else None
            unit.commit()
        return token


class ManualDispatcher:
    """Records what it was asked to do and runs nothing until a test says so.

    `run_once` drains through the real claim query rather than through the recorded ids, so a test
    that calls it exercises the queue -- including `run_after`, which is what makes "a retry waits
    for its backoff" assertable.
    """

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        runner: ItemRunner,
        lease_seconds: int = 300,
        worker_id: str = "manual",
    ) -> None:
        self._unit_of_work = unit_of_work
        self._runner = runner
        self._lease_seconds = lease_seconds
        self._worker_id = worker_id
        self.enqueued_items: list[str] = []
        self.enqueued_batches: list[tuple[str, list[str]]] = []

    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        self.enqueued_items.append(item_id)

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        self.enqueued_batches.append((batch_id, list(item_ids)))

    def start(self) -> None:
        return None

    def stop(self, *, timeout: float = 30.0) -> None:
        return None

    def run_once(self, *, limit: int = MAX_DRAIN_ITERATIONS) -> int:
        """Process every item claimable right now. Returns how many ran.

        Unlike the inline and pool dispatchers, this one lets an unexpected exception propagate.
        It drives tests and manual runs, where a surprise should be loud rather than logged and
        stepped over.
        """
        processed = 0
        while processed < limit:
            claim = self._claim_next()
            if claim is None:
                return processed
            item_id, token = claim
            self._runner.run(item_id, claim_token=token)
            processed += 1
        return processed

    def _claim_next(self) -> tuple[str, str] | None:
        with self._unit_of_work() as unit:
            item = unit.items.claim_next(
                worker_id=self._worker_id, lease_seconds=self._lease_seconds
            )
            claim = (item.id, item.claim_token) if item is not None else None
            unit.commit()
        return claim
