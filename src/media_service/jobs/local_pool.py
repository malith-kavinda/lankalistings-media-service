"""The in-process worker: a thread pool, a poller, and a reaper.

Threads, not asyncio. `pytesseract` shells out to a subprocess, Pillow is blocking C, and the
SQLAlchemy `Session` is blocking; an asyncio dispatcher would need `run_in_executor` around all
three plus an async driver, which is two concurrency models for no benefit. With a pool as the unit
of concurrency the provider adapters stay synchronous too, and nothing needs a bridge between the
two worlds.

**Claim on a free slot, never claim then queue.** The poller takes a semaphore permit *before* it
claims, so a lease is never created for work that has not started. Claiming first and queueing
after would start a clock on an item sitting in the executor's backlog, and the reaper would
eventually requeue work that was never lost.

Restart recovery has two modes with genuinely different correctness envelopes.
`worker_single_instance` requeues every in-flight row on startup regardless of lease -- safe only
because no other process can hold those claims, and it returns progress in milliseconds instead of
after a lease timeout. With it off, the reaper waits out lease expiry, which is the only correct
choice when another worker may be alive.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from socket import gethostname

from media_service.config import Settings
from media_service.db.uow import UnitOfWorkFactory
from media_service.domain.item_state import ItemStatus
from media_service.jobs.runner import ItemRunner

logger = logging.getLogger(__name__)

# Tesseract's own OpenMP threading fights a thread pool and costs multiples of throughput on a
# multi-core machine: every worker thread spawns a Tesseract that tries to use every core.
OMP_THREAD_LIMIT = "1"


class LocalPoolDispatcher:
    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        runner: ItemRunner,
        settings: Settings,
        worker_id: str | None = None,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._runner = runner
        self._settings = settings
        self._worker_id = worker_id or f"{gethostname()}:{os.getpid()}"

        self._executor: ThreadPoolExecutor | None = None
        self._slots = threading.BoundedSemaphore(settings.worker_concurrency)
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._threads: list[threading.Thread] = []
        self._started = False

    # -- lifecycle -----------------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        os.environ.setdefault("OMP_THREAD_LIMIT", OMP_THREAD_LIMIT)

        if self._settings.worker_single_instance:
            self._recover_in_flight()

        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.worker_concurrency, thread_name_prefix="ingest"
        )
        self._threads = [
            threading.Thread(target=self._poll_loop, name="ingest-poller", daemon=True),
            threading.Thread(target=self._reap_loop, name="ingest-reaper", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._started = True
        logger.info(
            "Worker %s started with %d slots.",
            self._worker_id,
            self._settings.worker_concurrency,
        )

    def stop(self, *, timeout: float = 30.0) -> None:
        if not self._started:
            return
        self._stopping.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._threads = []
        self._started = False

    # -- hints ---------------------------------------------------------------------------------

    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        self._wake.set()

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        self._wake.set()

    # -- loops ---------------------------------------------------------------------------------

    def _poll_loop(self) -> None:
        poll_seconds = self._settings.poll_interval_ms / 1000
        while not self._stopping.is_set():
            if not self._slots.acquire(timeout=poll_seconds):
                continue

            claim = self._claim_next()
            if claim is None:
                self._slots.release()
                # Nothing to do. Sleep until a hint arrives or the poll interval elapses -- the
                # poll is what makes a lost hint cost latency rather than correctness.
                self._wake.wait(timeout=poll_seconds)
                self._wake.clear()
                continue

            self._submit(*claim)

    def _reap_loop(self) -> None:
        interval = self._settings.reaper_interval_ms / 1000
        while not self._stopping.wait(timeout=interval):
            try:
                requeued = self._requeue_expired()
                if requeued:
                    logger.warning("Requeued %d item(s) whose lease expired.", requeued)
                    self._wake.set()
            except Exception:  # noqa: BLE001 - the reaper must outlive any single failure
                logger.exception("Reaper pass failed.")

    # -- database ------------------------------------------------------------------------------

    def _claim_next(self) -> tuple[str, str] | None:
        try:
            with self._unit_of_work() as unit:
                item = unit.items.claim_next(
                    worker_id=self._worker_id, lease_seconds=self._settings.lease_seconds
                )
                claim = (item.id, item.claim_token) if item is not None else None
                unit.commit()
            return claim
        except Exception:  # noqa: BLE001 - a database blip must not kill the poller
            logger.exception("Claim failed.")
            return None

    def _requeue_expired(self) -> int:
        with self._unit_of_work() as unit:
            expired = unit.items.find_expired_leases()
            for item in expired:
                unit.artifacts.abandon_running_ocr(
                    item_id=item.id, generation=item.pipeline_generation
                )
                unit.artifacts.abandon_running_llm_runs(
                    item_id=item.id, generation=item.pipeline_generation
                )
                if item.attempt_count >= self._settings.max_item_attempts:
                    # Out of budget. Park it for a person rather than looping on whatever keeps
                    # killing the worker that claims it.
                    unit.items.transition(
                        item,
                        target=ItemStatus.NEEDS_ATTENTION,
                        release_claim=True,
                        error_code="LEASE_EXPIRED",
                        error_message="The worker processing this item stopped responding.",
                    )
                else:
                    unit.items.requeue_abandoned(item)
            count = len(expired)
            unit.commit()
        return count

    def _recover_in_flight(self) -> int:
        with self._unit_of_work() as unit:
            stranded = unit.items.find_in_flight()
            for item in stranded:
                unit.artifacts.abandon_running_ocr(
                    item_id=item.id, generation=item.pipeline_generation
                )
                unit.artifacts.abandon_running_llm_runs(
                    item_id=item.id, generation=item.pipeline_generation
                )
                unit.items.requeue_abandoned(item)
            count = len(stranded)
            unit.commit()
        if count:
            logger.warning("Requeued %d item(s) left in flight by a previous run.", count)
        return count

    # -- execution -----------------------------------------------------------------------------

    def _submit(self, item_id: str, claim_token: str) -> None:
        executor = self._executor
        if executor is None:  # pragma: no cover - stop() raced with the poller
            self._slots.release()
            return

        def task() -> None:
            try:
                self._runner.run(item_id, claim_token=claim_token)
            finally:
                self._slots.release()
                # A finished slot may unblock the next item immediately; do not wait out the poll
                # interval to find that out.
                self._wake.set()

        executor.submit(task)
