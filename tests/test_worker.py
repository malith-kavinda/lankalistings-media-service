"""The worker: the pool, the poller, the reaper, and restart recovery.

Only the first test runs threads, and it waits on a condition rather than on a sleep. The recovery
paths are exercised by calling them directly -- what matters about the reaper is which rows it
changes, and driving that through a timer would make the test slow and flaky without testing
anything the direct call does not.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from media_service.config import Settings
from media_service.domain.item_state import BatchStatus, ItemStatus
from media_service.jobs.local_pool import LocalPoolDispatcher
from media_service.jobs.runner import ItemRunner
from media_service.services.ingestion import IngestionService
from tests.support import CountingExtractor, CountingPreprocessor, FakeOcrStep, uploads

pytestmark = pytest.mark.usefixtures("engine")

OPERATOR = "operator-1"
DEADLINE_SECONDS = 20.0


def _fast_settings(**overrides) -> Settings:  # type: ignore[no-untyped-def]
    return Settings(
        worker_concurrency=2,
        poll_interval_ms=25,
        reaper_interval_ms=50,
        lease_seconds=30,
        **overrides,
    )


def _build(unit_of_work, asset_store, settings):  # type: ignore[no-untyped-def]
    from media_service.domain.listings import LocalListingGateway

    runner = ItemRunner(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=settings,
        preprocess=CountingPreprocessor(),
        ocr=FakeOcrStep(),
        extraction=CountingExtractor(),
        gateway=LocalListingGateway(),
    )
    dispatcher = LocalPoolDispatcher(
        unit_of_work=unit_of_work, runner=runner, settings=settings, worker_id="test-worker"
    )
    service = IngestionService(
        unit_of_work=unit_of_work, store=asset_store, settings=settings, dispatcher=dispatcher
    )
    return service, dispatcher


def _await_status(service, batch_id: str, status: BatchStatus):  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + DEADLINE_SECONDS
    while time.monotonic() < deadline:
        progress = service.get_batch(batch_id)
        if progress.status is status:
            return progress
        time.sleep(0.05)
    raise AssertionError(
        f"Batch {batch_id} never reached {status}; last seen "
        f"{service.get_batch(batch_id).status} with {service.get_batch(batch_id).counts}"
    )


def test_the_pool_drains_a_batch(unit_of_work, asset_store) -> None:
    service, dispatcher = _build(unit_of_work, asset_store, _fast_settings())
    dispatcher.start()
    try:
        progress = service.create_batch(uploads(4), created_by=OPERATOR)
        finished = _await_status(service, progress.id, BatchStatus.COMPLETED)
    finally:
        dispatcher.stop(timeout=5)

    assert finished.counts["awaiting_review"] == 4


def test_work_committed_before_the_worker_started_is_still_found(
    unit_of_work, asset_store
) -> None:
    """The poller is the guarantee; the enqueue hint is only latency (AC-015)."""
    service, dispatcher = _build(unit_of_work, asset_store, _fast_settings())
    progress = service.create_batch(uploads(2), created_by=OPERATOR)

    dispatcher.start()
    try:
        finished = _await_status(service, progress.id, BatchStatus.COMPLETED)
    finally:
        dispatcher.stop(timeout=5)

    assert finished.counts["awaiting_review"] == 2


def test_stopping_a_worker_that_never_started_is_harmless(unit_of_work, asset_store) -> None:
    _, dispatcher = _build(unit_of_work, asset_store, _fast_settings())

    dispatcher.stop()  # no threads, no executor, no error


# -- recovery ----------------------------------------------------------------------------------


def _claimed_item(service, unit_of_work, *, lease_seconds: int):  # type: ignore[no-untyped-def]
    progress = service.create_batch(uploads(1), created_by=OPERATOR)
    with unit_of_work() as work:
        item = work.items.claim(
            progress.items[0].id, worker_id="dead-worker", lease_seconds=lease_seconds
        )
        claimed_id = item.id
        work.commit()
    return claimed_id


def test_the_reaper_requeues_an_item_whose_worker_stopped_responding(
    unit_of_work, asset_store
) -> None:
    service, dispatcher = _build(unit_of_work, asset_store, _fast_settings())
    item_id = _claimed_item(service, unit_of_work, lease_seconds=30)

    # Expire the lease rather than waiting for it.
    with unit_of_work() as work:
        work.items.get(item_id).lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        work.commit()

    assert dispatcher._requeue_expired() == 1  # noqa: SLF001

    requeued = service.get_item(item_id)
    assert requeued.status is ItemStatus.UPLOADED
    assert requeued.attempt_count == 1


def test_the_reaper_parks_an_item_that_has_used_its_attempts(unit_of_work, asset_store) -> None:
    settings = _fast_settings(max_item_attempts=1)
    service, dispatcher = _build(unit_of_work, asset_store, settings)
    item_id = _claimed_item(service, unit_of_work, lease_seconds=30)

    with unit_of_work() as work:
        item = work.items.get(item_id)
        item.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        work.commit()

    dispatcher._requeue_expired()  # noqa: SLF001

    parked = service.get_item(item_id)
    assert parked.status is ItemStatus.NEEDS_ATTENTION
    assert parked.error_code == "LEASE_EXPIRED"


def test_a_live_lease_survives_a_reaper_pass(unit_of_work, asset_store) -> None:
    service, dispatcher = _build(unit_of_work, asset_store, _fast_settings())
    item_id = _claimed_item(service, unit_of_work, lease_seconds=300)

    assert dispatcher._requeue_expired() == 0  # noqa: SLF001
    assert service.get_item(item_id).status is ItemStatus.PREPROCESSING


def test_a_single_instance_restart_requeues_everything_in_flight(
    unit_of_work, asset_store
) -> None:
    """Safe only because nothing else can hold those claims -- and it returns progress at once."""
    service, dispatcher = _build(unit_of_work, asset_store, _fast_settings())
    item_id = _claimed_item(service, unit_of_work, lease_seconds=300)

    assert dispatcher._recover_in_flight() == 1  # noqa: SLF001

    assert service.get_item(item_id).status is ItemStatus.UPLOADED


def test_a_multi_instance_worker_leaves_other_claims_alone(unit_of_work, asset_store) -> None:
    settings = _fast_settings(worker_single_instance=False)
    service, dispatcher = _build(unit_of_work, asset_store, settings)
    item_id = _claimed_item(service, unit_of_work, lease_seconds=300)

    dispatcher.start()
    try:
        # The live lease belongs to another worker as far as this one knows, so only its expiry
        # may release it.
        time.sleep(0.2)
        assert service.get_item(item_id).status is ItemStatus.PREPROCESSING
    finally:
        dispatcher.stop(timeout=5)
