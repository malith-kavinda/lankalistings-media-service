"""Repository behaviour against a real PostgreSQL.

These are the tests that would be meaningless on a different engine: `SKIP LOCKED`, partial unique
indexes, savepoint recovery, and array columns all behave differently or not at all elsewhere.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from media_service.api.errors import InvalidStatusTransitionError, ItemClaimLostError
from media_service.db.repositories.assets import AssetRepository
from media_service.db.repositories.batches import BatchRepository
from media_service.db.repositories.candidates import CandidateRepository
from media_service.db.repositories.items import ItemRepository
from media_service.db.tables import (
    AdvertisementProvenance,
    IngestionItem,
    MediaDerivative,
    OcrExtraction,
)
from media_service.db.uow import UnitOfWork
from media_service.domain.ids import new_derivative_id, new_ocr_extraction_id, new_provenance_id
from media_service.domain.item_state import BatchStatus, ItemStatus
from tests.factories import make_asset, seed_batch_with_items

pytestmark = pytest.mark.usefixtures("engine")

LEASE = 300


# -- transitions -------------------------------------------------------------------------------


def test_a_transition_moves_the_item_and_refreshes_the_batch(session) -> None:
    batch, items = seed_batch_with_items(session, count=2)
    repository = ItemRepository(session)

    repository.transition(items[0], target=ItemStatus.PREPROCESSING)
    session.commit()

    assert repository.get(items[0].id).status == ItemStatus.PREPROCESSING.value
    refreshed_batch = BatchRepository(session).get(batch.id)
    assert refreshed_batch.status == BatchStatus.PROCESSING.value
    assert refreshed_batch.status_counts["processing"] == 1
    assert refreshed_batch.status_counts["queued"] == 1


def test_an_illegal_transition_is_refused_before_any_write(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)

    with pytest.raises(InvalidStatusTransitionError) as caught:
        repository.transition(items[0], target=ItemStatus.COMPLETED)

    assert caught.value.status_code == 409
    assert caught.value.code == "INVALID_STATUS_TRANSITION"
    assert repository.get(items[0].id).status == ItemStatus.UPLOADED.value


def test_a_transition_whose_expected_status_is_stale_raises(session) -> None:
    """The compare-and-swap, seen from the losing side."""
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    item = items[0]

    repository.transition(item, target=ItemStatus.PREPROCESSING)
    session.commit()

    with pytest.raises(ItemClaimLostError):
        repository.transition(
            item, target=ItemStatus.PREPROCESSING, expected=ItemStatus.UPLOADED
        )


def test_a_worker_write_with_the_wrong_claim_token_affects_nothing(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)

    claimed = repository.claim(items[0].id, worker_id="worker-a", lease_seconds=LEASE)
    session.commit()

    with pytest.raises(ItemClaimLostError):
        repository.transition(
            claimed, target=ItemStatus.OCR_PROCESSING, claim_token="clm_someone-else"
        )
    assert repository.get(claimed.id).status == ItemStatus.PREPROCESSING.value


def test_completion_timestamps_are_cleared_when_an_item_moves_again(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    item = items[0]

    item = repository.transition(item, target=ItemStatus.PREPROCESSING)
    item = repository.transition(item, target=ItemStatus.FAILED)
    assert item.completed_at is not None

    item = repository.transition(item, target=ItemStatus.UPLOADED)
    assert item.completed_at is None


# -- claiming ----------------------------------------------------------------------------------


def test_claiming_takes_the_oldest_item_and_starts_a_lease(session) -> None:
    _, items = seed_batch_with_items(session, count=3)
    repository = ItemRepository(session)
    before = datetime.now(UTC)

    claimed = repository.claim_next(worker_id="worker-a", lease_seconds=LEASE)
    session.commit()

    assert claimed.id == items[0].id
    assert claimed.status == ItemStatus.PREPROCESSING.value
    assert claimed.claim_token is not None
    assert claimed.claimed_by == "worker-a"
    assert claimed.attempt_count == 1
    assert claimed.lease_expires_at >= before + timedelta(seconds=LEASE - 1)


def test_two_workers_claiming_at_once_take_different_items(session_factory) -> None:
    """The property `SKIP LOCKED` buys: no contention, and never the same row twice."""
    with session_factory() as setup:
        _, items = seed_batch_with_items(setup, count=2)
        expected = {item.id for item in items}

    with session_factory() as first, session_factory() as second:
        # Two open transactions, interleaved deliberately: the second claim runs while the first
        # still holds its lock.
        one = ItemRepository(first).claim_next(worker_id="worker-a", lease_seconds=LEASE)
        two = ItemRepository(second).claim_next(worker_id="worker-b", lease_seconds=LEASE)
        first.commit()
        second.commit()

    assert one is not None and two is not None
    assert {one.id, two.id} == expected


def test_claiming_an_empty_queue_returns_none(session) -> None:
    seed_batch_with_items(session, count=1)
    repository = ItemRepository(session)

    assert repository.claim_next(worker_id="worker-a", lease_seconds=LEASE) is not None
    assert repository.claim_next(worker_id="worker-a", lease_seconds=LEASE) is None


def test_an_item_scheduled_for_later_is_not_claimable_yet(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    future = datetime.now(UTC) + timedelta(minutes=5)
    items[0].run_after = future
    session.commit()

    assert repository.claim_next(worker_id="worker-a", lease_seconds=LEASE) is None
    assert repository.claimable_count() == 0
    assert repository.claim_next(
        worker_id="worker-a", lease_seconds=LEASE, now=future + timedelta(seconds=1)
    ) is not None


def test_heartbeat_extends_a_live_lease_and_refuses_a_lost_one(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    claimed = repository.claim(items[0].id, worker_id="worker-a", lease_seconds=LEASE)
    original_expiry = claimed.lease_expires_at

    later = datetime.now(UTC) + timedelta(seconds=60)
    assert repository.heartbeat(
        claimed.id, claim_token=claimed.claim_token, lease_seconds=LEASE, now=later
    )
    assert not repository.heartbeat(
        claimed.id, claim_token="clm_stale", lease_seconds=LEASE, now=later
    )

    session.expire_all()
    assert repository.get(claimed.id).lease_expires_at > original_expiry


# -- recovery ----------------------------------------------------------------------------------


def test_an_expired_lease_is_found_and_requeued(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    claimed = repository.claim(items[0].id, worker_id="worker-a", lease_seconds=LEASE)
    session.commit()

    after_expiry = claimed.lease_expires_at + timedelta(seconds=1)
    expired = repository.find_expired_leases(now=after_expiry)
    assert [item.id for item in expired] == [claimed.id]

    requeued = repository.requeue_abandoned(expired[0], now=after_expiry)
    session.commit()

    assert requeued.status == ItemStatus.UPLOADED.value
    assert requeued.claim_token is None
    assert requeued.claimed_by is None
    # The attempt is not refunded: a process that keeps dying mid-item must exhaust its budget.
    assert requeued.attempt_count == 1


def test_a_live_lease_is_not_reaped(session) -> None:
    _, items = seed_batch_with_items(session)
    repository = ItemRepository(session)
    repository.claim(items[0].id, worker_id="worker-a", lease_seconds=LEASE)
    session.commit()

    assert repository.find_expired_leases() == []


def test_in_flight_items_are_found_for_single_instance_restart_recovery(session) -> None:
    _, items = seed_batch_with_items(session, count=2)
    repository = ItemRepository(session)
    claimed = repository.claim(items[0].id, worker_id="worker-a", lease_seconds=LEASE)
    session.commit()

    in_flight = repository.find_in_flight()

    assert [item.id for item in in_flight] == [claimed.id]


# -- assets ------------------------------------------------------------------------------------


def test_identical_bytes_resolve_to_one_asset(session) -> None:
    repository = AssetRepository(session)
    first, created_first = repository.get_or_create(make_asset(content="same"))
    session.commit()

    second, created_second = repository.get_or_create(make_asset(content="same"))
    session.commit()

    assert created_first is True
    assert created_second is False
    assert second.id == first.id


def test_a_failed_insert_leaves_the_transaction_usable(session) -> None:
    """The reason every dedup path is wrapped in a savepoint.

    On PostgreSQL a failed statement aborts the entire transaction. Without the savepoint the
    second write here would fail with "current transaction is aborted" rather than succeeding.
    """
    repository = AssetRepository(session)
    repository.get_or_create(make_asset(content="same"))
    session.commit()

    repository.get_or_create(make_asset(content="same"))
    survivor, created = repository.get_or_create(make_asset(content="different"))
    session.commit()

    assert created is True
    assert repository.get(survivor.id) is not None


def test_a_derivative_is_reused_when_its_parameters_are_unchanged(session) -> None:
    repository = AssetRepository(session)
    asset, _ = repository.get_or_create(make_asset(content="page"))
    session.commit()

    def build() -> MediaDerivative:
        return MediaDerivative(
            id=new_derivative_id(),
            source_asset_id=asset.id,
            purpose="ocr_input",
            storage_key=f"derivatives/xx/{asset.id}/ocr_input/abc.png",
            content_type="image/png",
            byte_size=10,
            params_hash="abc",
        )

    first, created_first = repository.get_or_create_derivative(build())
    session.commit()
    second, created_second = repository.get_or_create_derivative(build())
    session.commit()

    assert created_first is True
    assert created_second is False
    assert second.id == first.id


# -- artifacts ---------------------------------------------------------------------------------


def _ocr_row(item_id: str, asset_id: str, *, attempt: int, status: str) -> OcrExtraction:
    return OcrExtraction(
        id=new_ocr_extraction_id(),
        ingestion_item_id=item_id,
        source_asset_id=asset_id,
        generation=1,
        attempt=attempt,
        status=status,
        raw_text="text",
    )


def test_only_one_ocr_result_per_generation_can_complete(session) -> None:
    _, items = seed_batch_with_items(session)
    item = items[0]
    unit = UnitOfWork(session)

    first, created_first = unit.artifacts.record_ocr(
        _ocr_row(item.id, item.source_asset_id, attempt=1, status="completed")
    )
    second, created_second = unit.artifacts.record_ocr(
        _ocr_row(item.id, item.source_asset_id, attempt=2, status="completed")
    )
    session.commit()

    assert created_first is True
    # The loser resumes from the winner's row instead of writing a second result.
    assert created_second is False
    assert second.id == first.id
    assert unit.artifacts.completed_ocr(item_id=item.id, generation=1).id == first.id


def test_failed_attempts_are_all_kept(session) -> None:
    _, items = seed_batch_with_items(session)
    item = items[0]
    unit = UnitOfWork(session)

    unit.artifacts.record_ocr(_ocr_row(item.id, item.source_asset_id, attempt=1, status="failed"))
    unit.artifacts.record_ocr(_ocr_row(item.id, item.source_asset_id, attempt=2, status="failed"))
    session.commit()

    assert unit.artifacts.next_ocr_attempt(item_id=item.id, generation=1) == 3
    assert unit.artifacts.completed_ocr(item_id=item.id, generation=1) is None


def test_a_running_ocr_row_left_by_a_dead_worker_is_abandoned(session) -> None:
    _, items = seed_batch_with_items(session)
    item = items[0]
    unit = UnitOfWork(session)
    unit.artifacts.record_ocr(_ocr_row(item.id, item.source_asset_id, attempt=1, status="running"))
    session.commit()

    assert unit.artifacts.abandon_running_ocr(item_id=item.id, generation=1) == 1
    session.commit()

    rows = session.scalars(
        select(OcrExtraction).where(OcrExtraction.ingestion_item_id == item.id)
    ).all()
    assert [row.status for row in rows] == ["abandoned"]


# -- candidates --------------------------------------------------------------------------------


def _candidate(item: IngestionItem, *, index: int, generation: int = 1) -> AdvertisementProvenance:
    return AdvertisementProvenance(
        id=new_provenance_id(),
        ingestion_batch_id=item.batch_id,
        ingestion_item_id=item.id,
        source_asset_id=item.source_asset_id,
        generation=generation,
        candidate_index=index,
    )


def test_replaying_a_generation_cannot_duplicate_its_candidates(session) -> None:
    _, items = seed_batch_with_items(session)
    item = items[0]
    repository = CandidateRepository(session)

    first, created_first = repository.record(_candidate(item, index=0))
    session.commit()
    second, created_second = repository.record(_candidate(item, index=0))
    session.commit()

    assert created_first is True
    assert created_second is False
    assert second.id == first.id
    assert repository.count_for(item.id, generation=1) == 1


def test_a_new_generation_supersedes_only_undecided_candidates(session) -> None:
    _, items = seed_batch_with_items(session)
    item = items[0]
    repository = CandidateRepository(session)

    undecided, _ = repository.record(_candidate(item, index=0))
    decided, _ = repository.record(_candidate(item, index=1))
    decided.candidate_state = "linked"
    session.commit()

    superseded = repository.supersede_undecided(item.id, new_generation=2)
    session.commit()
    session.expire_all()

    assert superseded == 1
    assert repository.get(undecided.id).candidate_state == "superseded"
    assert repository.get(decided.id).candidate_state == "linked"
    assert repository.has_decided_candidate(item.id) is True
