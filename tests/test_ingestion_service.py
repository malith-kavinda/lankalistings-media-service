"""Batch ingestion: acceptance, idempotency, deduplication, and retry."""

from __future__ import annotations

import pytest

from media_service.api.errors import (
    IdempotencyKeyConflictError,
    ItemNotFoundError,
    NothingToRetryError,
    ReprocessRefusedError,
)
from media_service.config import Settings
from media_service.db.tables import AdvertisementProvenance
from media_service.domain.ids import new_provenance_id
from media_service.domain.item_state import BatchStatus, ItemStatus
from media_service.domain.views import AssetView
from media_service.services.ingestion import IngestionService, request_fingerprint
from media_service.services.uploads import UploadRejectedError
from tests.support import png_bytes, upload, uploads

pytestmark = pytest.mark.usefixtures("engine")

OPERATOR = "operator-1"


# -- acceptance --------------------------------------------------------------------------------


def test_a_batch_is_accepted_with_one_item_per_file(ingestion_service, dispatcher) -> None:
    progress = ingestion_service.create_batch(uploads(3), created_by=OPERATOR)

    assert progress.total_items == 3
    assert progress.status is BatchStatus.QUEUED
    assert progress.counts["queued"] == 3
    assert [item.item_index for item in progress.items] == [0, 1, 2]
    assert all(item.status is ItemStatus.UPLOADED for item in progress.items)


def test_dispatch_happens_once_the_rows_are_committed(ingestion_service, dispatcher) -> None:
    """Publishing before the commit would let a worker claim a row nobody can see yet."""
    progress = ingestion_service.create_batch(uploads(2), created_by=OPERATOR)

    assert dispatcher.batches == [(progress.id, [item.id for item in progress.items])]
    # Every dispatched id is readable, which is what "after commit" buys.
    for item_id in dispatcher.batches[0][1]:
        assert ingestion_service.get_item(item_id) is not None


def test_the_batch_is_only_answerable_once_it_is_fully_written(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)

    assert progress.committed_at is not None
    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        assert work.batches.get(progress.id).response_snapshot["total_items"] == 1


def test_uploaded_bytes_land_in_content_addressed_storage(
    ingestion_service, asset_store
) -> None:
    progress = ingestion_service.create_batch(uploads(2), created_by=OPERATOR)

    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        assets = [
            AssetView.of(work.assets.get(item.source_asset_id)) for item in progress.items
        ]

    assert all(asset_store.exists(asset.storage_key) for asset in assets)
    assert all(asset.checksum_sha256 in asset.storage_key for asset in assets)


def test_the_detected_format_decides_the_content_type_not_the_client(ingestion_service) -> None:
    """A declared type is a claim; the decoder's answer is a fact."""
    progress = ingestion_service.create_batch(
        [upload(png_bytes(), filename="page.jpg", declared_content_type="image/jpeg")],
        created_by=OPERATOR,
    )
    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        asset = AssetView.of(work.assets.get(progress.items[0].source_asset_id))

    assert asset.content_type == "image/png"
    assert progress.items[0].declared_content_type == "image/jpeg"


# -- deduplication -----------------------------------------------------------------------------


def test_the_same_image_twice_in_one_batch_is_flagged_not_dropped(ingestion_service) -> None:
    payload = png_bytes()
    progress = ingestion_service.create_batch(
        [
            upload(payload, filename="scan.png"),
            upload(png_bytes(width=40), filename="other.png"),
            upload(payload, filename="scan-copy.png"),
        ],
        created_by=OPERATOR,
    )

    first, _, repeat = progress.items
    assert len(progress.items) == 3
    assert first.is_duplicate_in_batch is False
    assert repeat.is_duplicate_in_batch is True
    assert repeat.duplicate_of_item_id == first.id
    assert "DUPLICATE_IMAGE_IN_BATCH" in repeat.warning_codes
    # Flagged, never discarded: both remain independently retryable.
    assert repeat.status is ItemStatus.UPLOADED


def test_identical_bytes_are_stored_once_across_batches(ingestion_service) -> None:
    payload = png_bytes()
    first = ingestion_service.create_batch([upload(payload)], created_by=OPERATOR)
    second = ingestion_service.create_batch([upload(payload)], created_by=OPERATOR)

    assert first.items[0].source_asset_id == second.items[0].source_asset_id
    assert first.id != second.id


# -- validation --------------------------------------------------------------------------------


def test_an_empty_request_is_refused(ingestion_service) -> None:
    with pytest.raises(UploadRejectedError) as caught:
        ingestion_service.create_batch([], created_by=OPERATOR)

    assert caught.value.details[0]["code"] == "REQUIRED"


def test_too_many_images_is_refused_before_anything_is_staged(
    unit_of_work, asset_store, dispatcher
) -> None:
    service = IngestionService(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=Settings(max_images_per_batch=2),
        dispatcher=dispatcher,
    )

    with pytest.raises(UploadRejectedError) as caught:
        service.create_batch(uploads(3), created_by=OPERATOR)

    assert caught.value.details[0]["code"] == "MAX_COUNT"
    assert list(asset_store.iter_stored_keys()) == []


def test_the_batch_total_is_binding_even_when_every_file_is_legal(
    unit_of_work, asset_store, dispatcher
) -> None:
    """PRD 15.1: 25 images at the per-image cap exceed the total, and the total wins."""
    service = IngestionService(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=Settings(max_image_bytes=10_000, max_batch_bytes=120),
        dispatcher=dispatcher,
    )

    with pytest.raises(UploadRejectedError) as caught:
        service.create_batch(uploads(3), created_by=OPERATOR)

    assert [detail["code"] for detail in caught.value.details] == ["MAX_BATCH_SIZE"]


def test_an_oversized_image_names_itself_in_the_rejection(
    unit_of_work, asset_store, dispatcher
) -> None:
    service = IngestionService(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=Settings(max_image_bytes=50),
        dispatcher=dispatcher,
    )

    with pytest.raises(UploadRejectedError) as caught:
        service.create_batch(
            [upload(png_bytes(width=200, height=200), filename="huge.png")], created_by=OPERATOR
        )

    assert caught.value.details[0]["code"] == "MAX_SIZE"
    assert "huge.png" in caught.value.details[0]["message"]


def test_a_file_that_is_not_an_image_is_refused(ingestion_service) -> None:
    with pytest.raises(UploadRejectedError) as caught:
        ingestion_service.create_batch(
            [upload(b"this is not a picture", filename="notes.txt")], created_by=OPERATOR
        )

    assert caught.value.details[0]["code"] == "UNREADABLE_IMAGE"


def test_every_problem_in_one_upload_set_is_reported_together(ingestion_service) -> None:
    """A 25-file upload should be fixable in one pass, not one rejection at a time."""
    with pytest.raises(UploadRejectedError) as caught:
        ingestion_service.create_batch(
            [
                upload(png_bytes(), filename="fine.png"),
                upload(b"nope", filename="broken.png"),
                upload(b"", filename="empty.png"),
            ],
            created_by=OPERATOR,
        )

    assert [detail["code"] for detail in caught.value.details] == ["UNREADABLE_IMAGE", "REQUIRED"]


def test_a_rejected_upload_leaves_no_files_behind(ingestion_service, asset_store) -> None:
    with pytest.raises(UploadRejectedError):
        ingestion_service.create_batch(
            [upload(png_bytes()), upload(b"nope", filename="broken.png")], created_by=OPERATOR
        )

    assert list(asset_store.iter_stored_keys()) == []
    assert asset_store.purge_staging(min_age_seconds=0) == 0


# -- idempotency -------------------------------------------------------------------------------


def test_the_same_request_twice_under_one_key_creates_one_batch(ingestion_service) -> None:
    payload = png_bytes()
    first = ingestion_service.create_batch(
        [upload(payload)], created_by=OPERATOR, idempotency_key="key-1"
    )
    second = ingestion_service.create_batch(
        [upload(payload)], created_by=OPERATOR, idempotency_key="key-1"
    )

    assert second.replayed is True
    assert second.id == first.id
    assert [item.id for item in second.items] == [item.id for item in first.items]


def test_reusing_a_key_for_different_files_is_a_conflict(ingestion_service) -> None:
    ingestion_service.create_batch(
        [upload(png_bytes())], created_by=OPERATOR, idempotency_key="key-1"
    )

    with pytest.raises(IdempotencyKeyConflictError) as caught:
        ingestion_service.create_batch(
            [upload(png_bytes(width=40))], created_by=OPERATOR, idempotency_key="key-1"
        )

    assert caught.value.status_code == 409


def test_one_key_per_operator(ingestion_service) -> None:
    payload = png_bytes()
    first = ingestion_service.create_batch(
        [upload(payload)], created_by="operator-a", idempotency_key="key-1"
    )
    second = ingestion_service.create_batch(
        [upload(payload)], created_by="operator-b", idempotency_key="key-1"
    )

    assert second.id != first.id
    assert second.replayed is False


def test_file_order_does_not_change_what_a_request_is() -> None:
    forwards = request_fingerprint(OPERATOR, ["aa", "bb"])
    backwards = request_fingerprint(OPERATOR, ["bb", "aa"])
    different_operator = request_fingerprint("operator-2", ["aa", "bb"])

    assert forwards == backwards
    assert forwards != different_operator


# -- reads -------------------------------------------------------------------------------------


def test_progress_is_recomputed_rather_than_read_from_the_cache(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(2), created_by=OPERATOR)

    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        item = work.items.get(progress.items[0].id)
        work.items.transition(item, target=ItemStatus.PREPROCESSING)
        # Deliberately corrupt the cached projection: the reader must not believe it.
        batch = work.batches.get(progress.id)
        batch.status_counts = {"queued": 99}
        work.commit()

    fresh = ingestion_service.get_batch(progress.id)

    assert fresh.counts["queued"] == 1
    assert fresh.counts["processing"] == 1
    assert fresh.status is BatchStatus.PROCESSING


def test_listing_batches_returns_the_newest_first(ingestion_service) -> None:
    first = ingestion_service.create_batch(uploads(1, prefix="a"), created_by=OPERATOR)
    second = ingestion_service.create_batch(uploads(1, prefix="b"), created_by=OPERATOR)

    listed = [progress.id for progress in ingestion_service.list_batches()]

    assert listed == [second.id, first.id]


# -- retry -------------------------------------------------------------------------------------


def _fail(service: IngestionService, item_id: str) -> None:
    with service._unit_of_work() as work:  # noqa: SLF001
        item = work.items.get(item_id)
        item = work.items.transition(item, target=ItemStatus.PREPROCESSING)
        work.items.transition(
            item, target=ItemStatus.NEEDS_ATTENTION, error_code="OCR_FAILED", failed_stage="ocr"
        )
        work.commit()


def test_retrying_a_failed_item_requeues_it_and_clears_the_error(
    ingestion_service, dispatcher
) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _fail(ingestion_service, item_id)

    outcome = ingestion_service.retry_item(item_id, actor_id="moderator-1")

    assert outcome.items[0].status is ItemStatus.UPLOADED
    assert outcome.items[0].error_code is None
    assert outcome.items[0].failed_stage is None
    assert outcome.items[0].attempt_count == 0
    assert dispatcher.items == [item_id]


def test_a_retry_keeps_the_generation_so_prior_work_is_reused(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _fail(ingestion_service, item_id)

    outcome = ingestion_service.retry_item(item_id, actor_id="moderator-1")

    assert outcome.items[0].pipeline_generation == 1


def test_retrying_an_item_that_did_not_fail_is_refused(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)

    with pytest.raises(NothingToRetryError) as caught:
        ingestion_service.retry_item(progress.items[0].id, actor_id="moderator-1")

    assert caught.value.code == "NOTHING_TO_RETRY"


def test_retrying_an_unknown_item_is_a_404(ingestion_service) -> None:
    with pytest.raises(ItemNotFoundError):
        ingestion_service.retry_item("itm_missing", actor_id="moderator-1")


def test_a_retry_reopens_a_finished_batch(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _fail(ingestion_service, item_id)
    failed = ingestion_service.get_batch(progress.id)
    assert failed.status is BatchStatus.FAILED
    assert failed.completed_at is not None

    ingestion_service.retry_item(item_id, actor_id="moderator-1")
    reopened = ingestion_service.get_batch(progress.id)

    assert reopened.status is BatchStatus.QUEUED
    assert reopened.completed_at is None


def test_retrying_a_batch_touches_only_the_items_that_failed(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(3), created_by=OPERATOR)
    _fail(ingestion_service, progress.items[0].id)
    _fail(ingestion_service, progress.items[2].id)

    outcome = ingestion_service.retry_batch(progress.id, actor_id="moderator-1")

    assert {item.id for item in outcome.items} == {progress.items[0].id, progress.items[2].id}


def test_retrying_an_item_writes_an_audit_event(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _fail(ingestion_service, item_id)

    ingestion_service.retry_item(item_id, actor_id="moderator-1")

    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        events = [(event.action, event.actor_id) for event in work.events.list_for_item(item_id)]

    assert events == [("reprocessed", "moderator-1")]


# -- reprocess ---------------------------------------------------------------------------------


def _add_candidate(service: IngestionService, item_id: str, *, state: str) -> str:
    with service._unit_of_work() as work:  # noqa: SLF001
        item = work.items.get(item_id)
        provenance, _ = work.candidates.record(
            AdvertisementProvenance(
                id=new_provenance_id(),
                ingestion_batch_id=item.batch_id,
                ingestion_item_id=item.id,
                source_asset_id=item.source_asset_id,
                generation=item.pipeline_generation,
                candidate_index=0,
                candidate_state=state,
            )
        )
        work.commit()
        return provenance.id


def _finish(service: IngestionService, item_id: str) -> None:
    with service._unit_of_work() as work:  # noqa: SLF001
        item = work.items.get(item_id)
        for target in (
            ItemStatus.PREPROCESSING,
            ItemStatus.OCR_PROCESSING,
            ItemStatus.LLM_PROCESSING,
            ItemStatus.AWAITING_REVIEW,
        ):
            item = work.items.transition(item, target=target)
        work.commit()


def test_reprocessing_bumps_the_generation_and_supersedes_undecided_candidates(
    ingestion_service,
) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _finish(ingestion_service, item_id)
    candidate_id = _add_candidate(ingestion_service, item_id, state="pending_publish")

    outcome = ingestion_service.retry_item(item_id, mode="reprocess", actor_id="moderator-1")

    assert outcome.items[0].pipeline_generation == 2
    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        assert work.candidates.get(candidate_id).candidate_state == "superseded"


def test_reprocessing_is_refused_when_a_reviewer_already_decided(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _finish(ingestion_service, item_id)
    _add_candidate(ingestion_service, item_id, state="linked")

    with pytest.raises(ReprocessRefusedError):
        ingestion_service.retry_item(item_id, mode="reprocess", actor_id="moderator-1")


def test_forcing_a_reprocess_never_supersedes_a_decided_candidate(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _finish(ingestion_service, item_id)
    decided = _add_candidate(ingestion_service, item_id, state="linked")

    ingestion_service.retry_item(
        item_id, mode="reprocess", actor_id="moderator-1", force=True
    )

    with ingestion_service._unit_of_work() as work:  # noqa: SLF001
        assert work.candidates.get(decided).candidate_state == "linked"


def test_a_plain_retry_cannot_reprocess_a_finished_item(ingestion_service) -> None:
    progress = ingestion_service.create_batch(uploads(1), created_by=OPERATOR)
    item_id = progress.items[0].id
    _finish(ingestion_service, item_id)

    with pytest.raises(NothingToRetryError):
        ingestion_service.retry_item(item_id, actor_id="moderator-1")
