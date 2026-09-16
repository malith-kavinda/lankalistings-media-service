"""The pipeline end to end: stages, resume, failure, and the guarantees around them.

The expensive properties are asserted by call count. "OCR ran once across two attempts" and "the
extractor was never called again after the crash" are the whole reason the artifact tables exist,
and a count is the only way to see them.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from media_service.config import Settings
from media_service.db.tables import Advertisement, LlmExtractionRun, MediaDerivative, OcrExtraction
from media_service.domain.item_state import BatchStatus, ItemStatus
from media_service.jobs.stages import PillowPreprocessor, StageError
from tests.support import CountingExtractor, FakeOcrStep, build_pipeline, png_bytes, uploads

pytestmark = pytest.mark.usefixtures("engine")

OPERATOR = "operator-1"


def _item(pipeline, index: int = 0):  # type: ignore[no-untyped-def]
    progress = pipeline.service.create_batch(uploads(index + 1), created_by=OPERATOR)
    return progress, progress.items[index]


# -- the happy path ----------------------------------------------------------------------------


def test_an_item_runs_through_to_awaiting_review(pipeline) -> None:
    progress, item = _item(pipeline)

    assert pipeline.run() == 1

    finished = pipeline.service.get_item(item.id)
    assert finished.status is ItemStatus.AWAITING_REVIEW
    assert finished.candidate_count == 1
    assert finished.completed_at is not None
    assert finished.error_code is None


def test_the_batch_completes_when_every_item_does(pipeline) -> None:
    progress = pipeline.service.create_batch(uploads(3), created_by=OPERATOR)

    assert pipeline.run() == 3

    finished = pipeline.service.get_batch(progress.id)
    assert finished.status is BatchStatus.COMPLETED
    assert finished.counts["awaiting_review"] == 3
    assert finished.counts["queued"] == 0


def test_one_image_can_yield_several_independent_candidates(
    unit_of_work, asset_store, ingestion_settings
) -> None:
    """The requirement the prototype could not express: a page holds more than one advertisement."""
    pipeline = build_pipeline(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=ingestion_settings,
        extraction=CountingExtractor(count=3),
    )
    progress, item = _item(pipeline)

    pipeline.run()

    finished = pipeline.service.get_item(item.id)
    assert finished.candidate_count == 3
    with unit_of_work() as work:
        candidates = work.candidates.list_for_item(item.id)
        assert [candidate.candidate_index for candidate in candidates] == [0, 1, 2]
        # One image, one asset, three candidates (AC-002).
        assert {candidate.source_asset_id for candidate in candidates} == {item.source_asset_id}


def test_a_page_with_no_advertisements_is_not_a_failure(
    unit_of_work, asset_store, ingestion_settings
) -> None:
    pipeline = build_pipeline(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=ingestion_settings,
        extraction=CountingExtractor(count=0),
    )
    progress, item = _item(pipeline)

    pipeline.run()

    finished = pipeline.service.get_item(item.id)
    assert finished.status is ItemStatus.NO_ADS
    assert finished.candidate_count == 0
    assert finished.error_code is None


def test_candidates_are_never_publicly_readable(pipeline, session) -> None:
    """AC-010 / invariant 2: nothing machine-made reaches `active` without a person."""
    _item(pipeline)

    pipeline.run()

    statuses = set(session.scalars(select(Advertisement.status)))
    assert statuses == {"pending"}


def test_candidates_carry_their_evidence(pipeline, unit_of_work) -> None:
    progress, item = _item(pipeline)

    pipeline.run()

    with unit_of_work() as work:
        candidate = work.candidates.list_for_item(item.id)[0]
        assert candidate.ocr_extraction_id is not None
        assert candidate.llm_extraction_run_id is not None
        assert candidate.ingestion_batch_id == progress.id
        assert candidate.candidate_state == "pending_publish"


def test_preprocessing_writes_an_ocr_input_derivative(pipeline, session) -> None:
    _item(pipeline)

    pipeline.run()

    derivatives = session.scalars(select(MediaDerivative)).all()
    assert [derivative.purpose for derivative in derivatives] == ["ocr_input"]
    assert derivatives[0].preprocessing_version == "preprocess/v0"


def test_the_preprocessor_refuses_an_image_above_the_pixel_cap() -> None:
    """The second line of defence, for bytes that never went through upload validation."""
    with pytest.raises(StageError) as caught:
        PillowPreprocessor(max_pixels=50).run(png_bytes(width=40, height=40))

    assert caught.value.code == "IMAGE_TOO_LARGE"
    # Not worth retrying: the image will be exactly as large next time.
    assert caught.value.retryable is False


# -- resume ------------------------------------------------------------------------------------


def test_re_running_a_finished_item_repeats_no_work(pipeline, unit_of_work) -> None:
    """Every artifact of this generation is present, so the resume plan has nothing to run."""
    progress, item = _item(pipeline)
    pipeline.run()

    with unit_of_work() as work:
        finished = work.items.get(item.id)
        work.items.transition(finished, target=ItemStatus.UPLOADED)
        work.commit()
    assert pipeline.run() == 1

    assert pipeline.preprocess.calls == 1
    assert pipeline.ocr.calls == 1
    assert pipeline.extraction.calls == 1


def test_a_crash_after_ocr_does_not_run_ocr_again(pipeline, unit_of_work) -> None:
    progress, item = _item(pipeline)
    _crash_after_ocr(pipeline, item.id)
    assert pipeline.ocr.calls == 1

    assert pipeline.run() == 1

    assert pipeline.ocr.calls == 1
    assert pipeline.service.get_item(item.id).status is ItemStatus.AWAITING_REVIEW


def test_a_crash_after_the_extractor_validated_does_not_call_it_again(
    pipeline, unit_of_work, session
) -> None:
    """The expensive resume: the paid call is not repeated, only the rows it should have made."""
    progress, item = _item(pipeline)
    _crash_after_extraction(pipeline, item.id)
    assert pipeline.extraction.calls == 1

    assert pipeline.run() == 1

    assert pipeline.extraction.calls == 1
    finished = pipeline.service.get_item(item.id)
    assert finished.status is ItemStatus.AWAITING_REVIEW
    assert finished.candidate_count == 1
    assert len(session.scalars(select(LlmExtractionRun)).all()) == 1


def test_resuming_reuses_the_preprocessed_derivative(pipeline, session) -> None:
    progress, item = _item(pipeline)
    _crash_after_ocr(pipeline, item.id)

    pipeline.run()

    assert pipeline.preprocess.calls == 1
    assert len(session.scalars(select(MediaDerivative)).all()) == 1


def test_a_retry_creates_no_duplicate_candidates(pipeline, unit_of_work) -> None:
    """AC-006, stated as the constraint that enforces it rather than as a hope."""
    progress, item = _item(pipeline)
    pipeline.run()

    # Force the item back through the pipeline within the same generation.
    with unit_of_work() as work:
        finished = work.items.get(item.id)
        work.items.transition(finished, target=ItemStatus.UPLOADED)
        work.commit()
    pipeline.run()

    with unit_of_work() as work:
        assert work.candidates.count_for(item.id, generation=1) == 1
    assert pipeline.service.get_item(item.id).candidate_count == 1


def test_reprocessing_produces_a_second_generation_of_candidates(pipeline, unit_of_work) -> None:
    progress, item = _item(pipeline)
    pipeline.run()

    pipeline.service.retry_item(item.id, mode="reprocess", actor_id="moderator-1")
    pipeline.run()

    with unit_of_work() as work:
        first = [
            candidate.candidate_state
            for candidate in work.candidates.list_for_item(item.id, generation=1)
        ]
        second = [
            candidate.candidate_state
            for candidate in work.candidates.list_for_item(item.id, generation=2)
        ]

    assert first == ["superseded"]
    assert second == ["pending_publish"]
    # The new generation is a fresh extraction, so the engines ran again.
    assert pipeline.ocr.calls == 2


# -- failure -----------------------------------------------------------------------------------


def test_a_retryable_failure_requeues_the_item_with_backoff(pipeline) -> None:
    pipeline.ocr.fail_with = StageError("OCR_TIMEOUT", "Timed out.", retryable=True)
    progress, item = _item(pipeline)

    pipeline.run()

    requeued = pipeline.service.get_item(item.id)
    assert requeued.status is ItemStatus.UPLOADED
    assert requeued.error_code == "OCR_TIMEOUT"
    assert requeued.failed_stage == "ocr"
    assert requeued.run_after > datetime.now(UTC)
    # It is not claimable yet, so a drain does not spin on it.
    assert pipeline.run() == 0


def test_a_permanent_failure_parks_the_item_immediately(pipeline) -> None:
    pipeline.ocr.fail_with = StageError("IMAGE_UNREADABLE", "Not an image.", retryable=False)
    progress, item = _item(pipeline)

    pipeline.run()

    parked = pipeline.service.get_item(item.id)
    assert parked.status is ItemStatus.NEEDS_ATTENTION
    assert parked.error_code == "IMAGE_UNREADABLE"
    assert parked.attempt_count == 1


def test_attempts_are_bounded(unit_of_work, asset_store) -> None:
    settings = Settings(max_item_attempts=2)
    pipeline = build_pipeline(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=settings,
        ocr=FakeOcrStep(fail_with=StageError("OCR_TIMEOUT", "Timed out.")),
    )
    progress, item = _item(pipeline)

    for _ in range(3):
        _make_claimable(pipeline, item.id)
        pipeline.run()

    parked = pipeline.service.get_item(item.id)
    assert parked.status is ItemStatus.NEEDS_ATTENTION
    assert parked.attempt_count == 2
    assert pipeline.ocr.calls == 2


def test_one_failing_item_does_not_stop_the_rest_of_the_batch(
    unit_of_work, asset_store, ingestion_settings
) -> None:
    """AC-004."""
    pipeline = build_pipeline(
        unit_of_work=unit_of_work, store=asset_store, settings=ingestion_settings
    )
    progress = pipeline.service.create_batch(uploads(3), created_by=OPERATOR)

    failing = progress.items[1].id
    original = pipeline.ocr.run

    def run(image_bytes, *, content_type):  # type: ignore[no-untyped-def]
        # Fail only the second item, identified by the one distinguishing thing a stage sees.
        if _claimed_item_id(pipeline) == failing:
            raise StageError("IMAGE_UNREADABLE", "Not an image.", retryable=False)
        return original(image_bytes, content_type=content_type)

    pipeline.ocr.run = run  # type: ignore[method-assign]
    pipeline.run()

    finished = pipeline.service.get_batch(progress.id)
    assert finished.status is BatchStatus.PARTIAL_FAILED
    assert finished.counts["awaiting_review"] == 2
    assert finished.counts["needs_attention"] == 1


def test_an_abandoned_ocr_row_is_closed_out_on_failure(pipeline, session) -> None:
    pipeline.ocr.fail_with = StageError("OCR_TIMEOUT", "Timed out.")
    _item(pipeline)

    pipeline.run()

    assert session.scalars(select(OcrExtraction)).all() == []


# -- helpers -----------------------------------------------------------------------------------


def _claimed_item_id(pipeline) -> str | None:  # type: ignore[no-untyped-def]
    with pipeline.unit_of_work() as work:
        in_flight = work.items.find_in_flight()
        return in_flight[0].id if in_flight else None


def _make_claimable(pipeline, item_id: str) -> None:
    """Bring a backed-off item forward so a test does not have to wait for it."""
    with pipeline.unit_of_work() as work:
        item = work.items.get(item_id)
        if item.status == ItemStatus.UPLOADED.value:
            item.run_after = datetime.now(UTC) - timedelta(seconds=1)
        work.commit()


def _crash_after_ocr(pipeline, item_id: str) -> None:
    """Run as far as the OCR row, then simulate the process dying.

    The item is left claimed and in flight, exactly as a `kill -9` would leave it, and then
    requeued the way the reaper would.
    """
    original = pipeline.extraction.run

    def explode(result):  # type: ignore[no-untyped-def]
        raise _Crash

    pipeline.extraction.run = explode  # type: ignore[method-assign]
    with pytest.raises(_Crash):
        _run_raw(pipeline)
    pipeline.extraction.run = original  # type: ignore[method-assign]
    _requeue_in_flight(pipeline)


def _crash_after_extraction(pipeline, item_id: str) -> None:
    """Die after the extractor validated but before the candidates are committed."""
    original = pipeline.runner._finish  # noqa: SLF001

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise _Crash

    pipeline.runner._finish = explode  # type: ignore[method-assign]  # noqa: SLF001
    with pytest.raises(_Crash):
        _run_raw(pipeline)
    pipeline.runner._finish = original  # type: ignore[method-assign]  # noqa: SLF001
    _requeue_in_flight(pipeline)


class _Crash(BaseException):
    """Not an Exception: the runner catches those and turns them into item failures.

    A `kill -9` gives the process no chance to record anything, and this has to behave the same way
    or the test would be exercising the error path instead of the recovery path.
    """


def _run_raw(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Claim and run one item without the dispatcher's error handling."""
    with pipeline.unit_of_work() as work:
        item = work.items.claim_next(worker_id="crash-test", lease_seconds=300)
        claim = (item.id, item.claim_token)
        work.commit()
    pipeline.runner.run(claim[0], claim_token=claim[1])


def _requeue_in_flight(pipeline) -> None:  # type: ignore[no-untyped-def]
    """What the reaper or a single-instance restart would do to the stranded row."""
    with pipeline.unit_of_work() as work:
        for item in work.items.find_in_flight():
            work.artifacts.abandon_running_ocr(
                item_id=item.id, generation=item.pipeline_generation
            )
            work.items.requeue_abandoned(item)
        work.commit()


# -- failures the runner must absorb rather than propagate --------------------------------------


def test_a_failure_before_the_pipeline_starts_still_marks_the_item(pipeline, monkeypatch) -> None:
    """`_begin` can fail too, and the item must not be left claimed and silent.

    A pool worker never inspects the future it submits, so an exception escaping here would vanish
    with no log line; an inline dispatcher would abort the rest of an already-committed batch.
    """
    progress, item = _item(pipeline)

    def explode(item_id, claim_token):  # type: ignore[no-untyped-def]
        raise StageError("ASSET_MISSING", "The source asset is gone.", retryable=False)

    monkeypatch.setattr(pipeline.runner, "_begin", explode)
    assert pipeline.run() == 1

    parked = pipeline.service.get_item(item.id)
    assert parked.status is ItemStatus.NEEDS_ATTENTION
    assert parked.error_code == "ASSET_MISSING"


def test_an_unexpected_failure_before_the_pipeline_starts_is_retryable(
    pipeline, monkeypatch
) -> None:
    progress, item = _item(pipeline)

    def explode(item_id, claim_token):  # type: ignore[no-untyped-def]
        raise RuntimeError("the database went away")

    monkeypatch.setattr(pipeline.runner, "_begin", explode)
    assert pipeline.run() == 1

    requeued = pipeline.service.get_item(item.id)
    # An unknown fault says nothing about the item, so it is worth another attempt.
    assert requeued.status is ItemStatus.UPLOADED
    assert requeued.error_code == "INTERNAL_ERROR"


def test_inline_dispatch_does_not_strand_the_rest_of_a_batch(
    unit_of_work, asset_store, ingestion_settings
) -> None:
    """Inline mode has no poller, so an aborted loop leaves its items queued forever."""
    from media_service.domain.listings import LocalListingGateway
    from media_service.jobs.dispatchers import InlineDispatcher
    from media_service.jobs.runner import ItemRunner
    from media_service.services.ingestion import IngestionService
    from tests.support import CountingExtractor, CountingPreprocessor, FakeOcrStep

    runner = ItemRunner(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=ingestion_settings,
        preprocess=CountingPreprocessor(),
        ocr=FakeOcrStep(),
        extraction=CountingExtractor(),
        gateway=LocalListingGateway(),
    )
    attempted: list[str] = []
    original = runner.run

    def run(item_id, *, claim_token):  # type: ignore[no-untyped-def]
        attempted.append(item_id)
        if len(attempted) == 1:
            raise RuntimeError("something the runner could not handle")
        return original(item_id, claim_token=claim_token)

    runner.run = run  # type: ignore[method-assign]
    dispatcher = InlineDispatcher(unit_of_work=unit_of_work, runner=runner)
    service = IngestionService(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=ingestion_settings,
        dispatcher=dispatcher,
    )

    progress = service.create_batch(uploads(3), created_by=OPERATOR)

    # Every item was attempted, and the upload itself still succeeded.
    assert attempted == [item.id for item in progress.items]
    assert service.get_item(progress.items[2].id).status is ItemStatus.AWAITING_REVIEW
