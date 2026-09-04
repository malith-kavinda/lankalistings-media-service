"""Lifecycle rules (PRD 10.1, 10.2).

Pure functions, so the transition table is tested exhaustively rather than by sampling.
"""

from __future__ import annotations

import pytest

from media_service.domain.item_state import (
    CLAIMABLE,
    COUNT_KEYS,
    IN_FLIGHT,
    ITEM_TRANSITIONS,
    PIPELINE_ACTIVE,
    PIPELINE_FAILED,
    PIPELINE_SUCCEEDED,
    REPROCESSABLE,
    RETRYABLE,
    STAGE_OF,
    BatchStatus,
    ItemStatus,
    Stage,
    counts_from,
    derive_batch_status,
    empty_counts,
    is_legal_transition,
    is_terminal_batch_status,
    stage_of,
)


def test_every_status_has_a_transition_entry() -> None:
    """A missing entry would raise KeyError inside a worker rather than rejecting cleanly."""
    assert set(ITEM_TRANSITIONS) == set(ItemStatus)


def test_every_status_has_a_stage() -> None:
    assert set(STAGE_OF) == set(ItemStatus)


def test_every_status_has_a_count_key() -> None:
    """FR-JOB-001 requires all seven counts, so no status may be unrepresented."""
    counts = counts_from(dict.fromkeys(ItemStatus, 1))
    assert sum(counts.values()) == len(ItemStatus)
    assert set(counts) == set(COUNT_KEYS)


def test_uploaded_is_reported_as_queued() -> None:
    """PRD 10.2: there is no `queued` item state; it is a count over `uploaded` items."""
    assert counts_from({ItemStatus.UPLOADED: 3})["queued"] == 3


def test_the_three_in_flight_states_collapse_into_one_processing_count() -> None:
    counts = counts_from(
        {
            ItemStatus.PREPROCESSING: 1,
            ItemStatus.OCR_PROCESSING: 2,
            ItemStatus.LLM_PROCESSING: 3,
        }
    )
    assert counts["processing"] == 6


def test_happy_path_is_walkable() -> None:
    path = [
        ItemStatus.UPLOADED,
        ItemStatus.PREPROCESSING,
        ItemStatus.OCR_PROCESSING,
        ItemStatus.LLM_PROCESSING,
        ItemStatus.AWAITING_REVIEW,
        ItemStatus.COMPLETED,
    ]
    for current, target in zip(path, path[1:], strict=False):
        assert is_legal_transition(current, target), f"{current} -> {target} should be legal"


@pytest.mark.parametrize("stage", list(IN_FLIGHT))
def test_any_processing_stage_can_fail_or_need_attention(stage: ItemStatus) -> None:
    assert is_legal_transition(stage, ItemStatus.NEEDS_ATTENTION)
    assert is_legal_transition(stage, ItemStatus.FAILED)


def test_no_ads_is_reachable_only_from_llm_processing() -> None:
    """AC-003. Only the extraction stage can conclude that a page holds no advertisement."""
    sources = [status for status in ItemStatus if is_legal_transition(status, ItemStatus.NO_ADS)]
    assert sources == [ItemStatus.LLM_PROCESSING]


@pytest.mark.parametrize("status", sorted(RETRYABLE))
def test_retry_returns_a_failed_item_to_the_queue(status: ItemStatus) -> None:
    """Retry is a transition, not a state: the item goes back to `uploaded`."""
    assert is_legal_transition(status, ItemStatus.UPLOADED)


def test_ocr_cannot_skip_the_extraction_stage() -> None:
    assert not is_legal_transition(ItemStatus.OCR_PROCESSING, ItemStatus.AWAITING_REVIEW)


def test_an_item_cannot_jump_straight_to_completed() -> None:
    for status in ItemStatus:
        if status is ItemStatus.AWAITING_REVIEW:
            continue
        assert not is_legal_transition(status, ItemStatus.COMPLETED), (
            f"{status} must not reach completed without review"
        )


def test_uploaded_cannot_re_enter_itself() -> None:
    """Otherwise a retry loop could spin without ever claiming the item."""
    assert not is_legal_transition(ItemStatus.UPLOADED, ItemStatus.UPLOADED)


def test_claimable_and_in_flight_do_not_overlap() -> None:
    assert not (CLAIMABLE & IN_FLIGHT)


def test_pipeline_partitions_cover_every_status_exactly_once() -> None:
    """A status in two partitions, or none, would corrupt every batch count."""
    partitions = [PIPELINE_ACTIVE, PIPELINE_SUCCEEDED, PIPELINE_FAILED]
    union: set[ItemStatus] = set()
    for partition in partitions:
        assert not (union & partition), f"Overlapping partition: {union & partition}"
        union |= partition
    assert union == set(ItemStatus)


def test_stage_lookup() -> None:
    assert stage_of(ItemStatus.OCR_PROCESSING) is Stage.OCR
    assert stage_of(ItemStatus.COMPLETED) is Stage.DONE


# ---------------------------------------------------------------------------------------------
# Batch derivation (FR-JOB-003, FR-JOB-004)
# ---------------------------------------------------------------------------------------------


def test_empty_batch_is_queued() -> None:
    assert derive_batch_status({}) is BatchStatus.QUEUED


def test_batch_is_queued_while_every_item_waits() -> None:
    assert derive_batch_status({ItemStatus.UPLOADED: 3}) is BatchStatus.QUEUED


@pytest.mark.parametrize("stage", sorted(IN_FLIGHT))
def test_batch_is_processing_once_any_item_is_in_flight(stage: ItemStatus) -> None:
    """An all-items-in-OCR batch must not report `queued`."""
    assert derive_batch_status({stage: 3}) is BatchStatus.PROCESSING


def test_batch_is_processing_while_any_item_remains_active() -> None:
    assert (
        derive_batch_status({ItemStatus.UPLOADED: 1, ItemStatus.AWAITING_REVIEW: 2})
        is BatchStatus.PROCESSING
    )


def test_batch_completes_when_every_item_succeeded() -> None:
    assert (
        derive_batch_status({ItemStatus.AWAITING_REVIEW: 2, ItemStatus.NO_ADS: 1})
        is BatchStatus.COMPLETED
    )


def test_awaiting_review_counts_as_terminal_for_the_batch() -> None:
    """The batch tracks extraction, not review. Otherwise progress never finishes (PRD 10.1)."""
    assert derive_batch_status({ItemStatus.AWAITING_REVIEW: 5}) is BatchStatus.COMPLETED


def test_a_mixed_outcome_is_partial_failed() -> None:
    """FR-JOB-004."""
    assert (
        derive_batch_status({ItemStatus.AWAITING_REVIEW: 9, ItemStatus.NEEDS_ATTENTION: 1})
        is BatchStatus.PARTIAL_FAILED
    )


def test_a_batch_where_everything_failed_is_failed() -> None:
    assert (
        derive_batch_status({ItemStatus.FAILED: 2, ItemStatus.NEEDS_ATTENTION: 1})
        is BatchStatus.FAILED
    )


def test_retrying_an_item_reopens_a_finished_batch() -> None:
    """PRD 10.1 as corrected in v1.1: partial_failed and failed return to processing."""
    finished = {ItemStatus.AWAITING_REVIEW: 2, ItemStatus.FAILED: 1}
    assert derive_batch_status(finished) is BatchStatus.PARTIAL_FAILED

    after_retry = {ItemStatus.AWAITING_REVIEW: 2, ItemStatus.UPLOADED: 1}
    assert derive_batch_status(after_retry) is BatchStatus.PROCESSING


def test_terminal_batch_statuses() -> None:
    assert is_terminal_batch_status(BatchStatus.COMPLETED)
    assert is_terminal_batch_status(BatchStatus.PARTIAL_FAILED)
    assert is_terminal_batch_status(BatchStatus.FAILED)
    assert not is_terminal_batch_status(BatchStatus.QUEUED)
    assert not is_terminal_batch_status(BatchStatus.PROCESSING)


def test_empty_counts_has_every_key_at_zero() -> None:
    counts = empty_counts()
    assert set(counts) == set(COUNT_KEYS)
    assert set(counts.values()) == {0}


def test_reprocessable_and_retryable_are_distinct() -> None:
    """Retrying a successful item is a different operation from retrying a failed one (AC-006)."""
    assert not (RETRYABLE & REPROCESSABLE)
