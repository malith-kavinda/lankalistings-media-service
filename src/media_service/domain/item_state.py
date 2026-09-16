"""Item and batch lifecycle rules (PRD 10.1, 10.2).

Pure functions. No database, no session, no I/O -- so the rules can be tested exhaustively and are
cheap to reason about.

Enforcement is deliberately not a CHECK constraint or a trigger. A CHECK sees only the row being
written, not the row it replaces, so it cannot express "this transition is legal from that state". A
trigger could, but is invisible to Alembic autogenerate and hard to test. The pairing used instead
is this module for legality plus a compare-and-swap UPDATE for concurrency, which together cover
both.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ItemStatus(StrEnum):
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    OCR_PROCESSING = "ocr_processing"
    LLM_PROCESSING = "llm_processing"
    AWAITING_REVIEW = "awaiting_review"
    NO_ADS = "no_ads"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"
    COMPLETED = "completed"


class BatchStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"


class Stage(StrEnum):
    QUEUE = "queue"
    PREPROCESS = "preprocess"
    OCR = "ocr"
    LLM = "llm"
    REVIEW = "review"
    DONE = "done"


# `current_stage` is derived rather than stored (PRD 12.2). Two columns encoding one truth drift,
# and a stage disagreeing with a status cannot be reconciled after the fact.
STAGE_OF: Final[dict[ItemStatus, Stage]] = {
    ItemStatus.UPLOADED: Stage.QUEUE,
    ItemStatus.PREPROCESSING: Stage.PREPROCESS,
    ItemStatus.OCR_PROCESSING: Stage.OCR,
    ItemStatus.LLM_PROCESSING: Stage.LLM,
    ItemStatus.AWAITING_REVIEW: Stage.REVIEW,
    ItemStatus.NO_ADS: Stage.DONE,
    ItemStatus.COMPLETED: Stage.DONE,
    ItemStatus.NEEDS_ATTENTION: Stage.DONE,
    ItemStatus.FAILED: Stage.DONE,
}

# Retry is a transition, not a state (PRD 10.2). A retried item returns to `uploaded` and is claimed
# again; an item waiting to retry is `uploaded` with a non-zero attempt count and a future
# run_after. Reprocessing a finished item is allowed only through the explicit reprocess path,
# which is why `completed`, `no_ads`, and `awaiting_review` can also return to `uploaded`.
#
# Each in-flight state can also return to `uploaded`. That edge is *requeue after abnormal
# termination*: a worker was killed or its lease expired, so the row is claimable again. It is not a
# retry -- nobody asked for it and no review event is written -- which is why the repository exposes
# it as one named operation (`requeue_abandoned`) rather than as a general transition anyone may
# take.
ITEM_TRANSITIONS: Final[dict[ItemStatus, frozenset[ItemStatus]]] = {
    ItemStatus.UPLOADED: frozenset(
        {ItemStatus.PREPROCESSING, ItemStatus.NEEDS_ATTENTION, ItemStatus.FAILED}
    ),
    ItemStatus.PREPROCESSING: frozenset(
        {
            ItemStatus.OCR_PROCESSING,
            ItemStatus.NEEDS_ATTENTION,
            ItemStatus.FAILED,
            ItemStatus.UPLOADED,
        }
    ),
    ItemStatus.OCR_PROCESSING: frozenset(
        {
            ItemStatus.LLM_PROCESSING,
            ItemStatus.NEEDS_ATTENTION,
            ItemStatus.FAILED,
            ItemStatus.UPLOADED,
        }
    ),
    ItemStatus.LLM_PROCESSING: frozenset(
        {
            ItemStatus.AWAITING_REVIEW,
            ItemStatus.NO_ADS,
            ItemStatus.NEEDS_ATTENTION,
            ItemStatus.FAILED,
            ItemStatus.UPLOADED,
        }
    ),
    ItemStatus.AWAITING_REVIEW: frozenset({ItemStatus.COMPLETED, ItemStatus.UPLOADED}),
    ItemStatus.NO_ADS: frozenset({ItemStatus.UPLOADED}),
    ItemStatus.COMPLETED: frozenset({ItemStatus.UPLOADED}),
    ItemStatus.NEEDS_ATTENTION: frozenset({ItemStatus.UPLOADED, ItemStatus.FAILED}),
    ItemStatus.FAILED: frozenset({ItemStatus.UPLOADED}),
}

# States a worker may claim work from, and states where work is in flight.
CLAIMABLE: Final = frozenset({ItemStatus.UPLOADED})
IN_FLIGHT: Final = frozenset(
    {ItemStatus.PREPROCESSING, ItemStatus.OCR_PROCESSING, ItemStatus.LLM_PROCESSING}
)

# Terminal *for the batch*. `awaiting_review` counts: the batch tracks extraction progress, not
# review progress, or the batch progress view could never reach completion (PRD 10.1).
PIPELINE_ACTIVE: Final = frozenset({ItemStatus.UPLOADED}) | IN_FLIGHT
PIPELINE_SUCCEEDED: Final = frozenset(
    {ItemStatus.AWAITING_REVIEW, ItemStatus.NO_ADS, ItemStatus.COMPLETED}
)
PIPELINE_FAILED: Final = frozenset({ItemStatus.NEEDS_ATTENTION, ItemStatus.FAILED})

# Retry is meaningful only from these. Retrying a successful item is a different operation.
RETRYABLE: Final = frozenset({ItemStatus.NEEDS_ATTENTION, ItemStatus.FAILED})
REPROCESSABLE: Final = frozenset(
    {ItemStatus.AWAITING_REVIEW, ItemStatus.NO_ADS, ItemStatus.COMPLETED}
)

# The counts key used by the API. `queued` is a count, not an item state (PRD 13.1).
COUNT_KEY_OF: Final[dict[ItemStatus, str]] = {
    ItemStatus.UPLOADED: "queued",
    ItemStatus.PREPROCESSING: "processing",
    ItemStatus.OCR_PROCESSING: "processing",
    ItemStatus.LLM_PROCESSING: "processing",
    ItemStatus.AWAITING_REVIEW: "awaiting_review",
    ItemStatus.NO_ADS: "no_ads",
    ItemStatus.NEEDS_ATTENTION: "needs_attention",
    ItemStatus.FAILED: "failed",
    ItemStatus.COMPLETED: "completed",
}

COUNT_KEYS: Final = (
    "queued",
    "processing",
    "awaiting_review",
    "no_ads",
    "needs_attention",
    "failed",
    "completed",
)


def is_legal_transition(current: ItemStatus, target: ItemStatus) -> bool:
    return target in ITEM_TRANSITIONS[current]


def stage_of(status: ItemStatus) -> Stage:
    return STAGE_OF[status]


def empty_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def counts_from(statuses: dict[ItemStatus, int]) -> dict[str, int]:
    """Roll item statuses up into the seven API count keys."""
    counts = empty_counts()
    for status, quantity in statuses.items():
        counts[COUNT_KEY_OF[status]] += quantity
    return counts


def derive_batch_status(statuses: dict[ItemStatus, int]) -> BatchStatus:
    """Batch status is derived from its items, never assigned by a client (FR-JOB-003)."""
    total = sum(statuses.values())
    if total == 0:
        return BatchStatus.QUEUED

    waiting = statuses.get(ItemStatus.UPLOADED, 0)
    active = sum(quantity for status, quantity in statuses.items() if status in PIPELINE_ACTIVE)
    failed = sum(quantity for status, quantity in statuses.items() if status in PIPELINE_FAILED)

    # Nothing has started yet. This is also the state a batch returns to when its only remaining
    # work was re-queued by a retry.
    if waiting == total:
        return BatchStatus.QUEUED
    if active > 0:
        return BatchStatus.PROCESSING

    # Every item has stopped moving.
    if failed == total:
        return BatchStatus.FAILED
    if failed > 0:
        return BatchStatus.PARTIAL_FAILED
    return BatchStatus.COMPLETED


def is_terminal_batch_status(status: BatchStatus) -> bool:
    return status in {BatchStatus.COMPLETED, BatchStatus.PARTIAL_FAILED, BatchStatus.FAILED}
