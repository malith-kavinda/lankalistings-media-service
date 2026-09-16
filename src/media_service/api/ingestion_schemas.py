"""Wire shapes for the ingestion endpoints (PRD 13.1).

`counts` publishes all seven keys, always, including the zeroes. A client that has to distinguish
"no items failed" from "the server did not mention failures" ends up guessing, and the progress bar
built on that guess is wrong exactly when something has gone wrong.

`stage` is derived from `status` on the way out rather than stored beside it, so the two can never
disagree.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from media_service.domain.views import BatchView, ItemView


class ItemResponse(BaseModel):
    id: str
    batch_id: str
    item_index: int
    status: str
    stage: str
    original_filename: str
    source_asset_id: str
    pipeline_generation: int
    attempt_count: int
    candidate_count: int
    warning_codes: list[str] = Field(default_factory=list)
    is_duplicate_in_batch: bool
    duplicate_of_item_id: str | None
    failed_stage: str | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    @classmethod
    def of(cls, item: ItemView) -> ItemResponse:
        return cls(
            id=item.id,
            batch_id=item.batch_id,
            item_index=item.item_index,
            status=item.status.value,
            stage=item.stage.value,
            original_filename=item.original_filename,
            source_asset_id=item.source_asset_id,
            pipeline_generation=item.pipeline_generation,
            attempt_count=item.attempt_count,
            candidate_count=item.candidate_count,
            warning_codes=list(item.warning_codes),
            is_duplicate_in_batch=item.is_duplicate_in_batch,
            duplicate_of_item_id=item.duplicate_of_item_id,
            failed_stage=item.failed_stage,
            error_code=item.error_code,
            error_message=item.error_message,
            created_at=item.created_at,
            updated_at=item.updated_at,
            completed_at=item.completed_at,
        )


class BatchResponse(BaseModel):
    id: str
    status: str
    created_by: str
    total_items: int
    counts: dict[str, int]
    correlation_id: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    items: list[ItemResponse] = Field(default_factory=list)

    @classmethod
    def of(cls, batch: BatchView) -> BatchResponse:
        return cls(
            id=batch.id,
            status=batch.status.value,
            created_by=batch.created_by,
            total_items=batch.total_items,
            counts=batch.counts,
            correlation_id=batch.correlation_id,
            created_at=batch.created_at,
            updated_at=batch.updated_at,
            completed_at=batch.completed_at,
            items=[ItemResponse.of(item) for item in batch.items],
        )


class RetryResponse(BaseModel):
    mode: str
    items: list[ItemResponse] = Field(default_factory=list)
