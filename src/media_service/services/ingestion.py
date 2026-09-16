"""Batch ingestion: accept an upload set, make it durable, hand it to the workers.

The sequence is fixed, and each step is where it is for a reason:

1. **Stage and validate everything.** Nothing reaches the database until the whole set is known to
   be acceptable.
2. **Reserve the batch row and commit it immediately**, with `committed_at` still NULL. The unique
   index on `(created_by, idempotency_key)` -- not application logic -- decides who wins a race
   between two identical requests.
3. **Promote the blobs.** Files land before the rows that reference them, so a crash in between
   leaves a file nothing points at (the orphan sweep collects it) rather than a row pointing at a
   file that is not there.
4. **One transaction** for assets, items, and the batch's own completion, then commit.
5. **Dispatch**, after the commit, never before.

`committed_at` is what lets step 2 and step 4 be separate transactions without inventing a
`receiving` batch state: a concurrent duplicate that finds the row can tell "being written" from
"ready to replay" by looking at that one column.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from sqlalchemy.exc import IntegrityError

from media_service.api.errors import (
    BatchNotFoundError,
    IdempotencyInProgressError,
    IdempotencyKeyConflictError,
    ItemNotFoundError,
    NothingToRetryError,
    ReprocessRefusedError,
)
from media_service.config import Settings
from media_service.db.tables import IngestionBatch, IngestionItem, MediaAsset
from media_service.db.uow import UnitOfWork, UnitOfWorkFactory
from media_service.domain.ids import new_asset_id, new_batch_id, new_item_id
from media_service.domain.item_state import (
    REPROCESSABLE,
    RETRYABLE,
    BatchStatus,
    ItemStatus,
    counts_from,
    derive_batch_status,
)
from media_service.domain.views import BatchView, ItemView, RetryOutcome
from media_service.jobs.protocol import JobDispatcher
from media_service.services.uploads import (
    IncomingFile,
    StagedImage,
    UploadLimits,
    stage_upload_set,
)
from media_service.storage import FilesystemAssetStore, original_key

DUPLICATE_WARNING = "DUPLICATE_IMAGE_IN_BATCH"

# How long a duplicate request waits for the original to finish being written before giving up.
# Long enough to absorb a normal commit, short enough that a client is not left hanging.
IDEMPOTENCY_WAIT_SECONDS = 1.0
IDEMPOTENCY_POLL_SECONDS = 0.05


def request_fingerprint(created_by: str, checksums: Sequence[str]) -> str:
    """Identifies *what was asked for*, so a reused idempotency key can be judged.

    Sorted, because the same images uploaded in a different order are the same request; the item
    order is a property of the response, not of the intent.
    """
    material = "|".join([created_by, str(len(checksums)), *sorted(checksums)])
    return sha256(material.encode()).hexdigest()


class IngestionService:
    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        store: FilesystemAssetStore,
        settings: Settings,
        dispatcher: JobDispatcher,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._store = store
        self._settings = settings
        self._dispatcher = dispatcher

    # -- creation ------------------------------------------------------------------------------

    def create_batch(
        self,
        files: Sequence[IncomingFile],
        *,
        created_by: str,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> BatchView:
        staged = stage_upload_set(self._store, files, limits=self._limits())
        fingerprint = request_fingerprint(created_by, [image.checksum for image in staged])

        try:
            batch_id = self._reserve(
                created_by=created_by,
                idempotency_key=idempotency_key,
                fingerprint=fingerprint,
                correlation_id=correlation_id,
                total_items=len(staged),
            )
        except IntegrityError:
            self._discard(staged)
            if idempotency_key is None:
                # Nothing about a keyless request can legitimately collide, so this is a real
                # failure rather than the duplicate-request path.
                raise
            # Someone else holds this key. Either they asked for the same thing -- in which case
            # their answer is ours -- or they did not, which is a client bug worth a 409.
            return self._replay(
                created_by=created_by, idempotency_key=idempotency_key, fingerprint=fingerprint
            )

        try:
            progress = self._record(batch_id, staged, correlation_id=correlation_id)
        except BaseException:
            self._discard(staged)
            raise

        self._dispatcher.enqueue_batch(
            batch_id, [item.id for item in progress.items], correlation_id=correlation_id
        )
        return progress

    def _reserve(
        self,
        *,
        created_by: str,
        idempotency_key: str | None,
        fingerprint: str,
        correlation_id: str | None,
        total_items: int,
    ) -> str:
        """Claim the idempotency key, committing before any work depends on having won it."""
        batch_id = new_batch_id()
        with self._unit_of_work() as unit:
            unit.batches.add(
                IngestionBatch(
                    id=batch_id,
                    created_by=created_by,
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    correlation_id=correlation_id,
                    total_items=total_items,
                    status=BatchStatus.QUEUED.value,
                )
            )
            unit.commit()
        return batch_id

    def _record(
        self, batch_id: str, staged: Sequence[StagedImage], *, correlation_id: str | None
    ) -> BatchView:
        promoted = self._promote(staged)

        with self._unit_of_work() as unit:
            batch = unit.batches.get(batch_id)
            if batch is None:  # pragma: no cover - the row was committed a moment ago
                raise BatchNotFoundError(batch_id)

            assets = [self._asset_for(unit, image) for image in staged]
            # Items carry foreign keys to assets, and no ORM relationship exists to order the
            # flush by, so the parents are written first, explicitly.
            unit.flush()

            items = [
                unit.items.add(
                    IngestionItem(
                        id=new_item_id(),
                        batch_id=batch_id,
                        item_index=image.index,
                        source_asset_id=asset.id,
                        original_filename=image.filename,
                        declared_content_type=image.declared_content_type,
                        correlation_id=correlation_id,
                    )
                )
                for image, asset in zip(staged, assets, strict=True)
            ]
            unit.flush()

            self._flag_duplicates(items, staged)
            unit.flush()

            batch.total_items = len(items)
            unit.batches.refresh_projection(batch_id)
            unit.batches.mark_committed(
                batch, response_snapshot=_snapshot(batch, items, promoted_bytes=promoted)
            )
            unit.commit()
            return _progress(unit, batch)

    def _asset_for(self, unit: UnitOfWork, image: StagedImage) -> MediaAsset:
        """The row for these bytes, created once no matter how often they are uploaded."""
        asset, _ = unit.assets.get_or_create(
            MediaAsset(
                id=new_asset_id(),
                storage_key=original_key(image.checksum, image.content_type),
                content_type=image.content_type,
                image_format=image.image_format,
                byte_size=image.byte_size,
                checksum_sha256=image.checksum,
                width=image.width,
                height=image.height,
                exif_orientation=image.exif_orientation,
            )
        )
        return asset

    @staticmethod
    def _flag_duplicates(items: Sequence[IngestionItem], staged: Sequence[StagedImage]) -> None:
        """Mark repeats of the same image within one upload (FR-ING-008).

        Flagged, never dropped. An operator who uploads the same page twice by accident and one who
        does it deliberately look identical here, so both items stay independently visible and
        retryable and the decision is left to the person reviewing them.

        Applied after the items are flushed: `duplicate_of_item_id` points at a sibling row, and
        the foreign key needs that sibling to exist.
        """
        first_by_checksum: dict[str, str] = {}
        for item, image in zip(items, staged, strict=True):
            original = first_by_checksum.get(image.checksum)
            if original is None:
                first_by_checksum[image.checksum] = item.id
                continue
            item.is_duplicate_in_batch = True
            item.duplicate_of_item_id = original
            # Reassigned rather than appended: an ARRAY column is not mutation-tracked, so an
            # in-place append would never reach the UPDATE.
            item.warning_codes = [*(item.warning_codes or []), DUPLICATE_WARNING]

    def _promote(self, staged: Sequence[StagedImage]) -> int:
        """Move staged bytes to their content-addressed keys. Returns how many were written."""
        written = 0
        for image in staged:
            blob = self._store.promote(
                image.staged, original_key(image.checksum, image.content_type)
            )
            written += int(blob.was_written)
        return written

    def _discard(self, staged: Sequence[StagedImage]) -> None:
        for image in staged:
            self._store.discard(image.staged)

    def _replay(
        self, *, created_by: str, idempotency_key: str, fingerprint: str
    ) -> BatchView:
        deadline = time.monotonic() + IDEMPOTENCY_WAIT_SECONDS
        while True:
            with self._unit_of_work() as unit:
                winner = unit.batches.find_by_idempotency_key(
                    created_by=created_by, key=idempotency_key
                )
                if winner is None:  # pragma: no cover - the row that just conflicted with us
                    raise IdempotencyInProgressError

                if winner.request_fingerprint != fingerprint:
                    raise IdempotencyKeyConflictError

                if winner.committed_at is not None:
                    return _progress(unit, winner, replayed=True)

            # The winner is mid-write. Wait for its second commit rather than answering from a
            # half-built batch.
            if time.monotonic() >= deadline:
                raise IdempotencyInProgressError
            time.sleep(IDEMPOTENCY_POLL_SECONDS)

    # -- reads ---------------------------------------------------------------------------------

    def get_batch(self, batch_id: str) -> BatchView:
        with self._unit_of_work() as unit:
            batch = unit.batches.get(batch_id)
            if batch is None:
                raise BatchNotFoundError(batch_id)
            return _progress(unit, batch)

    def list_batches(self, *, limit: int = 50, offset: int = 0) -> list[BatchView]:
        with self._unit_of_work() as unit:
            return [
                _progress(unit, batch)
                for batch in unit.batches.list_recent(limit=limit, offset=offset)
            ]

    def get_item(self, item_id: str) -> ItemView:
        with self._unit_of_work() as unit:
            item = unit.items.get(item_id)
            if item is None:
                raise ItemNotFoundError(item_id)
            return ItemView.of(item)

    # -- retry ---------------------------------------------------------------------------------

    def retry_item(
        self,
        item_id: str,
        *,
        mode: str = "resume",
        actor_id: str,
        force: bool = False,
        correlation_id: str | None = None,
    ) -> RetryOutcome:
        """Put an item back in the queue.

        Two modes, because AC-006 leaves the distinction open and the two have different costs:

        *resume* keeps the generation, so every artifact already validated for it is reused -- a
        re-run after the LLM succeeded but the process died is free, and re-running an item that
        actually finished is a no-op the constraints enforce.

        *reprocess* bumps the generation, which is the only way to get new candidates. It is
        refused when a reviewer has already accepted or rejected something from this item, because
        superseding those would discard a human decision.
        """
        with self._unit_of_work() as unit:
            item = unit.items.get(item_id)
            if item is None:
                raise ItemNotFoundError(item_id)

            status = ItemStatus(item.status)
            allowed = RETRYABLE if mode == "resume" else RETRYABLE | REPROCESSABLE
            if status not in allowed:
                raise NothingToRetryError(item_id=item_id, status=status.value)

            fields: dict[str, Any] = {
                # A person asked for this run, so the automatic attempt budget starts again. The
                # cap exists to stop a failing item looping unattended, not to stop an operator.
                "attempt_count": 0,
                "error_code": None,
                "error_message": None,
                "failed_stage": None,
            }
            if mode == "reprocess":
                if unit.candidates.has_decided_candidate(item_id) and not force:
                    raise ReprocessRefusedError(
                        f"Item {item_id} has candidates a reviewer already decided on. "
                        "Reprocessing would supersede them; pass force=true to proceed."
                    )
                generation = item.pipeline_generation + 1
                unit.candidates.supersede_undecided(item_id, new_generation=generation)
                fields["pipeline_generation"] = generation

            item = unit.items.transition(
                item, target=ItemStatus.UPLOADED, expected=status, release_claim=True, **fields
            )
            unit.events.record(
                subject_type="ingestion_item",
                action="reprocessed",
                actor_id=actor_id,
                ingestion_item_id=item_id,
                ingestion_batch_id=item.batch_id,
                reason_code=mode.upper(),
                correlation_id=correlation_id,
            )
            unit.commit()
            view = ItemView.of(item)

        self._dispatcher.enqueue_item(item_id, correlation_id=correlation_id)
        return RetryOutcome(mode=mode, items=(view,))

    def retry_batch(
        self,
        batch_id: str,
        *,
        actor_id: str,
        correlation_id: str | None = None,
    ) -> RetryOutcome:
        """Requeue every item in a batch that stopped on a failure.

        Items that succeeded are left alone rather than refused: the operator's intent for a
        partly-failed batch is "run what did not work", and failing the whole call because one item
        is fine would make the button useless.
        """
        with self._unit_of_work() as unit:
            if unit.batches.get(batch_id) is None:
                raise BatchNotFoundError(batch_id)
            retryable = [
                item.id
                for item in unit.items.list_for_batch(batch_id)
                if ItemStatus(item.status) in RETRYABLE
            ]

        retried = tuple(
            self.retry_item(
                item_id, mode="resume", actor_id=actor_id, correlation_id=correlation_id
            ).items[0]
            for item_id in retryable
        )
        return RetryOutcome(mode="resume", items=retried)

    # -- helpers -------------------------------------------------------------------------------

    def _limits(self) -> UploadLimits:
        return UploadLimits(
            max_images_per_batch=self._settings.max_images_per_batch,
            max_image_bytes=self._settings.max_image_bytes,
            max_batch_bytes=self._settings.max_batch_bytes,
        )


def _progress(unit: UnitOfWork, batch: IngestionBatch, *, replayed: bool = False) -> BatchView:
    """Read a batch the way a caller should: counts recomputed, never trusted from the cache.

    One GROUP BY over at most 25 rows. The stored copy exists so the batch *list* endpoint can sort
    and filter; a reader of one batch has no reason to depend on it being fresh (FR-JOB-003).
    """
    statuses = unit.batches.status_counts(batch.id)
    return BatchView.of(
        batch,
        status=derive_batch_status(statuses),
        counts=counts_from(statuses),
        items=tuple(ItemView.of(item) for item in unit.items.list_for_batch(batch.id)),
        replayed=replayed,
    )


def _snapshot(
    batch: IngestionBatch, items: Sequence[IngestionItem], *, promoted_bytes: int
) -> dict[str, Any]:
    """The accepted-response body, stored so a repeat request is answered identically."""
    return {
        "batch_id": batch.id,
        "created_by": batch.created_by,
        "total_items": len(items),
        "stored_blobs": promoted_bytes,
        "items": [
            {
                "id": item.id,
                "item_index": item.item_index,
                "original_filename": item.original_filename,
                "is_duplicate_in_batch": item.is_duplicate_in_batch,
            }
            for item in items
        ],
    }
