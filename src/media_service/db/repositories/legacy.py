"""The prototype's repository, backed by PostgreSQL instead of a JSON file.

A temporary adapter with a known end date. The prototype's endpoints speak in three frozen
dataclasses and a base64 `image_url`; the new schema speaks in batches, items, assets, and
derivatives. This translates between them so `MEDIA_REPOSITORY=sql` moves the old endpoints onto
the real database without changing a line of their code or their responses -- and so the cutover is
one environment variable in each direction. Phase 4 deletes this file along with the endpoints it
serves.

Two translations are worth knowing about.

**Every extraction needs an item.** `ocr_extractions.ingestion_item_id` is not nullable, because an
extraction with no source is not evidence of anything. The prototype's single-image endpoint has no
batch, so one is synthesised per call with `kind='legacy_single'` -- which is exactly what that
value in the vocabulary is for.

**The base64 image is decoded back into bytes.** The prototype inlines the whole image in the
advertisement row; here it becomes a content-addressed blob and a foreign key, so the same picture
uploaded twice is stored once, and the data URL is rebuilt on read. That gets the legacy path under
invariant 12 rather than leaving it as the one place that still duplicates images.
"""

from __future__ import annotations

import base64
from hashlib import sha256

from sqlalchemy import select

from media_service.db import tables
from media_service.db.uow import UnitOfWorkFactory
from media_service.domain import models
from media_service.domain.ids import (
    new_asset_id,
    new_batch_id,
    new_item_id,
)
from media_service.storage import FilesystemAssetStore, original_key

LEGACY_OPERATOR = "legacy"
LEGACY_BATCH_KIND = "legacy_single"

# The prototype's two-value status enum against the stored vocabulary (PRD 10.3).
STATUS_TO_STORED = {
    models.AdvertisementStatus.PENDING_REVIEW: "pending",
    models.AdvertisementStatus.ACTIVE: "active",
}
STATUS_FROM_STORED = {
    "pending": models.AdvertisementStatus.PENDING_REVIEW,
    "active": models.AdvertisementStatus.ACTIVE,
}


class SqlAlchemyMediaRepository:
    """Implements the prototype's `MediaRepository` protocol over the ingestion schema."""

    def __init__(
        self,
        *,
        unit_of_work: UnitOfWorkFactory,
        store: FilesystemAssetStore,
        public_image_url_mode: str = "data_url",
    ) -> None:
        self._unit_of_work = unit_of_work
        self._store = store
        self._public_image_url_mode = public_image_url_mode

    # -- assets --------------------------------------------------------------------------------

    def save_asset(self, asset: models.MediaAsset) -> None:
        with self._unit_of_work() as unit:
            unit.assets.get_or_create(
                tables.MediaAsset(
                    id=asset.id,
                    storage_key=original_key(asset.checksum, asset.content_type),
                    content_type=asset.content_type,
                    byte_size=asset.byte_size,
                    checksum_sha256=asset.checksum,
                    width=asset.width,
                    height=asset.height,
                    # The prototype never wrote the bytes anywhere. The row records what is known
                    # about the image; `absent` is the honest answer to "can I fetch it?".
                    bytes_state="absent",
                    created_at=asset.created_at,
                )
            )
            unit.commit()

    def get_asset(self, asset_id: str) -> models.MediaAsset | None:
        with self._unit_of_work() as unit:
            row = unit.assets.get(asset_id)
            return None if row is None else _asset_of(row)

    # -- extractions ---------------------------------------------------------------------------

    def save_extraction(self, extraction: models.ExtractionResult) -> None:
        with self._unit_of_work() as unit:
            asset = unit.assets.get(extraction.asset_id)
            if asset is None:  # pragma: no cover - the service always saves the asset first
                return
            item = _legacy_item(unit, asset)
            unit.flush()

            unit.artifacts.record_ocr(
                tables.OcrExtraction(
                    id=extraction.id,
                    ingestion_item_id=item.id,
                    source_asset_id=asset.id,
                    generation=1,
                    attempt=1,
                    status=extraction.status.value,
                    raw_text=extraction.raw_text,
                    engine=extraction.engine,
                    engine_version=extraction.model_version,
                    languages=extraction.language,
                    width=asset.width,
                    height=asset.height,
                    duration_ms=extraction.processing_ms,
                    started_at=extraction.created_at,
                    completed_at=extraction.created_at,
                    created_at=extraction.created_at,
                )
            )
            unit.commit()

    def get_extraction(self, extraction_id: str) -> models.ExtractionResult | None:
        with self._unit_of_work() as unit:
            row = unit.artifacts.get_ocr(extraction_id)
            return None if row is None else _extraction_of(row)

    # -- advertisements ------------------------------------------------------------------------

    def save_advertisement(self, advertisement: models.Advertisement) -> None:
        with self._unit_of_work() as unit:
            asset_id = self._store_inline_image(unit, advertisement.image_url)
            existing = unit.session.get(tables.Advertisement, advertisement.id)
            row = existing or tables.Advertisement(id=advertisement.id)

            row.title = advertisement.title
            row.description = advertisement.description
            row.category = advertisement.category
            row.location = advertisement.location
            row.price = advertisement.price
            row.status = STATUS_TO_STORED[advertisement.status]
            row.origin = "ocr_heuristic" if advertisement.source_text else "manual"
            row.source_text = advertisement.source_text
            row.extraction_confidence = advertisement.extraction_confidence
            row.created_at = advertisement.created_at
            if asset_id is not None:
                row.image_asset_id = asset_id

            if existing is None:
                unit.session.add(row)
            unit.commit()

    def get_advertisement(self, advertisement_id: str) -> models.Advertisement | None:
        with self._unit_of_work() as unit:
            row = unit.session.get(tables.Advertisement, advertisement_id)
            return None if row is None else self._advertisement_of(unit, row)

    def list_advertisements(self) -> list[models.Advertisement]:
        with self._unit_of_work() as unit:
            rows = unit.session.scalars(
                select(tables.Advertisement).order_by(tables.Advertisement.created_at)
            ).all()
            return [self._advertisement_of(unit, row) for row in rows]

    # -- images --------------------------------------------------------------------------------

    def _store_inline_image(self, unit, data_url: str) -> str | None:  # type: ignore[no-untyped-def]
        """Turn the prototype's inline base64 image into a stored, deduplicated blob."""
        decoded = _decode_data_url(data_url)
        if decoded is None:
            return None
        content_type, payload = decoded
        checksum = sha256(payload).hexdigest()
        key = original_key(checksum, content_type)
        self._store.write_bytes(payload, key)

        asset, _ = unit.assets.get_or_create(
            tables.MediaAsset(
                id=new_asset_id(),
                storage_key=key,
                content_type=content_type,
                byte_size=len(payload),
                checksum_sha256=checksum,
            )
        )
        unit.flush()
        return asset.id

    def _advertisement_of(self, unit, row: tables.Advertisement) -> models.Advertisement:  # type: ignore[no-untyped-def]
        return models.Advertisement(
            id=row.id,
            title=row.title,
            price=row.price,
            category=row.category,
            location=row.location,
            description=row.description,
            image_url=self._image_url(unit, row),
            status=STATUS_FROM_STORED.get(row.status, models.AdvertisementStatus.PENDING_REVIEW),
            created_at=row.created_at,
            source_text=row.source_text,
            extraction_confidence=row.extraction_confidence,
        )

    def _image_url(self, unit, row: tables.Advertisement) -> str:  # type: ignore[no-untyped-def]
        """Rebuild what the public clients already read.

        `data_url` keeps `frontend-web` and `mobile` working with no change at all, which is the
        only reason the base64 round trip exists; `http_url` is the mode they move to once they can
        fetch an image by URL.
        """
        if row.image_asset_id is None:
            return ""
        if self._public_image_url_mode == "http_url":
            return f"/api/v1/media/assets/{row.image_asset_id}/original"

        asset = unit.assets.get(row.image_asset_id)
        if asset is None or asset.bytes_state != "present":
            return ""
        try:
            payload = self._store.read_bytes(asset.storage_key)
        except FileNotFoundError:
            return ""
        return f"data:{asset.content_type};base64,{base64.b64encode(payload).decode()}"


def _legacy_item(unit, asset: tables.MediaAsset) -> tables.IngestionItem:  # type: ignore[no-untyped-def]
    """The synthetic batch and item a single-image extraction hangs from."""
    existing = unit.session.scalar(
        select(tables.IngestionItem).where(tables.IngestionItem.source_asset_id == asset.id)
    )
    if existing is not None:
        return existing

    batch = tables.IngestionBatch(
        id=new_batch_id(),
        kind=LEGACY_BATCH_KIND,
        created_by=LEGACY_OPERATOR,
        total_items=1,
    )
    unit.batches.add(batch)
    unit.flush()

    return unit.items.add(
        tables.IngestionItem(
            id=new_item_id(),
            batch_id=batch.id,
            item_index=0,
            source_asset_id=asset.id,
            original_filename="legacy-upload",
            declared_content_type=asset.content_type,
            status="awaiting_review",
        )
    )


def _asset_of(row: tables.MediaAsset) -> models.MediaAsset:
    return models.MediaAsset(
        id=row.id,
        content_type=row.content_type,
        byte_size=row.byte_size,
        checksum=row.checksum_sha256,
        width=row.width,
        height=row.height,
        created_at=row.created_at,
    )


def _extraction_of(row: tables.OcrExtraction) -> models.ExtractionResult:
    return models.ExtractionResult(
        id=row.id,
        asset_id=row.source_asset_id,
        status=models.ExtractionStatus(_legacy_status(row.status)),
        raw_text=row.raw_text,
        engine=row.engine,
        model_version=row.engine_version or "",
        language=row.languages,
        confidence="high",
        processing_ms=row.duration_ms or 0,
        created_at=row.created_at,
    )


def _legacy_status(status: str) -> str:
    """Map the richer OCR vocabulary back onto the prototype's three values."""
    if status in {"completed", "empty"}:
        return "completed"
    if status == "unsupported":
        return "unsupported"
    return "failed"


def _decode_data_url(value: str) -> tuple[str, bytes] | None:
    if not value.startswith("data:") or ";base64," not in value:
        return None
    header, payload = value.split(";base64,", 1)
    content_type = header.removeprefix("data:") or "application/octet-stream"
    try:
        return content_type, base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error):
        return None
