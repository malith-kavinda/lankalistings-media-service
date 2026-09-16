"""Operational commands.

    python -m media_service.cli import-legacy-json [--dry-run] [--include-extractions]
    python -m media_service.cli storage-gc [--dry-run] [--min-age-seconds N]

Both are safe to run twice. The import is idempotent through `legacy_json_id` and the checksum
index, so a half-finished run is resumed by running it again rather than by cleaning up after it.
The sweep only ever removes files that no row references and that nothing has touched recently.

The JSON file is read and never written or deleted. Rolling back the cutover is
`MEDIA_REPOSITORY=json` and nothing else.
"""

from __future__ import annotations

import argparse
import base64
import sys
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from sqlalchemy import select

from media_service.config import Settings, get_settings
from media_service.db import tables
from media_service.db.base import utcnow
from media_service.db.engine import build_session_factory, cached_engine
from media_service.db.uow import UnitOfWork, UnitOfWorkFactory
from media_service.domain import models
from media_service.domain.ids import (
    new_advertisement_id,
    new_asset_id,
    new_batch_id,
    new_item_id,
    new_provenance_id,
)
from media_service.domain.repository import JsonMediaRepository
from media_service.storage import FilesystemAssetStore, original_key

LEGACY_BATCH_KIND = "legacy_import"
LEGACY_OPERATOR = "legacy-import"

# An orphan younger than this may simply be a file another request is between writing and
# committing. Files land before rows, so the gap is real and the age floor is what keeps the sweep
# from racing it.
DEFAULT_MIN_AGE_SECONDS = 3600


@dataclass
class ImportReport:
    advertisements: int = 0
    assets: int = 0
    extractions: int = 0
    skipped: int = 0
    deduplicated: int = 0
    warnings: list[str] = field(default_factory=list)

    def render(self, *, dry_run: bool) -> str:
        prefix = "Would import" if dry_run else "Imported"
        lines = [
            f"{prefix}: {self.advertisements} advertisement(s), "
            f"{self.assets} asset(s), {self.extractions} extraction(s).",
            f"Skipped {self.skipped} already present; {self.deduplicated} image(s) "
            "resolved to bytes already stored.",
        ]
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def import_legacy_json(
    *,
    unit_of_work: UnitOfWorkFactory,
    store: FilesystemAssetStore,
    path: Path,
    include_extractions: bool = False,
    dry_run: bool = False,
) -> ImportReport:
    report = ImportReport()
    if not path.exists():
        report.warnings.append(f"No legacy file at {path}; nothing to import.")
        return report

    source = JsonMediaRepository(path)
    advertisements = source.list_advertisements()

    with unit_of_work() as unit:
        batch = _import_batch(unit)

        for index, advertisement in enumerate(advertisements):
            if _already_imported(unit, advertisement.id):
                report.skipped += 1
                continue
            _import_advertisement(
                unit,
                store,
                batch=batch,
                advertisement=advertisement,
                index=index,
                report=report,
            )

        if include_extractions:
            _import_extractions(unit, source, batch=batch, report=report)

        batch.total_items = len(unit.items.list_for_batch(batch.id))
        unit.batches.refresh_projection(batch.id)

        if dry_run:
            # Everything above ran against a real transaction and is now discarded, so a dry run
            # reports what would actually happen rather than what the code guesses would.
            unit.rollback()
        else:
            unit.commit()

    return report


def _import_batch(unit: UnitOfWork) -> tables.IngestionBatch:
    """One batch for every legacy import, reused across runs so a re-run adds to it."""
    existing = unit.session.scalar(
        select(tables.IngestionBatch)
        .where(tables.IngestionBatch.kind == LEGACY_BATCH_KIND)
        .order_by(tables.IngestionBatch.id)
        .limit(1)
    )
    if existing is not None:
        return existing

    batch = unit.batches.add(
        tables.IngestionBatch(
            id=new_batch_id(),
            kind=LEGACY_BATCH_KIND,
            created_by=LEGACY_OPERATOR,
            committed_at=utcnow(),
        )
    )
    unit.flush()
    return batch


def _already_imported(unit: UnitOfWork, legacy_id: str) -> bool:
    return (
        unit.session.scalar(
            select(tables.Advertisement.id).where(
                tables.Advertisement.legacy_json_id == legacy_id
            )
        )
        is not None
    )


def _import_advertisement(
    unit: UnitOfWork,
    store: FilesystemAssetStore,
    *,
    batch: tables.IngestionBatch,
    advertisement: models.Advertisement,
    index: int,
    report: ImportReport,
) -> None:
    asset = _asset_from_data_url(unit, store, advertisement.image_url, report=report)
    item = _item_for(unit, batch=batch, asset=asset, index=index, filename="legacy-advertisement")
    unit.flush()

    row = tables.Advertisement(
        id=new_advertisement_id_for(advertisement.id),
        title=advertisement.title,
        description=advertisement.description,
        category=advertisement.category,
        location=advertisement.location,
        price=advertisement.price,
        status="active" if advertisement.status is models.AdvertisementStatus.ACTIVE else "pending",
        origin="legacy_import",
        source_text=advertisement.source_text,
        extraction_confidence=advertisement.extraction_confidence,
        image_asset_id=asset.id if asset is not None else None,
        legacy_json_id=advertisement.id,
        created_at=advertisement.created_at,
    )
    unit.session.add(row)
    unit.flush()
    report.advertisements += 1

    if item is not None:
        unit.candidates.record(
            tables.AdvertisementProvenance(
                id=new_provenance_id(),
                advertisement_id=row.id,
                ingestion_batch_id=batch.id,
                ingestion_item_id=item.id,
                source_asset_id=item.source_asset_id,
                generation=1,
                candidate_index=index,
                candidate_state="linked" if row.status == "active" else "pending_publish",
                extracted_values={"source": "legacy_json"},
            )
        )


def _import_extractions(
    unit: UnitOfWork,
    source: JsonMediaRepository,
    *,
    batch: tables.IngestionBatch,
    report: ImportReport,
) -> None:
    """Import the prototype's assets and OCR results.

    The bytes behind these are genuinely gone -- the prototype stored metadata only -- so the rows
    land with `bytes_state='absent'`. They are still worth having: an extraction is the evidence a
    moderator reads, and it outlives the pixels.
    """
    for extraction in _legacy_extractions(source):
        if unit.artifacts.get_ocr(extraction.id) is not None:
            report.skipped += 1
            continue

        legacy_asset = source.get_asset(extraction.asset_id)
        if legacy_asset is None:
            report.warnings.append(
                f"Extraction {extraction.id} references asset {extraction.asset_id}, "
                "which is not in the file."
            )
            continue

        asset, created = unit.assets.get_or_create(
            tables.MediaAsset(
                id=new_asset_id(),
                storage_key=original_key(legacy_asset.checksum, legacy_asset.content_type),
                content_type=legacy_asset.content_type,
                byte_size=legacy_asset.byte_size,
                checksum_sha256=legacy_asset.checksum,
                width=legacy_asset.width,
                height=legacy_asset.height,
                bytes_state="absent",
                created_at=legacy_asset.created_at,
            )
        )
        report.assets += int(created)
        report.deduplicated += int(not created)

        index = len(unit.items.list_for_batch(batch.id))
        item = _item_for(
            unit, batch=batch, asset=asset, index=index, filename="legacy-extraction"
        )
        unit.flush()
        if item is None:  # pragma: no cover - asset is never None here
            continue

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
                duration_ms=extraction.processing_ms,
                started_at=extraction.created_at,
                completed_at=extraction.created_at,
                created_at=extraction.created_at,
            )
        )
        report.extractions += 1


def _legacy_extractions(source: JsonMediaRepository) -> list[models.ExtractionResult]:
    # The protocol has no "list extractions"; this importer is the only caller that needs one, and
    # adding it to the protocol would outlive the file it reads.
    return list(source._extractions.values())  # noqa: SLF001


def _item_for(
    unit: UnitOfWork,
    *,
    batch: tables.IngestionBatch,
    asset: tables.MediaAsset | None,
    index: int,
    filename: str,
) -> tables.IngestionItem | None:
    if asset is None:
        return None
    existing = unit.session.scalar(
        select(tables.IngestionItem).where(
            tables.IngestionItem.batch_id == batch.id,
            tables.IngestionItem.source_asset_id == asset.id,
        )
    )
    if existing is not None:
        return existing

    return unit.items.add(
        tables.IngestionItem(
            id=new_item_id(),
            batch_id=batch.id,
            item_index=index,
            source_asset_id=asset.id,
            original_filename=filename,
            declared_content_type=asset.content_type,
            status="awaiting_review",
        )
    )


def _asset_from_data_url(
    unit: UnitOfWork,
    store: FilesystemAssetStore,
    data_url: str,
    *,
    report: ImportReport,
) -> tables.MediaAsset | None:
    """Recover the image the prototype inlined, and store it properly.

    This is the half of the legacy data that *is* recoverable, and it collapses duplicates on the
    way in: two advertisements carrying the same picture become one asset.
    """
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return None
    header, payload = data_url.split(";base64,", 1)
    content_type = header.removeprefix("data:") or "application/octet-stream"
    try:
        content = base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error):
        report.warnings.append("An advertisement carried an image that could not be decoded.")
        return None

    checksum = sha256(content).hexdigest()
    key = original_key(checksum, content_type)
    store.write_bytes(content, key)

    asset, created = unit.assets.get_or_create(
        tables.MediaAsset(
            id=new_asset_id(),
            storage_key=key,
            content_type=content_type,
            byte_size=len(content),
            checksum_sha256=checksum,
            bytes_state="present",
        )
    )
    report.assets += int(created)
    report.deduplicated += int(not created)
    return asset


def new_advertisement_id_for(legacy_id: str) -> str:
    """Keep the prototype's identifier when it fits, so existing links keep working."""
    return legacy_id if len(legacy_id) <= 40 else new_advertisement_id()


# -- storage sweep -----------------------------------------------------------------------------


@dataclass
class SweepReport:
    orphans: list[str] = field(default_factory=list)
    staging_removed: int = 0

    def render(self, *, dry_run: bool) -> str:
        verb = "Would remove" if dry_run else "Removed"
        return (
            f"{verb} {len(self.orphans)} orphaned blob(s) and "
            f"{self.staging_removed} abandoned staging file(s)."
        )


def storage_gc(
    *,
    unit_of_work: UnitOfWorkFactory,
    store: FilesystemAssetStore,
    min_age_seconds: float = DEFAULT_MIN_AGE_SECONDS,
    dry_run: bool = False,
) -> SweepReport:
    """Delete blobs no row references.

    Files are written before their rows commit, so a crash in between leaves a file nothing points
    at. That ordering is deliberate -- the alternative is a row pointing at a file that is not there
    -- and this is the other half of it.
    """
    with unit_of_work() as unit:
        known = unit.assets.list_storage_keys()

    report = SweepReport(orphans=store.collect_orphans(known, min_age_seconds=min_age_seconds))
    if not dry_run:
        for key in report.orphans:
            store.delete(key)
        report.staging_removed = store.purge_staging(min_age_seconds=min_age_seconds)
    return report


# -- entry point -------------------------------------------------------------------------------


def _context(settings: Settings) -> tuple[UnitOfWorkFactory, FilesystemAssetStore]:
    unit_of_work = UnitOfWorkFactory(
        build_session_factory(cached_engine(settings.database_url))
    )
    return unit_of_work, FilesystemAssetStore(settings.storage_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="media_service.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    importer = commands.add_parser("import-legacy-json", help="Load the prototype's JSON store.")
    importer.add_argument("--path", type=Path, default=None)
    importer.add_argument("--include-extractions", action="store_true")
    importer.add_argument("--dry-run", action="store_true")

    sweep = commands.add_parser("storage-gc", help="Delete blobs no row references.")
    sweep.add_argument("--min-age-seconds", type=float, default=DEFAULT_MIN_AGE_SECONDS)
    sweep.add_argument("--dry-run", action="store_true")

    arguments = parser.parse_args(argv)
    settings = get_settings()
    unit_of_work, store = _context(settings)

    if arguments.command == "import-legacy-json":
        report = import_legacy_json(
            unit_of_work=unit_of_work,
            store=store,
            path=arguments.path or settings.metadata_path,
            include_extractions=arguments.include_extractions,
            dry_run=arguments.dry_run,
        )
        print(report.render(dry_run=arguments.dry_run))
        return 0

    report = storage_gc(
        unit_of_work=unit_of_work,
        store=store,
        min_age_seconds=arguments.min_age_seconds,
        dry_run=arguments.dry_run,
    )
    print(report.render(dry_run=arguments.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
