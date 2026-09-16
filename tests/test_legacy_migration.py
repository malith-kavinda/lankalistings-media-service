"""The JSON-to-PostgreSQL cutover: the repository adapter and the import command.

Both halves of the same promise -- that turning `MEDIA_REPOSITORY=sql` on, or running the import,
changes where the prototype's data lives and nothing about what its endpoints return.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from media_service.cli import import_legacy_json, storage_gc
from media_service.db.repositories.legacy import SqlAlchemyMediaRepository
from media_service.db.tables import Advertisement, IngestionBatch, MediaAsset, OcrExtraction
from media_service.domain.models import Advertisement as LegacyAdvertisement
from media_service.domain.models import AdvertisementStatus, ExtractionResult, ExtractionStatus
from media_service.domain.models import MediaAsset as LegacyAsset
from media_service.main import create_app
from tests.support import StubOcrEngine, png_bytes

pytestmark = pytest.mark.usefixtures("engine")


def _data_url(payload: bytes | None = None) -> str:
    content = payload if payload is not None else png_bytes()
    return f"data:image/png;base64,{base64.b64encode(content).decode()}"


def _legacy_ad(**overrides) -> LegacyAdvertisement:  # type: ignore[no-untyped-def]
    defaults = {
        "id": str(uuid4()),
        "title": "Ocean View Apartment",
        "price": "Rs. 4,500,000",
        "category": "property",
        "location": "Colombo 05",
        "description": "Two bedrooms, sea facing.",
        "image_url": _data_url(),
        "status": AdvertisementStatus.PENDING_REVIEW,
        "created_at": datetime.now(UTC),
        "source_text": "Ocean View Apartment Colombo 05",
        "extraction_confidence": "high",
    }
    return LegacyAdvertisement(**{**defaults, **overrides})


@pytest.fixture
def sql_repository(unit_of_work, asset_store):  # type: ignore[no-untyped-def]
    return SqlAlchemyMediaRepository(unit_of_work=unit_of_work, store=asset_store)


# -- the repository adapter --------------------------------------------------------------------


def test_an_advertisement_round_trips_through_postgresql(sql_repository) -> None:
    advertisement = _legacy_ad()

    sql_repository.save_advertisement(advertisement)
    restored = sql_repository.get_advertisement(advertisement.id)

    assert restored.title == advertisement.title
    assert restored.status is AdvertisementStatus.PENDING_REVIEW
    assert restored.image_url == advertisement.image_url


def test_the_wire_status_is_still_pending_review(sql_repository, session) -> None:
    """Stored as `pending`, spoken as `pending_review` (PRD 10.3)."""
    advertisement = _legacy_ad()

    sql_repository.save_advertisement(advertisement)

    assert session.scalar(select(Advertisement.status)) == "pending"
    assert sql_repository.get_advertisement(advertisement.id).status.value == "pending_review"


def test_the_same_inline_image_twice_is_stored_once(sql_repository, session) -> None:
    """The legacy path joins invariant 12 instead of staying the one place that duplicates."""
    shared = _data_url()
    sql_repository.save_advertisement(_legacy_ad(image_url=shared))
    sql_repository.save_advertisement(_legacy_ad(image_url=shared))

    assert len(session.scalars(select(MediaAsset)).all()) == 1


def test_saving_the_same_advertisement_twice_updates_it(sql_repository, session) -> None:
    advertisement = _legacy_ad()
    sql_repository.save_advertisement(advertisement)

    sql_repository.save_advertisement(replace(advertisement, title="Corrected title"))

    assert len(session.scalars(select(Advertisement)).all()) == 1
    assert sql_repository.get_advertisement(advertisement.id).title == "Corrected title"


def test_an_extraction_gets_the_item_the_schema_requires(sql_repository, session) -> None:
    asset = LegacyAsset.create(
        asset_id=str(uuid4()),
        content_type="image/png",
        byte_size=120,
        checksum="a" * 64,
        width=12,
        height=8,
    )
    sql_repository.save_asset(asset)
    extraction = ExtractionResult(
        id=str(uuid4()),
        asset_id=asset.id,
        status=ExtractionStatus.COMPLETED,
        raw_text="Ocean View Apartment",
        engine="tesseract",
        model_version="5.4.0",
        language="sin+eng",
        confidence="high",
        processing_ms=42,
        created_at=datetime.now(UTC),
    )

    sql_repository.save_extraction(extraction)

    stored = session.scalars(select(OcrExtraction)).one()
    assert stored.ingestion_item_id is not None
    assert session.scalar(select(IngestionBatch.kind)) == "legacy_single"
    assert sql_repository.get_extraction(extraction.id).raw_text == "Ocean View Apartment"


def test_an_asset_the_prototype_never_stored_is_marked_absent(sql_repository, session) -> None:
    sql_repository.save_asset(
        LegacyAsset.create(
            asset_id=str(uuid4()),
            content_type="image/png",
            byte_size=120,
            checksum="b" * 64,
            width=12,
            height=8,
        )
    )

    assert session.scalar(select(MediaAsset.bytes_state)) == "absent"


def test_http_url_mode_points_at_the_asset_endpoint(unit_of_work, asset_store) -> None:
    repository = SqlAlchemyMediaRepository(
        unit_of_work=unit_of_work, store=asset_store, public_image_url_mode="http_url"
    )
    advertisement = _legacy_ad()
    repository.save_advertisement(advertisement)

    restored = repository.get_advertisement(advertisement.id)

    assert restored.image_url.startswith("/api/v1/media/assets/")
    assert restored.image_url.endswith("/original")


def test_the_prototype_endpoints_work_against_postgresql(
    api_settings, unit_of_work, asset_store
) -> None:
    """The point of the adapter: same endpoints, same responses, different storage."""
    app = create_app(
        settings=api_settings.model_copy(update={"media_repository": "sql"}),
        ocr_engine=StubOcrEngine(),
        unit_of_work=unit_of_work,
        store=asset_store,
    )
    client = TestClient(app)

    created = client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", png_bytes(), "image/png")},
    )
    listed = client.get("/api/v1/advertisements/review")

    assert created.status_code == 200
    assert created.json()["data"]["advertisement"]["status"] == "pending_review"
    assert [ad["id"] for ad in listed.json()["data"]] == [
        created.json()["data"]["advertisement"]["id"]
    ]


# -- the import command ------------------------------------------------------------------------


def _legacy_file(tmp_path, *, advertisements, assets=(), extractions=()):  # type: ignore[no-untyped-def]
    """Write a file in the prototype's own format rather than mocking its reader."""

    def serialise(value):  # type: ignore[no-untyped-def]
        data = asdict(value)
        data["created_at"] = value.created_at.isoformat()
        if hasattr(value, "status"):
            data["status"] = value.status.value
        return data

    path = tmp_path / "media_metadata.json"
    path.write_text(
        json.dumps(
            {
                "assets": {asset.id: serialise(asset) for asset in assets},
                "extractions": {
                    extraction.id: serialise(extraction) for extraction in extractions
                },
                "advertisements": {ad.id: serialise(ad) for ad in advertisements},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_importing_advertisements_recovers_their_images(
    tmp_path, unit_of_work, asset_store, session
) -> None:
    path = _legacy_file(tmp_path, advertisements=[_legacy_ad(), _legacy_ad()])

    report = import_legacy_json(unit_of_work=unit_of_work, store=asset_store, path=path)

    assert report.advertisements == 2
    rows = session.scalars(select(Advertisement)).all()
    assert {row.origin for row in rows} == {"legacy_import"}
    assert all(row.legacy_json_id is not None for row in rows)
    stored = session.scalars(select(MediaAsset)).all()
    assert all(asset_store.exists(asset.storage_key) for asset in stored)


def test_two_advertisements_sharing_an_image_collapse_to_one_asset(
    tmp_path, unit_of_work, asset_store, session
) -> None:
    shared = _data_url()
    path = _legacy_file(
        tmp_path, advertisements=[_legacy_ad(image_url=shared), _legacy_ad(image_url=shared)]
    )

    report = import_legacy_json(unit_of_work=unit_of_work, store=asset_store, path=path)

    assert report.assets == 1
    assert report.deduplicated == 1
    assert len(session.scalars(select(MediaAsset)).all()) == 1


def test_importing_twice_imports_nothing_the_second_time(
    tmp_path, unit_of_work, asset_store, session
) -> None:
    path = _legacy_file(tmp_path, advertisements=[_legacy_ad(), _legacy_ad()])
    import_legacy_json(unit_of_work=unit_of_work, store=asset_store, path=path)

    report = import_legacy_json(unit_of_work=unit_of_work, store=asset_store, path=path)

    assert report.advertisements == 0
    assert report.skipped == 2
    assert len(session.scalars(select(Advertisement)).all()) == 2


def test_a_dry_run_reports_without_writing(tmp_path, unit_of_work, asset_store, session) -> None:
    path = _legacy_file(tmp_path, advertisements=[_legacy_ad()])

    report = import_legacy_json(
        unit_of_work=unit_of_work, store=asset_store, path=path, dry_run=True
    )

    assert report.advertisements == 1
    assert session.scalars(select(Advertisement)).all() == []


def test_extractions_are_opt_in_and_admit_their_bytes_are_gone(
    tmp_path, unit_of_work, asset_store, session
) -> None:
    asset = LegacyAsset.create(
        asset_id=str(uuid4()),
        content_type="image/png",
        byte_size=900,
        checksum="c" * 64,
        width=900,
        height=520,
    )
    extraction = ExtractionResult(
        id=str(uuid4()),
        asset_id=asset.id,
        status=ExtractionStatus.COMPLETED,
        raw_text="ඕනෑම දැන්වීමක්",
        engine="tesseract",
        model_version="5.4.0",
        language="sin+eng",
        confidence="high",
        processing_ms=88,
        created_at=datetime.now(UTC),
    )
    path = _legacy_file(
        tmp_path, advertisements=[], assets=[asset], extractions=[extraction]
    )

    without = import_legacy_json(unit_of_work=unit_of_work, store=asset_store, path=path)
    assert without.extractions == 0

    with_them = import_legacy_json(
        unit_of_work=unit_of_work, store=asset_store, path=path, include_extractions=True
    )

    assert with_them.extractions == 1
    stored = session.scalars(select(OcrExtraction)).one()
    # Sinhala survives the round trip byte for byte (AC-012).
    assert stored.raw_text == "ඕනෑම දැන්වීමක්"
    assert session.scalar(select(MediaAsset.bytes_state)) == "absent"


def test_importing_a_file_that_is_not_there_says_so(tmp_path, unit_of_work, asset_store) -> None:
    report = import_legacy_json(
        unit_of_work=unit_of_work, store=asset_store, path=tmp_path / "absent.json"
    )

    assert report.advertisements == 0
    assert "nothing to import" in report.warnings[0]


# -- the storage sweep -------------------------------------------------------------------------


def test_the_sweep_removes_blobs_no_row_references(unit_of_work, asset_store) -> None:
    """The other half of writing files before rows: collecting what a crash left behind."""
    asset_store.write_bytes(b"orphaned", "originals/aa/bb/orphan.png")

    report = storage_gc(unit_of_work=unit_of_work, store=asset_store, min_age_seconds=0)

    assert report.orphans == ["originals/aa/bb/orphan.png"]
    assert not asset_store.exists("originals/aa/bb/orphan.png")


def test_the_sweep_leaves_referenced_blobs_alone(
    ingestion_service, unit_of_work, asset_store
) -> None:
    from tests.support import uploads

    progress = ingestion_service.create_batch(uploads(2), created_by="operator-1")

    report = storage_gc(unit_of_work=unit_of_work, store=asset_store, min_age_seconds=0)

    assert report.orphans == []
    with unit_of_work() as work:
        keys = [work.assets.get(item.source_asset_id).storage_key for item in progress.items]
    assert all(asset_store.exists(key) for key in keys)


def test_a_dry_sweep_deletes_nothing(unit_of_work, asset_store) -> None:
    asset_store.write_bytes(b"orphaned", "originals/aa/bb/orphan.png")

    report = storage_gc(
        unit_of_work=unit_of_work, store=asset_store, min_age_seconds=0, dry_run=True
    )

    assert report.orphans == ["originals/aa/bb/orphan.png"]
    assert asset_store.exists("originals/aa/bb/orphan.png")
