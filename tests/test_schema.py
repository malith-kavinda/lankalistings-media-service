"""Schema-level guarantees (PRD 12).

These tests protect properties that are invisible in ordinary use and expensive when they break: the
constraints that make retries safe, the naming rules that keep migrations reversible, and the two
PostgreSQL behaviours that most often surprise code written against a permissive engine.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError, StatementError
from sqlalchemy.orm import Session

from media_service.db.base import MAX_IDENTIFIER_LENGTH, Base
from media_service.db.engine import savepoint
from media_service.db.tables import (
    Advertisement,
    AdvertisementProvenance,
    IngestionBatch,
    IngestionItem,
    MediaAsset,
)
from media_service.domain.ids import (
    IdPrefix,
    has_prefix,
    new_advertisement_id,
    new_asset_id,
    new_batch_id,
    new_id,
    new_item_id,
    new_provenance_id,
)

pytestmark = pytest.mark.usefixtures("engine")


# ---------------------------------------------------------------------------------------------
# Metadata-only checks. These need no database.
# ---------------------------------------------------------------------------------------------


def test_every_constraint_and_index_name_fits_postgres_identifier_limit() -> None:
    """PostgreSQL silently truncates identifiers past 63 bytes, which collides names."""
    too_long: list[str] = []
    for table in Base.metadata.sorted_tables:
        for constraint in table.constraints:
            if constraint.name and len(str(constraint.name)) > MAX_IDENTIFIER_LENGTH:
                too_long.append(f"{table.name}.{constraint.name}")
        for index in table.indexes:
            if index.name and len(str(index.name)) > MAX_IDENTIFIER_LENGTH:
                too_long.append(f"{table.name}.{index.name}")
    assert not too_long, f"Identifiers exceeding {MAX_IDENTIFIER_LENGTH} bytes: {too_long}"


def test_index_names_are_globally_unique() -> None:
    """Index names are schema-scoped in PostgreSQL, so two tables cannot share one."""
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            name = str(index.name)
            if name in seen:
                duplicates.append(f"{name} on both {seen[name]} and {table.name}")
            seen[name] = table.name
    assert not duplicates, duplicates


def test_all_nine_tables_are_defined() -> None:
    assert set(Base.metadata.tables) == {
        "ingestion_batches",
        "ingestion_items",
        "media_assets",
        "media_derivatives",
        "ocr_extractions",
        "llm_extraction_runs",
        "advertisements",
        "advertisement_provenance",
        "review_events",
    }


def test_advertisements_has_no_foreign_keys_in_either_direction() -> None:
    """Invariant 14: the advertisement aggregate moves to the listing service.

    If anything grows a foreign key to or from this table, that move stops being a table drop and
    becomes a schema migration across every referencing table.
    """
    advertisements = Base.metadata.tables["advertisements"]
    assert not advertisements.foreign_keys, "advertisements must not reference other tables"

    inbound = [
        f"{table.name}.{key.parent.name}"
        for table in Base.metadata.sorted_tables
        for key in table.foreign_keys
        if key.column.table.name == "advertisements"
    ]
    assert not inbound, f"Nothing may reference advertisements by foreign key: {inbound}"


# ---------------------------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------------------------


def test_ids_are_prefixed_and_sortable() -> None:
    first = new_item_id()
    second = new_item_id()

    assert has_prefix(first, IdPrefix.ITEM)
    assert first < second, "ULIDs must sort chronologically so keyset pagination works"


def test_ids_fit_the_column() -> None:
    for prefix in IdPrefix:
        assert len(new_id(prefix)) <= 40


# ---------------------------------------------------------------------------------------------
# PostgreSQL behaviours that catch code written against a permissive engine
# ---------------------------------------------------------------------------------------------


def test_naive_datetime_is_rejected_at_the_persistence_boundary(session: Session) -> None:
    """A naive value compares incorrectly against timestamptz, far from where it was written."""
    batch = IngestionBatch(id=new_batch_id(), created_by="op")
    batch.created_at = datetime(2026, 9, 5, 12, 0, 0)  # noqa: DTZ001 - deliberately naive

    session.add(batch)
    # SQLAlchemy wraps the type decorator's ValueError in a StatementError on flush.
    with pytest.raises((ValueError, StatementError)):
        session.flush()


def test_timestamps_round_trip_as_utc_aware(session: Session) -> None:
    created = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    batch = IngestionBatch(id=new_batch_id(), created_by="op", created_at=created)

    session.add(batch)
    session.commit()
    session.expire_all()

    loaded = session.get(IngestionBatch, batch.id)
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None
    assert loaded.created_at == created


def test_string_length_is_enforced(session: Session) -> None:
    """Unlike a permissive engine, PostgreSQL rejects over-long values rather than truncating."""
    batch = IngestionBatch(id=new_batch_id(), created_by="o" * 200)
    session.add(batch)
    with pytest.raises(DBAPIError):
        session.flush()


def test_savepoint_lets_a_transaction_survive_a_constraint_violation(session: Session) -> None:
    """The single most important PostgreSQL difference.

    A failed statement aborts the whole transaction, so `except IntegrityError: pass` leaves a
    session that rejects everything afterwards. The savepoint helper confines the failure, which
    is what makes checksum dedup work.
    """
    first = MediaAsset(
        id=new_asset_id(),
        storage_key="originals/aa/bb/dup.png",
        content_type="image/png",
        byte_size=10,
        checksum_sha256="d" * 64,
    )
    session.add(first)
    session.flush()

    duplicate = MediaAsset(
        id=new_asset_id(),
        storage_key="originals/cc/dd/other.png",
        content_type="image/png",
        byte_size=20,
        checksum_sha256="d" * 64,
    )
    with pytest.raises(IntegrityError):
        with savepoint(session):
            session.add(duplicate)
            session.flush()

    # The transaction is still usable: this is the assertion that matters.
    existing = session.scalar(select(MediaAsset).where(MediaAsset.checksum_sha256 == "d" * 64))
    assert existing is not None
    assert existing.id == first.id

    session.commit()
    assert session.scalar(select(text("count(*)")).select_from(MediaAsset.__table__)) == 1


# ---------------------------------------------------------------------------------------------
# The constraints that make retries safe
# ---------------------------------------------------------------------------------------------


def make_asset(session: Session, *, checksum: str) -> MediaAsset:
    asset = MediaAsset(
        id=new_asset_id(),
        storage_key=f"originals/{checksum[:2]}/{checksum[2:4]}/{checksum}.png",
        content_type="image/png",
        byte_size=100,
        checksum_sha256=checksum,
    )
    session.add(asset)
    session.flush()
    return asset


def make_batch(session: Session, *, key: str | None = None) -> IngestionBatch:
    batch = IngestionBatch(id=new_batch_id(), created_by="operator-1", idempotency_key=key)
    session.add(batch)
    session.flush()
    return batch


def make_item(session: Session, batch: IngestionBatch, asset: MediaAsset, index: int = 0):  # type: ignore[no-untyped-def]
    item = IngestionItem(
        id=new_item_id(),
        batch_id=batch.id,
        item_index=index,
        source_asset_id=asset.id,
        original_filename=f"page-{index}.png",
    )
    session.add(item)
    session.flush()
    return item


def test_identical_bytes_are_stored_once(session: Session) -> None:
    """Invariant 12 / AC-014, enforced by the database rather than by application discipline."""
    make_asset(session, checksum="a" * 64)
    with pytest.raises(IntegrityError):
        with savepoint(session):
            make_asset(session, checksum="a" * 64)


def test_one_batch_per_operator_idempotency_key(session: Session) -> None:
    """FR-ING-006: a repeated upload must not create a second batch."""
    make_batch(session, key="key-1")
    with pytest.raises(IntegrityError):
        with savepoint(session):
            make_batch(session, key="key-1")


def test_the_same_idempotency_key_is_allowed_for_a_different_operator(session: Session) -> None:
    make_batch(session, key="shared")
    other = IngestionBatch(id=new_batch_id(), created_by="operator-2", idempotency_key="shared")
    session.add(other)
    session.flush()  # must not raise


def test_batches_without_an_idempotency_key_do_not_collide(session: Session) -> None:
    """NULLs are distinct in a unique index, so unkeyed uploads are unconstrained."""
    make_batch(session, key=None)
    make_batch(session, key=None)
    session.flush()


def test_a_candidate_index_cannot_repeat_within_one_generation(session: Session) -> None:
    """AC-006: retrying an item must not duplicate its candidates."""
    batch = make_batch(session)
    asset = make_asset(session, checksum="b" * 64)
    item = make_item(session, batch, asset)

    def add_candidate() -> None:
        session.add(
            AdvertisementProvenance(
                id=new_provenance_id(),
                advertisement_id=new_advertisement_id(),
                ingestion_batch_id=batch.id,
                ingestion_item_id=item.id,
                source_asset_id=asset.id,
                generation=1,
                candidate_index=0,
            )
        )
        session.flush()

    add_candidate()
    with pytest.raises(IntegrityError):
        with savepoint(session):
            add_candidate()


def test_a_new_generation_may_reuse_candidate_indexes(session: Session) -> None:
    """Reprocessing supersedes the previous generation rather than colliding with it."""
    batch = make_batch(session)
    asset = make_asset(session, checksum="c" * 64)
    item = make_item(session, batch, asset)

    for generation in (1, 2):
        session.add(
            AdvertisementProvenance(
                id=new_provenance_id(),
                advertisement_id=new_advertisement_id(),
                ingestion_batch_id=batch.id,
                ingestion_item_id=item.id,
                source_asset_id=asset.id,
                generation=generation,
                candidate_index=0,
            )
        )
    session.flush()


def test_many_reviewer_created_candidates_may_coexist(session: Session) -> None:
    """FR-REV-005. A manual candidate has no index; NULLs are distinct in a unique index."""
    batch = make_batch(session)
    asset = make_asset(session, checksum="e" * 64)
    item = make_item(session, batch, asset)

    for _ in range(3):
        session.add(
            AdvertisementProvenance(
                id=new_provenance_id(),
                advertisement_id=new_advertisement_id(),
                ingestion_batch_id=batch.id,
                ingestion_item_id=item.id,
                source_asset_id=asset.id,
                generation=1,
                candidate_index=None,
            )
        )
    session.flush()


def test_two_items_in_one_batch_may_share_an_asset(session: Session) -> None:
    """FR-ING-008: a duplicate image is flagged, not discarded, so both items must persist."""
    batch = make_batch(session)
    asset = make_asset(session, checksum="f" * 64)

    make_item(session, batch, asset, index=0)
    second = make_item(session, batch, asset, index=1)
    second.is_duplicate_in_batch = True
    session.flush()


def test_item_index_is_unique_within_a_batch(session: Session) -> None:
    batch = make_batch(session)
    asset = make_asset(session, checksum="1" * 64)
    make_item(session, batch, asset, index=0)

    with pytest.raises(IntegrityError):
        with savepoint(session):
            second_asset = make_asset(session, checksum="2" * 64)
            make_item(session, batch, second_asset, index=0)


def test_deleting_a_batch_cascades_to_its_items(session: Session) -> None:
    batch = make_batch(session)
    asset = make_asset(session, checksum="3" * 64)
    make_item(session, batch, asset)
    session.commit()

    session.delete(batch)
    session.commit()

    assert session.scalars(select(IngestionItem)).all() == []
    assert session.get(MediaAsset, asset.id) is not None, "The asset outlives the batch"


def test_status_check_constraints_reject_unknown_values(session: Session) -> None:
    batch = IngestionBatch(id=new_batch_id(), created_by="op", status="not_a_status")
    session.add(batch)
    with pytest.raises(IntegrityError):
        session.flush()


def test_advertisement_defaults_to_the_target_status_vocabulary(session: Session) -> None:
    """PRD 10.3: `pending` is stored; `pending_review` is produced at the wire edge."""
    advertisement = Advertisement(id=new_advertisement_id(), origin="llm_extraction")
    session.add(advertisement)
    session.commit()

    assert advertisement.status == "pending"


def test_advertisement_version_increments_on_update(session: Session) -> None:
    """PRD 13.2 optimistic version checking."""
    advertisement = Advertisement(id=new_advertisement_id(), title="Before")
    session.add(advertisement)
    session.commit()
    assert advertisement.version == 1

    advertisement.title = "After"
    session.commit()
    assert advertisement.version == 2


def test_claimable_index_is_partial(engine: Engine) -> None:
    """The queue index must not grow with completed work."""
    with engine.connect() as connection:
        definition = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_items__claimable'")
        ).scalar_one()
    assert "WHERE" in definition
    assert "uploaded" in definition


def test_only_one_completed_ocr_result_per_generation(engine: Engine) -> None:
    with engine.connect() as connection:
        definition = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'uq_ocr__item_generation_completed'"
            )
        ).scalar_one()
    assert "UNIQUE" in definition
    assert "completed" in definition


def test_only_one_validated_llm_run_per_generation(engine: Engine) -> None:
    """This is what stops a retry paying the provider a second time."""
    with engine.connect() as connection:
        definition = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'uq_llm__item_generation_validated'"
            )
        ).scalar_one()
    assert "UNIQUE" in definition
    assert "validated" in definition


def test_lease_expiry_can_be_queried_for_reaping(session: Session) -> None:
    batch = make_batch(session)
    asset = make_asset(session, checksum="4" * 64)
    item = make_item(session, batch, asset)

    item.claim_token = "clm_expired"
    item.status = "ocr_processing"
    item.lease_expires_at = datetime.now(UTC) - timedelta(seconds=5)
    session.commit()

    stale = session.scalars(
        select(IngestionItem).where(
            IngestionItem.claim_token.is_not(None),
            IngestionItem.lease_expires_at < datetime.now(UTC),
        )
    ).all()
    assert [found.id for found in stale] == [item.id]
