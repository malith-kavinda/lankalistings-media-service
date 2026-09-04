"""The nine ingestion tables (PRD 12).

Two structural decisions run through this module.

`advertisements` is a **shim**. It carries no foreign key in either direction, and every reference
to it is a plain indexed string. The target architecture gives advertisement ownership to a
separate listing
service (PRD 7.3, invariant 14); when that happens this one table is dropped and nothing else in the
schema changes. Constraints that make retries safe therefore live on `advertisement_provenance`,
which this service keeps, rather than on the advertisement row, which it does not.

`ingestion_items` **is** the work queue. There is no separate jobs table, because two rows
describing the same work would need to be kept in step, and the moment they disagree an item is
either processed twice or never. Claim and lease columns live on the item itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from media_service.db.base import Base, JsonDoc, utcnow

# ---------------------------------------------------------------------------------------------
# Closed vocabularies.
#
# Stored as String + a named CheckConstraint rather than a native PostgreSQL enum. Extending a
# native enum requires ALTER TYPE, which is awkward inside a migration; altering a CHECK is a
# single statement.
# ---------------------------------------------------------------------------------------------

BATCH_STATUSES = ("queued", "processing", "completed", "partial_failed", "failed")
BATCH_KINDS = ("operator_upload", "legacy_single", "legacy_import")

ITEM_STATUSES = (
    "uploaded",
    "preprocessing",
    "ocr_processing",
    "llm_processing",
    "awaiting_review",
    "no_ads",
    "needs_attention",
    "failed",
    "completed",
)

DERIVATIVE_PURPOSES = ("ocr_input", "page_preview", "thumbnail", "advertisement_crop")
BYTES_STATES = ("present", "absent", "purged")

OCR_STATUSES = ("running", "completed", "empty", "failed", "unsupported", "abandoned")
LLM_STATUSES = ("running", "validated", "schema_invalid", "provider_error", "timeout", "abandoned")
LLM_ATTEMPT_KINDS = ("primary", "retry", "repair")

ADVERTISEMENT_STATUSES = ("draft", "pending", "active", "rejected", "expired", "sold")
ADVERTISEMENT_ORIGINS = (
    "manual",
    "llm_extraction",
    "ocr_heuristic",
    "manual_reviewer",
    "legacy_import",
)

CANDIDATE_STATES = ("pending_publish", "linked", "superseded", "discarded")

REVIEW_ACTIONS = (
    "created_by_extraction",
    "edited",
    "approved",
    "rejected",
    "reprocessed",
    "superseded",
)
REVIEW_SUBJECTS = ("advertisement", "ingestion_item", "ingestion_batch")

ID = String(40)


def _in(column: str, allowed: tuple[str, ...]) -> str:
    values = ", ".join(f"'{value}'" for value in allowed)
    return f"{column} IN ({values})"


class IngestionBatch(Base):
    __tablename__ = "ingestion_batches"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), default="operator_upload")
    created_by: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="queued")

    idempotency_key: Mapped[str | None] = mapped_column(String(128), default=None)
    # sha256 over the operator plus the sorted checksums of the request's files. Lets a repeated
    # request with the same key be distinguished from a *different* request reusing that key.
    request_fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    # The exact 202 body, replayed verbatim when a duplicate request arrives.
    response_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)

    total_items: Mapped[int] = mapped_column(Integer, default=0)
    status_counts: Mapped[dict[str, Any]] = mapped_column(
        JsonDoc, default=dict, server_default=text("'{}'::jsonb")
    )

    correlation_id: Mapped[str | None] = mapped_column(String(64), default=None)

    # NULL while the request is still being written. A concurrent duplicate waits on this rather
    # than reading a half-built batch, which is why no extra `receiving` status is needed.
    committed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)

    __table_args__ = (
        UniqueConstraint("created_by", "idempotency_key", name="uq_batches__operator_key"),
        CheckConstraint(_in("status", BATCH_STATUSES), name="status"),
        CheckConstraint(_in("kind", BATCH_KINDS), name="kind"),
        CheckConstraint("total_items >= 0", name="total_items"),
        Index("ix_batches__kind_status_created", "kind", "status", "created_at"),
    )


class IngestionItem(Base):
    __tablename__ = "ingestion_items"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    batch_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ingestion_batches.id", ondelete="CASCADE")
    )
    # Not named `position`: that is a reserved word in SQL and needs quoting everywhere.
    item_index: Mapped[int] = mapped_column(Integer)

    source_asset_id: Mapped[str] = mapped_column(
        ID, ForeignKey("media_assets.id", ondelete="RESTRICT")
    )
    # Display metadata only. FR-ING-009: it must never influence a storage path.
    original_filename: Mapped[str] = mapped_column(String(255))
    declared_content_type: Mapped[str | None] = mapped_column(String(100), default=None)

    status: Mapped[str] = mapped_column(String(24), default="uploaded")
    # The idempotency scope for candidates. Bumped only when prior artifacts are declared invalid.
    pipeline_generation: Mapped[int] = mapped_column(Integer, default=1)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)

    # Which stage broke. Not derivable from `status`, unlike `current_stage` (PRD 12.2).
    failed_stage: Mapped[str | None] = mapped_column(String(24), default=None)
    error_code: Mapped[str | None] = mapped_column(String(48), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    warning_codes: Mapped[list[Any]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'::text[]")
    )

    # FR-ING-008: a repeated image is flagged, never discarded.
    is_duplicate_in_batch: Mapped[bool] = mapped_column(
        Boolean(create_constraint=False), default=False
    )
    duplicate_of_item_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("ingestion_items.id", ondelete="SET NULL"), default=None
    )

    # Queue columns.
    run_after: Mapped[datetime] = mapped_column(default=utcnow)
    claim_token: Mapped[str | None] = mapped_column(String(40), default=None)
    claimed_by: Mapped[str | None] = mapped_column(String(64), default=None)
    claimed_at: Mapped[datetime | None] = mapped_column(default=None)
    lease_expires_at: Mapped[datetime | None] = mapped_column(default=None)
    heartbeat_at: Mapped[datetime | None] = mapped_column(default=None)

    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    correlation_id: Mapped[str | None] = mapped_column(String(64), default=None)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)

    __table_args__ = (
        UniqueConstraint("batch_id", "item_index", name="uq_items__batch_index"),
        CheckConstraint(_in("status", ITEM_STATUSES), name="status"),
        CheckConstraint("attempt_count >= 0", name="attempt_count"),
        CheckConstraint("pipeline_generation >= 1", name="generation"),
        CheckConstraint("candidate_count >= 0", name="candidate_count"),
        # The claim query's index. Partial, because only claimable rows are ever scanned and the
        # index should not grow with completed work.
        Index(
            "ix_items__claimable",
            "run_after",
            "id",
            postgresql_where=text("status = 'uploaded' AND claim_token IS NULL"),
        ),
        # The reaper's index: rows currently leased.
        Index(
            "ix_items__lease",
            "lease_expires_at",
            postgresql_where=text("claim_token IS NOT NULL"),
        ),
        Index("ix_items__batch_id", "batch_id"),
        Index("ix_items__source_asset_id", "source_asset_id"),
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    storage_key: Mapped[str] = mapped_column(String(255))
    # The *detected* type, from decoding the bytes. Never the client's declared type.
    content_type: Mapped[str] = mapped_column(String(100))
    image_format: Mapped[str | None] = mapped_column(String(16), default=None)
    byte_size: Mapped[int] = mapped_column(Integer)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    exif_orientation: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    # Retention can delete pixels while keeping the evidence metadata a moderator needs.
    bytes_state: Mapped[str] = mapped_column(String(16), default="present")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (
        # Invariant 12 / AC-014: the same image is stored once, enforced by the database rather than
        # by a check the application could forget.
        UniqueConstraint("checksum_sha256", name="uq_assets__checksum"),
        UniqueConstraint("storage_key", name="uq_assets__storage_key"),
        CheckConstraint("byte_size > 0", name="byte_size"),
        CheckConstraint(_in("bytes_state", BYTES_STATES), name="bytes_state"),
    )


class MediaDerivative(Base):
    __tablename__ = "media_derivatives"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    source_asset_id: Mapped[str] = mapped_column(
        ID, ForeignKey("media_assets.id", ondelete="CASCADE")
    )
    purpose: Mapped[str] = mapped_column(String(24))
    storage_key: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    byte_size: Mapped[int] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    preprocessing_version: Mapped[str] = mapped_column(String(32), default="v0")
    params: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)
    crop_box: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)
    # Identifies the exact preprocessing that produced this file. Re-running with the same
    # parameters rewrites an identical file, which is what makes preprocessing idempotent under
    # retry; changing a parameter creates a new file instead of corrupting one another row already
    # references.
    params_hash: Mapped[str] = mapped_column(String(64))
    # Opaque candidate reference for crops. No foreign key: it points at the shim table.
    owner_ref: Mapped[str | None] = mapped_column(ID, default=None)
    bytes_state: Mapped[str] = mapped_column(String(16), default="present")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "source_asset_id", "purpose", "params_hash", name="uq_derivatives__asset_purpose_params"
        ),
        UniqueConstraint("storage_key", name="uq_derivatives__storage_key"),
        CheckConstraint(_in("purpose", DERIVATIVE_PURPOSES), name="purpose"),
        CheckConstraint(_in("bytes_state", BYTES_STATES), name="bytes_state"),
        Index("ix_derivatives__owner_ref", "owner_ref"),
    )


class OcrExtraction(Base):
    __tablename__ = "ocr_extractions"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    ingestion_item_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ingestion_items.id", ondelete="CASCADE")
    )
    source_asset_id: Mapped[str] = mapped_column(
        ID, ForeignKey("media_assets.id", ondelete="RESTRICT")
    )
    input_derivative_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("media_derivatives.id", ondelete="RESTRICT"), default=None
    )

    generation: Mapped[int] = mapped_column(Integer, default=1)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="running")

    raw_text: Mapped[str] = mapped_column(Text, default="")
    blocks: Mapped[list[Any] | None] = mapped_column(JsonDoc, default=None)
    # Denormalised so evidence validation does not have to parse the blocks document.
    block_count: Mapped[int] = mapped_column(Integer, default=0)
    max_block_id: Mapped[int | None] = mapped_column(Integer, default=None)

    engine: Mapped[str] = mapped_column(String(32), default="tesseract")
    engine_version: Mapped[str | None] = mapped_column(String(64), default=None)
    traineddata_version: Mapped[str | None] = mapped_column(String(64), default=None)
    languages: Mapped[str] = mapped_column(String(32), default="sin+eng")
    mean_confidence: Mapped[float | None] = mapped_column(default=None)
    low_confidence: Mapped[bool] = mapped_column(Boolean(create_constraint=False), default=False)

    width: Mapped[int | None] = mapped_column(Integer, default=None)
    height: Mapped[int | None] = mapped_column(Integer, default=None)
    preprocessing_version: Mapped[str] = mapped_column(String(32), default="v0")
    duration_ms: Mapped[int | None] = mapped_column(Integer, default=None)

    error_code: Mapped[str | None] = mapped_column(String(48), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ingestion_item_id", "generation", "attempt", name="uq_ocr__item_generation_attempt"
        ),
        # At most one *completed* OCR result per generation. A duplicate dispatch loses this race
        # and resumes from the winner's row instead of producing a second result.
        Index(
            "uq_ocr__item_generation_completed",
            "ingestion_item_id",
            "generation",
            unique=True,
            postgresql_where=text("status = 'completed'"),
        ),
        CheckConstraint(_in("status", OCR_STATUSES), name="status"),
        CheckConstraint(
            "mean_confidence IS NULL OR (mean_confidence >= 0 AND mean_confidence <= 1)",
            name="mean_confidence",
        ),
        Index("ix_ocr__item", "ingestion_item_id"),
    )


class LlmExtractionRun(Base):
    __tablename__ = "llm_extraction_runs"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    ingestion_item_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ingestion_items.id", ondelete="CASCADE")
    )
    ocr_extraction_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ocr_extractions.id", ondelete="RESTRICT")
    )

    generation: Mapped[int] = mapped_column(Integer, default=1)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    attempt_kind: Mapped[str] = mapped_column(String(16), default="primary")
    parent_run_id: Mapped[str | None] = mapped_column(ID, default=None)

    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    prompt_checksum: Mapped[str | None] = mapped_column(String(32), default=None)
    schema_version: Mapped[str] = mapped_column(String(16), default="1.0")
    category_catalog_version: Mapped[str | None] = mapped_column(String(32), default=None)
    # Lets a retry reuse an identical earlier result rather than paying the provider twice.
    request_hash: Mapped[str] = mapped_column(String(64))

    status: Mapped[str] = mapped_column(String(24), default="running")

    # Sanitised and size-capped. Persisted for provenance (PRD 12.5); never written to a log (15.4).
    raw_response_text: Mapped[str | None] = mapped_column(Text, default=None)
    validated_response: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)
    validation_errors: Mapped[list[Any] | None] = mapped_column(JsonDoc, default=None)
    candidate_count: Mapped[int | None] = mapped_column(Integer, default=None)

    input_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    output_tokens: Mapped[int | None] = mapped_column(Integer, default=None)
    cost_micros: Mapped[int | None] = mapped_column(BigInteger, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)

    error_code: Mapped[str | None] = mapped_column(String(48), default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)

    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "ingestion_item_id", "generation", "attempt", name="uq_llm__item_generation_attempt"
        ),
        # At most one validated run per generation, so a crash after the provider call but before
        # candidate creation resumes without paying for the call again.
        Index(
            "uq_llm__item_generation_validated",
            "ingestion_item_id",
            "generation",
            unique=True,
            postgresql_where=text("status = 'validated'"),
        ),
        CheckConstraint(_in("status", LLM_STATUSES), name="status"),
        CheckConstraint(_in("attempt_kind", LLM_ATTEMPT_KINDS), name="attempt_kind"),
        Index("ix_llm__item", "ingestion_item_id"),
        Index("ix_llm__request_hash", "request_hash"),
    )


class Advertisement(Base):
    """The shim. No foreign keys in either direction -- see the module docstring."""

    __tablename__ = "advertisements"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    reference: Mapped[str | None] = mapped_column(String(16), default=None)

    title: Mapped[str] = mapped_column(String(200), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    category: Mapped[str] = mapped_column(String(48), default="other")
    location: Mapped[str] = mapped_column(String(120), default="")

    # The legacy display string the management portal already reads. Kept alongside the structured
    # fields rather than replaced, so the portal keeps working during migration.
    price: Mapped[str] = mapped_column(String(64), default="")
    price_raw: Mapped[str | None] = mapped_column(String(64), default=None)
    price_amount_cents: Mapped[int | None] = mapped_column(BigInteger, default=None)
    currency: Mapped[str | None] = mapped_column(String(3), default=None)

    phones: Mapped[list[Any] | None] = mapped_column(JsonDoc, default=None)
    language: Mapped[str | None] = mapped_column(String(8), default=None)

    # The target vocabulary is stored; `pending_review` is produced at the wire edge (PRD 10.3).
    status: Mapped[str] = mapped_column(String(24), default="pending")
    origin: Mapped[str] = mapped_column(String(24), default="manual")

    source_text: Mapped[str] = mapped_column(Text, default="")
    extraction_confidence: Mapped[str] = mapped_column(String(16), default="manual")
    confidence_overall: Mapped[float | None] = mapped_column(default=None)
    warning_codes: Mapped[list[Any]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'::text[]")
    )

    image_asset_id: Mapped[str | None] = mapped_column(ID, default=None)
    image_derivative_id: Mapped[str | None] = mapped_column(ID, default=None)

    # Optimistic locking for PRD 13.2's version check on reviewer edits.
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(default=None)
    approved_at: Mapped[datetime | None] = mapped_column(default=None)
    rejected_at: Mapped[datetime | None] = mapped_column(default=None)
    published_at: Mapped[datetime | None] = mapped_column(default=None)
    expires_at: Mapped[datetime | None] = mapped_column(default=None)

    approved_by: Mapped[str | None] = mapped_column(String(64), default=None)
    rejected_by: Mapped[str | None] = mapped_column(String(64), default=None)
    rejection_reason_code: Mapped[str | None] = mapped_column(String(48), default=None)
    rejection_note: Mapped[str | None] = mapped_column(Text, default=None)

    # Makes importing the prototype's JSON store idempotent.
    legacy_json_id: Mapped[str | None] = mapped_column(String(64), default=None)

    __mapper_args__ = {"version_id_col": version}

    __table_args__ = (
        UniqueConstraint("reference", name="uq_ads__reference"),
        UniqueConstraint("legacy_json_id", name="uq_ads__legacy_json_id"),
        CheckConstraint(_in("status", ADVERTISEMENT_STATUSES), name="status"),
        CheckConstraint(_in("origin", ADVERTISEMENT_ORIGINS), name="origin"),
        CheckConstraint(
            "price_amount_cents IS NULL OR price_amount_cents >= 0", name="price_amount"
        ),
        # The public feed and the review queue both filter on status first.
        Index("ix_ads__status_created", "status", "created_at"),
        Index("ix_ads__category_status", "category", "status"),
        Index("ix_ads__confidence", "confidence_overall"),
    )


class AdvertisementProvenance(Base):
    """Where candidate idempotency lives.

    The uniqueness constraints that make a retry safe are here rather than on `advertisements`,
    because the service that performs the retry must own them, and the advertisement row is the part
    that later moves to another service.
    """

    __tablename__ = "advertisement_provenance"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    # Opaque. NULL until the listing owner assigns an identifier, which is the seam a remote gateway
    # needs.
    advertisement_id: Mapped[str | None] = mapped_column(ID, default=None)

    # These foreign keys carry explicit names. The convention would generate identifiers longer than
    # PostgreSQL's 63-byte limit for this table, and the server truncates rather than refusing --
    # which produces collisions and constraints a later migration cannot drop by name.
    ingestion_batch_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ingestion_batches.id", ondelete="CASCADE", name="fk_prov__batch")
    )
    ingestion_item_id: Mapped[str] = mapped_column(
        ID, ForeignKey("ingestion_items.id", ondelete="CASCADE", name="fk_prov__item")
    )
    source_asset_id: Mapped[str] = mapped_column(
        ID, ForeignKey("media_assets.id", ondelete="RESTRICT", name="fk_prov__asset")
    )
    ocr_extraction_id: Mapped[str | None] = mapped_column(
        ID,
        ForeignKey("ocr_extractions.id", ondelete="RESTRICT", name="fk_prov__ocr"),
        default=None,
    )
    llm_extraction_run_id: Mapped[str | None] = mapped_column(
        ID,
        ForeignKey("llm_extraction_runs.id", ondelete="RESTRICT", name="fk_prov__llm_run"),
        default=None,
    )
    crop_derivative_id: Mapped[str | None] = mapped_column(
        ID,
        ForeignKey("media_derivatives.id", ondelete="SET NULL", name="fk_prov__crop"),
        default=None,
    )

    generation: Mapped[int] = mapped_column(Integer, default=1)
    # NULL for a candidate a reviewer created by hand (FR-REV-005). NULLs are distinct in a unique
    # index, so many manual candidates can coexist for one item.
    candidate_index: Mapped[int | None] = mapped_column(Integer, default=None)

    source_block_ids: Mapped[list[Any]] = mapped_column(
        JsonDoc, default=list, server_default=text("'[]'::jsonb")
    )
    field_confidence: Mapped[dict[str, Any]] = mapped_column(
        JsonDoc, default=dict, server_default=text("'{}'::jsonb")
    )
    warnings: Mapped[list[Any]] = mapped_column(
        JsonDoc, default=list, server_default=text("'[]'::jsonb")
    )
    warning_codes: Mapped[list[Any]] = mapped_column(
        ARRAY(Text), default=list, server_default=text("'{}'::text[]")
    )

    # FR-REV-009 needs both: what the model produced, and what the reviewer accepted.
    extracted_values: Mapped[dict[str, Any]] = mapped_column(
        JsonDoc, default=dict, server_default=text("'{}'::jsonb")
    )
    accepted_values: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)

    candidate_state: Mapped[str] = mapped_column(String(24), default="pending_publish")
    superseded_by_generation: Mapped[int | None] = mapped_column(Integer, default=None)
    superseded_at: Mapped[datetime | None] = mapped_column(default=None)

    candidate_fingerprint: Mapped[str | None] = mapped_column(String(64), default=None)
    duplicate_of_advertisement_id: Mapped[str | None] = mapped_column(ID, default=None)

    reviewer_id: Mapped[str | None] = mapped_column(String(64), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    __table_args__ = (
        # FR-CAN-004: reprocessing one run cannot create a second copy of its candidates.
        UniqueConstraint(
            "llm_extraction_run_id", "candidate_index", name="uq_prov__run_candidate"
        ),
        # AC-006: nor can a retry within the same generation.
        UniqueConstraint(
            "ingestion_item_id",
            "generation",
            "candidate_index",
            name="uq_prov__item_generation_candidate",
        ),
        UniqueConstraint("advertisement_id", name="uq_prov__advertisement"),
        CheckConstraint(_in("candidate_state", CANDIDATE_STATES), name="candidate_state"),
        Index("ix_prov__batch", "ingestion_batch_id"),
        Index("ix_prov__item", "ingestion_item_id"),
        Index("ix_prov__fingerprint", "candidate_fingerprint"),
    )


class ReviewEvent(Base):
    """Append-only. No update, no delete."""

    __tablename__ = "review_events"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    # Reprocessing is item-scoped and may have no advertisement, so the subject is explicit
    # (PRD 12.7).
    subject_type: Mapped[str] = mapped_column(String(16))
    advertisement_id: Mapped[str | None] = mapped_column(ID, default=None)
    ingestion_item_id: Mapped[str | None] = mapped_column(
        ID, ForeignKey("ingestion_items.id", ondelete="SET NULL"), default=None
    )
    ingestion_batch_id: Mapped[str | None] = mapped_column(ID, default=None)

    action: Mapped[str] = mapped_column(String(32))
    actor_id: Mapped[str] = mapped_column(String(64))
    actor_role: Mapped[str | None] = mapped_column(String(24), default=None)

    reason_code: Mapped[str | None] = mapped_column(String(48), default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)

    before_values: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)
    after_values: Mapped[dict[str, Any] | None] = mapped_column(JsonDoc, default=None)
    # Stored rather than derived: PRD 15.5 reports correction rate *per field*, and diffing two JSON
    # documents at query time is not practical.
    changed_fields: Mapped[list[Any] | None] = mapped_column(JsonDoc, default=None)

    correlation_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    __table_args__ = (
        CheckConstraint(_in("action", REVIEW_ACTIONS), name="action"),
        CheckConstraint(_in("subject_type", REVIEW_SUBJECTS), name="subject_type"),
        Index("ix_events__ad_created", "advertisement_id", "created_at"),
        Index("ix_events__action_created", "action", "created_at"),
        Index("ix_events__item", "ingestion_item_id"),
    )
