"""Assets and derivatives, deduplicated by the database rather than by a prior read.

Both writes here are get-or-create against a unique index, and both do it in the same order:
attempt the insert inside a savepoint, and on conflict re-select the row that won. Checking first
and inserting second looks equivalent and is not -- two requests uploading the same image can both
find nothing and both insert, and one of them gets an IntegrityError at a point where the caller
has already moved on.

The savepoint is not optional on PostgreSQL. A failed statement aborts the whole transaction, so a
bare `except IntegrityError: pass` leaves a session that accepts no further statements; confining
the failure to a nested block is what lets the caller carry on with the row it re-selected.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from media_service.db.engine import savepoint
from media_service.db.tables import MediaAsset, MediaDerivative


class AssetRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    # -- assets --------------------------------------------------------------------------------

    def get(self, asset_id: str) -> MediaAsset | None:
        return self._session.get(MediaAsset, asset_id)

    def get_by_checksum(self, checksum: str) -> MediaAsset | None:
        return self._session.scalar(
            select(MediaAsset).where(MediaAsset.checksum_sha256 == checksum)
        )

    def get_or_create(self, asset: MediaAsset) -> tuple[MediaAsset, bool]:
        """Return the stored asset for these bytes, inserting it if this is the first sight.

        The boolean says whether this call created the row, which is how the caller knows an
        upload was a duplicate of something already held (invariant 12, AC-014).
        """
        try:
            with savepoint(self._session):
                self._session.add(asset)
                self._session.flush()
        except IntegrityError:
            existing = self.get_by_checksum(asset.checksum_sha256)
            if existing is None:  # pragma: no cover - only reachable if a different index fired
                raise
            return existing, False
        return asset, True

    def list_storage_keys(self) -> set[str]:
        """Every key the database believes is on disk, for the orphan sweep."""
        asset_keys = set(self._session.scalars(select(MediaAsset.storage_key)))
        derivative_keys = set(self._session.scalars(select(MediaDerivative.storage_key)))
        return asset_keys | derivative_keys

    # -- derivatives ---------------------------------------------------------------------------

    def get_derivative(
        self, *, asset_id: str, purpose: str, params_hash: str
    ) -> MediaDerivative | None:
        return self._session.scalar(
            select(MediaDerivative).where(
                MediaDerivative.source_asset_id == asset_id,
                MediaDerivative.purpose == purpose,
                MediaDerivative.params_hash == params_hash,
            )
        )

    def find_derivative(self, *, asset_id: str, purpose: str) -> MediaDerivative | None:
        """The most recent derivative of a purpose, when the caller does not know the params."""
        return self._session.scalar(
            select(MediaDerivative)
            .where(
                MediaDerivative.source_asset_id == asset_id,
                MediaDerivative.purpose == purpose,
            )
            .order_by(MediaDerivative.id.desc())
            .limit(1)
        )

    def get_or_create_derivative(
        self, derivative: MediaDerivative
    ) -> tuple[MediaDerivative, bool]:
        """Idempotent under retry: identical parameters resolve to the row already there."""
        try:
            with savepoint(self._session):
                self._session.add(derivative)
                self._session.flush()
        except IntegrityError:
            existing = self.get_derivative(
                asset_id=derivative.source_asset_id,
                purpose=derivative.purpose,
                params_hash=derivative.params_hash,
            )
            if existing is None:  # pragma: no cover - only reachable on a storage-key collision
                raise
            return existing, False
        return derivative, True
