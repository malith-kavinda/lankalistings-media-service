"""Serving stored bytes back to an operator.

The storage key always comes from the database, never from the URL. A caller names an asset and a
purpose; this resolves both to a row and reads the key off it, so no request can describe a path.
The store's own root check then runs anyway -- two independent controls, because path traversal is
the kind of bug where the cheap second check is the one that holds when the first is refactored.

A purged asset is a 410, not a 404. The row is still there and its metadata is still evidence; what
is gone is the pixels, and telling those apart saves a moderator from chasing a link that will
never work again.
"""

from __future__ import annotations

from dataclasses import dataclass

from media_service.api.errors import AssetNotFoundError, MediaBytesUnavailableError
from media_service.db.uow import UnitOfWorkFactory
from media_service.storage import DerivativePurpose, FilesystemAssetStore

ORIGINAL = "original"
PURPOSES = (ORIGINAL, *(purpose.value for purpose in DerivativePurpose))


@dataclass(frozen=True, slots=True)
class AssetBytes:
    content: bytes
    content_type: str
    etag: str
    filename: str


class AssetService:
    def __init__(
        self, *, unit_of_work: UnitOfWorkFactory, store: FilesystemAssetStore
    ) -> None:
        self._unit_of_work = unit_of_work
        self._store = store

    def read(self, asset_id: str, purpose: str = ORIGINAL) -> AssetBytes:
        with self._unit_of_work() as unit:
            asset = unit.assets.get(asset_id)
            if asset is None:
                raise AssetNotFoundError(asset_id)
            if asset.bytes_state != "present":
                raise MediaBytesUnavailableError(asset_id)

            if purpose == ORIGINAL:
                key, content_type, checksum = (
                    asset.storage_key,
                    asset.content_type,
                    asset.checksum_sha256,
                )
            else:
                derivative = unit.assets.find_derivative(asset_id=asset_id, purpose=purpose)
                if derivative is None:
                    raise AssetNotFoundError(f"{asset_id}/{purpose}")
                if derivative.bytes_state != "present":
                    raise MediaBytesUnavailableError(asset_id)
                key, content_type, checksum = (
                    derivative.storage_key,
                    derivative.content_type,
                    derivative.params_hash,
                )

        try:
            content = self._store.read_bytes(key)
        except FileNotFoundError as exc:
            # The row says the bytes are present and they are not. That is a real inconsistency,
            # reported as one rather than as a missing resource.
            raise MediaBytesUnavailableError(asset_id) from exc

        return AssetBytes(
            content=content,
            content_type=content_type,
            # Content-addressed storage makes a strong ETag free: the identifier *is* the digest.
            etag=f'"{checksum}"',
            filename=key.rsplit("/", 1)[-1],
        )
