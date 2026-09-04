"""Content-addressed blob storage.

Layout:

    <root>/tmp/<uuid>.part
    <root>/originals/<c0c1>/<c2c3>/<sha256>.<ext>
    <root>/derivatives/<a0a1>/<asset_id>/<purpose>/<params_hash>.<ext>

Addressing originals by content hash satisfies two requirements at once. Identical bytes land on the
same path, so the same image is stored once no matter how many times it is uploaded (invariant 12,
AC-014). And the path is a function of the bytes alone, so an uploaded filename cannot influence
where anything is written (FR-ING-009) -- which removes traversal as a class of problem rather
than filtering for it.

Derivatives are keyed by a hash of the parameters that produced them. Re-running preprocessing
with the same settings rewrites an identical file, which is what makes the preprocessing stage
idempotent under retry; changing a setting produces a new path instead of overwriting a file
another row still points at.
"""

from __future__ import annotations

import hashlib
import os
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Final
from uuid import uuid4

CHUNK_SIZE: Final = 1024 * 1024

EXTENSION_BY_CONTENT_TYPE: Final[dict[str, str]] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
}

# Windows reserves these regardless of extension.
_RESERVED_FILENAMES: Final = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{digit}" for digit in range(1, 10)}
    | {f"lpt{digit}" for digit in range(1, 10)}
)

MAX_FILENAME_LENGTH: Final = 255


class DerivativePurpose(StrEnum):
    OCR_INPUT = "ocr_input"
    PAGE_PREVIEW = "page_preview"
    THUMBNAIL = "thumbnail"
    ADVERTISEMENT_CROP = "advertisement_crop"


@dataclass(frozen=True, slots=True)
class StagedUpload:
    """Bytes written to the staging area, with their checksum, before any database row exists."""

    temp_path: Path
    checksum_sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class StoredBlob:
    storage_key: str
    checksum_sha256: str
    byte_size: int
    # False when identical bytes were already present, so the caller knows it reused rather than
    # wrote.
    was_written: bool


def extension_for(content_type: str) -> str:
    return EXTENSION_BY_CONTENT_TYPE.get(content_type, "bin")


def original_key(checksum: str, content_type: str) -> str:
    return f"originals/{checksum[:2]}/{checksum[2:4]}/{checksum}.{extension_for(content_type)}"


def derivative_key(
    asset_id: str, purpose: DerivativePurpose, params_hash: str, content_type: str
) -> str:
    return (
        f"derivatives/{asset_id[-2:]}/{asset_id}/{purpose.value}/"
        f"{params_hash[:16]}.{extension_for(content_type)}"
    )


def params_hash(purpose: DerivativePurpose, version: str, params: dict[str, object]) -> str:
    canonical = "|".join(f"{key}={params[key]!r}" for key in sorted(params))
    return hashlib.sha256(f"{purpose.value}|{version}|{canonical}".encode()).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Make an uploaded filename safe to store and to echo back.

    Display metadata only -- it never influences a storage path. The value still needs cleaning
    because it is rendered in the portal and returned in a Content-Disposition header.
    """
    normalized = unicodedata.normalize("NFC", filename)
    # Take the last path segment under both separators: a client may send either.
    normalized = normalized.replace("\\", "/").rsplit("/", 1)[-1]
    # Quotes and newlines would let a filename break out of a Content-Disposition header.
    forbidden = '"\r\n'
    cleaned = "".join(
        character
        for character in normalized
        if character.isprintable() and character not in forbidden
    ).strip()
    cleaned = cleaned.lstrip(".") or "upload"

    stem, dot, suffix = cleaned.rpartition(".")
    if dot and stem.lower() in _RESERVED_FILENAMES:
        cleaned = f"file-{cleaned}"
    elif not dot and cleaned.lower() in _RESERVED_FILENAMES:
        cleaned = f"file-{cleaned}"

    return cleaned[:MAX_FILENAME_LENGTH]


class FilesystemAssetStore:
    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, storage_key: str) -> Path:
        """Resolve a key to an absolute path, refusing anything that escapes the root.

        Keys are generated internally and never come from a client, so this is defence in depth
        rather than the primary control -- but it is cheap and it makes the guarantee local.
        """
        resolved = (self._root / storage_key).resolve()
        root = self._root.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Storage key escapes the storage root: {storage_key!r}")
        return resolved

    def exists(self, storage_key: str) -> bool:
        return self.path_for(storage_key).exists()

    def stage(self, stream: BinaryIO | Iterable[bytes]) -> StagedUpload:
        """Stream bytes to the staging area, hashing as they arrive.

        Never loads the whole upload into memory: at the configured limits a single batch can carry
        100 MiB, and `await file.read()` on every part would hold all of it at once.
        """
        staging = self._root / "tmp"
        staging.mkdir(parents=True, exist_ok=True)
        temp_path = staging / f"{uuid4().hex}.part"

        digest = hashlib.sha256()
        size = 0
        with open(temp_path, "wb") as handle:
            for chunk in _iter_chunks(stream):
                digest.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        return StagedUpload(
            temp_path=temp_path, checksum_sha256=digest.hexdigest(), byte_size=size
        )

    def promote(self, staged: StagedUpload, storage_key: str) -> StoredBlob:
        """Move staged bytes to their final key.

        If the key already holds bytes, the staged copy is discarded: identical content hashes to an
        identical key, so the existing file is the same file.
        """
        target = self.path_for(storage_key)

        if target.exists():
            staged.temp_path.unlink(missing_ok=True)
            return StoredBlob(
                storage_key=storage_key,
                checksum_sha256=staged.checksum_sha256,
                byte_size=staged.byte_size,
                was_written=False,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        # os.replace is atomic over an existing destination on Windows; os.rename is not.
        os.replace(staged.temp_path, target)
        _fsync_directory(target.parent)

        return StoredBlob(
            storage_key=storage_key,
            checksum_sha256=staged.checksum_sha256,
            byte_size=staged.byte_size,
            was_written=True,
        )

    def write_bytes(self, payload: bytes, storage_key: str) -> StoredBlob:
        staged = self.stage([payload])
        return self.promote(staged, storage_key)

    def read_bytes(self, storage_key: str) -> bytes:
        return self.path_for(storage_key).read_bytes()

    def delete(self, storage_key: str) -> bool:
        target = self.path_for(storage_key)
        if not target.exists():
            return False
        target.unlink()
        return True

    def discard(self, staged: StagedUpload) -> None:
        staged.temp_path.unlink(missing_ok=True)

    def iter_stored_keys(self) -> Iterator[str]:
        """Every key currently on disk, for reconciliation against the database."""
        for prefix in ("originals", "derivatives"):
            base = self._root / prefix
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file():
                    yield path.relative_to(self._root).as_posix()

    def collect_orphans(self, known_keys: set[str], *, min_age_seconds: float) -> list[str]:
        """Blobs with no database row.

        Files land before their rows commit, so a crash in between leaves a blob nothing references.
        The age floor keeps the sweep from deleting a file that is mid-write in another request.
        """
        import time

        now = time.time()
        orphans = []
        for key in self.iter_stored_keys():
            if key in known_keys:
                continue
            path = self.path_for(key)
            if now - path.stat().st_mtime >= min_age_seconds:
                orphans.append(key)
        return orphans

    def purge_staging(self, *, min_age_seconds: float) -> int:
        """Remove abandoned staging files left by interrupted uploads."""
        import time

        staging = self._root / "tmp"
        if not staging.exists():
            return 0

        now = time.time()
        removed = 0
        for path in staging.glob("*.part"):
            if now - path.stat().st_mtime >= min_age_seconds:
                path.unlink(missing_ok=True)
                removed += 1
        return removed


def _iter_chunks(stream: BinaryIO | Iterable[bytes]) -> Iterator[bytes]:
    if hasattr(stream, "read"):
        while True:
            chunk = stream.read(CHUNK_SIZE)  # type: ignore[union-attr]
            if not chunk:
                return
            yield chunk
    else:
        yield from stream  # type: ignore[misc]


def _fsync_directory(directory: Path) -> None:
    """Persist the directory entry itself, not just the file contents.

    Best effort: Windows does not support opening a directory for fsync, and it is not required
    there for the rename to be durable.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except (OSError, PermissionError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def staged_upload(store: FilesystemAssetStore, stream: BinaryIO) -> Iterator[StagedUpload]:
    """Stage bytes, and clean them up if the caller does not promote them."""
    staged = store.stage(stream)
    try:
        yield staged
    finally:
        store.discard(staged)
