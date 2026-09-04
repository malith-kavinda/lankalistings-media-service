"""Content-addressed blob storage (PRD 12.3, FR-ING-007/008/009, invariant 12)."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from media_service.storage.filesystem import (
    DerivativePurpose,
    FilesystemAssetStore,
    derivative_key,
    original_key,
    params_hash,
    sanitize_filename,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"payload" * 32


@pytest.fixture
def store(tmp_path: Path) -> FilesystemAssetStore:
    return FilesystemAssetStore(tmp_path / "media")


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------------------------


def test_original_key_is_derived_from_content_only() -> None:
    """FR-ING-009: an uploaded filename must not influence where anything is written."""
    checksum = sha256_of(PNG)
    assert original_key(checksum, "image/png") == original_key(checksum, "image/png")
    assert checksum in original_key(checksum, "image/png")


def test_original_key_fans_out_by_checksum_prefix() -> None:
    """Two levels keep directory sizes manageable and map onto object-store prefixes later."""
    checksum = "ab" + "c" * 62
    assert original_key(checksum, "image/png").startswith("originals/ab/cc/")


def test_key_extension_follows_the_detected_type_not_the_upload_name() -> None:
    checksum = sha256_of(PNG)
    assert original_key(checksum, "image/jpeg").endswith(".jpg")
    assert original_key(checksum, "image/png").endswith(".png")
    assert original_key(checksum, "application/x-evil").endswith(".bin")


def test_derivative_key_changes_when_parameters_change() -> None:
    """A parameter change must produce a new path rather than overwrite a referenced file."""
    first = params_hash(DerivativePurpose.OCR_INPUT, "v1", {"threshold": 128})
    second = params_hash(DerivativePurpose.OCR_INPUT, "v1", {"threshold": 200})
    assert first != second

    assert derivative_key(
        "ast_1", DerivativePurpose.OCR_INPUT, first, "image/png"
    ) != derivative_key("ast_1", DerivativePurpose.OCR_INPUT, second, "image/png")


def test_derivative_key_is_stable_for_identical_parameters() -> None:
    """This is what makes preprocessing idempotent under retry."""
    first = params_hash(DerivativePurpose.OCR_INPUT, "v1", {"a": 1, "b": 2})
    second = params_hash(DerivativePurpose.OCR_INPUT, "v1", {"b": 2, "a": 1})
    assert first == second, "Parameter ordering must not affect the hash"


# ---------------------------------------------------------------------------------------------
# Filenames
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("page-1.png", "page-1.png"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\config", "config"),
        ("/absolute/path/scan.jpg", "scan.jpg"),
        ("...", "upload"),
        ("", "upload"),
        ("   ", "upload"),
    ],
)
def test_sanitize_filename_strips_path_information(supplied: str, expected: str) -> None:
    assert sanitize_filename(supplied) == expected


def test_sanitize_filename_preserves_sinhala() -> None:
    """AC-012. Operators upload files named in Sinhala; the name is shown back to them."""
    assert sanitize_filename("පුවත්පත්-1.png") == "පුවත්පත්-1.png"


def test_sanitize_filename_removes_header_injection_characters() -> None:
    """The value is echoed in a Content-Disposition header."""
    cleaned = sanitize_filename('bad"name\r\nX-Injected: 1.png')
    assert '"' not in cleaned
    assert "\r" not in cleaned
    assert "\n" not in cleaned


def test_sanitize_filename_escapes_windows_reserved_names() -> None:
    assert sanitize_filename("CON.png") == "file-CON.png"
    assert sanitize_filename("nul") == "file-nul"


def test_sanitize_filename_is_length_capped() -> None:
    assert len(sanitize_filename("a" * 500 + ".png")) <= 255


# ---------------------------------------------------------------------------------------------
# Staging and promotion
# ---------------------------------------------------------------------------------------------


def test_stage_computes_checksum_and_size_while_streaming(store: FilesystemAssetStore) -> None:
    staged = store.stage(io.BytesIO(PNG))

    assert staged.checksum_sha256 == sha256_of(PNG)
    assert staged.byte_size == len(PNG)
    assert staged.temp_path.exists()
    assert staged.temp_path.read_bytes() == PNG


def test_stage_handles_input_larger_than_one_chunk(store: FilesystemAssetStore) -> None:
    payload = os.urandom(3 * 1024 * 1024 + 17)
    staged = store.stage(io.BytesIO(payload))

    assert staged.byte_size == len(payload)
    assert staged.checksum_sha256 == sha256_of(payload)


def test_promote_moves_bytes_to_the_content_addressed_key(store: FilesystemAssetStore) -> None:
    staged = store.stage(io.BytesIO(PNG))
    key = original_key(staged.checksum_sha256, "image/png")

    stored = store.promote(staged, key)

    assert stored.was_written is True
    assert store.exists(key)
    assert store.read_bytes(key) == PNG
    assert not staged.temp_path.exists(), "Staging file must not be left behind"


def test_identical_bytes_are_written_once(store: FilesystemAssetStore) -> None:
    """Invariant 12 / AC-014 at the filesystem layer."""
    first = store.stage(io.BytesIO(PNG))
    key = original_key(first.checksum_sha256, "image/png")
    store.promote(first, key)

    second = store.stage(io.BytesIO(PNG))
    stored = store.promote(second, key)

    assert stored.was_written is False, "The second upload must reuse the existing blob"
    assert not second.temp_path.exists()
    assert store.read_bytes(key) == PNG


def test_different_bytes_get_different_keys(store: FilesystemAssetStore) -> None:
    first = store.stage(io.BytesIO(PNG))
    second = store.stage(io.BytesIO(PNG + b"different"))

    assert original_key(first.checksum_sha256, "image/png") != original_key(
        second.checksum_sha256, "image/png"
    )


def test_discard_removes_a_staged_file(store: FilesystemAssetStore) -> None:
    staged = store.stage(io.BytesIO(PNG))
    store.discard(staged)
    assert not staged.temp_path.exists()


# ---------------------------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["../escape.png", "originals/../../escape.png", "originals/../../../etc/passwd"],
)
def test_keys_cannot_escape_the_storage_root(store: FilesystemAssetStore, key: str) -> None:
    with pytest.raises(ValueError, match="escapes the storage root"):
        store.path_for(key)


def test_paths_resolve_inside_the_root(store: FilesystemAssetStore) -> None:
    resolved = store.path_for("originals/ab/cd/file.png")
    assert resolved.is_relative_to(store.root.resolve())


# ---------------------------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------------------------


def test_orphan_collection_finds_blobs_with_no_database_row(store: FilesystemAssetStore) -> None:
    """Files land before rows commit, so a crash in between leaves a blob nothing references."""
    staged = store.stage(io.BytesIO(PNG))
    key = original_key(staged.checksum_sha256, "image/png")
    store.promote(staged, key)

    assert store.collect_orphans(known_keys={key}, min_age_seconds=0) == []
    assert store.collect_orphans(known_keys=set(), min_age_seconds=0) == [key]


def test_orphan_collection_respects_the_age_floor(store: FilesystemAssetStore) -> None:
    """A file being written by a concurrent request must not be swept away."""
    staged = store.stage(io.BytesIO(PNG))
    key = original_key(staged.checksum_sha256, "image/png")
    store.promote(staged, key)

    assert store.collect_orphans(known_keys=set(), min_age_seconds=3600) == []


def test_purge_staging_removes_abandoned_uploads(store: FilesystemAssetStore) -> None:
    store.stage(io.BytesIO(PNG))
    store.stage(io.BytesIO(PNG + b"x"))

    assert store.purge_staging(min_age_seconds=0) == 2
    assert store.purge_staging(min_age_seconds=0) == 0


def test_delete_reports_whether_anything_was_removed(store: FilesystemAssetStore) -> None:
    staged = store.stage(io.BytesIO(PNG))
    key = original_key(staged.checksum_sha256, "image/png")
    store.promote(staged, key)

    assert store.delete(key) is True
    assert store.delete(key) is False
