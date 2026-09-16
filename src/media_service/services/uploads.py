"""Staging and validating an upload set before any row is written.

The order of operations is the requirement, not an implementation detail. Every file is streamed to
disk and hashed, then the *whole set* is validated, and only then does anything reach the database.
Validating as each file arrives would accept the first twenty images of a batch that breaches the
total-size cap, leaving the caller with a partial ingestion it never asked for (PRD 15.1 against
FR-ING-004).

Bytes are streamed, never read whole. `await file.read()` across 25 files at the per-image limit
peaks at a quarter of a gigabyte of resident memory for a request that writes all of it to disk
anyway.

The content type is *detected* by decoding the bytes, never taken from the client. A declared type
is a claim about a file; the decoder's answer is a fact about it, and it is the one that picks the
stored extension.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import BinaryIO, Final

from PIL import Image, UnidentifiedImageError

from media_service.api.errors import ServiceError
from media_service.storage import FilesystemAssetStore, StagedUpload, sanitize_filename

SUPPORTED_IMAGE_TYPES: Final = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}

CONTENT_TYPE_BY_FORMAT: Final[dict[str, str]] = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
    "MPO": "image/jpeg",
}

EXIF_ORIENTATION_TAG: Final = 274


@dataclass(frozen=True, slots=True)
class IncomingFile:
    """One part of the multipart request, still unread."""

    filename: str
    declared_content_type: str | None
    stream: BinaryIO


@dataclass(frozen=True, slots=True)
class StagedImage:
    """A validated image on disk, with the facts its rows need and no bytes in memory."""

    index: int
    filename: str
    declared_content_type: str | None
    staged: StagedUpload
    content_type: str
    image_format: str
    width: int
    height: int
    exif_orientation: int | None

    @property
    def checksum(self) -> str:
        return self.staged.checksum_sha256

    @property
    def byte_size(self) -> int:
        return self.staged.byte_size


@dataclass(frozen=True, slots=True)
class UploadLimits:
    max_images_per_batch: int
    max_image_bytes: int
    max_batch_bytes: int
    max_image_pixels: int


class UploadRejectedError(ServiceError):
    """One 422 carrying every problem in the set, so a 25-file upload is fixed in one pass."""

    def __init__(self, details: list[dict[str, str | None]]) -> None:
        super().__init__(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=details,
        )


def stage_upload_set(
    store: FilesystemAssetStore,
    files: Sequence[IncomingFile],
    *,
    limits: UploadLimits,
) -> list[StagedImage]:
    """Stage every file, validate the set, and return it -- or stage nothing and raise.

    On any rejection the staged bytes are discarded before returning, so a refused request leaves
    no files behind for the orphan sweep to find later.
    """
    _reject_set_shape(files, limits=limits)

    staged: list[StagedImage] = []
    raw: list[StagedUpload] = []
    details: list[dict[str, str | None]] = []
    total_bytes = 0

    try:
        for index, incoming in enumerate(files):
            upload = store.stage(incoming.stream)
            raw.append(upload)
            total_bytes += upload.byte_size

            problem = _reject_file(incoming, upload, index=index, limits=limits)
            if problem is not None:
                details.append(problem)
                continue

            staged.append(_describe(incoming, upload, index=index))

        if total_bytes > limits.max_batch_bytes:
            details.append(
                {
                    "field": "images",
                    "code": "MAX_BATCH_SIZE",
                    "message": (
                        f"The upload totals {total_bytes} bytes; the limit for one batch is "
                        f"{limits.max_batch_bytes}."
                    ),
                }
            )

        if details:
            raise UploadRejectedError(details)
    except BaseException:
        for upload in raw:
            store.discard(upload)
        raise

    return staged


def _reject_set_shape(files: Sequence[IncomingFile], *, limits: UploadLimits) -> None:
    """Refuse a request whose shape is wrong before a single byte is written."""
    if not files:
        raise UploadRejectedError(
            [{"field": "images", "code": "REQUIRED", "message": "At least one image is required."}]
        )
    if len(files) > limits.max_images_per_batch:
        raise UploadRejectedError(
            [
                {
                    "field": "images",
                    "code": "MAX_COUNT",
                    "message": (
                        f"A batch takes at most {limits.max_images_per_batch} images; "
                        f"{len(files)} were sent."
                    ),
                }
            ]
        )


def _reject_file(
    incoming: IncomingFile,
    upload: StagedUpload,
    *,
    index: int,
    limits: UploadLimits,
) -> dict[str, str | None] | None:
    field = f"images[{index}]"

    if upload.byte_size == 0:
        return {"field": field, "code": "REQUIRED", "message": "The uploaded file is empty."}

    if upload.byte_size > limits.max_image_bytes:
        return {
            "field": field,
            "code": "MAX_SIZE",
            "message": (
                f"{incoming.filename} is {upload.byte_size} bytes; the limit for one image is "
                f"{limits.max_image_bytes}."
            ),
        }

    try:
        detected = _probe(upload)
    except UnidentifiedImageError:
        return {
            "field": field,
            "code": "UNREADABLE_IMAGE",
            "message": f"{incoming.filename} could not be decoded as an image.",
        }
    except Image.DecompressionBombError:
        # Pillow's own guard, surfaced as a validation failure rather than a 500: a pixel count
        # large enough to exhaust memory on decode is a bad request, not a service fault.
        return {
            "field": field,
            "code": "IMAGE_TOO_LARGE",
            "message": f"{incoming.filename} declares more pixels than this service will decode.",
        }

    content_type, _, width, height, _ = detected
    pixels = max(width, 1) * max(height, 1)
    if pixels > limits.max_image_pixels:
        # Checked here rather than left to Pillow, which raises only above *twice* its own limit
        # and merely warns below that. A file in the warning band is small on disk, passes the byte
        # cap, and then costs hundreds of megabytes to decode in the worker.
        return {
            "field": field,
            "code": "IMAGE_TOO_LARGE",
            "message": (
                f"{incoming.filename} is {width}x{height} ({pixels} pixels); the limit is "
                f"{limits.max_image_pixels}."
            ),
        }

    if content_type not in SUPPORTED_IMAGE_TYPES:
        return {
            "field": field,
            "code": "UNSUPPORTED_CONTENT_TYPE",
            "message": (
                f"{incoming.filename} is {content_type}; supported types are "
                f"{', '.join(sorted(SUPPORTED_IMAGE_TYPES))}."
            ),
        }
    return None


def _probe(upload: StagedUpload) -> tuple[str, str, int, int, int | None]:
    """Read format, dimensions, and orientation from the staged file."""
    with Image.open(upload.temp_path) as image:
        image_format = image.format or "UNKNOWN"
        width, height = image.size
        orientation = None
        # `getexif` is cheap and does not decode pixels; a file with no EXIF returns an empty map.
        try:
            orientation = image.getexif().get(EXIF_ORIENTATION_TAG)
        except (AttributeError, OSError, ValueError):
            orientation = None

    content_type = CONTENT_TYPE_BY_FORMAT.get(image_format, f"image/{image_format.lower()}")
    return content_type, image_format, width, height, orientation


def _describe(incoming: IncomingFile, upload: StagedUpload, *, index: int) -> StagedImage:
    content_type, image_format, width, height, orientation = _probe(upload)
    return StagedImage(
        index=index,
        filename=sanitize_filename(incoming.filename),
        declared_content_type=incoming.declared_content_type,
        staged=upload,
        content_type=content_type,
        image_format=image_format,
        width=width,
        height=height,
        exif_orientation=int(orientation) if orientation is not None else None,
    )
