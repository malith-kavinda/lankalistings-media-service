"""Preparing an image for recognition (FR-OCR-004/005/006).

The original is never touched. This produces the bytes stored as the `ocr_input` derivative, keyed
by a hash of the settings that produced them -- so re-running with the same settings rewrites an
identical file, and changing one setting creates a new file instead of overwriting one an existing
extraction still cites.

Every step past orientation is **off by default**, and that is a measured decision rather than
caution. Thresholding and denoising help a photographed, unevenly lit page and hurt a clean scan;
turned on blindly they would change every number in the regression corpus while claiming to be an
improvement. `OCR_PREPROCESS_*` turns them on for deployments whose input needs them, and
`preprocess_version` is recorded on every extraction so two results produced under different
settings are never silently compared.

Orientation is the exception, always applied: a page photographed sideways carries its rotation in
EXIF, and an engine reading raw pixels sees the text on its side. There is no input for which that
is the right answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Final

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from media_service.config import DEFAULT_MAX_IMAGE_PIXELS

PREPROCESS_VERSION: Final = "preprocess/v1"
OUTPUT_CONTENT_TYPE: Final = "image/png"

# Rotations tried when deskew is on, in degrees. Beyond a couple of degrees a scan is misfed rather
# than skewed, and rotating it further would not save the page.
DESKEW_RANGE: Final = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)


class PreprocessError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class PreprocessSettings:
    version: str = PREPROCESS_VERSION
    exif_transpose: bool = True
    grayscale: bool = True
    autocontrast: bool = False
    denoise: bool = False
    # 0-255, or None to leave the image in greyscale. A fixed threshold is deliberate: adaptive
    # thresholding varies with content, and a derivative that differs run to run is not cacheable.
    threshold: int | None = None
    deskew: bool = False
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS

    def as_params(self) -> dict[str, Any]:
        """The settings that identify this derivative, hashed into its storage key.

        `max_pixels` is excluded on purpose: it is a refusal limit, not a transformation, so two
        deployments with different limits that both accept an image produce identical bytes.
        """
        return {
            "exif_transpose": self.exif_transpose,
            "grayscale": self.grayscale,
            "autocontrast": self.autocontrast,
            "denoise": self.denoise,
            "threshold": self.threshold,
            "deskew": self.deskew,
        }


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    version: str
    params: dict[str, Any]
    applied: tuple[str, ...]


class ImagePreprocessor:
    def __init__(self, settings: PreprocessSettings | None = None) -> None:
        self._settings = settings or PreprocessSettings()

    @property
    def settings(self) -> PreprocessSettings:
        return self._settings

    @property
    def version(self) -> str:
        return self._settings.version

    def params(self) -> dict[str, Any]:
        return self._settings.as_params()

    def run(self, image_bytes: bytes) -> PreprocessResult:
        settings = self._settings
        applied: list[str] = []

        try:
            with Image.open(BytesIO(image_bytes)) as opened:
                self._reject_oversized(opened.size)
                image = opened.copy()
        except UnidentifiedImageError as exc:
            raise PreprocessError(
                "IMAGE_UNREADABLE", "The stored bytes could not be decoded."
            ) from exc
        except Image.DecompressionBombError as exc:
            raise PreprocessError(
                "IMAGE_TOO_LARGE",
                "The image declares more pixels than this service will decode.",
            ) from exc

        if settings.exif_transpose:
            image = ImageOps.exif_transpose(image) or image
            applied.append("exif_transpose")

        if settings.grayscale or settings.autocontrast or settings.threshold is not None:
            # Every later step needs a single channel; requesting any of them implies this one.
            image = image.convert("L")
            applied.append("grayscale")

        if settings.autocontrast:
            image = ImageOps.autocontrast(image)
            applied.append("autocontrast")

        if settings.denoise:
            image = image.filter(ImageFilter.MedianFilter(size=3))
            applied.append("denoise")

        if settings.deskew:
            angle = _estimate_skew(image)
            if angle:
                image = image.rotate(
                    angle, resample=Image.BICUBIC, expand=True, fillcolor=255
                )
                applied.append(f"deskew({angle:+.1f})")

        if settings.threshold is not None:
            limit = settings.threshold
            image = image.point(lambda value: 255 if value > limit else 0, mode="1")
            applied.append(f"threshold({limit})")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return PreprocessResult(
            image_bytes=buffer.getvalue(),
            content_type=OUTPUT_CONTENT_TYPE,
            width=image.width,
            height=image.height,
            version=settings.version,
            params=settings.as_params(),
            applied=tuple(applied),
        )

    def _reject_oversized(self, size: tuple[int, int]) -> None:
        pixels = max(size[0], 1) * max(size[1], 1)
        if pixels > self._settings.max_pixels:
            raise PreprocessError(
                "IMAGE_TOO_LARGE",
                f"The image is {size[0]}x{size[1]} ({pixels} pixels); "
                f"the limit is {self._settings.max_pixels}.",
            )


def _estimate_skew(image: Image.Image) -> float:
    """Pick the rotation whose horizontal projection has the sharpest peaks.

    Text lines produce dark rows and light gaps; when the page is straight those rows line up and
    the variance of the row sums is at its highest. Cheap, deterministic, and good enough for the
    fraction of a degree a scanner introduces -- which is all this is for.
    """
    sample = image.convert("L")
    # A fixed working width keeps the cost constant and the answer stable across page sizes.
    if sample.width > 1000:
        ratio = 1000 / sample.width
        sample = sample.resize((1000, max(int(sample.height * ratio), 1)), Image.BILINEAR)

    best_angle = 0.0
    best_score = -1.0
    for angle in DESKEW_RANGE:
        rotated = (
            sample
            if angle == 0.0
            else sample.rotate(angle, resample=Image.BILINEAR, fillcolor=255)
        )
        score = _row_variance(rotated)
        if score > best_score:
            best_score, best_angle = score, angle
    return best_angle


def _row_variance(image: Image.Image) -> float:
    pixels = list(image.getdata())
    width, height = image.size
    if width == 0 or height == 0:
        return 0.0
    sums = [
        sum(pixels[row * width : (row + 1) * width]) / width for row in range(height)
    ]
    mean = sum(sums) / len(sums)
    return sum((value - mean) ** 2 for value in sums) / len(sums)
