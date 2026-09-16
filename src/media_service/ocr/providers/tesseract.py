"""Tesseract, returning blocks rather than a wall of text.

The prototype asked Tesseract for a string. This asks for word rows with geometry and confidence,
which is what makes everything downstream possible: a candidate can cite the blocks it came from, a
reviewer can see the region an answer was read out of, and a page can be judged low confidence
before anyone reads its output.

Availability is cached with a short TTL. `/health` calls it on every request, and the old code
shelled out to `tesseract --list-langs` each time -- a subprocess spawn per health check, which a
load balancer polls every few seconds.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Final

from PIL import Image, UnidentifiedImageError

from media_service.api.errors import OcrUnavailableError, ServiceError
from media_service.ocr import blocks as block_assembly
from media_service.ocr import quality
from media_service.ocr.tesseract_cli import (
    TesseractCli,
    TesseractOptions,
    as_word_data,
    parse_tsv,
)
from media_service.ocr.types import BoxSource, OcrResult

AVAILABILITY_TTL_SECONDS: Final = 30.0
PROVIDER_NAME: Final = "tesseract"

SUPPORTED_IMAGE_TYPES: Final = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/tiff",
    "image/bmp",
}

EXTENSION_BY_TYPE: Final[dict[str, str]] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
}


class TesseractOcrProvider:
    def __init__(
        self,
        *,
        options: TesseractOptions | None = None,
        cli: TesseractCli | None = None,
        low_confidence_threshold: float = quality.DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        empty_text_min_chars: int = quality.DEFAULT_EMPTY_TEXT_MIN_CHARS,
        preprocess_version: str = "preprocess/v1",
        availability_ttl_seconds: float = AVAILABILITY_TTL_SECONDS,
    ) -> None:
        self._cli = cli or TesseractCli(options)
        self._low_confidence_threshold = low_confidence_threshold
        self._empty_text_min_chars = empty_text_min_chars
        self._preprocess_version = preprocess_version
        self._ttl = availability_ttl_seconds
        self._cached_reason: str | None = None
        self._checked_at: float | None = None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def languages(self) -> str:
        return self._cli.options.languages

    @property
    def engine_version(self) -> str | None:
        return self._cli.version()

    @property
    def cli(self) -> TesseractCli:
        """Exposed so the hybrid provider can recognise a crop with the same binary and settings.

        Composition rather than a second CLI built from the same configuration: two of those could
        drift apart, and then switching provider would silently change recognition.
        """
        return self._cli

    # -- availability --------------------------------------------------------------------------

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        """Why Tesseract cannot run here, or None.

        Nothing in this method writes to the process environment or to a library global. The old
        implementation did both, from a method `/health` calls on every request.
        """
        now = time.monotonic()
        if self._checked_at is not None and now - self._checked_at < self._ttl:
            return self._cached_reason

        reason = self._check()
        self._cached_reason = reason
        self._checked_at = now
        return reason

    def _check(self) -> str | None:
        if self._cli.resolve_command() is None:
            return "The Tesseract OCR executable is not installed or not on PATH."

        available = self._cli.languages()
        if available is None:
            return "The Tesseract executable could not be run."

        missing = [
            language
            for language in self._cli.options.languages.split("+")
            if language and language not in available
        ]
        if missing:
            # Named, not silently skipped: Sinhala missing means the service cannot do the job it
            # exists for, and that must be visible rather than showing up as poor accuracy.
            return f"Missing Tesseract language data: {', '.join(missing)}."
        return None

    # -- extraction ----------------------------------------------------------------------------

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "file",
                        "code": "UNSUPPORTED_CONTENT_TYPE",
                        "message": "Only JPEG, PNG, WebP, TIFF, and BMP images are supported.",
                    }
                ],
            )

        reason = self.availability_reason()
        if reason:
            raise OcrUnavailableError(reason)

        width, height = self._dimensions(image_bytes)
        started = time.monotonic()
        tsv = self._cli.image_to_tsv(
            image_bytes, suffix=EXTENSION_BY_TYPE.get(content_type, ".png")
        )
        duration_ms = int((time.monotonic() - started) * 1000)

        words = block_assembly.words_from_tesseract(as_word_data(parse_tsv(tsv)))
        assembled = block_assembly.assemble(
            words,
            page_width=width or 1,
            box_source=BoxSource.ENGINE,
            detector=PROVIDER_NAME,
        )
        text = block_assembly.page_text(assembled)
        confidence = quality.mean_confidence(assembled)

        return OcrResult(
            text=text,
            blocks=assembled,
            provider=PROVIDER_NAME,
            engine=PROVIDER_NAME,
            engine_version=self._cli.version(),
            languages=self._cli.options.languages,
            mean_confidence=confidence,
            low_confidence=quality.is_low_confidence(
                confidence, threshold=self._low_confidence_threshold
            ),
            width=width,
            height=height,
            preprocess_version=self._preprocess_version,
            duration_ms=duration_ms,
            warnings=quality.warnings_for(
                text=text,
                confidence=confidence,
                low_confidence_threshold=self._low_confidence_threshold,
                empty_text_min_chars=self._empty_text_min_chars,
            ),
        )

    @staticmethod
    def _dimensions(image_bytes: bytes) -> tuple[int | None, int | None]:
        """Read the header only. Block geometry lives in this space, so it has to be right."""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                return image.width, image.height
        except UnidentifiedImageError:
            return None, None
