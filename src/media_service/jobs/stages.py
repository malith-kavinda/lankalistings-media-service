"""The three pipeline stages, behind protocols so their engines can be replaced.

Each stage is a small object with one method, and none of them touches the database. The runner
owns transactions; a stage takes bytes or text and returns a result. That split is what lets the
long calls -- Tesseract, and later a provider over the network -- run with no transaction open, and
it is what lets Phase 2 swap in a structured OCR provider and Phase 3 an LLM extractor without the
runner changing at all.

`StageError` carries whether the failure is worth another attempt. A timeout is; an image that will
not decode is not, and retrying it three times only delays the moment a person is told.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Final, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

from media_service.api.errors import OcrUnavailableError, ServiceError
from media_service.config import DEFAULT_MAX_IMAGE_PIXELS
from media_service.domain.categories import DEFAULT_CATALOG
from media_service.domain.listings import CandidateDraft
from media_service.services.ocr import OcrEngine

# Phase 2 replaces this with the configurable pipeline it owns. The version string is recorded on
# every derivative and extraction, so results produced under different preprocessing are
# distinguishable after the fact rather than silently comparable.
PREPROCESS_VERSION: Final = "preprocess/v0"
PREPROCESS_PARAMS: Final[dict[str, Any]] = {"exif_transpose": True, "mode": "L"}

# Below this, the page is treated as carrying no readable text at all rather than as an extraction
# that happened to find very little.
MIN_TEXT_CHARS: Final = 8


class StageError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    image_bytes: bytes
    content_type: str
    width: int
    height: int
    version: str = PREPROCESS_VERSION
    params: dict[str, Any] = field(default_factory=lambda: dict(PREPROCESS_PARAMS))


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    engine: str
    engine_version: str
    languages: str
    confidence_label: str
    mean_confidence: float | None
    width: int | None
    height: int | None
    duration_ms: int
    blocks: list[dict[str, Any]] | None = None

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < MIN_TEXT_CHARS


class PreprocessStep(Protocol):
    def run(self, image_bytes: bytes) -> PreprocessResult: ...


class OcrStep(Protocol):
    def run(self, image_bytes: bytes, *, content_type: str) -> OcrResult: ...


class ExtractionStep(Protocol):
    def run(self, result: OcrResult) -> list[CandidateDraft]: ...


class PillowPreprocessor:
    """Orientation and greyscale, written once per asset and parameter set.

    `exif_transpose` first, always: a phone photo of a page carries its rotation in metadata, and an
    OCR engine reading the raw pixels sees the text sideways. The original file is never modified --
    this produces a separate `ocr_input` derivative, so the evidence a moderator reviews is still
    the image that was uploaded.

    The pixel cap is checked again here even though upload validation already applied it. This is
    the only place that decodes a full image buffer, and it is reachable by rows that never passed
    through an upload -- a legacy import, or an asset stored before the cap was lowered.
    """

    def __init__(self, *, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> None:
        self._max_pixels = max_pixels

    def run(self, image_bytes: bytes) -> PreprocessResult:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                self._reject_oversized(image.size)
                oriented = ImageOps.exif_transpose(image) or image
                greyscale = oriented.convert("L")
                buffer = BytesIO()
                greyscale.save(buffer, format="PNG")
                return PreprocessResult(
                    image_bytes=buffer.getvalue(),
                    content_type="image/png",
                    width=greyscale.width,
                    height=greyscale.height,
                )
        except UnidentifiedImageError as exc:
            raise StageError(
                "IMAGE_UNREADABLE", "The stored bytes could not be decoded.", retryable=False
            ) from exc
        except Image.DecompressionBombError as exc:
            raise StageError(
                "IMAGE_TOO_LARGE",
                "The image declares more pixels than this service will decode.",
                retryable=False,
            ) from exc

    def _reject_oversized(self, size: tuple[int, int]) -> None:
        pixels = max(size[0], 1) * max(size[1], 1)
        if pixels > self._max_pixels:
            raise StageError(
                "IMAGE_TOO_LARGE",
                f"The image is {size[0]}x{size[1]} ({pixels} pixels); "
                f"the limit is {self._max_pixels}.",
                retryable=False,
            )


class EngineOcrStep:
    """Adapts the existing OCR engine to the stage protocol.

    Deliberately thin. Phase 2 introduces the structured provider -- blocks, bounding boxes, and
    per-word confidence -- and replaces this class; everything around it stays as it is, which is
    the point of the protocol.
    """

    def __init__(self, engine: OcrEngine) -> None:
        self._engine = engine

    def run(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        started = time.monotonic()
        try:
            output = self._engine.extract_text(image_bytes, content_type=content_type)
        except OcrUnavailableError as exc:
            # The engine is missing or misconfigured. Nothing about this item is wrong, so it is
            # worth another attempt once the deployment is fixed.
            raise StageError("OCR_UNAVAILABLE", exc.message, retryable=True) from exc
        except ServiceError as exc:
            raise StageError(exc.code, exc.message, retryable=False) from exc

        return OcrResult(
            text=output.text,
            engine=output.engine,
            engine_version=output.model_version,
            languages=output.language,
            confidence_label=output.confidence,
            mean_confidence=None,
            width=output.width,
            height=output.height,
            duration_ms=int((time.monotonic() - started) * 1000),
        )


class RuleBasedExtractor:
    """Phase 1's stand-in extractor: heuristics over the OCR text, at most one candidate.

    It exists so the pipeline can be exercised end to end before Phase 3 wires a model in, and it
    is honest about what it is -- one advertisement per page, which is precisely the limitation the
    LLM stage is there to remove. Candidates it produces are marked `ocr_heuristic`, so a reviewer
    and any later analysis can tell them apart from model output.
    """

    def run(self, result: OcrResult) -> list[CandidateDraft]:
        if result.is_empty:
            # Not a failure. A page with no advertisements on it is a legitimate outcome, and the
            # item ends in `no_ads` rather than in an error state (AC-003).
            return []

        from media_service.services.advertisements import AdvertisementService

        fields = AdvertisementService._extract_advertisement_fields(result.text)  # noqa: SLF001
        category, unmapped = DEFAULT_CATALOG.resolve(fields["category"])

        return [
            CandidateDraft(
                index=0,
                title=fields["title"],
                description=fields["description"],
                category=category,
                location=fields["location"],
                price=fields["price"],
                confidence_label=result.confidence_label,
                confidence=result.mean_confidence,
                source_text=result.text,
                warning_codes=("CATEGORY_UNMAPPED",) if unmapped else (),
                extracted_values=dict(fields),
            )
        ]
