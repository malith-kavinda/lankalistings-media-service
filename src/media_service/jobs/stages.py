"""The pipeline stages, behind protocols so their engines can be replaced.

Each stage is a small object with one method, and none of them touches the database. The runner owns
transactions; a stage takes bytes or text and returns a result. That split is what lets the long
calls -- Tesseract, and later a provider over the network -- run with no transaction open, and it is
what let Phase 2 replace the whole OCR stage without the runner changing shape.

Since Phase 2 the OCR stage *is* `ocr.OcrProvider`: there is no wrapper protocol around it,
because a second protocol describing the same thing is a second place for the contract to drift.
Preprocessing moved to `ocr.preprocess` for the same reason -- it is part of how a page is read,
not part of how work is scheduled.

`StageError` carries whether the failure is worth another attempt. A timeout is; an image that will
not decode is not, and retrying it three times only delays the moment a person is told.
"""

from __future__ import annotations

from typing import Protocol

from media_service.api.errors import OcrUnavailableError, ServiceError
from media_service.domain.categories import DEFAULT_CATALOG
from media_service.domain.listings import CandidateDraft
from media_service.ocr import quality
from media_service.ocr.preprocess import (
    ImagePreprocessor,
    PreprocessError,
    PreprocessResult,
    PreprocessSettings,
)
from media_service.ocr.protocol import OcrProvider
from media_service.ocr.types import OcrResult

__all__ = [
    "ExtractionStep",
    "ImagePreprocessor",
    "OcrProvider",
    "OcrResult",
    "PreprocessResult",
    "PreprocessSettings",
    "PreprocessStep",
    "RuleBasedExtractor",
    "StageError",
    "to_stage_error",
]


class StageError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class PreprocessStep(Protocol):
    def run(self, image_bytes: bytes) -> PreprocessResult: ...

    @property
    def version(self) -> str: ...

    def params(self) -> dict[str, object]: ...


class ExtractionStep(Protocol):
    def run(self, result: OcrResult) -> list[CandidateDraft]: ...


# Failures the OCR layer raises, and whether running the same page again could plausibly help.
# An engine that is not installed may be installed by the next attempt; an image that will not
# decode will not decode next time either.
RETRYABLE_OCR_CODES = frozenset({"OCR_UNAVAILABLE", "OCR_TIMEOUT", "OCR_FAILED"})
TERMINAL_PREPROCESS_CODES = frozenset({"IMAGE_UNREADABLE", "IMAGE_TOO_LARGE"})


def to_stage_error(error: Exception) -> StageError:
    """Translate an OCR or preprocessing failure into the runner's vocabulary."""
    if isinstance(error, PreprocessError):
        return StageError(
            error.code, error.message, retryable=error.code not in TERMINAL_PREPROCESS_CODES
        )
    if isinstance(error, OcrUnavailableError):
        return StageError("OCR_UNAVAILABLE", error.message, retryable=True)
    if isinstance(error, ServiceError):
        return StageError(
            error.code, error.message, retryable=error.code in RETRYABLE_OCR_CODES
        )
    return StageError("INTERNAL_ERROR", str(error), retryable=True)


class RuleBasedExtractor:
    """Phase 1's stand-in extractor: heuristics over the OCR text, at most one candidate.

    It exists so the pipeline can be exercised end to end before Phase 3 wires a model in, and it
    is honest about what it is -- one advertisement per page, which is precisely the limitation the
    LLM stage is there to remove. Candidates it produces are marked `ocr_heuristic`, so a reviewer
    and any later analysis can tell them apart from model output.
    """

    def run(self, result: OcrResult) -> list[CandidateDraft]:
        if quality.is_empty(result.text):
            # Not a failure. A page with no advertisements on it is a legitimate outcome, and the
            # item ends in `no_ads` rather than in an error state (AC-003).
            return []

        from media_service.services.advertisements import AdvertisementService

        fields = AdvertisementService._extract_advertisement_fields(result.text)  # noqa: SLF001
        category, unmapped = DEFAULT_CATALOG.resolve(fields["category"])
        warnings = tuple(result.warnings)
        if unmapped:
            warnings = (*warnings, "CATEGORY_UNMAPPED")

        return [
            CandidateDraft(
                index=0,
                title=fields["title"],
                description=fields["description"],
                category=category,
                location=fields["location"],
                price=fields["price"],
                confidence_label=_confidence_label(result.mean_confidence),
                confidence=result.mean_confidence,
                source_text=result.text,
                # Every block, because this extractor cannot tell which part of the page an
                # advertisement came from. Phase 3's model cites the blocks it actually used.
                source_block_ids=tuple(block.id for block in result.blocks),
                warning_codes=warnings,
                extracted_values=dict(fields),
            )
        ]


def _confidence_label(value: float | None) -> str:
    from media_service.ocr.compat import confidence_to_word

    return confidence_to_word(value)
