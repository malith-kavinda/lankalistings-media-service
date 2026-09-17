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

Phase 3 removed the rule-based extractor that used to live here. It is now an LLM *provider*
(`LLM_PROVIDER=rule_based`), which puts it behind the same startup guard as `fake`: a heuristic that
invents advertisements without a model must not be constructible in production (PRD 11.6).
"""

from __future__ import annotations

from typing import Protocol

from media_service.api.errors import OcrUnavailableError, ServiceError
from media_service.llm.service import ExtractionContext, ExtractionOutput
from media_service.llm.types import RETRYABLE_CODES
from media_service.ocr.preprocess import (
    ImagePreprocessor,
    PreprocessError,
    PreprocessResult,
    PreprocessSettings,
)
from media_service.ocr.protocol import OcrProvider
from media_service.ocr.types import OcrResult

__all__ = [
    "ExtractionContext",
    "ExtractionOutput",
    "ExtractionStep",
    "ImagePreprocessor",
    "OcrProvider",
    "OcrResult",
    "PreprocessResult",
    "PreprocessSettings",
    "PreprocessStep",
    "StageError",
    "extraction_failure",
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
    """Turning OCR blocks into candidates.

    Takes a context as well as the result, because an extractor that calls a provider has to record
    what it did: one row per attempt, keyed by item and generation. `rebuild` re-derives candidates
    from a stored response with no provider call, which is what makes a resume free.
    """

    def run(self, result: OcrResult, context: ExtractionContext) -> ExtractionOutput: ...

    def rebuild(self, payload: dict[str, object], result: OcrResult) -> ExtractionOutput: ...

    def request_hash_for(self, result: OcrResult) -> str: ...


# Failures the OCR layer raises, and whether running the same page again could plausibly help.
# An engine that is not installed may be installed by the next attempt; an image that will not
# decode will not decode next time either.
RETRYABLE_OCR_CODES = frozenset({"OCR_UNAVAILABLE", "OCR_TIMEOUT", "OCR_FAILED"})
TERMINAL_PREPROCESS_CODES = frozenset({"IMAGE_UNREADABLE", "IMAGE_TOO_LARGE"})


def extraction_failure(output: ExtractionOutput) -> StageError:
    """Translate an extraction failure into the pipeline's vocabulary.

    Only transport-shaped failures are worth another automatic attempt. A response that failed the
    schema after its repair was spent will fail the same way next time, so the item goes to
    `needs_attention` for a person rather than burning the item's attempts on a certainty.
    """
    code = output.failure_code or "LLM_EXTRACTION_FAILED"
    return StageError(
        code,
        output.failure_detail or "Extraction failed.",
        retryable=code in RETRYABLE_CODES,
    )


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
