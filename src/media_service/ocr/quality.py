"""Judging an OCR result.

Separate from the providers because every provider is judged the same way. A page is not "low
confidence for Tesseract" -- it is low confidence, and a reviewer needs the same signal whichever
engine read it.

`mean_confidence` averages **block** confidences, not word confidences. A block is the unit a
reviewer looks at and the unit a candidate cites, and word-weighting would let one long, cleanly
printed paragraph hide a short garbled one. This also matches how the corpus fixtures were
captured, so a provider result and a fixture are directly comparable.
"""

from __future__ import annotations

from typing import Final

from media_service.ocr.types import OcrBlock

# Warning codes travel to the reviewer through the item and the candidate, so they are part of the
# contract rather than log text.
LOW_CONFIDENCE_WARNING: Final = "OCR_LOW_CONFIDENCE"
EMPTY_TEXT_WARNING: Final = "OCR_EMPTY_TEXT"
NO_CONFIDENCE_WARNING: Final = "OCR_CONFIDENCE_UNAVAILABLE"

DEFAULT_LOW_CONFIDENCE_THRESHOLD: Final = 0.60
DEFAULT_EMPTY_TEXT_MIN_CHARS: Final = 8


def mean_confidence(blocks: tuple[OcrBlock, ...] | list[OcrBlock]) -> float | None:
    """The page's confidence, or None when no block carried one.

    None rather than 0.0. Zero is a measurement meaning "the engine was certain of nothing"; absent
    means "nothing was measured", and a reviewer sorting by confidence must not see the second
    presented as the first.
    """
    scored = [block.confidence for block in blocks if block.confidence is not None]
    if not scored:
        return None
    return round(sum(scored) / len(scored), 4)


def is_low_confidence(
    value: float | None, *, threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
) -> bool:
    """An unmeasured page is not a low-confidence page; it is an unknown one."""
    return value is not None and value < threshold


def is_empty(text: str, *, min_chars: int = DEFAULT_EMPTY_TEXT_MIN_CHARS) -> bool:
    """Whether the page carries too little text to be worth extracting from.

    Counted after stripping, so a page of whitespace and stray marks reads as empty rather than as
    an extraction that happened to find very little.
    """
    return len(text.strip()) < min_chars


def warnings_for(
    *,
    text: str,
    confidence: float | None,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    empty_text_min_chars: int = DEFAULT_EMPTY_TEXT_MIN_CHARS,
) -> tuple[str, ...]:
    codes: list[str] = []
    if is_empty(text, min_chars=empty_text_min_chars):
        codes.append(EMPTY_TEXT_WARNING)
    if confidence is None:
        codes.append(NO_CONFIDENCE_WARNING)
    elif is_low_confidence(confidence, threshold=low_confidence_threshold):
        codes.append(LOW_CONFIDENCE_WARNING)
    return tuple(codes)
