"""Judging a result: confidence, emptiness, and the warnings a reviewer sees."""

from __future__ import annotations

import pytest

from media_service.ocr import quality
from media_service.ocr.types import OcrBlock


def block(confidence: float | None, *, identifier: int = 1) -> OcrBlock:
    return OcrBlock(id=identifier, text="text", confidence=confidence)


def test_confidence_is_the_mean_across_blocks() -> None:
    assert quality.mean_confidence([block(0.90), block(0.80)]) == 0.85


def test_an_unmeasured_page_has_no_confidence_rather_than_zero() -> None:
    """Zero means "certain of nothing"; absent means "nothing was measured"."""
    assert quality.mean_confidence([]) is None
    assert quality.mean_confidence([block(None)]) is None


def test_blocks_without_a_score_do_not_drag_the_average_down() -> None:
    assert quality.mean_confidence([block(0.90), block(None, identifier=2)]) == 0.90


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.59, True), (0.60, False), (0.61, False), (None, False)],
)
def test_the_low_confidence_boundary(value: float | None, expected: bool) -> None:
    assert quality.is_low_confidence(value) is expected


def test_an_unmeasured_page_is_not_a_low_confidence_page() -> None:
    """It is an unknown one, and a reviewer sorting by confidence must see the difference."""
    assert quality.is_low_confidence(None) is False
    assert quality.NO_CONFIDENCE_WARNING in quality.warnings_for(
        text="a page of text here", confidence=None
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", True), ("   \n\t ", True), ("short", True), ("long enough", False)],
)
def test_emptiness_is_measured_after_stripping(text: str, expected: bool) -> None:
    assert quality.is_empty(text) is expected


def test_a_clean_page_carries_no_warnings() -> None:
    assert quality.warnings_for(text="Ocean View Apartment", confidence=0.95) == ()


def test_a_poor_page_says_both_things_that_are_wrong_with_it() -> None:
    codes = quality.warnings_for(text="", confidence=0.2)

    assert quality.EMPTY_TEXT_WARNING in codes
    assert quality.LOW_CONFIDENCE_WARNING in codes


def test_thresholds_are_configurable() -> None:
    assert quality.is_low_confidence(0.7, threshold=0.8) is True
    assert quality.is_empty("abc", min_chars=2) is False
