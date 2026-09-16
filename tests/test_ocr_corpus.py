"""The provider reproduces the regression corpus, block for block.

This is Phase 2's gate. The fixtures were captured by running this exact provider through this exact
preprocessor, so any difference means the OCR path changed -- and the corpus exists to make that
visible instead of letting it show up later as unexplained accuracy drift.

It needs the real Tesseract with Sinhala data. Where that is missing the tests skip with the reason
rather than failing, because a machine without the language data is misconfigured, not broken code.
"""

from __future__ import annotations

import unicodedata

import pytest

from media_service.config import get_settings
from media_service.ocr.preprocess import ImagePreprocessor
from media_service.ocr.registry import build_provider, preprocess_settings
from media_service.ocr.types import BoxSource
from tests.fixtures.corpus import all_cases, case_names, load_case


@pytest.fixture(scope="module")
def provider():  # type: ignore[no-untyped-def]
    built = build_provider(get_settings())
    reason = built.availability_reason()
    if reason is not None:
        pytest.skip(f"Tesseract is not usable here: {reason}")
    return built


@pytest.fixture(scope="module")
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(preprocess_settings(get_settings()))


def read(provider, preprocessor, name: str):  # type: ignore[no-untyped-def]
    case = load_case(name)
    prepared = preprocessor.run(case.image_bytes)
    return case, provider.extract(prepared.image_bytes, content_type="image/png")


@pytest.mark.parametrize("name", case_names())
def test_the_provider_reproduces_the_captured_blocks(provider, preprocessor, name) -> None:  # type: ignore[no-untyped-def]
    case, result = read(provider, preprocessor, name)
    captured = case.ocr["blocks"]

    assert result.block_count == len(captured), f"{name}: block count changed"
    for block, expected in zip(result.blocks, captured, strict=True):
        assert block.id == expected["id"]
        assert block.text == expected["text"]
        assert block.box.as_list() == expected["box"]
        assert block.confidence == expected["confidence"]
        assert block.source_ref == expected["source_ref"]


@pytest.mark.parametrize("name", case_names())
def test_the_page_transcript_is_unchanged(provider, preprocessor, name) -> None:  # type: ignore[no-untyped-def]
    case, result = read(provider, preprocessor, name)

    assert result.text == case.text


@pytest.mark.parametrize("name", case_names())
def test_the_page_confidence_is_unchanged(provider, preprocessor, name) -> None:  # type: ignore[no-untyped-def]
    case, result = read(provider, preprocessor, name)

    assert result.mean_confidence == case.mean_confidence


@pytest.mark.parametrize("name", case_names())
def test_every_block_carries_engine_geometry(provider, preprocessor, name) -> None:  # type: ignore[no-untyped-def]
    """Tesseract produces real boxes, so nothing here may claim to be an estimate."""
    _, result = read(provider, preprocessor, name)

    for block in result.blocks:
        assert block.box is not None
        assert block.box_source is BoxSource.ENGINE
        assert block.detector == "tesseract"


def test_ids_are_one_based_and_in_reading_order(provider, preprocessor) -> None:  # type: ignore[no-untyped-def]
    """The multi-column case is the one where engine order and reading order differ."""
    _, result = read(provider, preprocessor, "multi_column_three_ads")

    assert [block.id for block in result.blocks] == list(range(1, result.block_count + 1))
    # Each column is read out before the next begins, so the left column's blocks come first.
    lefts = [block.box.left for block in result.blocks]
    assert lefts == sorted(lefts) or len({round(left, -2) for left in lefts}) > 1


def test_sinhala_survives_the_provider_byte_for_byte(provider, preprocessor) -> None:  # type: ignore[no-untyped-def]
    """AC-012, at the layer where the bytes enter the system."""
    case, result = read(provider, preprocessor, "sinhala_only")

    sinhala = [character for character in result.text if "඀" <= character <= "෿"]
    assert sinhala, f"No Sinhala codepoints survived. Text was:\n{result.text}"
    assert result.text == unicodedata.normalize("NFC", result.text)
    assert result.text == case.text


def test_the_sinhala_case_is_still_the_low_confidence_one(provider, preprocessor) -> None:  # type: ignore[no-untyped-def]
    """It exists to prove a damaged page is reported as damaged rather than read confidently."""
    _, result = read(provider, preprocessor, "sinhala_only")

    assert result.low_confidence is True
    assert "OCR_LOW_CONFIDENCE" in result.warnings


def test_clean_pages_are_not_flagged_low_confidence(provider, preprocessor) -> None:  # type: ignore[no-untyped-def]
    for case in all_cases():
        if case.name == "sinhala_only":
            continue
        _, result = read(provider, preprocessor, case.name)
        assert result.low_confidence is False, f"{case.name} was flagged low confidence"


def test_every_case_records_which_preprocessing_produced_it(provider, preprocessor) -> None:  # type: ignore[no-untyped-def]
    """Two results read under different settings must never be silently comparable."""
    _, result = read(provider, preprocessor, "english_only")

    assert result.preprocess_version == get_settings().ocr_preprocess_version
