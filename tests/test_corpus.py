"""Corpus integrity (PRD 18.3).

These tests do not measure extraction quality -- nothing extracts yet. They assert that the corpus
itself is well formed, so that when Phases 2 and 3 start measuring against it, a failure means the
pipeline regressed rather than that a fixture was malformed.
"""

from __future__ import annotations

import json
import unicodedata

import pytest

from tests.fixtures.corpus import CORPUS_DIR, all_cases, case_names, load_case

REQUIRED_CASES = {
    "sinhala_only",
    "english_only",
    "mixed_language",
    "multi_column_three_ads",
    "no_ads",
    "prompt_injection",
    "duplicate_phone",
    "price_ambiguous_o_for_zero",
    "rotated_exif",
    "low_resolution",
}


def test_every_required_case_is_present() -> None:
    assert REQUIRED_CASES <= set(case_names())


@pytest.mark.parametrize("name", case_names())
def test_case_files_exist(name: str) -> None:
    case = load_case(name)

    assert case.image_path.exists(), f"{name} has no image.png"
    assert (case.directory / "expected_ocr.json").exists()
    assert (case.directory / "expected.json").exists()
    assert case.image_bytes.startswith(b"\x89PNG"), f"{name} image is not a PNG"


@pytest.mark.parametrize("name", case_names())
def test_expected_ad_count_matches_advertisement_entries(name: str) -> None:
    case = load_case(name)
    assert case.expected_ad_count == len(case.expected_advertisements)


@pytest.mark.parametrize("name", case_names())
def test_block_ids_are_sequential_and_one_based(name: str) -> None:
    """Evidence references are only meaningful if block ids are stable and dense."""
    case = load_case(name)
    ids = [block.id for block in case.blocks]
    assert ids == list(range(1, len(ids) + 1)), f"{name} block ids are not 1..n in order"


@pytest.mark.parametrize("name", case_names())
def test_blocks_have_usable_geometry_and_confidence(name: str) -> None:
    case = load_case(name)
    for block in case.blocks:
        left, top, width, height = block.box
        assert width > 0 and height > 0, f"{name} block {block.id} has an empty box"
        assert left >= 0 and top >= 0, f"{name} block {block.id} has a negative origin"
        assert 0.0 <= block.confidence <= 1.0, f"{name} block {block.id} confidence out of range"
        assert block.text.strip(), f"{name} block {block.id} has no text"


@pytest.mark.parametrize("name", case_names())
def test_expected_evidence_references_real_blocks(name: str) -> None:
    """An expectation that cites a block the OCR never produced would never be satisfiable."""
    case = load_case(name)
    for position, advertisement in enumerate(case.expected_advertisements):
        cited = advertisement.get("evidence_within_blocks")
        if cited is None:
            continue
        unknown = set(cited) - case.block_ids
        assert not unknown, (
            f"{name} advertisement {position} cites unknown blocks {sorted(unknown)}"
        )


def test_multi_column_case_carries_three_independent_advertisements() -> None:
    """AC-002 depends on this case genuinely containing three separable advertisements."""
    case = load_case("multi_column_three_ads")

    assert case.expected_ad_count == 3
    assert case.expected["disjoint_evidence"] is True

    evidence = [set(ad["evidence_within_blocks"]) for ad in case.expected_advertisements]
    for first in range(len(evidence)):
        for second in range(first + 1, len(evidence)):
            assert not evidence[first] & evidence[second], (
                "Advertisements in this case must not share evidence blocks, or a merge would look "
                "correct to the evaluation."
            )


def test_no_ads_case_expects_no_candidates() -> None:
    case = load_case("no_ads")
    assert case.expected_ad_count == 0
    assert case.expected_advertisements == ()
    assert case.text.strip(), (
        "The no-ads case must still contain readable text, or it tests OCR_EMPTY"
    )


def test_prompt_injection_case_contains_live_injection_text() -> None:
    """AC-013 is only tested if the injected instructions actually survived OCR into the blocks."""
    case = load_case("prompt_injection")
    lowered = case.text.lower()

    assert any(
        marker.lower().split(":")[0] in lowered for marker in case.expected["injection_markers"]
    ), f"No injection marker survived OCR for this case. Captured text was:\n{case.text}"
    assert case.expected_ad_count == 1, "The injected page still holds one genuine advertisement"


def test_duplicate_phone_case_shares_one_number_across_two_advertisements() -> None:
    case = load_case("duplicate_phone")
    phones = [tuple(ad["phones"]) for ad in case.expected_advertisements]

    assert case.expected_ad_count == 2
    assert phones[0] == phones[1], "This case exists to share a contact number"
    assert "POSSIBLE_DUPLICATE" in case.expected_warning_codes


def test_sinhala_case_preserves_sinhala_codepoints_through_capture() -> None:
    """AC-012, at the first hop. Sinhala must survive OCR capture and JSON round-tripping."""
    case = load_case("sinhala_only")

    sinhala = [character for character in case.text if "඀" <= character <= "෿"]
    assert sinhala, f"No Sinhala codepoints survived capture. Text was:\n{case.text}"

    assert case.text == unicodedata.normalize("NFC", case.text), (
        "Captured Sinhala must be stored NFC-normalised. NFKC must never be used here: it "
        "decomposes Sinhala conjuncts and would corrupt the text."
    )

    raw = (case.directory / "expected_ocr.json").read_text(encoding="utf-8")
    assert "\\u0d" not in raw.lower(), (
        "Sinhala was escaped to ASCII in the fixture. Writers must use ensure_ascii=False."
    )


def test_sinhala_case_is_the_low_confidence_case() -> None:
    """FR-OCR-008 needs a case that genuinely produces low OCR confidence."""
    case = load_case("sinhala_only")

    assert case.expected["ocr"]["expect_low_confidence"] is True
    assert case.mean_confidence is not None
    assert case.mean_confidence < case.expected["ocr"]["max_mean_confidence"]


def test_price_and_phone_survive_where_surrounding_sinhala_does_not() -> None:
    """The value of the Sinhala case: digits stay readable while the script around them is damaged.

    This is what stops the extraction stage from being scored as a failure when it correctly
    reports a damaged title alongside a correct price.
    """
    case = load_case("sinhala_only")
    assert "8,500,000" in case.text
    assert "0771234567" in case.text


@pytest.mark.parametrize("name", case_names())
def test_fixture_json_is_utf8_without_ascii_escapes(name: str) -> None:
    case = load_case(name)
    for filename in ("expected.json", "expected_ocr.json"):
        payload = (case.directory / filename).read_text(encoding="utf-8")
        json.loads(payload)  # must parse
        assert not payload.startswith("﻿"), f"{name}/{filename} has a BOM"


def test_corpus_has_no_stray_directories() -> None:
    for path in CORPUS_DIR.iterdir():
        if path.is_dir():
            assert (path / "image.png").exists(), f"{path.name} is missing image.png"


def test_cases_cover_the_language_matrix() -> None:
    languages = set()
    for case in all_cases():
        for advertisement in case.expected_advertisements:
            if "language" in advertisement:
                languages.add(advertisement["language"])
            languages.update(advertisement.get("language_any_of", ()))
    assert {"si", "en"} <= languages
    assert "mixed" in languages, "FR-LLM-005 requires a mixed-language case"
