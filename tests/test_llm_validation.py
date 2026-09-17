"""Tier 2: semantic validation, which warns and drops but never retries."""

from __future__ import annotations

import pytest

from media_service.llm.schema import AdExtractionEnvelope
from media_service.llm.validation import (
    CANDIDATE_LIMIT_EXCEEDED,
    CATEGORY_UNMAPPED,
    EVIDENCE_BLOCK_UNKNOWN,
    PHONE_UNPARSEABLE,
    SCHEMA_VERSION_UNSUPPORTED,
    normalize_phone,
    validate_extraction,
)

KNOWN = frozenset({1, 2, 3})


def envelope(*advertisements, schema_version: str = "1.0") -> AdExtractionEnvelope:
    base = {
        "source_block_ids": [1],
        "language": "en",
        "title": "Toyota Prius 2016",
        "category": "vehicles",
        "confidence": {"overall": 0.9},
    }
    return AdExtractionEnvelope.model_validate(
        {
            "schema_version": schema_version,
            "advertisements": [{**base, **entry} for entry in (advertisements or ({},))],
        }
    )


def run(envelope_, *, max_candidates: int = 20, known=KNOWN):  # type: ignore[no-untyped-def]
    return validate_extraction(
        envelope_,
        known_block_ids=known,
        max_candidates=max_candidates,
        supported_schema_versions=frozenset({"1.0"}),
    )


# -- evidence ----------------------------------------------------------------------------------


def test_a_candidate_citing_real_blocks_survives() -> None:
    outcome = run(envelope({"source_block_ids": [1, 2]}))

    assert len(outcome.candidates) == 1
    assert outcome.candidates[0].source_block_ids == (1, 2)


def test_an_unknown_block_is_stripped_with_a_warning() -> None:
    outcome = run(envelope({"source_block_ids": [1, 99]}))

    assert outcome.candidates[0].source_block_ids == (1,)
    assert EVIDENCE_BLOCK_UNKNOWN in outcome.candidates[0].warning_codes


def test_a_candidate_with_no_real_evidence_is_discarded() -> None:
    """The concrete detector for an invented advertisement: a fabrication has nowhere to point.

    Dropped rather than warned about, because a reviewer shown an advertisement with no evidence
    has no way to tell it from a real one.
    """
    outcome = run(envelope({"source_block_ids": [99, 100]}))

    assert outcome.candidates == ()
    assert outcome.dropped == 1


def test_duplicate_citations_are_collapsed() -> None:
    outcome = run(envelope({"source_block_ids": [1, 1, 2]}))

    assert outcome.candidates[0].source_block_ids == (1, 2)


# -- category ----------------------------------------------------------------------------------


def test_an_unknown_category_maps_to_other_with_a_warning() -> None:
    """FR-LLM-011, and the reason `category` is not a Literal in the schema."""
    outcome = run(envelope({"category": "antique_maps"}))

    assert outcome.candidates[0].category == "other"
    assert CATEGORY_UNMAPPED in outcome.candidates[0].warning_codes


def test_a_catalog_category_is_kept_and_normalised() -> None:
    outcome = run(envelope({"category": "Vehicles"}))

    assert outcome.candidates[0].category == "vehicles"
    assert CATEGORY_UNMAPPED not in outcome.candidates[0].warning_codes


# -- candidate cap -----------------------------------------------------------------------------


def test_excess_candidates_are_dropped_with_a_page_warning() -> None:
    outcome = run(envelope({}, {}, {}), max_candidates=2)

    assert len(outcome.candidates) == 2
    assert CANDIDATE_LIMIT_EXCEEDED in outcome.warning_codes


def test_candidate_indexes_are_sequential_after_drops() -> None:
    """A dropped candidate must not leave a gap a reviewer would read as a missing advertisement."""
    outcome = run(envelope({"source_block_ids": [99]}, {"source_block_ids": [1]}))

    assert [candidate.index for candidate in outcome.candidates] == [0]


# -- phones ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0771234567", "0771234567"),
        ("077 123 4567", "0771234567"),
        ("+94771234567", "0771234567"),
        ("94771234567", "0771234567"),
        ("771234567", "0771234567"),
        ("011 2345678", "0112345678"),
    ],
)
def test_phone_numbers_are_normalised(raw: str, expected: str) -> None:
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize("raw", ["", "not a number", "12", "0" * 20])
def test_an_unreadable_number_is_not_invented(raw: str) -> None:
    assert normalize_phone(raw) is None


def test_an_unparseable_number_is_kept_with_a_warning() -> None:
    """A number this cannot parse is still the number on the page, and a reviewer can read it."""
    outcome = run(envelope({"contacts": {"phones": ["call me maybe"]}}))

    candidate = outcome.candidates[0]
    assert PHONE_UNPARSEABLE in candidate.warning_codes
    assert candidate.phones == ("call me maybe",)


def test_duplicate_numbers_collapse() -> None:
    outcome = run(envelope({"contacts": {"phones": ["0771234567", "+94771234567"]}}))

    assert outcome.candidates[0].phones == ("0771234567",)


# -- text --------------------------------------------------------------------------------------


def test_control_and_bidi_characters_are_stripped_from_text() -> None:
    """A right-to-left override makes rendered text read as something other than its bytes."""
    outcome = run(envelope({"title": "Toyota‮Prius"}))

    assert outcome.candidates[0].advertisement.title == "ToyotaPrius"


def test_sinhala_survives_validation_unchanged() -> None:
    """NFC, never NFKC: compatibility normalisation decomposes conjuncts (AC-012)."""
    import unicodedata

    title = "ටොයොටා ප්‍රියස් 2016"
    outcome = run(envelope({"title": title}))

    survived = outcome.candidates[0].advertisement.title
    assert survived == unicodedata.normalize("NFC", title)
    assert "ර" in survived


def test_an_empty_string_becomes_none_rather_than_a_blank_field() -> None:
    outcome = run(envelope({"location": "   "}))

    assert outcome.candidates[0].advertisement.location is None


# -- schema version ----------------------------------------------------------------------------


def test_an_unsupported_schema_version_warns_rather_than_rejecting() -> None:
    """The payload already validated, so it is usable. What the mismatch shows is drift."""
    outcome = run(envelope(schema_version="9.9"))

    assert SCHEMA_VERSION_UNSUPPORTED in outcome.warning_codes
    assert len(outcome.candidates) == 1


def test_model_supplied_warnings_reach_the_candidate() -> None:
    outcome = run(
        envelope(
            {
                "warnings": [
                    {"code": "OCR_AMBIGUOUS_CHARACTER", "field": "price", "message": "O for 0"}
                ]
            }
        )
    )

    assert "OCR_AMBIGUOUS_CHARACTER" in outcome.candidates[0].warning_codes
