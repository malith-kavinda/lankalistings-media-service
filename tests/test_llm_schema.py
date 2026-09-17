"""Tier 1: the structural contract, and the two type choices that carry policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from media_service.llm.schema import AdExtractionEnvelope, json_schema


def envelope(**advertisement) -> dict:  # type: ignore[no-untyped-def]
    base = {
        "source_block_ids": [1],
        "language": "en",
        "title": "Toyota Prius 2016",
        "category": "vehicles",
        "confidence": {"overall": 0.9},
    }
    return {"schema_version": "1.0", "advertisements": [{**base, **advertisement}]}


def test_a_well_formed_response_validates() -> None:
    parsed = AdExtractionEnvelope.model_validate(envelope())

    assert len(parsed.advertisements) == 1
    assert parsed.advertisements[0].category == "vehicles"


def test_an_empty_response_is_valid() -> None:
    """AC-003: a page with no advertisement is an outcome, not a malformed answer."""
    parsed = AdExtractionEnvelope.model_validate(
        {"schema_version": "1.0", "advertisements": []}
    )

    assert parsed.advertisements == []


def test_an_invented_field_is_rejected() -> None:
    """Drift that passes silently is worse than a failure: the data goes nowhere, unnoticed."""
    with pytest.raises(ValidationError) as caught:
        AdExtractionEnvelope.model_validate(envelope(invented_field="surprise"))

    assert "invented_field" in str(caught.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_confidence_is_rejected(value: float) -> None:
    """PRD 11.5's "confidence values are finite", enforced by the model rather than by a check.

    A bounds test alone would not catch NaN: both `NaN <= 1.0` and `NaN >= 0.0` are False, so it
    would be rejected with a message about the wrong thing -- and any code that merely checked for
    None would let it straight through.
    """
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(envelope(confidence={"overall": value}))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_outside_zero_to_one_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(envelope(confidence={"overall": value}))


def test_a_negative_price_is_rejected() -> None:
    """Not a reading of anything on a page -- it is a parse that went wrong."""
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(envelope(price={"amount": -5}))


def test_a_currency_outside_scope_is_rejected() -> None:
    """A second currency should be a visible schema decision, not one a model can make."""
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(envelope(price={"amount": 10, "currency": "USD"}))


def test_an_unknown_language_is_rejected() -> None:
    """Unlike a category, an unknown language is drift: there is no policy to map it onto."""
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(envelope(language="kl"))


def test_an_unknown_category_is_accepted_by_tier_one() -> None:
    """The single most important type choice in the schema.

    As a `Literal`, this would be a structural failure, which triggers a paid repair request -- for
    something FR-LLM-011 says to map to `other` with a warning. Tier 2 handles it instead.
    """
    parsed = AdExtractionEnvelope.model_validate(envelope(category="antique_maps"))

    assert parsed.advertisements[0].category == "antique_maps"


def test_phone_numbers_stay_strings() -> None:
    """A leading zero is part of a Sri Lankan number, and a numeric type eats it."""
    parsed = AdExtractionEnvelope.model_validate(
        envelope(contacts={"phones": ["0771234567"]})
    )

    assert parsed.advertisements[0].contacts.phones == ["0771234567"]


def test_a_missing_confidence_is_rejected() -> None:
    """Every candidate has to carry one: a reviewer sorts by it."""
    with pytest.raises(ValidationError):
        AdExtractionEnvelope.model_validate(
            {"schema_version": "1.0", "advertisements": [{"source_block_ids": [1]}]}
        )


def test_the_published_schema_is_generated_from_the_model() -> None:
    """Hand-written copies are how a schema drifts from the validator that enforces it."""
    schema = json_schema()

    assert schema["properties"]["advertisements"]["type"] == "array"
    assert schema["additionalProperties"] is False
