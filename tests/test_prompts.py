"""Versioned prompt templates (PRD 11.1-11.3, FR-LLM-004)."""

from __future__ import annotations

import pytest

from media_service.domain.categories import DEFAULT_CATALOG
from media_service.llm.prompts.registry import PromptError, load_prompt

# Editing a prompt without bumping its version silently invalidates the regression corpus and every
# accuracy measurement taken against it, with nothing in provenance showing that anything changed.
# When this test fails, either bump to a new version directory or update the value here
# deliberately.
FROZEN_PROMPT_CHECKSUMS = {
    "ad_extraction/v1": "50b3dd178e21",
}


def flat(text: str) -> str:
    """Collapse whitespace so assertions survive the templates being re-wrapped."""
    return " ".join(text.lower().split())


def test_v1_loads() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    assert prompt.version == "ad-extraction/v1"
    assert prompt.schema_version == "1.0"
    assert prompt.max_ocr_chars > 0


def test_prompt_checksum_is_frozen() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    expected = FROZEN_PROMPT_CHECKSUMS["ad_extraction/v1"]
    assert prompt.checksum == expected, (
        f"The v1 prompt changed (checksum {prompt.checksum}, expected {expected}). "
        "Bump the prompt version, or update FROZEN_PROMPT_CHECKSUMS on purpose."
    )


def test_system_prompt_states_the_trust_boundary() -> None:
    """PRD 11.1: OCR text is data, never instructions."""
    system = flat(load_prompt("ad_extraction", "v1").system)
    assert "untrusted" in system
    assert "never an instruction" in system or "not as something to obey" in system


def test_system_prompt_forbids_merging_and_splitting() -> None:
    """PRD 11.2 rule: independent advertisements must stay separate."""
    system = flat(load_prompt("ad_extraction", "v1").system)
    assert "never merge" in system
    assert "never split" in system


def test_system_prompt_forbids_invented_values() -> None:
    system = flat(load_prompt("ad_extraction", "v1").system)
    assert "never invent" in system
    assert "null" in system


def test_system_prompt_requires_json_only() -> None:
    system = flat(load_prompt("ad_extraction", "v1").system)
    assert "only json" in system
    assert "markdown" in system


def test_user_template_declares_exactly_the_expected_placeholders() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    assert set(prompt.required_placeholders) == {
        "ingestion_item_id",
        "category_catalog",
        "schema_version",
        "ocr_blocks",
    }


def test_user_template_renders_with_the_real_catalog() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    rendered = prompt.render_user(
        {
            "ingestion_item_id": "itm_123",
            "category_catalog": DEFAULT_CATALOG.as_prompt_list(),
            "schema_version": "1.0",
            "ocr_blocks": "[1] (x=0,y=0,w=10,h=10) conf=0.90\nToyota Prius 2016",
        }
    )
    assert "{{" not in rendered
    assert "itm_123" in rendered
    assert "vehicles" in rendered
    assert "home_garden" in rendered
    assert "<ocr_blocks>" in rendered and "</ocr_blocks>" in rendered


def test_rendering_rejects_a_missing_value() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    with pytest.raises(PromptError, match="missing values"):
        prompt.render_user({"ingestion_item_id": "itm_123"})


def test_rendering_rejects_an_unused_value() -> None:
    """A renamed placeholder would otherwise ship a prompt with an empty section."""
    prompt = load_prompt("ad_extraction", "v1")
    with pytest.raises(PromptError, match="does not use"):
        prompt.render_user(
            {
                "ingestion_item_id": "itm_123",
                "category_catalog": "vehicles",
                "schema_version": "1.0",
                "ocr_blocks": "[1] text",
                "typo_field": "x",
            }
        )


def test_repair_template_renders() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    errors = "- advertisements.0.confidence: required"
    rendered = prompt.render_repair({"validation_errors": errors})
    assert "{{" not in rendered
    assert "confidence" in rendered


def test_repair_template_does_not_ask_for_field_values() -> None:
    """PRD 11.6/15.4: a repair carries error paths, never OCR-derived values."""
    repair = flat(load_prompt("ad_extraction", "v1").repair)
    assert "do not change any extracted fact" in repair


def test_unknown_prompt_version_fails_loudly() -> None:
    with pytest.raises(PromptError):
        load_prompt("ad_extraction", "v99")


def test_prompt_schema_version_matches_manifest_declaration() -> None:
    prompt = load_prompt("ad_extraction", "v1")
    assert prompt.schema_version == "1.0"
