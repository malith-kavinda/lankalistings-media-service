"""Category catalog (PRD 11.7, FR-LLM-011)."""

from __future__ import annotations

import pytest

from media_service.domain.categories import (
    CATEGORY_CATALOG_VERSION,
    DEFAULT_CATALOG,
    OTHER,
    CategoryCatalog,
)

LOCKED_SLUGS = (
    "vehicles",
    "property",
    "land",
    "jobs",
    "electronics",
    "services",
    "home_garden",
    "fashion",
    "other",
)


def test_catalog_is_the_nine_locked_categories() -> None:
    assert DEFAULT_CATALOG.slugs == LOCKED_SLUGS


def test_catalog_contains_other() -> None:
    """FR-LLM-011 makes `other` the destination for every unmapped category."""
    assert OTHER in DEFAULT_CATALOG.slugs


def test_catalog_is_versioned() -> None:
    """Provenance records the catalog version, so a taxonomy change must be visible."""
    assert DEFAULT_CATALOG.version == CATEGORY_CATALOG_VERSION
    assert CATEGORY_CATALOG_VERSION.startswith("categories/")


@pytest.mark.parametrize("slug", LOCKED_SLUGS)
def test_every_slug_resolves_to_itself(slug: str) -> None:
    resolved, unmapped = DEFAULT_CATALOG.resolve(slug)
    assert resolved == slug
    assert unmapped is False


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Vehicles", "vehicles"),
        ("  vehicles  ", "vehicles"),
        ("VEHICLES", "vehicles"),
        ("Home & Garden", "home_garden"),
        ("home garden", "home_garden"),
        ("home-garden", "home_garden"),
        ("Home", "home_garden"),
        ("Land", "land"),
    ],
)
def test_resolve_is_tolerant_of_case_spacing_and_separators(supplied: str, expected: str) -> None:
    resolved, unmapped = DEFAULT_CATALOG.resolve(supplied)
    assert resolved == expected
    assert unmapped is False


def test_unknown_category_maps_to_other_and_reports_it() -> None:
    """FR-LLM-011: map and warn. Never raise, or a paid repair is spent on a semantic problem."""
    resolved, unmapped = DEFAULT_CATALOG.resolve("Livestock")
    assert resolved == OTHER
    assert unmapped is True


@pytest.mark.parametrize("supplied", [None, "", "   "])
def test_missing_category_maps_to_other(supplied: str | None) -> None:
    resolved, unmapped = DEFAULT_CATALOG.resolve(supplied)
    assert resolved == OTHER
    assert unmapped is True


def test_resolve_never_raises_on_hostile_input() -> None:
    """Category text arrives from a model reading untrusted OCR."""
    for hostile in ("../../etc/passwd", "'; DROP TABLE ads; --", "\x00\x01", "ට" * 500):
        resolved, _ = DEFAULT_CATALOG.resolve(hostile)
        assert resolved in LOCKED_SLUGS


def test_prompt_list_contains_every_slug() -> None:
    """The prompt is the only thing constraining model output to the taxonomy."""
    rendered = DEFAULT_CATALOG.as_prompt_list()
    for slug in LOCKED_SLUGS:
        assert slug in rendered


def test_legacy_prototype_categories_all_resolve() -> None:
    """The prototype shipped these six; stored advertisements still carry them."""
    for legacy in ("Vehicles", "Property", "Electronics", "Jobs", "Home", "Land"):
        _, unmapped = DEFAULT_CATALOG.resolve(legacy)
        assert unmapped is False, f"Legacy category {legacy!r} would be silently mapped to other"


def test_labels_are_available_for_display() -> None:
    assert DEFAULT_CATALOG.label_for("home_garden") == "Home & Garden"
    assert DEFAULT_CATALOG.label_for("vehicles") == "Vehicles"
    assert DEFAULT_CATALOG.label_for("nonexistent") == "Other"


def test_catalog_is_immutable() -> None:
    catalog = CategoryCatalog()
    with pytest.raises(AttributeError):
        catalog.version = "categories/v2"  # type: ignore[misc]
