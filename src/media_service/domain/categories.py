"""The advertisement category catalog.

This is the single definition of the category taxonomy for the whole service. It is injected into
the extraction prompt and used to validate model output, so duplicating it elsewhere -- in a
route, in a client, or in a prompt file -- will produce drift that is invisible until extraction
quality drops.

See PRD 11.7. The nine top-level categories are locked by the platform data model.
"""

from dataclasses import dataclass
from typing import ClassVar, Final

CATEGORY_CATALOG_VERSION: Final = "categories/v1"

# The category a value is mapped to when nothing else applies. Its presence is required:
# FR-LLM-011 makes it the destination for every unrecognised category, so a catalog without it
# cannot satisfy the policy.
OTHER: Final = "other"


@dataclass(frozen=True, slots=True)
class Category:
    slug: str
    label: str


CATEGORIES: Final[tuple[Category, ...]] = (
    Category("vehicles", "Vehicles"),
    Category("property", "Property"),
    Category("land", "Land"),
    Category("jobs", "Jobs"),
    Category("electronics", "Electronics"),
    Category("services", "Services"),
    Category("home_garden", "Home & Garden"),
    Category("fashion", "Fashion"),
    Category(OTHER, "Other"),
)

# Values the prototype used that are not catalog slugs. Mapped rather than rejected so that existing
# advertisements and the current management portal keep resolving during migration.
_LEGACY_ALIASES: Final[dict[str, str]] = {
    "home": "home_garden",
    "home and garden": "home_garden",
    "home & garden": "home_garden",
    "house": "property",
    "houses": "property",
    "vehicle": "vehicles",
    "car": "vehicles",
    "cars": "vehicles",
    "job": "jobs",
    "vacancy": "jobs",
    "vacancies": "jobs",
    "service": "services",
    "electronic": "electronics",
    "lands": "land",
}


def _normalize(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())


@dataclass(frozen=True, slots=True)
class CategoryCatalog:
    version: str = CATEGORY_CATALOG_VERSION
    categories: tuple[Category, ...] = CATEGORIES

    OTHER: ClassVar[str] = OTHER

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(category.slug for category in self.categories)

    def contains(self, value: str) -> bool:
        """Whether `value` names a catalog category, ignoring case, spacing, and separators."""
        return self._lookup(value) is not None

    def resolve(self, value: str | None) -> tuple[str, bool]:
        """Map a model-supplied category onto the catalog.

        Returns the canonical slug and whether the value had to be mapped to `other`. A caller that
        receives `True` must attach a CATEGORY_UNMAPPED warning (FR-LLM-011).

        Mapping never raises. An unknown category is a semantic problem, not a structural one:
        raising here would turn it into a schema failure and spend the candidate's single repair
        request on
        something the PRD says to accept with a warning (PRD 11.5).
        """
        if value is None:
            return OTHER, True
        resolved = self._lookup(value)
        if resolved is None:
            return OTHER, True
        return resolved, False

    def label_for(self, slug: str) -> str:
        for category in self.categories:
            if category.slug == slug:
                return category.label
        return "Other"

    def as_prompt_list(self) -> str:
        """The catalog rendered for prompt injection (PRD 11.3 `category_catalog`)."""
        return ", ".join(self.slugs)

    def _lookup(self, value: str) -> str | None:
        normalized = _normalize(value)
        if not normalized:
            return None
        collapsed = normalized.replace(" ", "_")
        for category in self.categories:
            if collapsed == category.slug or normalized == _normalize(category.label):
                return category.slug
        return _LEGACY_ALIASES.get(normalized)


DEFAULT_CATALOG: Final = CategoryCatalog()
