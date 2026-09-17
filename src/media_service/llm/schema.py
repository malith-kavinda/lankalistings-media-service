"""Tier 1: the structural contract a response must satisfy (PRD 11.4, 11.5).

`extra="forbid"` everywhere. A field the model invented is prompt or schema drift, and drift that
passes silently is worse than a failure: the extra data goes nowhere, and nobody learns the prompt
and the schema have diverged.

`allow_inf_nan=False` is not decoration. Providers really do emit `NaN` for a confidence, and
`NaN <= 1.0` is False *and* `NaN >= 0.0` is False, so a bounds check alone would reject it with a
confusing message — while `float("nan")` would sail through any code that only checked `is not
None`. This flag *is* PRD 11.5's "confidence values are finite".

**`category` is `str`, not a `Literal`, and that is the single most important line in this file.**
As a `Literal`, a category outside the catalog becomes a Tier 1 structural failure, which triggers
a **paid repair request** -- for something FR-LLM-011 says to map to `other` with a warning.
Typing it loosely here is what lets Tier 2 handle it as the semantic matter it is.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

# Locked in the PRD rather than left to the model (FR-LLM-005). A `Literal` is right here, unlike
# for `category`: an unknown language is drift, not a taxonomy miss, and there is no "map it to
# something sensible" policy to apply.
Language = Literal["si", "en", "ta", "mixed", "unknown"]

Confidence01 = Annotated[float, Field(ge=0.0, le=1.0)]

# Long enough for a real classified advertisement, short enough that a runaway generation is caught
# structurally rather than filling a column that would reject it later.
MAX_TITLE_CHARS = 300
MAX_DESCRIPTION_CHARS = 4000
MAX_LOCATION_CHARS = 200
MAX_PHONE_CHARS = 40
MAX_WARNING_CHARS = 500


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=False)


class ExtractedPrice(StrictModel):
    # The source wording, kept verbatim so a reviewer can check the number against what was printed.
    raw: str | None = Field(default=None, max_length=200)
    # Integer minor-unit-free amount. Non-negative because a negative price is not a reading of
    # anything on a page -- it is a parse that went wrong.
    amount: int | None = Field(default=None, ge=0)
    # Sri Lankan marketplace scope (PRD 11.5). A second currency is a schema version bump, which is
    # the point: it should be a visible decision.
    currency: Literal["LKR"] | None = None


class ExtractedContacts(StrictModel):
    # Strings, never integers: a leading zero is part of a Sri Lankan number, and any numeric type
    # would eat it.
    phones: list[Annotated[str, Field(max_length=MAX_PHONE_CHARS)]] = Field(default_factory=list)


class ExtractedConfidence(StrictModel):
    overall: Confidence01
    title: Confidence01 | None = None
    description: Confidence01 | None = None
    category: Confidence01 | None = None
    price: Confidence01 | None = None
    location: Confidence01 | None = None
    contacts: Confidence01 | None = None


class ExtractionWarning(StrictModel):
    code: str = Field(max_length=64)
    field: str | None = Field(default=None, max_length=64)
    message: str = Field(max_length=MAX_WARNING_CHARS)


class ExtractedAdvertisement(StrictModel):
    # The evidence trail. Tier 2 checks these against the stored OCR record, which is the concrete
    # detector for an advertisement the model invented.
    source_block_ids: list[int] = Field(default_factory=list)
    language: Language = "unknown"
    title: str | None = Field(default=None, max_length=MAX_TITLE_CHARS)
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_CHARS)
    # Deliberately `str` -- see the module docstring.
    category: str = Field(default="other", max_length=64)
    price: ExtractedPrice | None = None
    location: str | None = Field(default=None, max_length=MAX_LOCATION_CHARS)
    contacts: ExtractedContacts | None = None
    confidence: ExtractedConfidence
    warnings: list[ExtractionWarning] = Field(default_factory=list)


class AdExtractionEnvelope(StrictModel):
    schema_version: str = Field(max_length=16)
    advertisements: list[ExtractedAdvertisement] = Field(default_factory=list)


def json_schema() -> dict[str, Any]:
    """The schema as the providers are told about it.

    Generated from the model rather than hand-written, so the two cannot disagree -- a hand-written
    copy is exactly how a schema drifts from the validator that enforces it.
    """
    return AdExtractionEnvelope.model_json_schema()
