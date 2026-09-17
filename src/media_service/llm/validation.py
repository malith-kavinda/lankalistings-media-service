"""Tier 2: semantic validation (PRD 11.5).

The tier split is the load-bearing decision of this whole phase, and it is about **failure
semantics, not severity**. Tier 1 asks "is this the shape we asked for?" — a failure there means the
model misunderstood the schema, which another attempt can fix, so it is repairable. Tier 2 asks "is
this a plausible reading of the page?" — a failure there is the model being wrong about the world,
which asking again does not fix. So Tier 2 **never retries and never repairs**: it warns, strips, or
drops a single candidate, and lets a person decide.

Getting that backwards is expensive in a specific way. If an unknown category were a structural
failure, every advertisement in an unusual trade would burn the item's one paid repair request on
something FR-LLM-011 says to accept with a warning.

The most important check here is `source_block_ids ⊆ the recorded OCR blocks`. It is the concrete
detector for an advertisement the model invented: a fabrication has nowhere real to point. A
candidate whose evidence is entirely unknown is discarded rather than warned about, because there is
nothing for a reviewer to check it against.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from media_service.domain.categories import DEFAULT_CATALOG, CategoryCatalog
from media_service.llm.rendering import sanitize
from media_service.llm.schema import AdExtractionEnvelope, ExtractedAdvertisement

CANDIDATE_LIMIT_EXCEEDED: Final = "CANDIDATE_LIMIT_EXCEEDED"
EVIDENCE_BLOCK_UNKNOWN: Final = "EVIDENCE_BLOCK_UNKNOWN"
EVIDENCE_MISSING: Final = "EVIDENCE_MISSING"
CATEGORY_UNMAPPED: Final = "CATEGORY_UNMAPPED"
PHONE_UNPARSEABLE: Final = "PHONE_UNPARSEABLE"
SCHEMA_VERSION_UNSUPPORTED: Final = "SCHEMA_VERSION_UNSUPPORTED"

# Sri Lankan numbers: 9 or 10 national digits, or an international form. Deliberately permissive --
# an unparseable number is kept with a warning, never dropped, because a reviewer can read a number
# this regex cannot.
_DIGITS: Final = re.compile(r"\d+")
_NATIONAL_LENGTHS: Final = frozenset({9, 10})
_COUNTRY_CODE: Final = "94"


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    """One advertisement that survived Tier 2, with what had to be changed recorded."""

    index: int
    advertisement: ExtractedAdvertisement
    source_block_ids: tuple[int, ...]
    warning_codes: tuple[str, ...]
    category: str
    phones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    candidates: tuple[ValidatedCandidate, ...] = ()
    # Page-level problems: a candidate cap that bit, an unsupported schema version.
    warning_codes: tuple[str, ...] = field(default_factory=tuple)
    dropped: int = 0


def validate_extraction(
    envelope: AdExtractionEnvelope,
    *,
    known_block_ids: frozenset[int],
    max_candidates: int,
    catalog: CategoryCatalog = DEFAULT_CATALOG,
    supported_schema_versions: frozenset[str] | None = None,
) -> ValidationOutcome:
    page_warnings: list[str] = []

    if (
        supported_schema_versions is not None
        and envelope.schema_version not in supported_schema_versions
    ):
        # A warning, not a rejection: the response already parsed against our model, so the payload
        # is usable. What the mismatch tells us is that the prompt and the schema have drifted.
        page_warnings.append(SCHEMA_VERSION_UNSUPPORTED)

    advertisements = list(envelope.advertisements)
    if len(advertisements) > max_candidates:
        page_warnings.append(CANDIDATE_LIMIT_EXCEEDED)
        advertisements = advertisements[:max_candidates]

    candidates: list[ValidatedCandidate] = []
    dropped = 0
    for advertisement in advertisements:
        candidate = _validate_one(
            advertisement,
            index=len(candidates),
            known_block_ids=known_block_ids,
            catalog=catalog,
        )
        if candidate is None:
            dropped += 1
            continue
        candidates.append(candidate)

    return ValidationOutcome(
        candidates=tuple(candidates),
        warning_codes=tuple(page_warnings),
        dropped=dropped,
    )


def _validate_one(
    advertisement: ExtractedAdvertisement,
    *,
    index: int,
    known_block_ids: frozenset[int],
    catalog: CategoryCatalog,
) -> ValidatedCandidate | None:
    warnings: list[str] = [warning.code for warning in advertisement.warnings]

    cited = tuple(dict.fromkeys(advertisement.source_block_ids))
    kept = tuple(block_id for block_id in cited if block_id in known_block_ids)
    if len(kept) != len(cited):
        warnings.append(EVIDENCE_BLOCK_UNKNOWN)
    if not kept:
        # Nothing on the page supports this. Dropping it is the point of the check -- a reviewer
        # shown an advertisement with no evidence has no way to tell it apart from a real one.
        return None

    category, unmapped = catalog.resolve(advertisement.category)
    if unmapped:
        warnings.append(CATEGORY_UNMAPPED)

    phones, phone_warning = _normalize_phones(advertisement)
    if phone_warning:
        warnings.append(phone_warning)

    return ValidatedCandidate(
        index=index,
        advertisement=_sanitized(advertisement),
        source_block_ids=kept,
        warning_codes=tuple(dict.fromkeys(warnings)),
        category=category,
        phones=phones,
    )


def _sanitized(advertisement: ExtractedAdvertisement) -> ExtractedAdvertisement:
    """NFC, no control characters, no bidi overrides -- on every string a person will read."""
    price = advertisement.price
    return advertisement.model_copy(
        update={
            "title": _clean(advertisement.title),
            "description": _clean(advertisement.description),
            "location": _clean(advertisement.location),
            "price": (
                price.model_copy(update={"raw": _clean(price.raw)}) if price is not None else None
            ),
        }
    )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = sanitize(value).strip()
    return cleaned or None


def _normalize_phones(
    advertisement: ExtractedAdvertisement,
) -> tuple[tuple[str, ...], str | None]:
    if advertisement.contacts is None:
        return (), None

    normalized: list[str] = []
    unparseable = False
    for raw in advertisement.contacts.phones:
        candidate = normalize_phone(raw)
        if candidate is None:
            unparseable = True
            # Kept as written. A number this cannot parse is still the number on the page, and a
            # reviewer can read it.
            cleaned = _clean(raw)
            if cleaned:
                normalized.append(cleaned)
            continue
        normalized.append(candidate)

    return tuple(dict.fromkeys(normalized)), PHONE_UNPARSEABLE if unparseable else None


def normalize_phone(raw: str) -> str | None:
    """Return a national-format Sri Lankan number, or None when it cannot be read as one.

    Stays a string throughout: a leading zero is part of the number, and any numeric type eats it.
    """
    digits = "".join(_DIGITS.findall(unicodedata.normalize("NFC", raw)))
    if not digits:
        return None

    if digits.startswith(_COUNTRY_CODE) and len(digits) == 11:
        digits = "0" + digits[2:]
    elif len(digits) == 9 and not digits.startswith("0"):
        digits = "0" + digits

    if len(digits) not in _NATIONAL_LENGTHS or not digits.startswith("0"):
        return None
    return digits
