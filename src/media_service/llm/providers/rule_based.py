"""A deterministic extractor that looks like a provider.

This is the prototype's regex heuristic, moved here from `AdvertisementService` and given the
provider interface so the pipeline can run end to end with no network, no credentials, and no cost.
It is honest about what it is: **at most one advertisement per page**, which is exactly the
limitation a real model removes.

It is available for automated tests and explicit local demos only. PRD 11.6 forbids falling back to
heuristic advertisement creation in production, and the registry enforces that at startup rather
than leaving it to convention -- see `llm.registry.guard_offline_provider`.

The one intentional difference from the prototype is the unmatched-category fallback. The prototype
files anything it cannot classify as `Home`, a real category, which silently mis-files every
unrecognised advertisement (PRD 11.7). Here the caller names the fallback, and the pipeline passes
`Other`. The prototype endpoint still passes `Home`, because its behaviour is pinned until Phase 4
removes it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from media_service.llm.rendering import CLOSE_DELIMITER, OPEN_DELIMITER
from media_service.llm.schema import SCHEMA_VERSION
from media_service.llm.types import LlmRequest, LlmResponse, LlmUsage

PROVIDER_NAME: Final = "rule_based"
MODEL_NAME: Final = "rule_based/v1"

PRICE_PATTERN: Final = re.compile(r"(?:rs\.?|රු\.?)\s*[\d,]+", re.IGNORECASE)
LOCATION_PATTERN: Final = re.compile(
    r"\b(?:Colombo|Kandy|Galle|Matara|Jaffna|Kurunegala|Negombo|Gampaha)\s*\d{0,2}\b",
    re.IGNORECASE,
)
LOCATION_KEYWORDS: Final = (
    "Colombo",
    "Kandy",
    "Galle",
    "Matara",
    "Jaffna",
    "Kurunegala",
    "Negombo",
    "Gampaha",
)
CATEGORY_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "Vehicles": ("vehicle", "car", "van", "bike", "prius", "toyota", "honda", "වාහන"),
    "Property": ("house", "land", "rent", "apartment", "property", "නිවස", "ඉඩම"),
    "Electronics": ("phone", "iphone", "laptop", "tv", "electronics", "දුරකථන"),
    "Jobs": ("job", "vacancy", "executive", "driver", "රැකියා", "ඇබෑර්තු"),
}

# A block header rendered into the prompt, e.g. "[4] (x=1,y=2,w=3,h=4) conf=0.94".
_BLOCK_HEADER: Final = re.compile(r"^\[(\d+)\](?:\s*\([^)]*\))?(?:\s*conf=[\d.]+)?\s*$")

PHONE_PATTERN: Final = re.compile(r"(?:\+?94|0)\d{8,9}")


def extract_fields(text: str, *, unmatched_category: str = "Other") -> dict[str, str]:
    """The prototype's heuristic, with the fallback category made a parameter."""
    lines = [line.strip(" -:") for line in text.splitlines() if line.strip(" -:")]
    joined = " ".join(lines)

    price_match = PRICE_PATTERN.search(joined)
    price = price_match.group(0).replace("Rs ", "Rs. ") if price_match else "Price pending"
    title = next(
        (line for line in lines if not PRICE_PATTERN.search(line) and len(line) > 3),
        "Newspaper advertisement",
    )
    location_match = LOCATION_PATTERN.search(joined)
    location = (
        location_match.group(0).strip()
        if location_match
        else next(
            (name for name in LOCATION_KEYWORDS if name.lower() in joined.lower()),
            "Location pending",
        )
    )

    return {
        "title": title[:120],
        "price": price,
        "category": detect_category(joined, unmatched=unmatched_category),
        "location": location,
        "description": joined or "OCR text pending reviewer verification.",
    }


def detect_category(text: str, *, unmatched: str = "Other") -> str:
    normalized = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return category
    return unmatched


def ocr_section(user_prompt: str) -> tuple[str, tuple[int, ...]]:
    """Recover the plain OCR text and the block ids from a rendered prompt.

    Reading the prompt back rather than taking the text through a side channel keeps `LlmRequest`
    meaning exactly one thing -- what a model sees. An offline provider that got privileged access
    to the original blocks would be testing a path no real provider takes.
    """
    start = user_prompt.find(OPEN_DELIMITER)
    end = user_prompt.find(CLOSE_DELIMITER)
    if start == -1 or end == -1 or end < start:
        return user_prompt, ()

    body = user_prompt[start + len(OPEN_DELIMITER) : end]
    lines: list[str] = []
    block_ids: list[int] = []
    for line in body.splitlines():
        header = _BLOCK_HEADER.match(line.strip())
        if header:
            block_ids.append(int(header.group(1)))
            continue
        if line.strip():
            lines.append(line)
    return "\n".join(lines), tuple(block_ids)


class RuleBasedLlmProvider:
    """Produces at most one candidate, citing every block it was shown."""

    def __init__(self, *, unmatched_category: str = "Other") -> None:
        self._unmatched_category = unmatched_category

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return MODEL_NAME

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str | None:
        return None

    def complete(self, request: LlmRequest) -> LlmResponse:
        payload = self.extract(request.user)
        return LlmResponse(
            raw_text=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            provider=PROVIDER_NAME,
            model=MODEL_NAME,
            usage=LlmUsage(cost_micros=0),
        )

    def extract(self, user_prompt: str) -> dict[str, Any]:
        text, block_ids = ocr_section(user_prompt)
        if not text.strip():
            return {"schema_version": SCHEMA_VERSION, "advertisements": []}

        fields = extract_fields(text, unmatched_category=self._unmatched_category)
        phones = list(dict.fromkeys(PHONE_PATTERN.findall(text)))
        amount = _amount_of(fields["price"])

        return {
            "schema_version": SCHEMA_VERSION,
            "advertisements": [
                {
                    "source_block_ids": list(block_ids),
                    "language": _dominant_language(text),
                    "title": fields["title"],
                    "description": fields["description"],
                    "category": fields["category"],
                    "price": {
                        "raw": fields["price"],
                        "amount": amount,
                        "currency": "LKR" if amount is not None else None,
                    },
                    "location": fields["location"],
                    "contacts": {"phones": phones},
                    # Fixed and modest on purpose. A heuristic cannot know how sure it is, and a
                    # high number here would read to a reviewer as a model's judgement.
                    "confidence": {"overall": 0.5},
                    "warnings": [
                        {
                            "code": "HEURISTIC_EXTRACTION",
                            "field": None,
                            "message": "Produced by the rule-based extractor, not a model.",
                        }
                    ],
                }
            ],
        }


def _amount_of(price: str) -> int | None:
    digits = "".join(character for character in price if character.isdigit())
    return int(digits) if digits else None


def _dominant_language(text: str) -> str:
    sinhala = sum(1 for character in text if "඀" <= character <= "෿")
    latin = sum(1 for character in text if character.isascii() and character.isalpha())
    if sinhala and latin:
        return "mixed" if min(sinhala, latin) / max(sinhala, latin) > 0.25 else (
            "si" if sinhala > latin else "en"
        )
    if sinhala:
        return "si"
    if latin:
        return "en"
    return "unknown"
