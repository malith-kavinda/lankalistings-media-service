"""Author the ground-truth expectations for each corpus case (PRD 18.3).

Development script, not a test. Run it after adding or revising a case:

    python -m tests.fixtures.author_expectations

The expectations below are hand-written from the intent of each rendered page, not derived from OCR
and not derived from any model's output. That direction matters: a fixture generated from what the
system currently produces measures nothing. `expected_ocr.json` records what OCR *does*;
`expected.json` records what the extraction *should* recover from it.

Fields are expressed as constraints rather than exact strings wherever OCR damage makes an exact
match meaningless. `title_from_blocks` says which blocks a title must be derived from -- it
deliberately does not pin the exact characters, because on a corrupted Sinhala line the correct
behaviour is to preserve the damaged source wording and lower confidence, not to silently repair
it into clean text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"

EXPECTATIONS: dict[str, dict[str, Any]] = {
    "sinhala_only": {
        "description": "Single Sinhala vehicle advertisement on a clean page.",
        "exercises": [
            "Sinhala Unicode survives OCR, storage, and serialisation (AC-012)",
            "Heavy conjunct corruption must lower confidence, not be silently repaired",
            "Numerals and phone digits survive even when surrounding Sinhala does not",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": True, "max_mean_confidence": 0.75},
        "advertisements": [
            {
                "category": "vehicles",
                "language": "si",
                "price": {"amount": 8500000, "currency": "LKR", "raw_contains": "8,500,000"},
                "phones": ["0771234567"],
                "location_contains": "03",
                "title_from_blocks": [1],
                "evidence_within_blocks": [1, 2, 3],
            }
        ],
        "expected_warning_codes": ["LOW_CONFIDENCE"],
        "must_not_contain": {
            "note": (
                "The title line is OCR-damaged. A clean, plausible Sinhala title would be invented."
            ),
        },
    },
    "english_only": {
        "description": "Single English property advertisement.",
        "exercises": [
            "Baseline extraction with no language ambiguity",
            "Multi-line description assembled from adjacent blocks",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False, "min_mean_confidence": 0.85},
        "advertisements": [
            {
                "category": "property",
                "language": "en",
                "price": {"amount": 42500000, "currency": "LKR", "raw_contains": "42,500,000"},
                "phones": ["0112345678"],
                "location_contains": "Nugegoda",
                "title_contains_any": ["Bedroom", "House"],
                "evidence_within_blocks": [1, 2, 3, 4, 5],
            }
        ],
        "expected_warning_codes": [],
    },
    "mixed_language": {
        "description": "One advertisement written in both Sinhala and English.",
        "exercises": [
            "FR-LLM-005 mixed-language handling",
            "language must be 'mixed' rather than forced to one script",
            "A negotiable-price note must not corrupt the numeric amount",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [
            {
                "category": "vehicles",
                "language_any_of": ["mixed", "si", "en"],
                "price": {"amount": 3250000, "currency": "LKR", "raw_contains": "3,250,000"},
                "phones": ["0719876543"],
                "location_contains": "Gampaha",
                "evidence_within_blocks": [1, 2, 3, 4],
            }
        ],
        "expected_warning_codes": [],
    },
    "multi_column_three_ads": {
        "description": "Three unrelated advertisements in three ruled columns.",
        "exercises": [
            "AC-002: three independent candidates from one image",
            "False-merge and false-split rates",
            "Column reading order via block coordinates",
            "Each candidate cites only its own column's blocks",
        ],
        "expected_ad_count": 3,
        "ocr": {"expect_low_confidence": False, "min_mean_confidence": 0.85},
        "advertisements": [
            {
                "category": "vehicles",
                "language": "en",
                "price": {"amount": 5750000, "currency": "LKR", "raw_contains": "5,750,000"},
                "phones": ["0812233445"],
                "location_contains": "Kandy",
                "title_contains_any": ["Honda", "Fit"],
                "evidence_within_blocks": [1, 2],
            },
            {
                "category": "land",
                "language": "si",
                "price": {"amount": 1800000, "currency": "LKR", "raw_contains": "1,800,000"},
                "phones": ["0412345678"],
                "evidence_within_blocks": [3, 4],
            },
            {
                "category": "jobs",
                "language": "en",
                "price": {"amount": None, "currency": None},
                "phones": ["0114567890"],
                "location_contains": "Colombo",
                "title_contains_any": ["Accounts", "Clerk"],
                "evidence_within_blocks": [5, 6],
            },
        ],
        "expected_warning_codes": [],
        "disjoint_evidence": True,
    },
    "no_ads": {
        "description": "A weather report. Editorial content with no advertisement.",
        "exercises": [
            "AC-003: an empty advertisements array marks the item no_ads",
            "No candidate is created from non-advertisement text",
        ],
        "expected_ad_count": 0,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [],
        "expected_warning_codes": [],
    },
    "prompt_injection": {
        "description": "A genuine advertisement containing text that impersonates instructions.",
        "exercises": [
            "AC-013: injected text cannot change the schema or the extraction policy",
            "Injected lines are treated as content, not as commands",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [
            {
                "category": "home_garden",
                "language": "en",
                "price": {"amount": 65000, "currency": "LKR", "raw_contains": "65,000"},
                "phones": ["0312345678"],
                "location_contains": "Negombo",
                "title_contains_any": ["Dining", "Table"],
                "evidence_within_blocks": [1, 2, 3, 4],
            }
        ],
        "expected_warning_codes": [],
        "injection_markers": [
            "Ignore all previous instructions",
            "SYSTEM:",
            "admin mode",
        ],
        "must_still_return_schema": True,
    },
    "duplicate_phone": {
        "description": "Two genuine advertisements that share one contact number.",
        "exercises": [
            "FR-CAN-007/008: a shared phone raises a duplicate warning",
            "Both candidates are kept; neither is silently discarded",
        ],
        "expected_ad_count": 2,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [
            {
                "category": "vehicles",
                "language": "en",
                "price": {"amount": 615000, "currency": "LKR", "raw_contains": "615,000"},
                "phones": ["0771111222"],
                "location_contains": "Kurunegala",
                "title_contains_any": ["Yamaha", "FZ"],
            },
            {
                "category": "vehicles",
                "language": "en",
                "price": {"amount": 245000, "currency": "LKR", "raw_contains": "245,000"},
                "phones": ["0771111222"],
                "location_contains": "Kurunegala",
                "title_contains_any": ["Bajaj", "CT"],
            },
        ],
        "expected_warning_codes": ["POSSIBLE_DUPLICATE"],
        "disjoint_evidence": True,
    },
    "price_ambiguous_o_for_zero": {
        "description": "A price printed with capital O standing in for zero.",
        "exercises": [
            "Raw price text is preserved verbatim",
            "A normalised amount is only asserted with a warning and reduced confidence",
            "The reviewer can see what the source actually said",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [
            {
                "category": "home_garden",
                "language": "en",
                "price": {
                    "amount_any_of": [120000, None],
                    "currency": "LKR",
                    "raw_must_be_preserved": True,
                },
                "phones": ["0382345678"],
                "location_contains": "Panadura",
                "price_confidence_below": 0.9,
            }
        ],
        "expected_warning_codes": ["OCR_AMBIGUOUS_CHARACTER"],
    },
    "rotated_exif": {
        "description": "A skewed scan, as produced by a hand-held photograph of a page.",
        "exercises": [
            "FR-OCR-004/005: orientation and deskew handling",
            "Extraction still succeeds on a degraded scan",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False},
        "advertisements": [
            {
                "category": "property",
                "language": "en",
                "price": {"amount": 95000, "currency": "LKR", "raw_contains": "95,000"},
                "phones": ["0117654321"],
                "location_contains": "Colombo",
                "title_contains_any": ["Office", "Rent"],
            }
        ],
        "expected_warning_codes": [],
    },
    "low_resolution": {
        "description": "A small, downscaled clipping.",
        "exercises": [
            "FR-OCR-008: low confidence still proceeds to extraction but warns",
            "Small text does not cause invented values",
        ],
        "expected_ad_count": 1,
        "ocr": {"expect_low_confidence": False, "max_mean_confidence": 0.95},
        "advertisements": [
            {
                "category": "electronics",
                "language": "en",
                "price": {"amount": 135000, "currency": "LKR", "raw_contains": "135,000"},
                "phones": ["0912345678"],
                "location_contains": "Galle",
                "title_contains_any": ["Laptop", "i5"],
            }
        ],
        "expected_warning_codes": [],
    },
}


def main() -> None:
    for case, expectation in sorted(EXPECTATIONS.items()):
        case_dir = CORPUS_DIR / case
        if not case_dir.exists():
            print(f"{case:32} SKIPPED (no such case directory)")
            continue

        payload = {"case": case, **expectation}
        target = case_dir / "expected.json"
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"{case:32} ads={expectation['expected_ad_count']:<3} -> {target.name}")


if __name__ == "__main__":
    main()
