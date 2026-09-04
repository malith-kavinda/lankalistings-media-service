"""Load the regression corpus (PRD 18.3).

Test-facing API. Each case bundles a source image, the OCR output actually captured from it, and the
ground-truth expectations authored for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


@dataclass(frozen=True, slots=True)
class CorpusBlock:
    id: int
    text: str
    box: tuple[int, int, int, int]
    confidence: float
    line_count: int
    source_ref: str


@dataclass(frozen=True, slots=True)
class CorpusCase:
    name: str
    directory: Path
    expected: dict[str, Any]
    ocr: dict[str, Any]

    @property
    def image_path(self) -> Path:
        return self.directory / "image.png"

    @property
    def image_bytes(self) -> bytes:
        return self.image_path.read_bytes()

    @property
    def expected_ad_count(self) -> int:
        return int(self.expected["expected_ad_count"])

    @property
    def text(self) -> str:
        return str(self.ocr["text"])

    @property
    def mean_confidence(self) -> float | None:
        value = self.ocr.get("mean_confidence")
        return None if value is None else float(value)

    @property
    def blocks(self) -> tuple[CorpusBlock, ...]:
        return tuple(
            CorpusBlock(
                id=int(block["id"]),
                text=str(block["text"]),
                box=tuple(int(value) for value in block["box"]),  # type: ignore[arg-type]
                confidence=float(block["confidence"]),
                line_count=int(block["line_count"]),
                source_ref=str(block["source_ref"]),
            )
            for block in self.ocr["blocks"]
        )

    @property
    def block_ids(self) -> frozenset[int]:
        return frozenset(block.id for block in self.blocks)

    @property
    def expected_advertisements(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.expected.get("advertisements", ()))

    @property
    def expected_warning_codes(self) -> tuple[str, ...]:
        return tuple(self.expected.get("expected_warning_codes", ()))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def load_case(name: str) -> CorpusCase:
    directory = CORPUS_DIR / name
    if not directory.is_dir():
        raise FileNotFoundError(f"No corpus case named {name!r} in {CORPUS_DIR}.")
    return CorpusCase(
        name=name,
        directory=directory,
        expected=_load_json(directory / "expected.json"),
        ocr=_load_json(directory / "expected_ocr.json"),
    )


@cache
def case_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.name
            for path in CORPUS_DIR.iterdir()
            if path.is_dir() and (path / "expected.json").exists()
        )
    )


def all_cases() -> tuple[CorpusCase, ...]:
    return tuple(load_case(name) for name in case_names())
