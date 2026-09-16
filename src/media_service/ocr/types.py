"""What an OCR provider returns.

Frozen and slotted: these cross a thread boundary from the worker pool and are serialised into the
`ocr_extractions` row that every later stage treats as evidence. A mutable result would let a stage
"correct" something after the fact, and the persisted record would no longer be what the engine
actually said.

`box_source` is the load-bearing field, and it is here from the first line of Phase 2 rather than
added later. A bounding box can come from the engine that recognised the text, from a model that
estimated it, or from nothing at all -- and `vision_llm` cannot produce engine geometry by
construction. A crop taken from an estimated box is not evidence of the same quality as one taken
from engine geometry, so FR-CAN-006's `crop_needs_review` path keys off exactly this value. Storing
a box without saying where it came from would erase that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BoxSource(StrEnum):
    ENGINE = "engine"
    MODEL_ESTIMATE = "model_estimate"
    SYNTHETIC = "synthetic"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel geometry in the coordinate space of the image the engine actually read."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return max(self.width, 0) * max(self.height, 0)

    def as_list(self) -> list[int]:
        """The corpus fixtures store `[left, top, width, height]`."""
        return [self.left, self.top, self.width, self.height]

    @classmethod
    def from_edges(cls, left: int, top: int, right: int, bottom: int) -> BoundingBox:
        return cls(left=left, top=top, width=right - left, height=bottom - top)

    @classmethod
    def from_list(cls, values: list[int] | tuple[int, ...]) -> BoundingBox:
        left, top, width, height = values
        return cls(left=int(left), top=int(top), width=int(width), height=int(height))

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox.from_edges(
            min(self.left, other.left),
            min(self.top, other.top),
            max(self.right, other.right),
            max(self.bottom, other.bottom),
        )

    def intersection_area(self, other: BoundingBox) -> int:
        width = min(self.right, other.right) - max(self.left, other.left)
        height = min(self.bottom, other.bottom) - max(self.top, other.top)
        return max(width, 0) * max(height, 0)

    def iou(self, other: BoundingBox) -> float:
        overlap = self.intersection_area(other)
        if overlap == 0:
            return 0.0
        return overlap / (self.area + other.area - overlap)


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    box: BoundingBox | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class OcrBlock:
    """One paragraph-level region of recognised text.

    `id` is ours, not the engine's: assigned 1-based in reading order by `blocks.assign_ids`.
    Tesseract's own numbering restarts per page, can be zero, and does not follow columns, so an
    advertisement citing "block 3" would mean something different after any re-run. The engine's
    value is kept verbatim in `source_ref` so a result can still be traced back to it.

    **The persisted id is authoritative.** Evidence validation compares a candidate's
    `source_block_ids` against the stored OCR record, never against a freshly recomputed one.
    """

    id: int
    text: str
    confidence: float | None = None
    box: BoundingBox | None = None
    box_source: BoxSource = BoxSource.NONE
    lines: tuple[OcrLine, ...] = ()
    source_ref: str = ""
    detector: str | None = None

    @property
    def line_count(self) -> int:
        return len(self.lines) or (len(self.text.splitlines()) if self.text else 0)

    def as_document(self) -> dict[str, Any]:
        """The JSONB shape stored on `ocr_extractions.blocks`.

        Matches the corpus fixture layout, so a stored record and a captured fixture can be
        compared field for field rather than through a translation nobody maintains.
        """
        document: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "box": self.box.as_list() if self.box else None,
            "box_source": self.box_source.value,
            "confidence": self.confidence,
            "line_count": self.line_count,
            "source_ref": self.source_ref,
        }
        if self.detector:
            document["detector"] = self.detector
        return document


@dataclass(frozen=True, slots=True)
class OcrResult:
    """Everything one OCR pass produced, and everything needed to judge it.

    `degraded` says the result was produced by a fallback rather than the configured path -- the
    hybrid provider running without its detector, for instance. It is not a failure and does not
    stop the pipeline, but a reviewer and any accuracy measurement need to know that this page was
    not read the way the configuration says pages are read.
    """

    text: str
    blocks: tuple[OcrBlock, ...] = ()
    provider: str = "tesseract"
    engine: str = "tesseract"
    engine_version: str | None = None
    traineddata_version: str | None = None
    languages: str = "sin+eng"
    mean_confidence: float | None = None
    low_confidence: bool = False
    width: int | None = None
    height: int | None = None
    preprocess_version: str = "preprocess/v1"
    duration_ms: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)
    degraded: bool = False

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def max_block_id(self) -> int | None:
        return max((block.id for block in self.blocks), default=None)

    def block_documents(self) -> list[dict[str, Any]]:
        return [block.as_document() for block in self.blocks]
