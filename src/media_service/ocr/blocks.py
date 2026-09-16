"""Turning engine word rows into ordered, stable blocks.

Pure functions over plain data. No engine is imported here, which is what lets the same assembly run
behind Tesseract, behind the Paddle hybrid, and in the fixture-capture script -- so the corpus and
the provider cannot drift apart by construction.

Three decisions, each fixing something the engine gets wrong for this purpose.

**Rows with `conf = -1` or empty text are dropped before anything is averaged.** Tesseract emits
them to describe layout, not recognised text. Averaging them in produces a confidence figure that
looks precise and means nothing.

**Reading order is computed, not taken from the engine.** Tesseract's `block_num` restarts per page,
can be zero, and follows its own segmentation rather than the columns a person reads. Classified
pages are columns; ordering by column band and then down the band is what makes "block 3" the same
region a reviewer would point at.

**Ids are ours and sequential from 1.** Every candidate cites `source_block_ids`, so an id that
shifted between runs would silently invalidate every stored citation. The engine's own identifiers
are preserved verbatim in `source_ref` for tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrLine

# A block whose left edge sits further than this fraction of the page width from the current
# column's starts a new column. Newspaper columns are separated by far more than the jitter of
# indentation within one column, so the threshold is not delicate.
COLUMN_GAP_RATIO: Final = 0.15

# Tesseract reports confidence as 0-100. Everything downstream -- the schema check, the low-
# confidence threshold, the stored column -- works in 0-1.
CONFIDENCE_SCALE: Final = 100.0


@dataclass(slots=True)
class WordRow:
    """One recognised word, as every engine can describe it."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    page: int = 1
    block: int = 0
    paragraph: int = 0
    line: int = 0

    @property
    def box(self) -> BoundingBox:
        return BoundingBox(self.left, self.top, self.width, self.height)


@dataclass(slots=True)
class _Group:
    lines: dict[int, list[WordRow]] = field(default_factory=dict)
    confidences: list[float] = field(default_factory=list)
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def add(self, word: WordRow) -> None:
        if not self.confidences:
            self.left, self.top = word.left, word.top
            self.right, self.bottom = word.left + word.width, word.top + word.height
        else:
            self.left = min(self.left, word.left)
            self.top = min(self.top, word.top)
            self.right = max(self.right, word.left + word.width)
            self.bottom = max(self.bottom, word.top + word.height)
        self.lines.setdefault(word.line, []).append(word)
        self.confidences.append(word.confidence)


def words_from_tesseract(data: dict[str, list[Any]]) -> list[WordRow]:
    """Read `image_to_data` output, discarding the rows that are not recognised text."""
    words: list[WordRow] = []
    for index in range(len(data["text"])):
        text = str(data["text"][index]).strip()
        if not text:
            continue
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        words.append(
            WordRow(
                text=text,
                left=int(data["left"][index]),
                top=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
                confidence=confidence,
                page=int(data.get("page_num", [1])[index]),
                block=int(data["block_num"][index]),
                paragraph=int(data["par_num"][index]),
                line=int(data["line_num"][index]),
            )
        )
    return words


def group_into_blocks(
    words: list[WordRow],
    *,
    box_source: BoxSource = BoxSource.ENGINE,
    detector: str | None = None,
) -> list[OcrBlock]:
    """Collapse words into paragraph-level blocks, unordered and unnumbered.

    Ids are assigned separately by `assign_ids`, because ordering needs the page width and a caller
    that merges blocks from several passes must number them once, at the end.
    """
    grouped: dict[tuple[int, int, int], _Group] = {}
    for word in words:
        grouped.setdefault((word.page, word.block, word.paragraph), _Group()).add(word)

    blocks: list[OcrBlock] = []
    for (page, block_num, paragraph), group in grouped.items():
        lines = tuple(_line_of(row) for _, row in sorted(group.lines.items()))
        blocks.append(
            OcrBlock(
                id=0,
                text="\n".join(line.text for line in lines),
                confidence=_mean(group.confidences),
                box=BoundingBox.from_edges(group.left, group.top, group.right, group.bottom),
                box_source=box_source,
                lines=lines,
                source_ref=f"page={page};block={block_num};par={paragraph}",
                detector=detector,
            )
        )
    return blocks


def reading_order(blocks: list[OcrBlock], *, page_width: int) -> list[OcrBlock]:
    """Sort into column bands, then down each band.

    Blocks with no geometry cannot be placed in a column, so they keep their relative order and
    follow everything that can -- a `vision_llm` result, where no box exists at all, comes back in
    the order the model produced it.
    """
    placed = [block for block in blocks if block.box is not None]
    unplaced = [block for block in blocks if block.box is None]
    if not placed:
        return unplaced

    threshold = max(1, int(page_width * COLUMN_GAP_RATIO))
    by_left = sorted(placed, key=lambda block: block.box.left)  # type: ignore[union-attr]

    columns: list[list[OcrBlock]] = [[by_left[0]]]
    for block in by_left[1:]:
        previous = columns[-1][-1].box
        if block.box.left - previous.left > threshold:  # type: ignore[union-attr]
            columns.append([block])
        else:
            columns[-1].append(block)

    ordered: list[OcrBlock] = []
    for column in columns:
        ordered.extend(
            sorted(column, key=lambda block: (block.box.top, block.box.left))  # type: ignore[union-attr]
        )
    return [*ordered, *unplaced]


def assign_ids(blocks: list[OcrBlock]) -> tuple[OcrBlock, ...]:
    """Number blocks 1..n in the order given. The only place an id is ever set."""
    from dataclasses import replace

    return tuple(
        replace(block, id=position) for position, block in enumerate(blocks, start=1)
    )


def assemble(
    words: list[WordRow],
    *,
    page_width: int,
    box_source: BoxSource = BoxSource.ENGINE,
    detector: str | None = None,
) -> tuple[OcrBlock, ...]:
    """Words in, ordered and numbered blocks out."""
    grouped = group_into_blocks(words, box_source=box_source, detector=detector)
    return assign_ids(reading_order(grouped, page_width=page_width))


def page_text(blocks: tuple[OcrBlock, ...] | list[OcrBlock]) -> str:
    """The flat transcript, blocks separated by a blank line.

    Reading order matters here: this string is what the extraction stage sees, and an advertisement
    split across a column break reads as nonsense if the blocks arrive in engine order.
    """
    return "\n\n".join(block.text for block in blocks)


def _line_of(words: list[WordRow]) -> OcrLine:
    box = words[0].box
    for word in words[1:]:
        box = box.union(word.box)
    return OcrLine(
        text=" ".join(word.text for word in words),
        box=box,
        confidence=_mean([word.confidence for word in words]),
    )


def _mean(confidences: list[float]) -> float | None:
    """Confidence in 0-1, or None when nothing survived filtering -- never a misleading zero."""
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences) / CONFIDENCE_SCALE, 4)
