"""Merging detected text lines into advertisement-sized regions.

A text detector returns one box per **line**. That is the wrong unit for this pipeline: feeding
hundreds of line crops to recognition destroys advertisement boundaries, and `source_block_ids`
stops meaning anything because every citation points at a fragment.

So lines are merged back into regions before recognition, using the two properties that actually
separate classified advertisements on a page:

* **Horizontal overlap.** Two lines belong to the same advertisement only if they sit in the same
  column. Lines in adjacent columns can be vertically adjacent and have nothing to do with each
  other, which is precisely the false merge that would join two unrelated ads.
* **Vertical gap, measured in line heights.** Within a column, the gap between lines of one
  advertisement is a fraction of a line; the gap between advertisements is larger. Measuring in
  line heights rather than pixels keeps the rule working at any scan resolution.

Pure functions over boxes, with no detector imported, so the heuristics can be tested exhaustively
without installing one.
"""

from __future__ import annotations

from typing import Final

from media_service.ocr.blocks import COLUMN_GAP_RATIO
from media_service.ocr.types import BoundingBox

# A gap wider than this many line heights starts a new region. Tuned to sit between the intra-ad
# line spacing of a classified column and the rule or whitespace between advertisements.
DEFAULT_GAP_IN_LINE_HEIGHTS: Final = 1.2

# Two lines are in the same column when this much of the narrower one lies within the wider one.
# Generous, because a short price line under a long title legitimately overlaps only partly.
DEFAULT_HORIZONTAL_OVERLAP: Final = 0.35


def horizontal_overlap_ratio(first: BoundingBox, second: BoundingBox) -> float:
    """Shared width as a fraction of the narrower box."""
    overlap = min(first.right, second.right) - max(first.left, second.left)
    if overlap <= 0:
        return 0.0
    narrower = min(first.width, second.width)
    return overlap / narrower if narrower > 0 else 0.0


def merge_lines(
    boxes: list[BoundingBox],
    *,
    gap_in_line_heights: float = DEFAULT_GAP_IN_LINE_HEIGHTS,
    horizontal_overlap: float = DEFAULT_HORIZONTAL_OVERLAP,
    merge_iou: float = 0.0,
) -> list[BoundingBox]:
    """Collapse line boxes into region boxes, top to bottom within each column.

    `merge_iou` is a second chance for boxes the column-and-gap rule separated: two boxes that
    genuinely overlap by that much are the same region whatever their vertical spacing says, which
    catches a detector that emitted a line twice at slightly different bounds.
    """
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda box: (box.top, box.left))
    typical_height = _median([box.height for box in ordered]) or 1
    gap_limit = typical_height * gap_in_line_heights

    regions: list[BoundingBox] = []
    for box in ordered:
        target = _absorbing_region(
            regions,
            box,
            gap_limit=gap_limit,
            horizontal_overlap=horizontal_overlap,
            merge_iou=merge_iou,
        )
        if target is None:
            regions.append(box)
        else:
            regions[target] = regions[target].union(box)

    return _settle(
        regions,
        gap_limit=gap_limit,
        horizontal_overlap=horizontal_overlap,
        merge_iou=merge_iou,
    )


def reading_order_boxes(boxes: list[BoundingBox], *, page_width: int) -> list[BoundingBox]:
    """Column band, then down the band -- the same order block assembly uses.

    Shares `COLUMN_GAP_RATIO` with it rather than repeating the number. Two copies that agree today
    would let a later tuning change desync the hybrid's reading order from the plain provider's,
    and nothing would catch it.
    """
    if not boxes:
        return []
    threshold = max(1, int(page_width * COLUMN_GAP_RATIO))
    by_left = sorted(boxes, key=lambda box: box.left)

    columns: list[list[BoundingBox]] = [[by_left[0]]]
    for box in by_left[1:]:
        if box.left - columns[-1][-1].left > threshold:
            columns.append([box])
        else:
            columns[-1].append(box)

    ordered: list[BoundingBox] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda box: (box.top, box.left)))
    return ordered


def clamp(box: BoundingBox, *, width: int, height: int, padding: int = 0) -> BoundingBox:
    """Grow a region slightly and keep it inside the page.

    Detectors fit boxes tightly to the ink; recognition does better with a little quiet space
    around the glyphs, and a crop that runs off the page edge is an error rather than a smaller
    crop.
    """
    left = max(box.left - padding, 0)
    top = max(box.top - padding, 0)
    right = min(box.right + padding, width)
    bottom = min(box.bottom + padding, height)
    return BoundingBox.from_edges(left, top, max(right, left), max(bottom, top))


def _absorbing_region(
    regions: list[BoundingBox],
    box: BoundingBox,
    *,
    gap_limit: float,
    horizontal_overlap: float,
    merge_iou: float,
) -> int | None:
    for index, region in enumerate(regions):
        if merge_iou > 0 and region.iou(box) >= merge_iou:
            return index
        if horizontal_overlap_ratio(region, box) < horizontal_overlap:
            continue
        # Negative gap means the boxes already overlap vertically.
        gap = box.top - region.bottom
        if gap <= gap_limit:
            return index
    return None


def _settle(
    regions: list[BoundingBox],
    *,
    gap_limit: float,
    horizontal_overlap: float,
    merge_iou: float,
) -> list[BoundingBox]:
    """Re-merge until nothing changes.

    One pass is not enough: absorbing a line grows a region, and the grown region may now reach a
    neighbour it did not before. Without this, the result would depend on the order the detector
    happened to emit its boxes.
    """
    current = regions
    while True:
        merged: list[BoundingBox] = []
        changed = False
        for box in current:
            target = _absorbing_region(
                merged,
                box,
                gap_limit=gap_limit,
                horizontal_overlap=horizontal_overlap,
                merge_iou=merge_iou,
            )
            if target is None:
                merged.append(box)
            else:
                merged[target] = merged[target].union(box)
                changed = True
        if not changed:
            return merged
        current = merged


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
