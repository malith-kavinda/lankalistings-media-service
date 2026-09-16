"""Merging detected lines back into advertisement-sized regions.

The heuristic that decides where one classified advertisement ends and the next begins. Tested with
synthetic boxes rather than a detector, because the rule is about geometry and needs to hold for any
detector that produces line boxes.
"""

from __future__ import annotations

from media_service.ocr import regions
from media_service.ocr.types import BoundingBox


def line(top: int, *, left: int = 10, width: int = 200, height: int = 20) -> BoundingBox:
    return BoundingBox(left=left, top=top, width=width, height=height)


def test_lines_of_one_advertisement_merge_into_one_region() -> None:
    merged = regions.merge_lines([line(0), line(24), line(48)])

    assert len(merged) == 1
    assert merged[0] == BoundingBox.from_edges(10, 0, 210, 68)


def test_a_wide_vertical_gap_starts_a_new_region() -> None:
    """The gap between advertisements is larger than the gap between their lines."""
    merged = regions.merge_lines([line(0), line(24), line(300)])

    assert len(merged) == 2


def test_lines_in_different_columns_never_merge() -> None:
    """The false merge that matters: two ads side by side are not one ad."""
    merged = regions.merge_lines([line(0, left=10), line(4, left=400)])

    assert len(merged) == 2
    assert {box.left for box in merged} == {10, 400}


def test_a_short_line_under_a_long_one_still_belongs_to_it() -> None:
    """A price line is narrower than its title and is part of the same advertisement."""
    merged = regions.merge_lines([line(0, width=300), line(24, width=80)])

    assert len(merged) == 1


def test_the_gap_is_measured_in_line_heights_not_pixels() -> None:
    """The same layout at twice the scan resolution must merge the same way."""
    small = regions.merge_lines([line(0, height=20), line(24, height=20)])
    large = regions.merge_lines(
        [line(0, height=40, width=400), line(48, height=40, width=400)]
    )

    assert len(small) == len(large) == 1


def test_merging_does_not_depend_on_the_order_boxes_arrive_in() -> None:
    """A region that grows can reach a neighbour it could not before, so one pass is not enough."""
    boxes = [line(0), line(48), line(24)]

    forwards = regions.merge_lines(boxes)
    backwards = regions.merge_lines(list(reversed(boxes)))

    assert forwards == backwards
    assert len(forwards) == 1


def test_heavily_overlapping_boxes_merge_whatever_the_gap_rule_says() -> None:
    """A detector that emitted the same line twice at slightly different bounds."""
    merged = regions.merge_lines(
        [line(0), BoundingBox(left=12, top=2, width=198, height=20)], merge_iou=0.5
    )

    assert len(merged) == 1


def test_no_boxes_means_no_regions() -> None:
    assert regions.merge_lines([]) == []


def test_regions_are_ordered_by_column_then_down() -> None:
    ordered = regions.reading_order_boxes(
        [line(5, left=600), line(200, left=10), line(10, left=12)], page_width=1000
    )

    # Column band first, then down the band -- so the box higher up its column comes first even
    # though another box in the same column starts further left.
    assert [(box.left, box.top) for box in ordered] == [(12, 10), (10, 200), (600, 5)]


def test_a_region_is_padded_but_never_runs_off_the_page() -> None:
    padded = regions.clamp(line(0, left=0), width=100, height=100, padding=6)

    assert padded.left == 0
    assert padded.top == 0
    assert padded.right <= 100
    assert padded.bottom <= 100


def test_horizontal_overlap_is_measured_against_the_narrower_box() -> None:
    wide = BoundingBox(left=0, top=0, width=400, height=20)
    narrow = BoundingBox(left=100, top=0, width=100, height=20)

    assert regions.horizontal_overlap_ratio(wide, narrow) == 1.0
    assert regions.horizontal_overlap_ratio(narrow, wide) == 1.0


def test_boxes_that_do_not_overlap_horizontally_score_zero() -> None:
    assert (
        regions.horizontal_overlap_ratio(
            BoundingBox(0, 0, 50, 20), BoundingBox(100, 0, 50, 20)
        )
        == 0.0
    )
