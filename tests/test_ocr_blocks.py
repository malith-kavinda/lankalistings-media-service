"""Block assembly: filtering, grouping, reading order, and id assignment.

Pure functions over word rows, so every case here is exact rather than approximate. These are the
rules that decide what "block 3" means, and a candidate's `source_block_ids` is only worth anything
if they are stable.
"""

from __future__ import annotations

from media_service.ocr import blocks as assembly
from media_service.ocr.blocks import WordRow
from media_service.ocr.types import BoundingBox, BoxSource


def word(
    text: str,
    *,
    left: int = 0,
    top: int = 0,
    width: int = 40,
    height: int = 12,
    conf: float = 90.0,
    block: int = 1,
    paragraph: int = 1,
    line: int = 1,
) -> WordRow:
    return WordRow(
        text=text,
        left=left,
        top=top,
        width=width,
        height=height,
        confidence=conf,
        block=block,
        paragraph=paragraph,
        line=line,
    )


# -- filtering ---------------------------------------------------------------------------------


def test_layout_rows_are_dropped_before_anything_is_averaged() -> None:
    """Tesseract marks structural rows with conf -1; averaging them in means nothing."""
    data = {
        "text": ["Colombo", "", "  ", "Kandy"],
        "conf": [96.0, -1, -1, 90.0],
        "left": [0, 0, 0, 0],
        "top": [0, 0, 0, 20],
        "width": [40, 0, 0, 40],
        "height": [12, 0, 0, 12],
        "page_num": [1, 1, 1, 1],
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 1, 2],
    }

    words = assembly.words_from_tesseract(data)

    assert [row.text for row in words] == ["Colombo", "Kandy"]


def test_a_block_with_no_scored_words_has_no_confidence_rather_than_zero() -> None:
    grouped = assembly.group_into_blocks([])

    assert grouped == []
    assert assembly._mean([]) is None  # noqa: SLF001


def test_confidence_is_rescaled_to_zero_to_one() -> None:
    grouped = assembly.group_into_blocks([word("Colombo", conf=96.0), word("Kandy", conf=90.0)])

    assert grouped[0].confidence == 0.93


# -- grouping ----------------------------------------------------------------------------------


def test_words_group_by_paragraph_and_keep_their_lines() -> None:
    words = [
        word("Honda", line=1),
        word("Fit", left=45, line=1),
        word("Kandy", top=20, line=2),
        word("Vacancy", paragraph=2, top=60),
    ]

    grouped = assembly.group_into_blocks(words)

    texts = sorted(block.text for block in grouped)
    assert texts == ["Honda Fit\nKandy", "Vacancy"]


def test_a_block_box_covers_every_word_in_it() -> None:
    words = [word("Honda", left=10, top=10), word("Kandy", left=100, top=30, width=50)]

    grouped = assembly.group_into_blocks(words)

    assert grouped[0].box == BoundingBox.from_edges(10, 10, 150, 42)


def test_the_engine_reference_is_preserved_for_tracing() -> None:
    grouped = assembly.group_into_blocks([word("Honda", block=4, paragraph=2)])

    assert grouped[0].source_ref == "page=1;block=4;par=2"


def test_the_box_source_travels_with_the_block() -> None:
    grouped = assembly.group_into_blocks(
        [word("Honda")], box_source=BoxSource.MODEL_ESTIMATE, detector="vision"
    )

    assert grouped[0].box_source is BoxSource.MODEL_ESTIMATE
    assert grouped[0].detector == "vision"


# -- reading order -----------------------------------------------------------------------------


def test_columns_are_read_one_at_a_time_not_across_the_page() -> None:
    """The reason ids are computed rather than taken from the engine.

    Left column top, left column bottom, then right column -- not row by row across both, which is
    what would splice two unrelated advertisements into one run of text.
    """
    left_top = assembly.group_into_blocks([word("LeftTop", left=10, top=10)])[0]
    left_bottom = assembly.group_into_blocks([word("LeftBottom", left=12, top=200, block=2)])[0]
    right_top = assembly.group_into_blocks([word("RightTop", left=600, top=5, block=3)])[0]

    ordered = assembly.reading_order(
        [right_top, left_bottom, left_top], page_width=1000
    )

    assert [block.text for block in ordered] == ["LeftTop", "LeftBottom", "RightTop"]


def test_blocks_without_geometry_keep_their_order_and_come_last() -> None:
    """A vision model returns no boxes; its output still has to be orderable."""
    from dataclasses import replace

    placed = assembly.group_into_blocks([word("Placed", left=10)])[0]
    floating = replace(placed, text="Floating", box=None)

    ordered = assembly.reading_order([floating, placed], page_width=1000)

    assert [block.text for block in ordered] == ["Placed", "Floating"]


def test_ids_are_one_based_and_sequential_in_the_given_order() -> None:
    grouped = [
        assembly.group_into_blocks([word("A", left=10)])[0],
        assembly.group_into_blocks([word("B", left=20)])[0],
    ]

    numbered = assembly.assign_ids(grouped)

    assert [block.id for block in numbered] == [1, 2]


def test_assemble_orders_then_numbers() -> None:
    words = [
        word("Right", left=700, top=10, block=2),
        word("Left", left=10, top=10, block=1),
    ]

    assembled = assembly.assemble(words, page_width=1000)

    assert [(block.id, block.text) for block in assembled] == [(1, "Left"), (2, "Right")]


def test_page_text_separates_blocks_by_a_blank_line() -> None:
    assembled = assembly.assemble(
        [word("Left", left=10), word("Right", left=700, block=2)], page_width=1000
    )

    assert assembly.page_text(assembled) == "Left\n\nRight"
