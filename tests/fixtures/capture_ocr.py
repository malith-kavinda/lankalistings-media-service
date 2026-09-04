"""Capture Tesseract output for every corpus case (PRD 18.3).

Development script, not a test. Run it after regenerating images or changing the OCR configuration:

    python -m tests.fixtures.capture_ocr

Writes `expected_ocr.json` next to each `image.png`. What is captured is what Tesseract *actually*
produces, including its mistakes -- not an idealised transcript. That is the point: the corpus
documents real OCR behaviour so the extraction stage can be measured on realistic input, and so a
change in OCR configuration shows up as a visible diff rather than a silent quality shift.

The block assembly here is deliberately self-contained. Phase 2 introduces the real OcrProvider;
when it lands, re-run this script and the diff shows exactly how the structured-OCR change
affected every case.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESSDATA_DIR = PROJECT_ROOT / ".tessdata"

LANGUAGES = "sin+eng"

# A block whose left edge is more than this fraction of the page width away from the previous
# block's starts a new column. Newspaper columns are far wider apart than the jitter within one
# column.
COLUMN_GAP_RATIO = 0.15


@dataclass(frozen=True, slots=True)
class CapturedBlock:
    id: int
    text: str
    box: list[int]
    confidence: float
    line_count: int
    source_ref: str


def _configure_tesseract() -> object:
    import pytesseract

    for candidate in (
        os.getenv("TESSERACT_CMD"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            break

    if TESSDATA_DIR.exists():
        os.environ.setdefault("TESSDATA_PREFIX", str(TESSDATA_DIR))
    return pytesseract


def _group_words(data: dict[str, list]) -> list[dict]:
    """Collapse Tesseract word rows into paragraph-level blocks."""
    grouped: dict[tuple[int, int, int], dict] = {}

    for index in range(len(data["text"])):
        text = str(data["text"][index]).strip()
        if not text:
            continue

        # Tesseract emits -1 for rows that carry layout structure rather than recognised text.
        # Averaging those into a confidence produces a meaningless number, so they are dropped
        # before aggregation.
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        key = (
            int(data["page_num"][index]),
            int(data["block_num"][index]),
            int(data["par_num"][index]),
        )
        left = int(data["left"][index])
        top = int(data["top"][index])
        right = left + int(data["width"][index])
        bottom = top + int(data["height"][index])
        line = int(data["line_num"][index])

        entry = grouped.setdefault(
            key,
            {
                "lines": {},
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "confidences": [],
            },
        )
        entry["lines"].setdefault(line, []).append(text)
        entry["left"] = min(entry["left"], left)
        entry["top"] = min(entry["top"], top)
        entry["right"] = max(entry["right"], right)
        entry["bottom"] = max(entry["bottom"], bottom)
        entry["confidences"].append(confidence)

    blocks = []
    for (page, block_num, par_num), entry in grouped.items():
        lines = [" ".join(words) for _, words in sorted(entry["lines"].items())]
        blocks.append(
            {
                "text": "\n".join(lines),
                "left": entry["left"],
                "top": entry["top"],
                "right": entry["right"],
                "bottom": entry["bottom"],
                "confidence": sum(entry["confidences"]) / len(entry["confidences"]),
                "line_count": len(lines),
                "source_ref": f"page={page};block={block_num};par={par_num}",
            }
        )
    return blocks


def _reading_order(blocks: list[dict], page_width: int) -> list[dict]:
    """Order blocks by column, then down the column.

    Tesseract's own block numbering is not a reliable reading order across columns, and unstable ids
    would silently invalidate every `source_block_ids` reference in the corpus.
    """
    if not blocks:
        return []

    threshold = max(1, int(page_width * COLUMN_GAP_RATIO))
    by_left = sorted(blocks, key=lambda block: block["left"])

    columns: list[list[dict]] = [[by_left[0]]]
    for block in by_left[1:]:
        if block["left"] - columns[-1][-1]["left"] > threshold:
            columns.append([block])
        else:
            columns[-1].append(block)

    ordered: list[dict] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda block: (block["top"], block["left"])))
    return ordered


def capture(image_path: Path) -> dict:
    pytesseract = _configure_tesseract()
    from pytesseract import Output

    image = ImageOps.exif_transpose(Image.open(image_path))
    data = pytesseract.image_to_data(image, lang=LANGUAGES, output_type=Output.DICT)

    ordered = _reading_order(_group_words(data), image.width)

    blocks = [
        CapturedBlock(
            id=position,
            text=block["text"],
            box=[
                block["left"],
                block["top"],
                block["right"] - block["left"],
                block["bottom"] - block["top"],
            ],
            confidence=round(block["confidence"] / 100.0, 4),
            line_count=block["line_count"],
            source_ref=block["source_ref"],
        )
        for position, block in enumerate(ordered, start=1)
    ]

    mean_confidence = (
        round(sum(block.confidence for block in blocks) / len(blocks), 4) if blocks else None
    )

    return {
        "engine": "tesseract",
        "engine_version": str(pytesseract.get_tesseract_version()),
        "languages": LANGUAGES,
        "width": image.width,
        "height": image.height,
        "mean_confidence": mean_confidence,
        "block_count": len(blocks),
        "text": "\n\n".join(block.text for block in blocks),
        "blocks": [asdict(block) for block in blocks],
    }


def main() -> None:
    cases = sorted(path for path in CORPUS_DIR.iterdir() if path.is_dir())
    for case in cases:
        image_path = case / "image.png"
        if not image_path.exists():
            print(f"{case.name:32} SKIPPED (no image.png)")
            continue

        captured = capture(image_path)
        target = case / "expected_ocr.json"
        # newline="\n" keeps fixtures byte-stable across platforms, so re-running this script on
        # Windows does not produce a line-ending-only diff for every case.
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(captured, ensure_ascii=False, indent=2) + "\n")
        confidence = captured["mean_confidence"]
        rendered = f"{confidence:.2f}" if confidence is not None else "n/a"
        print(f"{case.name:32} blocks={captured['block_count']:<3} mean_conf={rendered}")


if __name__ == "__main__":
    main()
