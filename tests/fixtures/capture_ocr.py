"""Capture Tesseract output for every corpus case (PRD 18.3).

Development script, not a test. Run it after regenerating images or changing OCR configuration:

    python -m tests.fixtures.capture_ocr

Writes `expected_ocr.json` next to each `image.png`. What is captured is what Tesseract *actually*
produces, including its mistakes -- not an idealised transcript. That is the point: the corpus
documents real OCR behaviour so the extraction stage can be measured on realistic input, and so a
change in OCR configuration shows up as a visible diff rather than a silent quality shift.

Since Phase 2 this runs the **production** preprocessor and provider rather than its own copy of the
block algorithm. That is what makes the corpus meaningful: a fixture is now exactly what the
pipeline sees, and `tests/test_ocr_corpus.py` asserts the provider still reproduces it. Two
implementations of the same assembly could drift apart without either being wrong on its own terms;
one cannot.
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from pathlib import Path

from media_service.config import get_settings
from media_service.ocr.preprocess import ImagePreprocessor
from media_service.ocr.providers.tesseract import TesseractOcrProvider
from media_service.ocr.registry import tesseract_options

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"


def build_provider() -> TesseractOcrProvider:
    return TesseractOcrProvider(options=tesseract_options(get_settings()))


def capture(image_path: Path) -> dict:
    settings = get_settings()
    preprocessed = ImagePreprocessor().run(image_path.read_bytes())
    result = build_provider().extract(preprocessed.image_bytes, content_type="image/png")

    return {
        "engine": result.engine,
        "engine_version": result.engine_version,
        "languages": result.languages,
        "preprocess_version": result.preprocess_version,
        "width": result.width,
        "height": result.height,
        "mean_confidence": result.mean_confidence,
        "block_count": result.block_count,
        # NFC, never NFKC: NFKC decomposes Sinhala conjuncts and the text stops round-tripping.
        "text": unicodedata.normalize("NFC", result.text),
        "blocks": result.block_documents(),
        "low_confidence_threshold": settings.ocr_low_confidence_threshold,
    }


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
        # Windows consoles default to cp1252, which cannot print Sinhala and would crash the run
        # after the files were already written.
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    cases = sorted(path for path in CORPUS_DIR.iterdir() if path.is_dir())
    for case in cases:
        image = case / "image.png"
        if not image.exists():
            print(f"skip {case.name}: no image.png")
            continue

        captured = capture(image)
        target = case / "expected_ocr.json"
        target.write_text(
            json.dumps(captured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{case.name:30s} blocks={captured['block_count']:2d} "
            f"confidence={captured['mean_confidence']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
