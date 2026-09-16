"""PaddleOCR detection with Tesseract recognition.

**Paddle detects, Tesseract recognises. Always.** PaddleOCR has no Sinhala recognition model at all,
so full PaddleOCR was never an option for this product; what its detector gives us is
language-agnostic text localisation, which is the part Tesseract's page segmentation does worst on
a dense classified page. Recognition stays with the one engine that can read Sinhala.

The dependency is optional and loaded in three stages, so a deployment that does not use this
provider never pays for it and a deployment that does never blocks startup on it:

1. `__init__` imports nothing.
2. `availability_reason()` asks `importlib.util.find_spec` whether the package exists -- a metadata
   lookup, not an import.
3. `extract()` builds the detector on first use, behind a lock, and reuses it afterwards. First use
   may download model weights, which is why it must not happen during a health check.

With `paddleocr` absent and `OCR_PADDLE_FALLBACK_TO_TESSERACT` on (the default), the service boots
healthy and every result carries `degraded=True` and a `DETECTOR_UNAVAILABLE` warning. Degraded is
not failed: the page is still read, by whole-page Tesseract, and the record says so rather than
quietly presenting a different pipeline's output as this one's.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Protocol

from PIL import Image

from media_service.ocr import blocks as block_assembly
from media_service.ocr import quality, regions
from media_service.ocr.providers.tesseract import TesseractOcrProvider
from media_service.ocr.tesseract_cli import TesseractCli, parse_tsv
from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrResult

PROVIDER_NAME: Final = "paddle_tesseract"
DETECTOR_NAME: Final = "paddle"

DETECTOR_UNAVAILABLE: Final = "DETECTOR_UNAVAILABLE"
DETECTOR_FAILED: Final = "DETECTOR_FAILED"
NO_REGIONS_DETECTED: Final = "NO_REGIONS_DETECTED"
TOO_MANY_REGIONS: Final = "TOO_MANY_REGIONS"

# Quiet space added around a detected region before recognition. Detectors fit tightly to the ink;
# Tesseract reads a little better with a margin.
REGION_PADDING: Final = 6


class TextDetector(Protocol):
    """Anything that can propose text boxes. Exists so the hybrid is testable without PaddleOCR."""

    def detect(self, image_bytes: bytes) -> list[BoundingBox]: ...

    def availability_reason(self) -> str | None: ...


class PaddleTextDetector:
    """The real detector, imported no earlier than first use."""

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        device: str = "cpu",
        box_threshold: float = 0.5,
    ) -> None:
        self._model_dir = model_dir
        self._device = device
        self._box_threshold = box_threshold
        self._engine: Any | None = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def availability_reason(self) -> str | None:
        # find_spec reads package metadata; it does not execute the package.
        if find_spec("paddleocr") is None:
            return "The paddleocr package is not installed."
        return None

    def detect(self, image_bytes: bytes) -> list[BoundingBox]:
        engine = self._detector()
        return _boxes_from_paddle(self._infer(engine, self._as_array(image_bytes)))

    def _infer(self, engine: Any, array: Any) -> Any:
        """Run detection, one caller at a time.

        Serialised deliberately, and not only during construction. A PaddleInference predictor is a
        single native object and is not safe to call from several threads at once -- the guidance is
        one predictor per thread, or an explicit clone. The worker pool drives `worker_concurrency`
        threads through this one detector, so without this lock two items would enter the same
        predictor together and risk a crash or, worse, geometry that is quietly wrong. Geometry is
        exactly what must not be quietly wrong here: it becomes the region boxes behind block ids,
        and block ids are what candidates cite.

        Serialising detection costs little, because it is the cheap half. Recognition still runs
        concurrently -- each region is read by its own Tesseract process.
        """
        with self._inference_lock:
            return engine.ocr(array, det=True, rec=False, cls=False)

    @staticmethod
    def _as_array(image_bytes: bytes) -> Any:
        with Image.open(BytesIO(image_bytes)) as image:
            import numpy  # noqa: PLC0415 - only reachable when paddleocr is installed

            return numpy.array(image.convert("RGB"))

    def _detector(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                from paddleocr import PaddleOCR  # noqa: PLC0415 - deliberately deferred

                options: dict[str, Any] = {
                    "det": True,
                    "rec": False,
                    "cls": False,
                    "show_log": False,
                    "det_db_box_thresh": self._box_threshold,
                    "use_gpu": self._device != "cpu",
                }
                if self._model_dir is not None:
                    options["det_model_dir"] = str(self._model_dir)
                self._engine = PaddleOCR(**options)
        return self._engine


class PaddleTesseractOcrProvider:
    def __init__(
        self,
        *,
        tesseract: TesseractOcrProvider,
        detector: TextDetector | None = None,
        model_dir: Path | None = None,
        device: str = "cpu",
        box_threshold: float = 0.5,
        merge_iou: float = 0.1,
        max_regions: int = 40,
        fallback_to_tesseract: bool = True,
        region_psm: int = 6,
    ) -> None:
        self._tesseract = tesseract
        self._detector = detector or PaddleTextDetector(
            model_dir=model_dir, device=device, box_threshold=box_threshold
        )
        self._merge_iou = merge_iou
        self._max_regions = max_regions
        self._fallback = fallback_to_tesseract
        self._region_psm = region_psm

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        """Recognition is what this provider cannot do without.

        A missing detector is a degradation, not an outage, so it is only reported here when the
        fallback is switched off -- otherwise `/health` would call the service unhealthy while it is
        still reading every page it is given.
        """
        recognition = self._tesseract.availability_reason()
        if recognition is not None:
            return recognition
        detector = self._detector.availability_reason()
        if detector is not None and not self._fallback:
            return f"{detector} Set OCR_PADDLE_FALLBACK_TO_TESSERACT=true to run without it."
        return None

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        reason = self._detector.availability_reason()
        if reason is not None:
            return self._degraded(image_bytes, content_type, DETECTOR_UNAVAILABLE)

        try:
            detected = self._detector.detect(image_bytes)
        except Exception:  # noqa: BLE001 - a detector fault must not lose the page
            return self._degraded(image_bytes, content_type, DETECTOR_FAILED)

        if not detected:
            return self._degraded(image_bytes, content_type, NO_REGIONS_DETECTED)

        merged = regions.merge_lines(detected, merge_iou=self._merge_iou)
        if len(merged) > self._max_regions:
            # Recognising hundreds of fragments would be slow and would produce citations nobody
            # can use. Falling back reads the whole page instead of dropping regions -- losing text
            # silently is the one outcome worse than a degraded result.
            return self._degraded(image_bytes, content_type, TOO_MANY_REGIONS)

        return self._recognise(image_bytes, merged)

    # -- paths ---------------------------------------------------------------------------------

    def _degraded(self, image_bytes: bytes, content_type: str, warning: str) -> OcrResult:
        result = self._tesseract.extract(image_bytes, content_type=content_type)
        return replace(
            result,
            provider=PROVIDER_NAME,
            degraded=True,
            warnings=(*result.warnings, warning),
        )

    def _recognise(self, image_bytes: bytes, boxes: list[BoundingBox]) -> OcrResult:
        with Image.open(BytesIO(image_bytes)) as opened:
            image = opened.convert("L")
            width, height = image.size
            ordered = regions.reading_order_boxes(boxes, page_width=width)

            assembled: list[OcrBlock] = []
            for position, box in enumerate(ordered, start=1):
                region = regions.clamp(box, width=width, height=height, padding=REGION_PADDING)
                if region.area <= 0:
                    continue
                text, confidence = self._read_region(image, region)
                if not text:
                    continue
                assembled.append(
                    OcrBlock(
                        # Numbered by `assign_ids` below, which is the only place an id is set.
                        # Regions that recognised no text are skipped, so a position counted here
                        # would not survive as the final id anyway.
                        id=0,
                        text=text,
                        confidence=confidence,
                        box=region,
                        # The geometry is a real engine's, just a different engine from the one
                        # that read the glyphs -- which is what `detector` records.
                        box_source=BoxSource.ENGINE,
                        source_ref=f"detector={DETECTOR_NAME};region={position}",
                        detector=DETECTOR_NAME,
                    )
                )

        numbered = block_assembly.assign_ids(assembled)
        text = block_assembly.page_text(numbered)
        confidence = quality.mean_confidence(numbered)

        return OcrResult(
            text=text,
            blocks=numbered,
            provider=PROVIDER_NAME,
            engine="tesseract",
            engine_version=self._tesseract.engine_version,
            languages=self._tesseract.languages,
            mean_confidence=confidence,
            low_confidence=quality.is_low_confidence(confidence),
            width=width,
            height=height,
            duration_ms=0,
            warnings=quality.warnings_for(text=text, confidence=confidence),
        )

    def _read_region(self, image: Image.Image, box: BoundingBox) -> tuple[str, float | None]:
        crop = image.crop((box.left, box.top, box.right, box.bottom))
        buffer = BytesIO()
        crop.save(buffer, format="PNG")

        region_cli = TesseractCli(
            replace(self._tesseract.cli.options, psm=self._region_psm)
        )
        rows = parse_tsv(region_cli.image_to_tsv(buffer.getvalue()))

        words = [row for row in rows if row.text.strip() and row.conf >= 0]
        if not words:
            return "", None

        lines: dict[tuple[int, int], list[str]] = {}
        for row in words:
            lines.setdefault((row.block_num, row.line_num), []).append(row.text.strip())
        text = "\n".join(" ".join(parts) for _, parts in sorted(lines.items()))
        confidence = round(
            sum(row.conf for row in words) / len(words) / block_assembly.CONFIDENCE_SCALE, 4
        )
        return text, confidence


def _boxes_from_paddle(detected: Any) -> list[BoundingBox]:
    """Convert PaddleOCR's polygon output into axis-aligned boxes.

    Paddle returns quadrilaterals -- four (x, y) points per line -- and wraps them in a per-page
    list. Recognition crops rectangles, and a rotated quad's bounding rectangle is what a crop
    needs anyway.

    Written defensively because the exact nesting has changed between PaddleOCR releases: anything
    that does not look like a list of points is skipped rather than guessed at. A detector that
    returns a shape we do not recognise degrades to whole-page Tesseract, which is the same
    outcome as the detector being absent.
    """
    for candidate in _candidate_polygon_lists(detected):
        boxes = [box for box in (_box_from_polygon(polygon) for polygon in candidate) if box]
        if boxes:
            return boxes
    return []


def _candidate_polygon_lists(detected: Any) -> list[Any]:
    """The detected value itself, and its first element -- covering both nestings Paddle uses."""
    if not isinstance(detected, list | tuple) or not detected:
        return []
    candidates: list[Any] = [detected]
    first = detected[0]
    if isinstance(first, list | tuple) and first and isinstance(first[0], list | tuple):
        candidates.insert(0, first)
    return candidates


def _box_from_polygon(polygon: Any) -> BoundingBox | None:
    if not isinstance(polygon, list | tuple) or len(polygon) < 2:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for point in polygon:
        if not isinstance(point, list | tuple) or len(point) < 2:
            return None
        try:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        except (TypeError, ValueError):
            return None
    box = BoundingBox.from_edges(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
    return box if box.area > 0 else None
