"""The prototype's OCR shape, kept alive for the endpoints that still speak it.

`OcrEngine` and `OcrOutput` are what the single-image endpoints were built against: a string, a
three-valued confidence word, and no geometry. They are deprecated, and Phase 4 removes them along
with those endpoints.

The Tesseract implementation that used to live here is gone, replaced by
`media_service.ocr.providers.tesseract`. It is not a rename -- the old one had two defects that the
shape itself encouraged:

* It wrote `os.environ["TESSDATA_PREFIX"]` and `pytesseract.tesseract_cmd`, process-wide globals,
  from inside `availability_reason()` -- a method `/health` calls on **every request**.
* It shelled out to `tesseract --list-langs` on every one of those calls, spawning a subprocess per
  health check.

The prototype endpoints now read pages through the real provider, wrapped back into this shape by
`ocr.compat.as_legacy_engine`, so both paths recognise text identically and there is only one
implementation to fix.
"""

from dataclasses import dataclass

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}


@dataclass(frozen=True)
class OcrOutput:
    text: str
    engine: str
    model_version: str
    language: str
    confidence: str
    processing_ms: int
    width: int | None
    height: int | None


class OcrEngine:
    """The deprecated interface. New code takes an `ocr.OcrProvider` instead."""

    def is_available(self) -> bool:
        raise NotImplementedError

    def availability_reason(self) -> str | None:
        raise NotImplementedError

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        raise NotImplementedError
