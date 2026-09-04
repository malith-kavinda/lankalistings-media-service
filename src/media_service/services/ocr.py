from dataclasses import dataclass
from io import BytesIO
from os import environ
from pathlib import Path
from shutil import which
from time import perf_counter

from PIL import Image, UnidentifiedImageError

from media_service.api.errors import OcrUnavailableError, ServiceError

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
    def is_available(self) -> bool:
        raise NotImplementedError

    def availability_reason(self) -> str | None:
        raise NotImplementedError

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        raise NotImplementedError


class TesseractOcrEngine(OcrEngine):
    def __init__(
        self,
        *,
        tesseract_cmd: str | None = None,
        language: str = "eng",
        tessdata_dir: Path | None = None,
    ) -> None:
        self._tesseract_cmd = tesseract_cmd
        self._language = language
        self._tessdata_dir = tessdata_dir

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        try:
            import pytesseract
        except ImportError:
            return "The pytesseract Python package is not installed."

        command = self._tesseract_cmd or which("tesseract")
        if not command:
            return "The Tesseract OCR executable is not installed or not on PATH."

        self._apply_runtime_config(pytesseract)

        missing_languages = [
            language
            for language in self._language.split("+")
            if language not in pytesseract.get_languages(config="")
        ]
        if missing_languages:
            return f"Missing Tesseract language data: {', '.join(missing_languages)}."

        return None

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "file",
                        "code": "UNSUPPORTED_CONTENT_TYPE",
                        "message": "Only JPEG, PNG, WebP, TIFF, and BMP images are supported.",
                    }
                ],
            )

        reason = self.availability_reason()
        if reason:
            raise OcrUnavailableError(reason)

        try:
            import pytesseract

            image = Image.open(BytesIO(image_bytes))
            width, height = image.size
            started_at = perf_counter()
            detected_text = pytesseract.image_to_string(
                image,
                lang=self._language,
                config="",
            ).strip()
            elapsed_ms = int((perf_counter() - started_at) * 1000)
        except UnidentifiedImageError as exc:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "file",
                        "code": "INVALID_IMAGE",
                        "message": "Uploaded file could not be decoded as an image.",
                    }
                ],
            ) from exc

        return OcrOutput(
            text=detected_text,
            engine="tesseract",
            model_version="tesseract-local",
            language=self._language,
            confidence="low" if not detected_text else "medium",
            processing_ms=elapsed_ms,
            width=width,
            height=height,
        )

    def _apply_runtime_config(self, pytesseract: object) -> None:
        if self._tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        if self._tessdata_dir:
            environ["TESSDATA_PREFIX"] = str(self._tessdata_dir)
