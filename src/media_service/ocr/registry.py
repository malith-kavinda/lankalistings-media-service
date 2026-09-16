"""Choosing an OCR provider from configuration.

An explicit builder table with **function-local imports**, not a decorator registry. A decorator
registry only contains what has been imported, so it would force every adapter module to be
imported at startup to populate itself -- and importing `paddle_tesseract` pulls in PaddleOCR, which
is exactly what must not happen on a deployment that does not use it. Here, choosing `tesseract`
never touches the Paddle module at all.

An unknown name has already been refused by `Settings.validate_startup`, so this cannot fall back to
a default. A typo that silently selected a different engine would change every extraction the
service produces with nothing in the output saying so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from media_service.config import OCR_PROVIDERS, Settings
from media_service.ocr.preprocess import PreprocessSettings
from media_service.ocr.tesseract_cli import TesseractOptions

if TYPE_CHECKING:
    from media_service.ocr.protocol import OcrProvider


def tesseract_options(settings: Settings) -> TesseractOptions:
    return TesseractOptions(
        command=settings.tesseract_cmd,
        tessdata_dir=settings.tesseract_data_dir,
        languages=settings.ocr_languages,
        psm=settings.ocr_tesseract_psm,
        oem=settings.ocr_tesseract_oem,
        timeout_seconds=settings.ocr_timeout_seconds,
    )


def preprocess_settings(settings: Settings) -> PreprocessSettings:
    return PreprocessSettings(
        version=settings.ocr_preprocess_version,
        grayscale=settings.ocr_preprocess_grayscale,
        autocontrast=settings.ocr_preprocess_autocontrast,
        denoise=settings.ocr_preprocess_denoise,
        threshold=settings.ocr_preprocess_threshold,
        deskew=settings.ocr_preprocess_deskew,
        max_pixels=settings.max_image_pixels,
    )


def build_tesseract(settings: Settings) -> OcrProvider:
    return build_tesseract_provider(settings)


def build_tesseract_provider(settings: Settings):  # type: ignore[no-untyped-def]
    """The concrete Tesseract provider, which the hybrid composes rather than re-derives."""
    from media_service.ocr.providers.tesseract import TesseractOcrProvider

    return TesseractOcrProvider(
        options=tesseract_options(settings),
        low_confidence_threshold=settings.ocr_low_confidence_threshold,
        empty_text_min_chars=settings.ocr_empty_text_min_chars,
        preprocess_version=settings.ocr_preprocess_version,
    )


def build_paddle_tesseract(settings: Settings) -> OcrProvider:
    from media_service.ocr.providers.paddle_tesseract import PaddleTesseractOcrProvider

    return PaddleTesseractOcrProvider(
        # Composed, not constructed separately: the hybrid recognises with the same Tesseract the
        # plain provider uses, so switching between them cannot change recognition behaviour.
        tesseract=build_tesseract_provider(settings),
        region_psm=settings.ocr_tesseract_region_psm,
        model_dir=settings.ocr_paddle_model_dir,
        device=settings.ocr_paddle_device,
        box_threshold=settings.ocr_paddle_box_thresh,
        merge_iou=settings.ocr_paddle_merge_iou,
        max_regions=settings.ocr_paddle_max_regions,
        fallback_to_tesseract=settings.ocr_paddle_fallback_to_tesseract,
    )


def build_vision_llm(settings: Settings) -> OcrProvider:
    from media_service.ocr.providers.vision_llm import VisionLlmOcrProvider

    return VisionLlmOcrProvider()


BUILDERS = {
    "tesseract": build_tesseract,
    "paddle_tesseract": build_paddle_tesseract,
    "vision_llm": build_vision_llm,
}

assert set(BUILDERS) == set(OCR_PROVIDERS), "every configurable provider needs a builder"


def build_provider(settings: Settings) -> OcrProvider:
    builder = BUILDERS.get(settings.ocr_provider)
    if builder is None:  # pragma: no cover - validate_startup refuses this first
        raise ValueError(
            f"OCR_PROVIDER={settings.ocr_provider!r} is not one of "
            f"{', '.join(sorted(BUILDERS))}."
        )
    return builder(settings)
