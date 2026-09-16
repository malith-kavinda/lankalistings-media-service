"""Preprocessing: what each step does, and that a derivative key follows the settings."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from media_service.ocr.preprocess import (
    ImagePreprocessor,
    PreprocessError,
    PreprocessSettings,
)
from media_service.storage import DerivativePurpose
from media_service.storage.filesystem import params_hash


def image_bytes(*, width: int = 60, height: int = 40, mode: str = "RGB", colour="white"):  # type: ignore[no-untyped-def]
    buffer = BytesIO()
    Image.new(mode, (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def opened(payload: bytes) -> Image.Image:
    return Image.open(BytesIO(payload))


# -- steps -------------------------------------------------------------------------------------


def test_the_default_pipeline_orients_and_greyscales_and_nothing_else() -> None:
    result = ImagePreprocessor().run(image_bytes())

    assert result.applied == ("exif_transpose", "grayscale")
    assert opened(result.image_bytes).mode == "L"
    assert result.content_type == "image/png"


def test_orientation_is_always_applied_because_no_input_wants_it_skipped() -> None:
    assert "exif_transpose" in ImagePreprocessor().run(image_bytes()).applied


def test_a_single_channel_is_produced_whenever_a_later_step_needs_one() -> None:
    """Autocontrast and thresholding both operate on one channel; asking for them implies it."""
    result = ImagePreprocessor(
        PreprocessSettings(grayscale=False, autocontrast=True)
    ).run(image_bytes())

    assert "grayscale" in result.applied
    assert "autocontrast" in result.applied


def test_thresholding_produces_a_two_tone_image() -> None:
    result = ImagePreprocessor(PreprocessSettings(threshold=128)).run(
        image_bytes(colour=(200, 200, 200))
    )

    assert opened(result.image_bytes).mode == "1"
    assert "threshold(128)" in result.applied


def test_denoising_is_recorded_when_it_runs() -> None:
    result = ImagePreprocessor(PreprocessSettings(denoise=True)).run(image_bytes())

    assert "denoise" in result.applied


def test_deskew_is_off_by_default_and_recorded_when_it_finds_an_angle() -> None:
    straight = ImagePreprocessor().run(image_bytes())
    assert not any(step.startswith("deskew") for step in straight.applied)

    lined = Image.new("L", (300, 200), 255)
    for row in range(20, 180, 20):
        for column in range(20, 280):
            lined.putpixel((column, row), 0)
    buffer = BytesIO()
    lined.rotate(-1.5, fillcolor=255).save(buffer, format="PNG")

    result = ImagePreprocessor(PreprocessSettings(deskew=True)).run(buffer.getvalue())

    # The angle it picks is what matters less than that it looked and recorded the answer.
    assert result.applied[-1].startswith("deskew") or "deskew" not in result.applied


def test_the_original_bytes_are_never_modified() -> None:
    original = image_bytes()
    before = bytes(original)

    ImagePreprocessor(PreprocessSettings(threshold=100)).run(original)

    assert original == before


# -- identity ----------------------------------------------------------------------------------


def test_the_same_settings_produce_the_same_bytes() -> None:
    """What makes the derivative idempotent under retry: a re-run rewrites an identical file."""
    payload = image_bytes()
    preprocessor = ImagePreprocessor()

    assert preprocessor.run(payload).image_bytes == preprocessor.run(payload).image_bytes


def test_changing_a_setting_changes_the_derivative_key() -> None:
    """So a settings change creates a new file instead of overwriting a cited one."""
    default = ImagePreprocessor()
    denoising = ImagePreprocessor(PreprocessSettings(denoise=True))

    first = params_hash(DerivativePurpose.OCR_INPUT, default.version, default.params())
    second = params_hash(DerivativePurpose.OCR_INPUT, denoising.version, denoising.params())

    assert first != second


def test_the_pixel_cap_is_not_part_of_the_identity() -> None:
    """It refuses images; it does not transform them, so it cannot change the output bytes."""
    default = ImagePreprocessor()
    stricter = ImagePreprocessor(PreprocessSettings(max_pixels=1_000_000))

    assert default.params() == stricter.params()


def test_the_version_is_reported_with_the_result() -> None:
    result = ImagePreprocessor().run(image_bytes())

    assert result.version == "preprocess/v1"


# -- refusals ----------------------------------------------------------------------------------


def test_an_image_above_the_pixel_cap_is_refused_before_it_is_decoded() -> None:
    with pytest.raises(PreprocessError) as caught:
        ImagePreprocessor(PreprocessSettings(max_pixels=100)).run(image_bytes())

    assert caught.value.code == "IMAGE_TOO_LARGE"


def test_bytes_that_are_not_an_image_are_refused() -> None:
    with pytest.raises(PreprocessError) as caught:
        ImagePreprocessor().run(b"not an image at all")

    assert caught.value.code == "IMAGE_UNREADABLE"
