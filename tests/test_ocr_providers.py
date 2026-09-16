"""Provider selection, the compatibility shim, and the hybrid's degraded path.

None of this needs Tesseract or PaddleOCR installed: the point of the seams is that the wiring is
testable without the engines behind them.
"""

from __future__ import annotations

import pytest

from media_service.api.errors import OcrUnavailableError
from media_service.config import Settings
from media_service.ocr.compat import (
    LegacyEngineProvider,
    ProviderLegacyEngine,
    as_legacy_engine,
    as_provider,
    confidence_to_word,
    word_to_confidence,
)
from media_service.ocr.protocol import OcrProvider
from media_service.ocr.providers.paddle_tesseract import (
    DETECTOR_FAILED,
    DETECTOR_UNAVAILABLE,
    NO_REGIONS_DETECTED,
    TOO_MANY_REGIONS,
    PaddleTesseractOcrProvider,
)
from media_service.ocr.providers.tesseract import TesseractOcrProvider
from media_service.ocr.registry import build_provider
from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrResult
from media_service.services.ocr import OcrEngine, OcrOutput
from tests.support import StubOcrEngine, png_bytes


class StubProvider:
    """A provider that records what it was asked and returns a fixed result."""

    def __init__(self, *, result: OcrResult | None = None, reason: str | None = None) -> None:
        self._result = result or OcrResult(text="Ocean View Apartment", mean_confidence=0.9)
        self._reason = reason
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return self._reason is None

    def availability_reason(self) -> str | None:
        return self._reason

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        self.calls += 1
        if self._reason:
            raise OcrUnavailableError(self._reason)
        return self._result


class StubDetector:
    def __init__(self, *, boxes=None, reason=None, explode=False) -> None:  # type: ignore[no-untyped-def]
        self._boxes = boxes or []
        self._reason = reason
        self._explode = explode

    def availability_reason(self) -> str | None:
        return self._reason

    def detect(self, image_bytes: bytes):  # type: ignore[no-untyped-def]
        if self._explode:
            raise RuntimeError("the detector fell over")
        return self._boxes


# -- registry ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["tesseract", "paddle_tesseract", "vision_llm"])
def test_every_configurable_provider_can_be_built(name: str) -> None:
    provider = build_provider(Settings(ocr_provider=name))

    assert isinstance(provider, OcrProvider)
    assert provider.name == name


def test_an_unknown_provider_stops_the_service_starting_and_names_the_valid_ones() -> None:
    """A typo must not silently select a different engine: every extraction would change."""
    with pytest.raises(ValueError, match="OCR_PROVIDER") as caught:
        Settings(ocr_provider="teseract").validate_startup()

    assert "tesseract" in str(caught.value)
    assert "paddle_tesseract" in str(caught.value)


def test_choosing_tesseract_never_imports_paddle() -> None:
    """The reason the registry is a builder table rather than a decorator registry."""
    import sys

    sys.modules.pop("media_service.ocr.providers.paddle_tesseract", None)
    build_provider(Settings(ocr_provider="tesseract"))

    assert "media_service.ocr.providers.paddle_tesseract" not in sys.modules


def test_the_vision_provider_exists_but_says_what_it_is_waiting_for() -> None:
    provider = build_provider(Settings(ocr_provider="vision_llm"))

    assert provider.is_available() is False
    assert "Phase 3" in provider.availability_reason()
    with pytest.raises(OcrUnavailableError):
        provider.extract(png_bytes(), content_type="image/png")


# -- compatibility -----------------------------------------------------------------------------


def test_a_legacy_engine_can_drive_the_new_pipeline() -> None:
    provider = as_provider(StubOcrEngine(text="Ocean View Apartment Colombo"))

    result = provider.extract(png_bytes(), content_type="image/png")

    assert result.text == "Ocean View Apartment Colombo"
    assert result.block_count == 1
    assert result.mean_confidence == word_to_confidence("high")


def test_a_legacy_engine_invents_no_geometry() -> None:
    """A full-page rectangle would look like evidence while being none."""
    provider = as_provider(StubOcrEngine())

    block = provider.extract(png_bytes(), content_type="image/png").blocks[0]

    assert block.box is None
    assert block.box_source is BoxSource.NONE


def test_an_empty_legacy_result_produces_no_blocks_and_no_confidence() -> None:
    provider = as_provider(StubOcrEngine(text="   "))

    result = provider.extract(png_bytes(), content_type="image/png")

    assert result.blocks == ()
    assert result.mean_confidence is None


def test_a_provider_can_drive_the_prototype_endpoints() -> None:
    engine = as_legacy_engine(
        StubProvider(
            result=OcrResult(text="Ocean View", mean_confidence=0.91, engine_version="5.4")
        )
    )

    output = engine.extract_text(png_bytes(), content_type="image/png")

    assert isinstance(output, OcrOutput)
    assert output.text == "Ocean View"
    assert output.confidence == "high"
    assert output.model_version == "5.4"


def test_wrapping_is_skipped_when_the_shape_already_matches() -> None:
    provider = StubProvider()
    engine = StubOcrEngine()

    assert as_provider(provider) is provider
    assert as_legacy_engine(engine) is engine
    assert isinstance(as_provider(engine), LegacyEngineProvider)
    assert isinstance(as_legacy_engine(provider), ProviderLegacyEngine)


@pytest.mark.parametrize(
    ("value", "word"), [(0.95, "high"), (0.7, "medium"), (0.2, "low"), (None, "low")]
)
def test_confidence_words_map_onto_the_numeric_scale(value: float | None, word: str) -> None:
    assert confidence_to_word(value) == word


def test_an_ocr_engine_is_not_mistaken_for_a_provider() -> None:
    """The protocols differ by more than a method name, and isinstance has to see that."""
    assert not isinstance(StubOcrEngine(), OcrProvider)
    assert isinstance(StubProvider(), OcrProvider)


# -- the hybrid --------------------------------------------------------------------------------


def hybrid(detector: StubDetector, **kwargs) -> PaddleTesseractOcrProvider:  # type: ignore[no-untyped-def]
    return PaddleTesseractOcrProvider(
        tesseract=as_provider(StubOcrEngine(text="Whole page text")),  # type: ignore[arg-type]
        detector=detector,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("detector", "warning"),
    [
        (StubDetector(reason="The paddleocr package is not installed."), DETECTOR_UNAVAILABLE),
        (StubDetector(explode=True), DETECTOR_FAILED),
        (StubDetector(boxes=[]), NO_REGIONS_DETECTED),
    ],
)
def test_the_hybrid_degrades_to_whole_page_rather_than_losing_it(
    detector: StubDetector, warning: str
) -> None:
    """Degraded is not failed: the page is still read, and the record says how."""
    result = hybrid(detector).extract(png_bytes(), content_type="image/png")

    assert result.degraded is True
    assert warning in result.warnings
    assert result.text == "Whole page text"
    assert result.provider == "paddle_tesseract"


def test_too_many_regions_falls_back_instead_of_dropping_any() -> None:
    """Losing text silently is the one outcome worse than a degraded result."""
    boxes = [BoundingBox(left=10, top=index * 200, width=100, height=20) for index in range(8)]

    result = hybrid(StubDetector(boxes=boxes), max_regions=2).extract(
        png_bytes(), content_type="image/png"
    )

    assert result.degraded is True
    assert TOO_MANY_REGIONS in result.warnings


def test_a_missing_detector_does_not_make_the_service_unhealthy() -> None:
    """With the fallback on, pages are still being read -- health should say so."""
    provider = hybrid(StubDetector(reason="not installed"), fallback_to_tesseract=True)

    assert provider.availability_reason() is None


def test_a_missing_detector_is_reported_when_the_fallback_is_switched_off() -> None:
    provider = hybrid(StubDetector(reason="not installed"), fallback_to_tesseract=False)

    reason = provider.availability_reason()

    assert reason is not None
    assert "OCR_PADDLE_FALLBACK_TO_TESSERACT" in reason


def test_the_hybrid_cannot_run_without_recognition() -> None:
    """Paddle has no Sinhala model, so a missing Tesseract is fatal however the detector is."""
    provider = PaddleTesseractOcrProvider(
        tesseract=StubProvider(reason="Missing Tesseract language data: sin."),  # type: ignore[arg-type]
        detector=StubDetector(),
    )

    assert "sin" in provider.availability_reason()


def test_paddle_polygons_become_axis_aligned_boxes() -> None:
    from media_service.ocr.providers.paddle_tesseract import _boxes_from_paddle

    detected = [[[[10, 20], [110, 22], [110, 42], [10, 40]]]]

    assert _boxes_from_paddle(detected) == [BoundingBox.from_edges(10, 20, 110, 42)]


def test_a_detector_shape_we_do_not_recognise_yields_nothing_rather_than_a_guess() -> None:
    from media_service.ocr.providers.paddle_tesseract import _boxes_from_paddle

    assert _boxes_from_paddle(None) == []
    assert _boxes_from_paddle([]) == []
    assert _boxes_from_paddle(["unexpected"]) == []


# -- availability caching ----------------------------------------------------------------------


def test_availability_is_not_rechecked_on_every_call() -> None:
    """`/health` calls this per request; the old code spawned a subprocess each time."""
    calls = 0

    class CountingCli:
        def resolve_command(self) -> str:
            return "tesseract"

        def languages(self):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return frozenset({"eng", "sin"})

        options = type("Options", (), {"languages": "sin+eng"})()

    provider = TesseractOcrProvider(cli=CountingCli())  # type: ignore[arg-type]

    for _ in range(5):
        assert provider.is_available()

    assert calls == 1


def test_an_expired_cache_is_rechecked() -> None:
    class CountingCli:
        def __init__(self) -> None:
            self.calls = 0

        def resolve_command(self) -> str:
            return "tesseract"

        def languages(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            return frozenset({"eng", "sin"})

        options = type("Options", (), {"languages": "sin+eng"})()

    cli = CountingCli()
    provider = TesseractOcrProvider(cli=cli, availability_ttl_seconds=0.0)  # type: ignore[arg-type]

    provider.is_available()
    provider.is_available()

    assert cli.calls == 2


def test_missing_language_data_is_named_rather_than_skipped() -> None:
    class EnglishOnlyCli:
        def resolve_command(self) -> str:
            return "tesseract"

        def languages(self):  # type: ignore[no-untyped-def]
            return frozenset({"eng"})

        options = type("Options", (), {"languages": "sin+eng"})()

    provider = TesseractOcrProvider(cli=EnglishOnlyCli())  # type: ignore[arg-type]

    assert provider.availability_reason() == "Missing Tesseract language data: sin."


def test_an_unsupported_content_type_is_refused_before_the_engine_runs() -> None:
    provider = TesseractOcrProvider()

    with pytest.raises(Exception) as caught:
        provider.extract(b"payload", content_type="application/pdf")

    assert caught.value.code == "VALIDATION_FAILED"  # type: ignore[attr-defined]


def test_a_block_document_round_trips_its_geometry() -> None:
    block = OcrBlock(
        id=3,
        text="Honda Fit",
        confidence=0.9,
        box=BoundingBox(10, 20, 100, 30),
        box_source=BoxSource.ENGINE,
        source_ref="page=1;block=2;par=1",
        detector="tesseract",
    )

    document = block.as_document()

    assert document["box"] == [10, 20, 100, 30]
    assert document["box_source"] == "engine"
    assert document["detector"] == "tesseract"


def test_a_legacy_engine_is_still_an_engine() -> None:
    assert isinstance(as_legacy_engine(StubProvider()), OcrEngine)


# -- the rule-based extractor -------------------------------------------------------------------


def test_the_rule_based_extractor_cites_the_blocks_it_read() -> None:
    """It cannot tell which region an ad came from, so it cites all of them -- never none."""
    from media_service.jobs.stages import RuleBasedExtractor

    result = OcrResult(
        text="Toyota Prius 2016\nRs. 5,750,000\nKandy",
        blocks=(
            OcrBlock(id=1, text="Toyota Prius 2016", confidence=0.9),
            OcrBlock(id=2, text="Rs. 5,750,000 Kandy", confidence=0.9),
        ),
        mean_confidence=0.9,
    )

    drafts = RuleBasedExtractor().run(result)

    assert len(drafts) == 1
    assert drafts[0].source_block_ids == (1, 2)


def test_the_rule_based_extractor_returns_nothing_for_an_empty_page() -> None:
    """`no_ads` is an outcome, not a failure (AC-003)."""
    from media_service.jobs.stages import RuleBasedExtractor

    assert RuleBasedExtractor().run(OcrResult(text="   ")) == []


def test_ocr_warnings_reach_the_candidate() -> None:
    """A low-confidence page has to stay visible as one all the way to the reviewer."""
    from media_service.jobs.stages import RuleBasedExtractor

    result = OcrResult(
        text="Toyota Prius 2016 Rs. 5,750,000 Kandy",
        blocks=(OcrBlock(id=1, text="Toyota Prius 2016", confidence=0.4),),
        mean_confidence=0.4,
        warnings=("OCR_LOW_CONFIDENCE",),
    )

    draft = RuleBasedExtractor().run(result)[0]

    assert "OCR_LOW_CONFIDENCE" in draft.warning_codes
    assert draft.confidence_label == "low"
