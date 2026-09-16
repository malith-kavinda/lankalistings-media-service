"""Adapters between the prototype's `OcrEngine` and the provider protocol.

Two shapes have to coexist for one more phase. The prototype's endpoints -- and the tests that pin
their behaviour -- speak `OcrEngine`, which returns a string and a confidence *word*. The pipeline
speaks `OcrProvider`, which returns blocks with geometry and a number.

These adapters are what let the new pipeline be exercised with the existing `FakeOcrEngine` without
editing a single prototype test, and what let the prototype endpoints keep working while the real
provider does the reading underneath. Phase 4 deletes both directions along with the endpoints.

The lossy direction is `as_legacy_engine`, and it is lossy on purpose: geometry and per-block
confidence have nowhere to go in a shape that predates them. The lossless direction,
`as_provider`, invents nothing -- a legacy engine has no boxes, so its single block carries
`box_source=none` rather than a fabricated rectangle covering the page.
"""

from __future__ import annotations

from media_service.ocr import quality
from media_service.ocr.protocol import OcrProvider
from media_service.ocr.types import BoxSource, OcrBlock, OcrResult
from media_service.services.ocr import OcrEngine, OcrOutput

# The prototype's three-valued confidence, mapped onto the numeric scale at the boundary between
# them. The thresholds are the midpoints of what each word was used to mean.
CONFIDENCE_WORDS: dict[str, float] = {"high": 0.9, "medium": 0.75, "low": 0.4}
DEFAULT_WORD_CONFIDENCE = 0.75


def word_to_confidence(word: str) -> float:
    return CONFIDENCE_WORDS.get(word.lower(), DEFAULT_WORD_CONFIDENCE)


def confidence_to_word(value: float | None) -> str:
    if value is None:
        return "low"
    if value >= 0.85:
        return "high"
    if value >= 0.60:
        return "medium"
    return "low"


class LegacyEngineProvider:
    """An `OcrEngine` presented as an `OcrProvider`."""

    def __init__(self, engine: OcrEngine, *, name: str = "legacy") -> None:
        self._engine = engine
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._engine.is_available()

    def availability_reason(self) -> str | None:
        return self._engine.availability_reason()

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        output = self._engine.extract_text(image_bytes, content_type=content_type)
        confidence = word_to_confidence(output.confidence)
        text = output.text.strip()

        # One block for the whole page, with no box. A legacy engine cannot say where anything was,
        # and a synthetic full-page rectangle would look like evidence while being none.
        blocks = (
            (
                OcrBlock(
                    id=1,
                    text=text,
                    confidence=confidence,
                    box=None,
                    box_source=BoxSource.NONE,
                    source_ref=f"engine={output.engine}",
                    detector=None,
                ),
            )
            if text
            else ()
        )

        return OcrResult(
            text=text,
            blocks=blocks,
            provider=self._name,
            engine=output.engine,
            engine_version=output.model_version,
            languages=output.language,
            mean_confidence=confidence if text else None,
            low_confidence=quality.is_low_confidence(confidence) if text else False,
            width=output.width,
            height=output.height,
            duration_ms=output.processing_ms,
            warnings=quality.warnings_for(
                text=text, confidence=confidence if text else None
            ),
        )


class ProviderLegacyEngine(OcrEngine):
    """An `OcrProvider` presented as an `OcrEngine`, for the prototype's endpoints."""

    def __init__(self, provider: OcrProvider) -> None:
        self._provider = provider

    def is_available(self) -> bool:
        return self._provider.is_available()

    def availability_reason(self) -> str | None:
        return self._provider.availability_reason()

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        result = self._provider.extract(image_bytes, content_type=content_type)
        return OcrOutput(
            text=result.text,
            engine=result.engine,
            model_version=result.engine_version or "unknown",
            language=result.languages,
            confidence=confidence_to_word(result.mean_confidence),
            processing_ms=result.duration_ms,
            width=result.width,
            height=result.height,
        )


def is_legacy_engine(candidate: object) -> bool:
    """Structural, not `isinstance(candidate, OcrEngine)`.

    A nominal check would wrap any engine that satisfies the interface without inheriting from it --
    every test double, for a start -- and then a test would be exercising two adapters stacked on
    each other instead of the thing it meant to test. `extract_text` is the method that
    distinguishes the two shapes.
    """
    return hasattr(candidate, "extract_text")


def as_provider(engine: OcrEngine | OcrProvider, *, name: str = "legacy") -> OcrProvider:
    """Accept either shape and return a provider, wrapping only when needed."""
    if not is_legacy_engine(engine):
        return engine  # type: ignore[return-value]
    return LegacyEngineProvider(engine, name=name)  # type: ignore[arg-type]


def as_legacy_engine(provider: OcrProvider | OcrEngine) -> OcrEngine:
    if is_legacy_engine(provider):
        return provider  # type: ignore[return-value]
    return ProviderLegacyEngine(provider)  # type: ignore[arg-type]
