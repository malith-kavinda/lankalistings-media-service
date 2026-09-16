"""Test doubles and helpers shared across the ingestion tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from media_service.services.uploads import IncomingFile


@dataclass
class RecordingDispatcher:
    """A dispatcher that remembers what it was asked to do and does none of it.

    Lets a test assert the *contract* -- that dispatch happens once, after the commit, with the
    right ids -- without a thread pool making the assertion racy.
    """

    items: list[str] = field(default_factory=list)
    batches: list[tuple[str, list[str]]] = field(default_factory=list)
    started: int = 0
    stopped: int = 0

    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        self.items.append(item_id)

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        self.batches.append((batch_id, list(item_ids)))

    def start(self) -> None:
        self.started += 1

    def stop(self, *, timeout: float = 30.0) -> None:
        self.stopped += 1


def png_bytes(*, width: int = 12, height: int = 8, colour: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(
    payload: bytes,
    *,
    filename: str = "page.png",
    declared_content_type: str | None = "image/png",
) -> IncomingFile:
    return IncomingFile(
        filename=filename,
        declared_content_type=declared_content_type,
        stream=BytesIO(payload),
    )


def uploads(count: int, *, prefix: str = "page") -> list[IncomingFile]:
    """`count` visually distinct images, so none of them deduplicate against another."""
    return [
        upload(
            png_bytes(width=12 + index, height=8 + index),
            filename=f"{prefix}-{index}.png",
        )
        for index in range(count)
    ]


@dataclass
class FakeOcrStep:
    """An OCR stage whose output and failures a test chooses, and that counts its calls.

    The call count is the assertion that matters for resume: "this ran once across two attempts" is
    the whole point of keeping the artifacts.
    """

    text: str = "Ocean View Apartment Colombo 05 Rs. 4,500,000"
    fail_with: object | None = None
    calls: int = 0

    def run(self, image_bytes: bytes, *, content_type: str):  # type: ignore[no-untyped-def]
        from media_service.jobs.stages import OcrResult

        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with

        return OcrResult(
            text=self.text,
            engine="fake",
            engine_version="fake/1.0",
            languages="sin+eng",
            confidence_label="high",
            mean_confidence=0.94,
            width=12,
            height=8,
            duration_ms=4,
        )


@dataclass
class CountingExtractor:
    """Produces a fixed number of candidates and records how often it was asked."""

    count: int = 1
    calls: int = 0

    def run(self, result):  # type: ignore[no-untyped-def]
        from media_service.domain.listings import CandidateDraft

        self.calls += 1
        return [
            CandidateDraft(
                index=index,
                title=f"Advertisement {index}",
                description="Extracted body text",
                category="property",
                location="Colombo",
                price="Rs. 4,500,000",
                source_text=result.text,
            )
            for index in range(self.count)
        ]


@dataclass
class CountingPreprocessor:
    calls: int = 0

    def run(self, image_bytes: bytes):  # type: ignore[no-untyped-def]
        from media_service.jobs.stages import PillowPreprocessor

        self.calls += 1
        return PillowPreprocessor().run(image_bytes)


@dataclass
class Pipeline:
    """Everything wired together, with the fakes still reachable for assertions."""

    service: object
    dispatcher: object
    runner: object
    preprocess: CountingPreprocessor
    ocr: FakeOcrStep
    extraction: CountingExtractor
    unit_of_work: object

    def run(self) -> int:
        return self.dispatcher.run_once()  # type: ignore[attr-defined]


def build_pipeline(*, unit_of_work, store, settings, ocr=None, extraction=None):  # type: ignore[no-untyped-def]
    from media_service.domain.listings import LocalListingGateway
    from media_service.jobs.dispatchers import ManualDispatcher
    from media_service.jobs.runner import ItemRunner
    from media_service.services.ingestion import IngestionService

    preprocess = CountingPreprocessor()
    ocr_step = ocr or FakeOcrStep()
    extraction_step = extraction or CountingExtractor()

    runner = ItemRunner(
        unit_of_work=unit_of_work,
        store=store,
        settings=settings,
        preprocess=preprocess,
        ocr=ocr_step,
        extraction=extraction_step,
        gateway=LocalListingGateway(),
    )
    dispatcher = ManualDispatcher(
        unit_of_work=unit_of_work, runner=runner, lease_seconds=settings.lease_seconds
    )
    service = IngestionService(
        unit_of_work=unit_of_work, store=store, settings=settings, dispatcher=dispatcher
    )
    return Pipeline(
        service=service,
        dispatcher=dispatcher,
        runner=runner,
        preprocess=preprocess,
        ocr=ocr_step,
        extraction=extraction_step,
        unit_of_work=unit_of_work,
    )
