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
    """An OCR provider whose output and failures a test chooses, counting its calls.

    The call count is the assertion that matters for resume: "this ran once across two attempts" is
    the whole point of keeping the artifacts.
    """

    text: str = "Ocean View Apartment Colombo 05 Rs. 4,500,000"
    fail_with: object | None = None
    calls: int = 0
    blocks_per_page: int = 2

    @property
    def name(self) -> str:
        return "fake"

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str | None:
        return None

    def extract(self, image_bytes: bytes, *, content_type: str):  # type: ignore[no-untyped-def]
        from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrResult

        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with

        lines = self.text.split(" ") if self.text else []
        chunk = max(len(lines) // max(self.blocks_per_page, 1), 1)
        blocks = tuple(
            OcrBlock(
                id=index + 1,
                text=" ".join(lines[index * chunk : (index + 1) * chunk]),
                confidence=0.94,
                box=BoundingBox(0, index * 20, 100, 18),
                box_source=BoxSource.ENGINE,
                source_ref=f"page=1;block={index + 1};par=1",
                detector="fake",
            )
            for index in range(self.blocks_per_page)
            if " ".join(lines[index * chunk : (index + 1) * chunk])
        )

        return OcrResult(
            text=self.text,
            blocks=blocks,
            provider="fake",
            engine="fake",
            engine_version="fake/1.0",
            languages="sin+eng",
            mean_confidence=0.94,
            low_confidence=False,
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
        # Cites the blocks it was given, like the real extractors do. A double that produced
        # uncited candidates would let a regression in citation handling pass unnoticed.
        cited = tuple(block.id for block in result.blocks)
        return [
            CandidateDraft(
                index=index,
                title=f"Advertisement {index}",
                description="Extracted body text",
                category="property",
                location="Colombo",
                price="Rs. 4,500,000",
                source_text=result.text,
                source_block_ids=cited,
            )
            for index in range(self.count)
        ]


@dataclass
class CountingPreprocessor:
    """The real preprocessor, counting how often it actually ran."""

    calls: int = 0

    @property
    def version(self) -> str:
        from media_service.ocr.preprocess import ImagePreprocessor

        return ImagePreprocessor().version

    def params(self) -> dict:
        from media_service.ocr.preprocess import ImagePreprocessor

        return ImagePreprocessor().params()

    def run(self, image_bytes: bytes):  # type: ignore[no-untyped-def]
        from media_service.ocr.preprocess import ImagePreprocessor

        self.calls += 1
        return ImagePreprocessor().run(image_bytes)


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


@dataclass
class StubOcrEngine:
    """An `OcrEngine` for tests that go through `create_app`.

    Separate from the one in `test_media_service`, which pins the prototype's behaviour and is
    deliberately left untouched.
    """

    text: str = "Ocean View Apartment Colombo 05 Rs. 4,500,000"
    available: bool = True
    reason: str | None = None

    def is_available(self) -> bool:
        return self.available

    def availability_reason(self) -> str | None:
        return self.reason

    def extract_text(self, image_bytes: bytes, *, content_type: str):  # type: ignore[no-untyped-def]
        from media_service.services.ocr import OcrOutput

        return OcrOutput(
            text=self.text,
            engine="stub",
            model_version="stub/1.0",
            language="sin+eng",
            confidence="high",
            processing_ms=2,
            width=12,
            height=8,
        )
