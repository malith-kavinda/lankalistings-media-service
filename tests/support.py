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
