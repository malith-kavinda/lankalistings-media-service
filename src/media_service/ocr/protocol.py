"""The OCR seam.

A `Protocol`, not a base class with `raise NotImplementedError`. The old base class gave no static
checking at all -- a provider that misspelled a method still subclassed cleanly and failed at run
time, on a worker thread, halfway through a batch.

**Sync, deliberately.** Two of the three implementations are CPU or subprocess bound; an async
protocol would force every adapter to carry its own thread plumbing for no gain. Concurrency lives
in exactly one place -- the worker pool that calls this -- so an adapter never thinks about it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from media_service.ocr.types import OcrResult


@runtime_checkable
class OcrProvider(Protocol):
    @property
    def name(self) -> str:
        """The `OCR_PROVIDER` value that selects this implementation."""

    def is_available(self) -> bool: ...

    def availability_reason(self) -> str | None:
        """Why this provider cannot run, or None when it can.

        Returns a reason a person can act on -- "GEMINI_API_KEY is not set", never the key itself.
        This string reaches `/health`, which is unauthenticated.
        """

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult: ...
