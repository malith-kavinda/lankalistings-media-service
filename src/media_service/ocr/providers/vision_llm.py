"""Reading a page with a vision model, rather than an OCR engine.

The seam exists now; the implementation arrives with Phase 3. That is not an oversight -- this
provider is built on the same schema-agnostic LLM adapters the extraction stage uses, and building
a second HTTP client here would guarantee the two drift apart before they were ever merged.

The name is reserved so `OCR_PROVIDER=vision_llm` is *recognised and refused with a reason*, rather
than rejected as a typo alongside genuinely invalid names. Startup fails, loudly, instead of the
service booting and failing on the first page.

One property of this provider is already fixed by the design and is why `BoxSource` exists from day
one: a vision model returns text, not engine geometry. Its blocks carry `box_source` of
`model_estimate` or `none`, never `engine`, and FR-CAN-006's `crop_needs_review` path keys off
exactly that.
"""

from __future__ import annotations

from typing import Final

from media_service.api.errors import OcrUnavailableError
from media_service.ocr.types import OcrResult

PROVIDER_NAME: Final = "vision_llm"

UNAVAILABLE_REASON: Final = (
    "The vision_llm OCR provider needs the LLM provider layer, which arrives in Phase 3. "
    "Use OCR_PROVIDER=tesseract or paddle_tesseract."
)


class VisionLlmOcrProvider:
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return False

    def availability_reason(self) -> str | None:
        return UNAVAILABLE_REASON

    def extract(self, image_bytes: bytes, *, content_type: str) -> OcrResult:
        raise OcrUnavailableError(UNAVAILABLE_REASON)
