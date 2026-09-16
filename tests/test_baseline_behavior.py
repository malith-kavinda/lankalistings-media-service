"""Pin the prototype's behaviour before the pipeline replaces it (PRD 19, Phase 0).

These tests describe what the service does *today*, including the parts that are wrong. They exist
so the change is visible: when Phases 1-3 land, each of these either still passes because
compatibility was preserved on purpose, or fails loudly and is rewritten in the same commit that
changes the behaviour.

Several assertions here encode known defects. Each is marked so it is not mistaken for desired
behaviour.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from fastapi.testclient import TestClient

from media_service.config import Settings
from media_service.domain.repository import InMemoryMediaRepository
from media_service.main import create_app
from media_service.services.ocr import OcrEngine, OcrOutput

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@dataclass
class StubOcrEngine(OcrEngine):
    text: str = ""

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str | None:
        return None

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        return OcrOutput(
            text=self.text,
            engine="stub",
            model_version="stub",
            language="sin+eng",
            confidence="medium",
            processing_ms=1,
            width=1,
            height=1,
        )


def client_for(text: str) -> TestClient:
    return TestClient(
        create_app(
            settings=Settings(max_upload_bytes=1024 * 1024),
            ocr_engine=StubOcrEngine(text=text),
            repository=InMemoryMediaRepository(),
        )
    )


THREE_ADS = (
    "Honda Fit 2014\nRs. 5,750,000\nKandy\n0812233445\n"
    "Toyota Axio 2017\nRs. 9,100,000\nGalle\n0771234567\n"
    "Accounts Clerk vacancy\nColombo 07\n0114567890"
)


def test_baseline_creates_exactly_one_advertisement_from_a_three_advertisement_page() -> None:
    """KNOWN DEFECT. This is the core reason the pipeline is being built.

    A page holding three unrelated advertisements collapses into a single listing whose
    description is the whole page. Phase 3 must turn this into three independent candidates
    (AC-002).
    """
    response = client_for(THREE_ADS).post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    advertisement = response.json()["data"]["advertisement"]

    assert advertisement["title"] == "Honda Fit 2014"
    assert "Toyota Axio" in advertisement["description"]
    assert "Accounts Clerk" in advertisement["description"]


def test_baseline_returns_a_single_object_not_a_collection() -> None:
    """The response shape itself cannot express more than one advertisement."""
    response = client_for(THREE_ADS).post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    data = response.json()["data"]
    assert isinstance(data["advertisement"], dict)
    assert "advertisements" not in data


def test_baseline_embeds_the_source_image_as_a_base64_data_url() -> None:
    """KNOWN DEFECT. Invariant 12 wants the bytes stored once and referenced.

    Today every advertisement carries its own copy of the full image inline, so a review queue
    response grows with the number of candidates.
    """
    response = client_for("House for sale\nRs. 100,000\nKandy").post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    image_url = response.json()["data"]["advertisement"]["image_url"]
    assert image_url.startswith("data:image/png;base64,")


def test_baseline_never_produces_zero_advertisements() -> None:
    """KNOWN DEFECT. AC-003 requires a no-advertisement page to create nothing.

    Today even a weather report becomes a pending draft, so a moderator must reject content that
    should never have been a candidate.
    """
    response = client_for("Showers are expected in the Western province this afternoon.").post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["data"]["advertisement"]["status"] == "pending_review"


def test_baseline_uses_a_category_vocabulary_that_omits_other() -> None:
    """KNOWN DEFECT, resolved in PRD v1.1 section 11.7.

    Nothing matches the heuristic keyword lists here, so the advertisement is filed as `Home` -- a
    real category -- rather than `Other`. FR-LLM-011 requires the unmatched case to be visible.
    """
    response = client_for("Antique brass telescope, working condition\nRs. 40,000").post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    assert response.json()["data"]["advertisement"]["category"] == "Home"


def test_baseline_exposes_confidence_as_a_string_bucket() -> None:
    """Wire compatibility: the portal reads this string.

    FR-OCR-009 adds a numeric field alongside it.
    """
    response = client_for("Phone for sale\nRs. 100,000").post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    confidence = response.json()["data"]["advertisement"]["extraction_confidence"]
    assert isinstance(confidence, str)
    assert confidence in {"low", "medium", "high"}


def test_baseline_pending_review_is_the_wire_status() -> None:
    """PRD v1.1 makes `pending` canonical and keeps `pending_review` as the wire alias."""
    response = client_for("Land for sale\nRs. 2,000,000\nMatara").post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    assert response.json()["data"]["advertisement"]["status"] == "pending_review"


def test_the_batch_endpoints_now_exist_alongside_the_single_ad_path() -> None:
    """Phase 1 filled the gap this file recorded, without moving the prototype's endpoints.

    The assertion is only that the routes are mounted -- their behaviour belongs to the ingestion
    tests. What matters here is that adding them left the single-ad path exactly as it was, which
    every other test in this file is still checking.
    """
    client = client_for("")

    # 422 rather than 404: the route exists and is refusing a request that carries no files.
    # Deliberately the only call made here -- reading a batch would need a database, and this file
    # is about the prototype, which never had one.
    assert client.post("/api/v1/ingestion-batches").status_code == 422


def test_baseline_review_queue_has_no_filters_or_pagination() -> None:
    """PRD 13.2 requires batch, status, warning, category, and confidence filters."""
    client = client_for("Bicycle for sale\nRs. 25,000\nGalle")
    client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("page.png", PNG_1X1, "image/png")},
    )

    response = client.get("/api/v1/advertisements/review?category=vehicles&limit=1")

    assert response.status_code == 200
    body = response.json()["data"]
    assert isinstance(body, list)
    assert len(body) == 1, "Query parameters are accepted but ignored today"
