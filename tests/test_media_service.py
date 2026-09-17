import base64
from dataclasses import dataclass

from fastapi.testclient import TestClient

from media_service.config import Settings
from media_service.domain.repository import InMemoryMediaRepository, JsonMediaRepository
from media_service.main import create_app
from media_service.services.ocr import OcrEngine, OcrOutput

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@dataclass
class FakeOcrEngine(OcrEngine):
    available: bool = True
    reason: str | None = None
    text: str = "Ocean View Apartment Colombo"

    def is_available(self) -> bool:
        return self.available

    def availability_reason(self) -> str | None:
        return self.reason

    def extract_text(self, image_bytes: bytes, *, content_type: str) -> OcrOutput:
        if not self.available:
            from media_service.api.errors import OcrUnavailableError

            raise OcrUnavailableError(self.reason or "OCR unavailable in test.")

        return OcrOutput(
            text=self.text,
            engine="fake",
            model_version="fake-test",
            language="eng",
            confidence="high",
            processing_ms=3,
            width=1,
            height=1,
        )


def make_client(engine: OcrEngine) -> TestClient:
    app = create_app(
        settings=Settings(max_upload_bytes=1024),
        ocr_engine=engine,
        repository=InMemoryMediaRepository(),
    )
    return TestClient(app)


def test_health_reports_ocr_status() -> None:
    client = make_client(FakeOcrEngine(available=False, reason="Tesseract is missing."))

    response = client.get("/health", headers={"X-Correlation-Id": "test-health"})

    assert response.status_code == 200
    assert response.json()["data"]["service"] == "media-service"
    assert response.json()["data"]["ocr_available"] is False
    assert response.json()["data"]["ocr_unavailable_reason"] == "Tesseract is missing."


def test_extract_rejects_missing_file() -> None:
    client = make_client(FakeOcrEngine())

    response = client.post("/api/v1/media/extract", headers={"X-Correlation-Id": "test-missing"})

    body = response.json()
    assert response.status_code == 422
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert body["error"]["correlation_id"] == "test-missing"


def test_extract_returns_ocr_unavailable_error() -> None:
    client = make_client(
        FakeOcrEngine(available=False, reason="The Tesseract binary is not installed.")
    )

    response = client.post(
        "/api/v1/media/extract",
        headers={"X-Correlation-Id": "test-unavailable"},
        files={"file": ("flyer.png", PNG_1X1, "image/png")},
    )

    body = response.json()
    assert response.status_code == 503
    assert body["data"] is None
    assert body["error"]["code"] == "OCR_UNAVAILABLE"
    assert body["error"]["correlation_id"] == "test-unavailable"
    assert body["error"]["details"][0]["code"] == "OCR_RUNTIME_MISSING"


def test_extract_returns_detected_text_and_metadata() -> None:
    client = make_client(FakeOcrEngine(text="Rs. 45,000,000\nColombo 07"))

    response = client.post(
        "/api/v1/media/extract",
        headers={"X-Correlation-Id": "test-success"},
        files={"file": ("flyer.png", PNG_1X1, "image/png")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["error"] is None
    assert body["data"]["detected_text"] == "Rs. 45,000,000\nColombo 07"
    assert body["data"]["asset"]["content_type"] == "image/png"
    assert body["data"]["asset"]["byte_size"] == len(PNG_1X1)
    assert body["data"]["asset"]["checksum"]
    assert body["data"]["metadata"]["engine"] == "fake"


def test_create_advertisement_accepts_image_upload() -> None:
    client = make_client(FakeOcrEngine())

    response = client.post(
        "/api/v1/advertisements",
        data={
            "title": "Honda Civic 2018 EX",
            "price": "Rs. 8,500,000",
            "category": "Vehicles",
            "location": "Colombo 03",
            "description": "Clean vehicle with full service records.",
        },
        files={"image": ("honda.png", PNG_1X1, "image/png")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["error"] is None
    assert body["data"]["title"] == "Honda Civic 2018 EX"
    assert body["data"]["status"] == "active"
    assert body["data"]["image_url"].startswith("data:image/png;base64,")


def test_list_advertisements_returns_newest_first() -> None:
    client = make_client(FakeOcrEngine())

    client.post(
        "/api/v1/advertisements",
        data={
            "title": "First ad",
            "price": "Rs. 100,000",
            "category": "Electronics",
            "location": "Galle",
            "description": "First description.",
        },
        files={"image": ("first.png", PNG_1X1, "image/png")},
    )
    client.post(
        "/api/v1/advertisements",
        data={
            "title": "Second ad",
            "price": "Rs. 200,000",
            "category": "Property",
            "location": "Kandy",
            "description": "Second description.",
        },
        files={"image": ("second.png", PNG_1X1, "image/png")},
    )

    response = client.get("/api/v1/advertisements")

    body = response.json()
    assert response.status_code == 200
    assert [advertisement["title"] for advertisement in body["data"]] == ["Second ad", "First ad"]


def test_create_advertisement_rejects_invalid_image() -> None:
    client = make_client(FakeOcrEngine())

    response = client.post(
        "/api/v1/advertisements",
        headers={"X-Correlation-Id": "test-invalid-ad-image"},
        data={
            "title": "Broken image ad",
            "price": "Rs. 100,000",
            "category": "Electronics",
            "location": "Galle",
            "description": "This image payload is not decodable.",
        },
        files={"image": ("broken.png", b"not-an-image", "image/png")},
    )

    body = response.json()
    assert response.status_code == 422
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_FAILED"
    assert body["error"]["details"][0]["code"] == "INVALID_IMAGE"


def test_extract_newspaper_article_creates_pending_review_draft() -> None:
    client = make_client(
        FakeOcrEngine(
            text=(
                "Toyota Prius 2016\n"
                "මිල රු 8500000\n"
                "Colombo 03\n"
                "ඉතා හොඳ තත්වයේ වාහනයක්. Call 0771234567"
            )
        )
    )

    response = client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("article.png", PNG_1X1, "image/png")},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["error"] is None
    assert body["data"]["detected_text"].startswith("Toyota Prius")
    assert body["data"]["advertisement"]["title"] == "Toyota Prius 2016"
    assert body["data"]["advertisement"]["price"] == "රු 8500000"
    assert body["data"]["advertisement"]["category"] == "Vehicles"
    assert body["data"]["advertisement"]["location"] == "Colombo 03"
    assert body["data"]["advertisement"]["status"] == "pending_review"


def test_review_queue_lists_only_pending_review_advertisements() -> None:
    client = make_client(FakeOcrEngine(text="House for rent\nRs. 120000\nKandy"))
    create_response = client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("article.png", PNG_1X1, "image/png")},
    )
    pending_id = create_response.json()["data"]["advertisement"]["id"]
    client.post(f"/api/v1/legacy/advertisements/{pending_id}/approve")
    client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("second.png", PNG_1X1, "image/png")},
    )

    response = client.get("/api/v1/legacy/advertisements/review")

    body = response.json()
    assert response.status_code == 200
    assert len(body["data"]) == 1
    assert body["data"][0]["status"] == "pending_review"


def test_update_then_approve_draft_publishes_to_public_feed() -> None:
    client = make_client(FakeOcrEngine(text="Phone for sale\nRs 265000\nGalle"))
    create_response = client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("article.png", PNG_1X1, "image/png")},
    )
    advertisement_id = create_response.json()["data"]["advertisement"]["id"]

    update_response = client.patch(
        f"/api/v1/legacy/advertisements/{advertisement_id}",
        json={
            "title": "iPhone 14 Pro 256GB",
            "price": "Rs. 265,000",
            "category": "Electronics",
            "location": "Galle",
            "description": "Edited by reviewer before approval.",
        },
    )
    approve_response = client.post(f"/api/v1/legacy/advertisements/{advertisement_id}/approve")
    public_response = client.get("/api/v1/advertisements")

    assert update_response.status_code == 200
    assert approve_response.status_code == 200
    assert approve_response.json()["data"]["status"] == "active"
    assert public_response.status_code == 200
    assert public_response.json()["data"][0]["title"] == "iPhone 14 Pro 256GB"


def test_public_feed_excludes_pending_review_advertisements() -> None:
    client = make_client(FakeOcrEngine(text="Pending listing\nRs 100000\nColombo"))
    client.post(
        "/api/v1/newspaper-articles/extract",
        files={"image": ("article.png", PNG_1X1, "image/png")},
    )

    response = client.get("/api/v1/advertisements")

    assert response.status_code == 200
    assert response.json()["data"] == []


def test_get_extraction_returns_previous_result() -> None:
    client = make_client(FakeOcrEngine(text="Luxury villa Galle"))
    create_response = client.post(
        "/api/v1/media/extract",
        files={"file": ("flyer.png", PNG_1X1, "image/png")},
    )
    extraction_id = create_response.json()["data"]["extraction_id"]

    response = client.get(
        f"/api/v1/media/extractions/{extraction_id}",
        headers={"X-Correlation-Id": "test-get"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["error"] is None
    assert body["data"]["extraction_id"] == extraction_id
    assert body["data"]["detected_text"] == "Luxury villa Galle"
    assert body["data"]["metadata"]["model_version"] == "fake-test"


def test_get_extraction_returns_not_found_envelope() -> None:
    client = make_client(FakeOcrEngine())

    response = client.get(
        "/api/v1/media/extractions/missing",
        headers={"X-Correlation-Id": "test-not-found"},
    )

    body = response.json()
    assert response.status_code == 404
    assert body["data"] is None
    assert body["error"]["code"] == "EXTRACTION_NOT_FOUND"
    assert body["error"]["correlation_id"] == "test-not-found"


def test_json_repository_persists_extraction_between_app_instances(tmp_path) -> None:
    metadata_path = tmp_path / "media_metadata.json"
    first_app = create_app(
        settings=Settings(max_upload_bytes=1024, metadata_path=metadata_path),
        ocr_engine=FakeOcrEngine(text="Persisted OCR text"),
        repository=JsonMediaRepository(metadata_path),
    )
    first_client = TestClient(first_app)
    create_response = first_client.post(
        "/api/v1/media/extract",
        files={"file": ("flyer.png", PNG_1X1, "image/png")},
    )
    extraction_id = create_response.json()["data"]["extraction_id"]

    second_app = create_app(
        settings=Settings(max_upload_bytes=1024, metadata_path=metadata_path),
        ocr_engine=FakeOcrEngine(),
        repository=JsonMediaRepository(metadata_path),
    )
    second_client = TestClient(second_app)
    response = second_client.get(f"/api/v1/media/extractions/{extraction_id}")

    assert response.status_code == 200
    assert response.json()["data"]["detected_text"] == "Persisted OCR text"


def test_json_repository_persists_advertisements_between_app_instances(tmp_path) -> None:
    metadata_path = tmp_path / "media_metadata.json"
    first_app = create_app(
        settings=Settings(max_upload_bytes=1024, metadata_path=metadata_path),
        ocr_engine=FakeOcrEngine(),
        repository=JsonMediaRepository(metadata_path),
    )
    first_client = TestClient(first_app)
    first_client.post(
        "/api/v1/advertisements",
        data={
            "title": "Persisted listing",
            "price": "Rs. 12,000,000",
            "category": "Property",
            "location": "Colombo",
            "description": "Saved through JSON repository.",
        },
        files={"image": ("persisted.png", PNG_1X1, "image/png")},
    )

    second_app = create_app(
        settings=Settings(max_upload_bytes=1024, metadata_path=metadata_path),
        ocr_engine=FakeOcrEngine(),
        repository=JsonMediaRepository(metadata_path),
    )
    second_client = TestClient(second_app)
    response = second_client.get("/api/v1/advertisements")

    assert response.status_code == 200
    assert response.json()["data"][0]["title"] == "Persisted listing"
