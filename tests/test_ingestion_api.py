"""The ingestion endpoints, through the app the service actually assembles.

These go through `create_app`, not through a hand-built stack, so the wiring is under test too: a
route that reads the wrong piece of `app.state`, or a dispatcher that is never given the runner,
fails here rather than in production.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from media_service.config import Settings
from media_service.main import create_app
from tests.support import StubOcrEngine, png_bytes

pytestmark = pytest.mark.usefixtures("engine")


def _files(count: int = 1) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("images", (f"page-{index}.png", png_bytes(width=12 + index), "image/png"))
        for index in range(count)
    ]


def _upload(client: TestClient, count: int = 1, **kwargs):  # type: ignore[no-untyped-def]
    return client.post("/api/v1/ingestion-batches", files=_files(count), **kwargs)


# -- upload ------------------------------------------------------------------------------------


def test_uploading_a_batch_is_accepted_not_completed(api_client) -> None:
    """202 with a Location header: the work is durable, and it has not been done yet."""
    response = _upload(api_client, 3)

    assert response.status_code == 202
    body = response.json()["data"]
    assert response.headers["Location"] == f"/api/v1/ingestion-batches/{body['id']}"
    assert body["status"] == "queued"
    assert body["total_items"] == 3
    assert len(body["items"]) == 3


def test_the_counts_publish_every_key_including_the_zeroes(api_client) -> None:
    body = _upload(api_client).json()["data"]

    assert body["counts"] == {
        "queued": 1,
        "processing": 0,
        "awaiting_review": 0,
        "no_ads": 0,
        "needs_attention": 0,
        "failed": 0,
        "completed": 0,
    }


def test_an_upload_with_no_files_is_refused(api_client) -> None:
    response = api_client.post("/api/v1/ingestion-batches")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


def test_an_unreadable_file_names_itself_in_the_error(api_client) -> None:
    response = api_client.post(
        "/api/v1/ingestion-batches",
        files=[("images", ("notes.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert details[0]["code"] == "UNREADABLE_IMAGE"
    assert "notes.txt" in details[0]["message"]


def test_repeating_a_request_under_one_key_does_not_create_a_second_batch(api_client) -> None:
    headers = {"Idempotency-Key": "key-1"}
    first = api_client.post("/api/v1/ingestion-batches", files=_files(2), headers=headers)
    second = api_client.post("/api/v1/ingestion-batches", files=_files(2), headers=headers)

    assert first.status_code == 202
    # Not 202 again: the second request accepted nothing, it replayed the first answer.
    assert second.status_code == 200
    assert second.json()["data"]["id"] == first.json()["data"]["id"]


def test_reusing_a_key_for_a_different_request_is_a_conflict(api_client) -> None:
    headers = {"Idempotency-Key": "key-1"}
    api_client.post("/api/v1/ingestion-batches", files=_files(2), headers=headers)
    response = api_client.post("/api/v1/ingestion-batches", files=_files(1), headers=headers)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"


# -- progress ----------------------------------------------------------------------------------


def test_progress_reflects_the_pipeline(api_client) -> None:
    batch_id = _upload(api_client, 2).json()["data"]["id"]

    api_client.app.state.dispatcher.run_once()
    body = api_client.get(f"/api/v1/ingestion-batches/{batch_id}").json()["data"]

    assert body["status"] == "completed"
    assert body["counts"]["awaiting_review"] == 2
    assert [item["stage"] for item in body["items"]] == ["review", "review"]
    assert all(item["candidate_count"] == 1 for item in body["items"])


def test_an_unknown_batch_is_a_404_with_a_correlation_id(api_client) -> None:
    response = api_client.get(
        "/api/v1/ingestion-batches/bat_missing", headers={"X-Correlation-Id": "trace-1"}
    )

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "BATCH_NOT_FOUND"
    assert error["correlation_id"] == "trace-1"


def test_batches_can_be_listed(api_client) -> None:
    first = _upload(api_client).json()["data"]["id"]
    second = _upload(api_client).json()["data"]["id"]

    listed = api_client.get("/api/v1/ingestion-batches?limit=10").json()["data"]

    assert [batch["id"] for batch in listed] == [second, first]


def test_an_item_can_be_read_on_its_own(api_client) -> None:
    item_id = _upload(api_client).json()["data"]["items"][0]["id"]

    body = api_client.get(f"/api/v1/ingestion-items/{item_id}").json()["data"]

    assert body["id"] == item_id
    assert body["status"] == "uploaded"
    assert body["stage"] == "queue"


# -- retry -------------------------------------------------------------------------------------


def test_retrying_an_item_that_has_not_failed_is_refused(api_client) -> None:
    item_id = _upload(api_client).json()["data"]["items"][0]["id"]

    response = api_client.post(f"/api/v1/ingestion-items/{item_id}/retry")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NOTHING_TO_RETRY"


def test_an_unknown_retry_mode_is_refused(api_client) -> None:
    item_id = _upload(api_client).json()["data"]["items"][0]["id"]

    response = api_client.post(f"/api/v1/ingestion-items/{item_id}/retry?mode=sideways")

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "mode"


def test_a_finished_item_can_be_reprocessed(api_client) -> None:
    batch = _upload(api_client).json()["data"]
    api_client.app.state.dispatcher.run_once()

    response = api_client.post(
        f"/api/v1/ingestion-items/{batch['items'][0]['id']}/retry?mode=reprocess"
    )

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["mode"] == "reprocess"
    assert body["items"][0]["status"] == "uploaded"
    assert body["items"][0]["pipeline_generation"] == 2


def test_retrying_a_batch_reports_what_it_touched(api_client) -> None:
    batch_id = _upload(api_client, 2).json()["data"]["id"]

    response = api_client.post(f"/api/v1/ingestion-batches/{batch_id}/retry")

    assert response.status_code == 200
    # Nothing failed, so nothing was requeued -- and that is a 200 with an empty list, not an error.
    assert response.json()["data"]["items"] == []


# -- assets ------------------------------------------------------------------------------------


def test_stored_bytes_are_served_with_a_content_addressed_etag(api_client) -> None:
    asset_id = _upload(api_client).json()["data"]["items"][0]["source_asset_id"]

    response = api_client.get(f"/api/v1/media/assets/{asset_id}/original")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["etag"].strip('"') in response.headers["content-disposition"]


def test_an_unchanged_asset_is_not_sent_twice(api_client) -> None:
    asset_id = _upload(api_client).json()["data"]["items"][0]["source_asset_id"]
    etag = api_client.get(f"/api/v1/media/assets/{asset_id}/original").headers["etag"]

    response = api_client.get(
        f"/api/v1/media/assets/{asset_id}/original", headers={"If-None-Match": etag}
    )

    assert response.status_code == 304
    assert response.content == b""


def test_the_preprocessed_input_is_servable_once_it_exists(api_client) -> None:
    asset_id = _upload(api_client).json()["data"]["items"][0]["source_asset_id"]
    assert api_client.get(f"/api/v1/media/assets/{asset_id}/ocr_input").status_code == 404

    api_client.app.state.dispatcher.run_once()

    assert api_client.get(f"/api/v1/media/assets/{asset_id}/ocr_input").status_code == 200


def test_an_invented_representation_is_not_found(api_client) -> None:
    asset_id = _upload(api_client).json()["data"]["items"][0]["source_asset_id"]

    response = api_client.get(f"/api/v1/media/assets/{asset_id}/../../etc/passwd")

    assert response.status_code == 404


# -- authentication ----------------------------------------------------------------------------


@pytest.fixture
def secured_client(api_settings, unit_of_work, asset_store):  # type: ignore[no-untyped-def]
    settings = api_settings.model_copy(
        update={"operator_auth_mode": "static_token", "operator_api_token": "s3cret"}
    )
    app = create_app(
        settings=settings,
        ocr_engine=StubOcrEngine(),
        unit_of_work=unit_of_work,
        store=asset_store,
    )
    return TestClient(app)


def test_an_unauthenticated_request_is_401(secured_client) -> None:
    response = _upload(secured_client)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_a_wrong_token_is_403_not_401(secured_client) -> None:
    """The distinction matters: one says identify yourself, the other says you did and it failed."""
    response = _upload(secured_client, headers={"Authorization": "Bearer wrong"})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_a_valid_token_is_accepted_and_names_the_operator(secured_client, unit_of_work) -> None:
    response = _upload(
        secured_client,
        headers={"Authorization": "Bearer s3cret", "X-Operator-Id": "moderator-7"},
    )

    assert response.status_code == 202
    assert response.json()["data"]["created_by"] == "moderator-7"


# -- startup -----------------------------------------------------------------------------------


def test_a_misspelled_switch_stops_the_service_starting() -> None:
    """An unknown mode must not fall back to a default -- for auth, that would fail open."""
    with pytest.raises(ValueError, match="OPERATOR_AUTH_MODE"):
        create_app(settings=Settings(operator_auth_mode="noen"))


def test_a_static_token_mode_without_a_token_stops_the_service_starting() -> None:
    with pytest.raises(ValueError, match="OPERATOR_API_TOKEN"):
        create_app(settings=Settings(operator_auth_mode="static_token"))


def test_open_access_is_refused_outside_local_and_test() -> None:
    with pytest.raises(ValueError, match="only allowed in local and test"):
        create_app(settings=Settings(environment="production", operator_auth_mode="none"))


# -- health ------------------------------------------------------------------------------------


def test_health_names_the_configured_provider(api_client) -> None:
    """"OCR is available" means something different per provider, so say which one is running."""
    body = api_client.get("/health").json()["data"]

    assert body["ocr_provider"] == "legacy"
    assert body["ocr_available"] is True
    assert body["ocr_unavailable_reason"] is None


def test_the_service_publishes_the_limits_it_enforces(api_client) -> None:
    """So the portal's pre-flight check and the server's answer stay one rule, not two copies."""
    response = api_client.get("/api/v1/ingestion-limits")

    assert response.status_code == 200
    limits = response.json()["data"]
    assert limits["max_images_per_batch"] > 0
    assert limits["max_image_bytes"] > 0
    assert "image/png" in limits["supported_content_types"]
