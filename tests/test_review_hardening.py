"""The review gate under adversarial and concurrent conditions.

Separate from `test_review_api.py` because these are not "does the feature work" tests. Each one
here corresponds to a way the gate was, or could be, got around: a second endpoint that publishes
without reviewing, two requests racing for the same decision, an unbounded field, a header that
reaches a log line intact.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from media_service.db.tables import Advertisement, AdvertisementProvenance, MediaAsset, ReviewEvent
from media_service.domain.listings import LocalListingGateway, ReviewerEdits
from media_service.services.review import ReviewService
from tests.factories import make_candidate, seed_batch_with_items

pytestmark = pytest.mark.usefixtures("engine")


@pytest.fixture
def seeded(session):  # type: ignore[no-untyped-def]
    batch, items = seed_batch_with_items(session, count=1)
    item = items[0]
    asset = session.get(MediaAsset, item.source_asset_id)
    advertisement, provenance = make_candidate(session, batch=batch, item=item, asset=asset)
    return batch, item, advertisement, provenance


def _app(api_settings, unit_of_work, asset_store, **overrides):  # type: ignore[no-untyped-def]
    from media_service.main import create_app
    from tests.support import StubOcrEngine

    return create_app(
        settings=api_settings.model_copy(update=overrides),
        ocr_engine=StubOcrEngine(),
        unit_of_work=unit_of_work,
        store=asset_store,
    )


# -- the gate cannot be walked around --------------------------------------------------------


def test_the_prototype_approve_route_cannot_publish_a_pipeline_candidate(
    api_settings, unit_of_work, asset_store, session, seeded
) -> None:
    """The bypass this guard exists for.

    `MEDIA_REPOSITORY=sql` puts the prototype's endpoints on the same `advertisements` table the
    review queue reads. Without the guard, its approve route sets `status=active` with no field
    validation, no version check, no row lock and no audit event -- a second way to publish a
    candidate, which makes invariant 2 a convention rather than a guarantee.
    """
    _, _, advertisement, _ = seeded
    client = TestClient(_app(api_settings, unit_of_work, asset_store, media_repository="sql"))

    response = client.post(f"/api/v1/legacy/advertisements/{advertisement.id}/approve")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "REVIEW_REQUIRED"

    session.expire_all()
    assert session.get(Advertisement, advertisement.id).status == "pending"


def test_the_prototype_edit_route_cannot_touch_a_pipeline_candidate(
    api_settings, unit_of_work, asset_store, session, seeded
) -> None:
    _, _, advertisement, _ = seeded
    client = TestClient(_app(api_settings, unit_of_work, asset_store, media_repository="sql"))

    response = client.patch(
        f"/api/v1/legacy/advertisements/{advertisement.id}",
        json={
            "title": "Rewritten without review",
            "price": "",
            "category": "vehicles",
            "location": "",
            "description": "",
        },
    )

    assert response.status_code == 409
    session.expire_all()
    assert session.get(Advertisement, advertisement.id).title == "Honda Fit 2014"


def test_the_prototype_write_routes_require_an_operator(
    api_settings, unit_of_work, asset_store, seeded
) -> None:
    """They had no auth at all, while every review route beside them did."""
    _, _, advertisement, _ = seeded
    client = TestClient(
        _app(
            api_settings,
            unit_of_work,
            asset_store,
            media_repository="sql",
            operator_auth_mode="static_token",
            operator_api_token="s3cret",
            environment="test",
        )
    )

    unauthenticated = [
        client.post(f"/api/v1/legacy/advertisements/{advertisement.id}/approve"),
        client.get("/api/v1/legacy/advertisements/review"),
        client.post(
            "/api/v1/newspaper-articles/extract",
            files={"image": ("p.png", b"x", "image/png")},
        ),
    ]

    assert [response.status_code for response in unauthenticated] == [401, 401, 401]


def test_the_review_routes_require_an_operator(
    api_settings, unit_of_work, asset_store, seeded
) -> None:
    _, _, advertisement, _ = seeded
    client = TestClient(
        _app(
            api_settings,
            unit_of_work,
            asset_store,
            operator_auth_mode="static_token",
            operator_api_token="s3cret",
            environment="test",
        )
    )

    responses = [
        client.get("/api/v1/advertisements/review"),
        client.get(f"/api/v1/advertisements/{advertisement.id}"),
        client.patch(f"/api/v1/advertisements/{advertisement.id}", json={"title": "x"}),
        client.post(f"/api/v1/advertisements/{advertisement.id}/approve", json={}),
        client.post(
            f"/api/v1/advertisements/{advertisement.id}/reject",
            json={"reason_code": "duplicate"},
        ),
        client.get("/api/v1/advertisements/rejection-reasons"),
        client.get("/api/v1/categories"),
    ]

    assert {response.status_code for response in responses} == {401}


# -- concurrency -------------------------------------------------------------------------------


def test_two_simultaneous_approvals_produce_one_winner(
    unit_of_work, session, seeded
) -> None:
    """The row lock, exercised as a race rather than as two sequential calls.

    A sequential double-approve only proves the `candidate_state` guard: the second call sees the
    first one's committed state either way. This one runs both against the real database at once,
    so a version of `_for_decision` that read without `FOR UPDATE` would let both transactions see
    `pending_publish` and both proceed.
    """
    _, _, advertisement, _ = seeded
    service = ReviewService(unit_of_work=unit_of_work, gateway=LocalListingGateway())

    def approve(actor: str) -> str:
        try:
            service.approve(advertisement.id, expected_version=None, actor_id=actor)
            return "approved"
        except Exception as exc:  # noqa: BLE001 - the test is about which failure, not the type
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(approve, ["reviewer-a", "reviewer-b"]))

    assert outcomes == ["InvalidReviewActionError", "approved"]

    session.expire_all()
    assert session.get(Advertisement, advertisement.id).status == "active"
    # One decision, so exactly one audit record and one reviewer on the provenance.
    assert session.query(ReviewEvent).filter_by(action="approved").count() == 1
    provenance = session.query(AdvertisementProvenance).filter_by(
        advertisement_id=advertisement.id
    ).one()
    assert provenance.candidate_state == "linked"


def test_a_simultaneous_approve_and_reject_do_not_both_apply(
    unit_of_work, session, seeded
) -> None:
    _, _, advertisement, _ = seeded
    service = ReviewService(unit_of_work=unit_of_work, gateway=LocalListingGateway())

    def decide(action: str) -> str:
        try:
            if action == "approve":
                service.approve(advertisement.id, expected_version=None, actor_id="reviewer-a")
            else:
                service.reject(
                    advertisement.id,
                    reason_code="duplicate",
                    expected_version=None,
                    actor_id="reviewer-b",
                )
            return action
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(decide, ["approve", "reject"]))

    assert outcomes.count("InvalidReviewActionError") == 1

    session.expire_all()
    saved = session.get(Advertisement, advertisement.id)
    # Exactly one decision landed, and the row is internally consistent with it.
    assert saved.status in {"active", "rejected"}
    if saved.status == "active":
        assert saved.rejected_at is None and saved.rejection_reason_code is None
    else:
        assert saved.approved_at is None and saved.published_at is None


def test_edits_from_the_same_version_do_not_silently_overwrite(
    unit_of_work, session, seeded
) -> None:
    """Two reviewers, one screen each, both loaded at version 1."""
    _, _, advertisement, _ = seeded
    service = ReviewService(unit_of_work=unit_of_work, gateway=LocalListingGateway())
    version = advertisement.version

    def save(title: str) -> str:
        try:
            service.apply_edits(
                advertisement.id,
                edits=ReviewerEdits(title=title),
                expected_version=version,
                actor_id="reviewer",
            )
            return "saved"
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(save, ["Title from A", "Title from B"]))

    assert outcomes == ["VersionConflictError", "saved"]


# -- input bounds ------------------------------------------------------------------------------


def test_an_unbounded_description_is_refused(api_client, seeded) -> None:
    """It is stored once and then re-served on every queue and detail read."""
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}", json={"description": "x" * 20_000}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_FAILED"


def test_too_many_phone_numbers_are_refused(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}",
        json={"phones": [f"07{index:08d}" for index in range(50)]},
    )

    assert response.status_code == 422


def test_an_over_long_phone_number_is_refused(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}", json={"phones": ["0" * 500]}
    )

    assert response.status_code == 422


def test_an_unknown_category_is_refused_on_save_not_only_on_approval(
    api_client, session, seeded
) -> None:
    """Storing it would only defer the failure to approval, with no explanation attached."""
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}", json={"category": "spaceships"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "category"
    session.expire_all()
    assert session.get(Advertisement, advertisement.id).category == "vehicles"


def test_a_deep_offset_is_refused(api_client, seeded) -> None:
    assert api_client.get("/api/v1/advertisements/review?offset=99999999").status_code == 422


# -- the audit trail ---------------------------------------------------------------------------


def test_an_operator_id_carrying_a_newline_is_refused(api_client, seeded) -> None:
    """It reaches log lines and a String(64) column; a newline forges log entries."""
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={},
        headers={"X-Operator-Id": "reviewer\nINFO forged log line"},
    )

    assert response.status_code in {400, 403}


def test_an_over_long_operator_id_is_refused_rather_than_crashing(api_client, seeded) -> None:
    """The column is String(64): an over-long value used to be a 500 on flush."""
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={},
        headers={"X-Operator-Id": "r" * 300},
    )

    assert response.status_code == 403


# -- counts ------------------------------------------------------------------------------------


def test_counts_describe_the_same_candidates_as_the_page(
    api_client, session, seeded
) -> None:
    """Counts beside a filtered page must count the filtered set, not the whole deployment."""
    batch, item, _, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    make_candidate(session, batch=batch, item=item, asset=asset, candidate_index=1, category="jobs")

    all_body = api_client.get("/api/v1/advertisements/review").json()["data"]
    assert all_body["total"] == 2
    assert sum(all_body["counts"].values()) == 2

    filtered = api_client.get("/api/v1/advertisements/review?category=jobs").json()["data"]
    assert filtered["total"] == 1
    assert sum(filtered["counts"].values()) == 1, "counts must follow the same filters as the page"


def test_counts_span_every_status_not_just_the_undecided_ones(
    api_client, session, seeded
) -> None:
    """The page shows outstanding work; the counts show what happened to the rest of the batch."""
    batch, item, first, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    make_candidate(session, batch=batch, item=item, asset=asset, candidate_index=1)
    api_client.post(f"/api/v1/advertisements/{first.id}/approve", json={})

    body = api_client.get("/api/v1/advertisements/review").json()["data"]

    assert body["total"] == 1, "the approved candidate left the queue"
    assert body["counts"]["active"] == 1, "but is still counted"
    assert body["counts"]["pending_review"] == 1
