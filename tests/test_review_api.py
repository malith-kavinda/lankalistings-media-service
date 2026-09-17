"""The review queue and the decisions taken from it (PRD 13.2, FR-REV-001..010).

The tests that matter most here are the atomicity ones. Approve carries the reviewer's edits and
the transition in one request, so there is a state the old two-call portal could reach that this
must not: edits saved, candidate still pending, nothing on screen saying which half happened.
`test_a_refused_approval_saves_nothing` is the one that would catch a regression back to it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from media_service.config import Settings
from media_service.db.tables import (
    Advertisement,
    AdvertisementProvenance,
    MediaAsset,
    ReviewEvent,
)
from tests.factories import make_candidate, seed_batch_with_items

pytestmark = pytest.mark.usefixtures("engine")


@pytest.fixture
def seeded(session):  # type: ignore[no-untyped-def]
    """One batch, one image, one pending candidate -- the shape of every test below."""
    batch, items = seed_batch_with_items(session, count=1)
    item = items[0]
    asset = session.get(MediaAsset, item.source_asset_id)
    advertisement, provenance = make_candidate(session, batch=batch, item=item, asset=asset)
    return batch, item, advertisement, provenance


def _client(api_settings, unit_of_work, asset_store, **overrides):  # type: ignore[no-untyped-def]
    from media_service.main import create_app
    from tests.support import StubOcrEngine

    settings: Settings = api_settings.model_copy(update=overrides) if overrides else api_settings
    return TestClient(
        create_app(
            settings=settings,
            ocr_engine=StubOcrEngine(),
            unit_of_work=unit_of_work,
            store=asset_store,
        )
    )


# -- the queue -----------------------------------------------------------------------------------


def test_the_queue_returns_pending_candidates_with_their_provenance(api_client, seeded) -> None:
    _, item, advertisement, _ = seeded

    response = api_client.get("/api/v1/advertisements/review")

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["total"] == 1
    row = body["candidates"][0]
    assert row["id"] == advertisement.id
    assert row["item_id"] == item.id
    assert row["candidate_state"] == "pending_publish"
    # The stored vocabulary is `pending`; the portal reads `pending_review` (PRD 10.3).
    assert row["status"] == "pending_review"


def test_the_queue_puts_the_least_certain_candidate_first(
    api_client, session, seeded
) -> None:
    """Ascending confidence. A moderator working top-down spends attention where it changes most."""
    batch, item, first, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    low, _ = make_candidate(
        session, batch=batch, item=item, asset=asset, candidate_index=1, confidence=0.2
    )
    unmeasured, _ = make_candidate(
        session, batch=batch, item=item, asset=asset, candidate_index=2, confidence=None
    )

    body = api_client.get("/api/v1/advertisements/review").json()["data"]

    # NULLS LAST: no confidence means unmeasured, not maximally uncertain.
    assert [row["id"] for row in body["candidates"]] == [low.id, first.id, unmeasured.id]


def test_the_queue_filters_by_batch_category_warning_and_confidence(
    api_client, session, seeded
) -> None:
    batch, item, target, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    other, _ = make_candidate(
        session,
        batch=batch,
        item=item,
        asset=asset,
        candidate_index=1,
        category="jobs",
        confidence=0.1,
        provenance_warnings=("LOW_OCR_CONFIDENCE",),
    )

    def ids(query: str) -> list[str]:
        body = api_client.get(f"/api/v1/advertisements/review?{query}").json()["data"]
        return [row["id"] for row in body["candidates"]]

    assert ids(f"batch_id={batch.id}") == [other.id, target.id]
    assert ids("batch_id=bat_nothing") == []
    assert ids("category=jobs") == [other.id]
    assert ids("warning=LOW_OCR_CONFIDENCE") == [other.id]
    assert ids("min_confidence=0.5") == [target.id]
    assert ids("max_confidence=0.5") == [other.id]
    assert ids("q=Honda") == [other.id, target.id]
    assert ids("status=pending_review") == [other.id, target.id]


def test_the_queue_paginates(api_client, session, seeded) -> None:
    batch, item, _, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    for index in range(1, 4):
        make_candidate(
            session,
            batch=batch,
            item=item,
            asset=asset,
            candidate_index=index,
            confidence=0.1 * index,
        )

    body = api_client.get("/api/v1/advertisements/review?limit=2&offset=1").json()["data"]

    assert body["total"] == 4
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert len(body["candidates"]) == 2


def test_the_queue_refuses_a_limit_beyond_the_maximum(api_client, seeded) -> None:
    """An unbounded page is how a review queue becomes a full table scan."""
    assert api_client.get("/api/v1/advertisements/review?limit=5000").status_code == 422


# -- detail --------------------------------------------------------------------------------------


def test_detail_carries_the_evidence_a_reviewer_needs(api_client, seeded) -> None:
    """FR-REV-003: source text, block ids, per-field confidence and warnings, together."""
    _, item, advertisement, _ = seeded

    response = api_client.get(f"/api/v1/advertisements/{advertisement.id}")

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["candidate"]["id"] == advertisement.id
    assert body["evidence"]["source_block_ids"] == [1, 2]
    assert body["evidence"]["field_confidence"] == {"title": 0.9, "price": 0.7}
    assert body["evidence"]["extracted_values"]["title"] == "Honda Fit 2014"
    assert body["source_filename"] == item.original_filename
    assert body["item_id"] == item.id


def test_detail_positions_a_candidate_among_its_siblings(api_client, session, seeded) -> None:
    """"Ad 2 of 3 from page-0.png" -- the pager in PRD 14.3."""
    batch, item, first, _ = seeded
    asset = session.get(MediaAsset, item.source_asset_id)
    second, _ = make_candidate(session, batch=batch, item=item, asset=asset, candidate_index=1)
    make_candidate(session, batch=batch, item=item, asset=asset, candidate_index=2)

    body = api_client.get(f"/api/v1/advertisements/{second.id}").json()["data"]

    assert body["sibling_count"] == 3
    assert body["position"] == 2
    assert body["siblings"][0] == first.id


def test_detail_of_an_unknown_advertisement_is_a_named_404(api_client) -> None:
    response = api_client.get("/api/v1/advertisements/adv_nothing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ADVERTISEMENT_NOT_FOUND"


# -- editing -------------------------------------------------------------------------------------


def test_an_edit_saves_the_change_and_records_who_made_it(api_client, session, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}",
        json={"title": "Honda Fit GP5 2014", "version": advertisement.version},
        headers={"X-Operator-Id": "reviewer-7"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["candidate"]["title"] == "Honda Fit GP5 2014"

    session.expire_all()
    assert session.get(Advertisement, advertisement.id).title == "Honda Fit GP5 2014"
    event = session.query(ReviewEvent).filter_by(advertisement_id=advertisement.id).one()
    assert event.action == "edited"
    assert event.actor_id == "reviewer-7"
    assert event.changed_fields == ["title"]
    assert event.before_values["title"] == "Honda Fit 2014"


def test_an_absent_field_is_left_alone_rather_than_cleared(api_client, session, seeded) -> None:
    """A portal that sends only the field it changed must not blank the other five."""
    _, _, advertisement, _ = seeded

    api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}", json={"title": "Edited title"}
    )

    session.expire_all()
    saved = session.get(Advertisement, advertisement.id)
    assert saved.title == "Edited title"
    assert saved.location == "Kandy"
    assert saved.phones == ["0812233445"]


def test_an_edit_that_changes_nothing_records_no_event(api_client, session, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}", json={"title": advertisement.title}
    )

    assert response.status_code == 200
    assert session.query(ReviewEvent).count() == 0


def test_a_stale_version_is_refused_rather_than_overwriting(api_client, seeded) -> None:
    """Optimistic locking: a reviewer must not silently overwrite a colleague's correction."""
    _, _, advertisement, _ = seeded

    response = api_client.patch(
        f"/api/v1/advertisements/{advertisement.id}",
        json={"title": "Written from a stale screen", "version": advertisement.version + 5},
    )

    assert response.status_code == 409
    body = response.json()["error"]
    assert body["code"] == "VERSION_CONFLICT"
    assert str(advertisement.version) in body["message"]


# -- approval ------------------------------------------------------------------------------------


def test_approve_applies_edits_and_publishes_in_one_request(
    api_client, session, seeded
) -> None:
    """The fix for the portal's update-then-approve pair."""
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={"edits": {"title": "Honda Fit GP5 2014", "location": "Kandy Town"}},
        headers={"X-Operator-Id": "reviewer-7"},
    )

    assert response.status_code == 200
    candidate = response.json()["data"]["candidate"]
    assert candidate["status"] == "active"
    assert candidate["candidate_state"] == "linked"
    assert candidate["title"] == "Honda Fit GP5 2014"

    session.expire_all()
    saved = session.get(Advertisement, advertisement.id)
    assert saved.status == "active"
    assert saved.approved_by == "reviewer-7"
    assert saved.approved_at is not None
    assert saved.published_at is not None

    provenance = (
        session.query(AdvertisementProvenance).filter_by(advertisement_id=saved.id).one()
    )
    # FR-REV-009: what the model produced and what the person accepted, side by side.
    assert provenance.extracted_values["title"] == "Honda Fit 2014"
    assert provenance.accepted_values["title"] == "Honda Fit GP5 2014"
    assert provenance.reviewer_id == "reviewer-7"

    event = session.query(ReviewEvent).filter_by(action="approved").one()
    assert sorted(event.changed_fields) == ["location", "title"]


def test_a_refused_approval_saves_nothing(api_client, session, seeded) -> None:
    """The atomicity guarantee, stated as the state that must be unreachable.

    Edits arrive with the approval; the approval fails validation; neither may survive. The old
    two-call portal could reach exactly the opposite -- edits saved, still pending, no feedback.
    """
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={"edits": {"title": "  ", "location": "Galle"}},
    )

    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "APPROVAL_VALIDATION_FAILED"
    assert [detail["field"] for detail in body["details"]] == ["title"]

    session.expire_all()
    saved = session.get(Advertisement, advertisement.id)
    assert saved.title == "Honda Fit 2014", "the rejected edit must not have been written"
    assert saved.location == "Kandy", "nor the edit that was fine on its own"
    assert saved.status == "pending"
    assert session.query(ReviewEvent).count() == 0


def test_approval_reports_every_missing_field_at_once(api_client, session, seeded) -> None:
    """FR-REV-008. One field per round trip is how a queue stops being clearable."""
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={
            "edits": {
                "title": "",
                "category": "",
                "location": "",
                "description": "",
                "phones": [],
            }
        },
    )

    assert response.status_code == 422
    fields = [detail["field"] for detail in response.json()["error"]["details"]]
    assert fields == ["title", "category", "phones"]


def test_an_unknown_category_blocks_publication(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={"edits": {"category": "spaceships"}},
    )

    assert response.status_code == 422
    detail = response.json()["error"]["details"][0]
    assert detail["field"] == "category"
    assert detail["code"] == "UNKNOWN_VALUE"


def test_an_approved_candidate_leaves_the_queue(api_client, seeded) -> None:
    """FR-REV-010, the half of it this service owns."""
    _, _, advertisement, _ = seeded

    api_client.post(f"/api/v1/advertisements/{advertisement.id}/approve", json={})

    body = api_client.get("/api/v1/advertisements/review").json()["data"]
    assert body["total"] == 0


def test_a_decided_candidate_cannot_be_decided_again(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded
    api_client.post(f"/api/v1/advertisements/{advertisement.id}/approve", json={})

    response = api_client.post(f"/api/v1/advertisements/{advertisement.id}/approve", json={})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "INVALID_REVIEW_ACTION"


def test_approve_refuses_two_disagreeing_versions(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/approve",
        json={"version": 1, "edits": {"title": "New", "version": 9}},
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "version"


# -- rejection -----------------------------------------------------------------------------------


def test_reject_records_the_reason_and_discards_the_candidate(
    api_client, session, seeded
) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/reject",
        json={"reason_code": "not_an_advertisement", "note": "Masthead, not an ad."},
        headers={"X-Operator-Id": "reviewer-3"},
    )

    assert response.status_code == 200
    assert response.json()["data"]["candidate"]["candidate_state"] == "discarded"

    session.expire_all()
    saved = session.get(Advertisement, advertisement.id)
    assert saved.status == "rejected"
    assert saved.rejection_reason_code == "not_an_advertisement"
    assert saved.rejected_by == "reviewer-3"

    event = session.query(ReviewEvent).filter_by(action="rejected").one()
    assert event.reason_code == "not_an_advertisement"
    assert event.note == "Masthead, not an ad."


def test_an_unknown_reason_code_is_refused(api_client, seeded) -> None:
    """A closed vocabulary is what makes PRD 15.5 able to group rejections at all."""
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/reject",
        json={"reason_code": "i_just_dont_like_it"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "reason_code"


def test_other_requires_a_note(api_client, seeded) -> None:
    _, _, advertisement, _ = seeded

    response = api_client.post(
        f"/api/v1/advertisements/{advertisement.id}/reject", json={"reason_code": "other"}
    )

    assert response.status_code == 422
    assert response.json()["error"]["details"][0]["field"] == "note"


def test_the_reason_catalog_is_published_for_the_portal(api_client) -> None:
    """So the dropdown cannot drift from the set the server actually accepts."""
    response = api_client.get("/api/v1/advertisements/rejection-reasons")

    assert response.status_code == 200
    codes = {reason["code"] for reason in response.json()["data"]}
    assert "not_an_advertisement" in codes
    assert "other" in codes


# -- wire compatibility --------------------------------------------------------------------------


def test_the_canonical_wire_publishes_the_stored_vocabulary(
    api_settings, unit_of_work, asset_store, seeded
) -> None:
    client = _client(
        api_settings, unit_of_work, asset_store, advertisement_status_wire="canonical"
    )

    body = client.get("/api/v1/advertisements/review").json()["data"]

    assert body["candidates"][0]["status"] == "pending"


def test_an_unknown_wire_setting_is_refused_at_startup(api_settings) -> None:
    with pytest.raises(ValueError, match="ADVERTISEMENT_STATUS_WIRE"):
        api_settings.model_copy(
            update={"advertisement_status_wire": "nonsense"}
        ).validate_startup()


def test_an_approved_candidate_reaches_the_public_feed(
    api_settings, unit_of_work, asset_store, seeded
) -> None:
    """The gate: approve here, and the listing becomes readable there (AC-010 in reverse)."""
    _, _, advertisement, _ = seeded
    client = _client(api_settings, unit_of_work, asset_store, media_repository="sql")

    assert client.get("/api/v1/advertisements").json()["data"] == []

    approved = client.post(f"/api/v1/advertisements/{advertisement.id}/approve", json={})
    assert approved.status_code == 200

    feed = client.get("/api/v1/advertisements").json()["data"]
    assert [row["id"] for row in feed] == [advertisement.id]
