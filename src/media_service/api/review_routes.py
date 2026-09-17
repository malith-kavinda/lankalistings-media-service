"""Review endpoints (PRD 13.2).

Plain `def` handlers, for the same reason as the ingestion routes: everything below them is a
blocking database driver, so FastAPI runs them in its threadpool instead of on the event loop.

Route order matters here. `/advertisements/review` and `/advertisements/rejection-reasons` are
declared **before** `/advertisements/{advertisement_id}`, because Starlette matches in declaration
order and would otherwise read "review" as an advertisement id and answer 404.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Query, Request, status

from media_service.api.deps import Operator
from media_service.api.errors import ServiceError, correlation_id_from
from media_service.api.review_schemas import (
    ApproveRequest,
    CandidateDetailResponse,
    CandidateEditRequest,
    CandidatePageResponse,
    RejectionReasonResponse,
    RejectRequest,
    stored_status,
)
from media_service.api.schemas import SuccessEnvelope
from media_service.config import Settings
from media_service.db.repositories.review import DEFAULT_LIMIT, MAX_LIMIT, ReviewFilters
from media_service.services.review import ReviewService

router = APIRouter(prefix="/api/v1", tags=["review"])

# What the queue shows when the caller says nothing: work still outstanding. Passing an explicit
# `candidate_state` widens it, which is how a reviewer looks back at what they already decided.
DEFAULT_CANDIDATE_STATES = ("pending_publish",)


@router.get("/advertisements/rejection-reasons", response_model=SuccessEnvelope)
def list_rejection_reasons(request: Request, operator: Operator) -> dict[str, object]:
    return {
        "data": [reason.model_dump(mode="json") for reason in RejectionReasonResponse.catalog()],
        "error": None,
    }


@router.get("/advertisements/review", response_model=SuccessEnvelope)
def list_review_queue(
    request: Request,
    operator: Operator,
    batch_id: Annotated[str | None, Query()] = None,
    item_id: Annotated[str | None, Query()] = None,
    review_status: Annotated[str | None, Query(alias="status")] = None,
    category: Annotated[str | None, Query()] = None,
    warning: Annotated[str | None, Query()] = None,
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    candidate_state: Annotated[list[str] | None, Query()] = None,
    q: Annotated[str | None, Query(max_length=120)] = None,
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    service, status_wire = _service(request)
    page = service.queue(
        ReviewFilters(
            batch_id=batch_id,
            item_id=item_id,
            # A client filtering by `pending_review` is speaking the legacy vocabulary it was
            # handed; translate it back rather than returning an empty page.
            status=stored_status(review_status, mode=status_wire) if review_status else None,
            category=category,
            warning=warning,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            candidate_states=(
                tuple(candidate_state) if candidate_state else DEFAULT_CANDIDATE_STATES
            ),
            query=q,
        ),
        limit=limit,
        offset=offset,
    )
    return _envelope(CandidatePageResponse.of(page, status_wire=status_wire))


@router.get("/advertisements/{advertisement_id}", response_model=SuccessEnvelope)
def get_review_candidate(
    request: Request, operator: Operator, advertisement_id: str
) -> dict[str, object]:
    service, status_wire = _service(request)
    detail = service.detail(advertisement_id)
    return _envelope(CandidateDetailResponse.of(detail, status_wire=status_wire))


@router.patch("/advertisements/{advertisement_id}", response_model=SuccessEnvelope)
def update_review_candidate(
    request: Request,
    operator: Operator,
    advertisement_id: str,
    payload: Annotated[CandidateEditRequest, Body()],
) -> dict[str, object]:
    service, status_wire = _service(request)
    detail = service.apply_edits(
        advertisement_id,
        edits=payload.to_edits(),
        expected_version=payload.version,
        actor_id=operator,
        correlation_id=correlation_id_from(request),
    )
    return _envelope(CandidateDetailResponse.of(detail, status_wire=status_wire))


@router.post(
    "/advertisements/{advertisement_id}/approve",
    response_model=SuccessEnvelope,
    status_code=status.HTTP_200_OK,
)
def approve_review_candidate(
    request: Request,
    operator: Operator,
    advertisement_id: str,
    payload: Annotated[ApproveRequest | None, Body()] = None,
) -> dict[str, object]:
    """Apply the reviewer's corrections and publish, in one transaction.

    The portal used to PATCH and then POST. A failure between the two left the edits saved and the
    candidate still pending, with nothing on screen to say which half had happened.
    """
    service, status_wire = _service(request)
    payload = payload or ApproveRequest()
    edits = payload.edits.to_edits() if payload.edits and payload.edits.has_changes else None
    # A version on the envelope and one inside `edits` must not disagree about what the reviewer
    # was looking at.
    expected_version = payload.version
    if payload.edits is not None and payload.edits.version is not None:
        if expected_version is not None and expected_version != payload.edits.version:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "version",
                        "code": "CONFLICTING_VALUE",
                        "message": "version and edits.version must match when both are sent.",
                    }
                ],
            )
        expected_version = payload.edits.version

    detail = service.approve(
        advertisement_id,
        edits=edits,
        expected_version=expected_version,
        actor_id=operator,
        correlation_id=correlation_id_from(request),
    )
    return _envelope(CandidateDetailResponse.of(detail, status_wire=status_wire))


@router.post("/advertisements/{advertisement_id}/reject", response_model=SuccessEnvelope)
def reject_review_candidate(
    request: Request,
    operator: Operator,
    advertisement_id: str,
    payload: Annotated[RejectRequest, Body()],
) -> dict[str, object]:
    service, status_wire = _service(request)
    detail = service.reject(
        advertisement_id,
        reason_code=payload.reason_code,
        note=payload.note,
        expected_version=payload.version,
        actor_id=operator,
        correlation_id=correlation_id_from(request),
    )
    return _envelope(CandidateDetailResponse.of(detail, status_wire=status_wire))


def _service(request: Request) -> tuple[ReviewService, str]:
    settings: Settings = request.app.state.settings
    service: ReviewService = request.app.state.review_service
    return service, settings.advertisement_status_wire


def _envelope(payload: object) -> dict[str, object]:
    return {"data": payload.model_dump(mode="json"), "error": None}  # type: ignore[attr-defined]
