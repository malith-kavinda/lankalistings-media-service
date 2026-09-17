from http import HTTPStatus
from uuid import uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: list[dict[str, str | None]] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or []


class OcrUnavailableError(ServiceError):
    def __init__(self, reason: str) -> None:
        super().__init__(
            status_code=503,
            code="OCR_UNAVAILABLE",
            message="OCR is not available on this service instance.",
            details=[{"field": "file", "code": "OCR_RUNTIME_MISSING", "message": reason}],
        )


def correlation_id_from(request: Request) -> str:
    return request.headers.get("X-Correlation-Id") or str(uuid4())


def error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: list[dict[str, str | None]] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details or [],
                "correlation_id": correlation_id_from(request),
            },
        },
    )


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        details.append(
            {
                "field": field or None,
                "code": "INVALID_FIELD",
                "message": error.get("msg", "Invalid field"),
            }
        )

    return error_response(
        request,
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        code="VALIDATION_FAILED",
        message="Request validation failed.",
        details=details,
    )


class NotFoundError(ServiceError):
    """A resource addressed by the URL does not exist.

    The code is per-resource rather than a single NOT_FOUND, because the portal distinguishes a
    stale batch link from a stale item link (PRD 13.4).
    """

    def __init__(self, *, code: str, message: str, field: str | None = None) -> None:
        super().__init__(
            status_code=404,
            code=code,
            message=message,
            details=[{"field": field, "code": "NOT_FOUND", "message": message}],
        )


class BatchNotFoundError(NotFoundError):
    def __init__(self, batch_id: str) -> None:
        super().__init__(
            code="BATCH_NOT_FOUND",
            message=f"No ingestion batch exists with id {batch_id}.",
            field="batch_id",
        )


class ItemNotFoundError(NotFoundError):
    def __init__(self, item_id: str) -> None:
        super().__init__(
            code="ITEM_NOT_FOUND",
            message=f"No ingestion item exists with id {item_id}.",
            field="item_id",
        )


class AssetNotFoundError(NotFoundError):
    def __init__(self, asset_id: str) -> None:
        super().__init__(
            code="ASSET_NOT_FOUND",
            message=f"No media asset exists with id {asset_id}.",
            field="asset_id",
        )


class MediaBytesUnavailableError(ServiceError):
    """The row survives but its pixels do not -- retention purged them (PRD 13.4).

    410 rather than 404: the distinction tells a client that retrying is pointless while the
    evidence metadata it already holds is still valid.
    """

    def __init__(self, asset_id: str) -> None:
        super().__init__(
            status_code=410,
            code="MEDIA_BYTES_UNAVAILABLE",
            message=f"The stored bytes for asset {asset_id} are no longer available.",
        )


class InvalidStatusTransitionError(ServiceError):
    def __init__(self, *, current: str, target: str) -> None:
        super().__init__(
            status_code=409,
            code="INVALID_STATUS_TRANSITION",
            message=f"An item cannot move from {current} to {target}.",
        )


class ItemClaimLostError(ServiceError):
    """A worker write did not match the row it expected.

    Raised when a compare-and-swap affects no rows: the item was reclaimed after a lease expiry, or
    another writer moved it. The worker must stop touching the item rather than continue on a row
    someone else now owns.
    """

    def __init__(self, item_id: str) -> None:
        super().__init__(
            status_code=409,
            code="ITEM_CLAIM_LOST",
            message=f"The claim on item {item_id} is no longer held by this worker.",
        )


class NothingToRetryError(ServiceError):
    def __init__(self, *, item_id: str, status: str) -> None:
        super().__init__(
            status_code=409,
            code="NOTHING_TO_RETRY",
            message=f"Item {item_id} is {status} and has nothing to retry.",
        )


class ReprocessRefusedError(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(status_code=409, code="REPROCESS_REFUSED", message=message)


class IdempotencyKeyConflictError(ServiceError):
    """The key was reused for a genuinely different request (FR-ING-006)."""

    def __init__(self) -> None:
        super().__init__(
            status_code=409,
            code="IDEMPOTENCY_KEY_CONFLICT",
            message=(
                "This idempotency key was already used for a different set of files. "
                "Use a new key, or resend the original request unchanged."
            ),
        )


class IdempotencyInProgressError(ServiceError):
    """A duplicate arrived while the first request was still being written."""

    def __init__(self) -> None:
        super().__init__(
            status_code=409,
            code="IDEMPOTENCY_IN_PROGRESS",
            message="An identical request is still being processed. Retry shortly.",
        )


class UnauthenticatedError(ServiceError):
    def __init__(self, message: str = "Operator credentials are required.") -> None:
        super().__init__(status_code=401, code="UNAUTHENTICATED", message=message)


class ForbiddenError(ServiceError):
    def __init__(self, message: str = "Operator credentials are not valid.") -> None:
        super().__init__(status_code=403, code="FORBIDDEN", message=message)


class AdvertisementNotFoundError(NotFoundError):
    def __init__(self, advertisement_id: str) -> None:
        super().__init__(
            code="ADVERTISEMENT_NOT_FOUND",
            message=f"No advertisement exists with id {advertisement_id}.",
            field="advertisement_id",
        )


class VersionConflictError(ServiceError):
    """Someone else edited this candidate since the reviewer loaded it (PRD 13.2).

    409 with both versions, so the client can say what happened rather than silently overwriting a
    colleague's correction -- which is the outcome optimistic locking exists to prevent.
    """

    def __init__(self, *, expected: int, actual: int) -> None:
        super().__init__(
            status_code=409,
            code="VERSION_CONFLICT",
            message=(
                f"This candidate has changed since it was loaded (you have version {expected}, "
                f"the current version is {actual}). Reload before saving."
            ),
            details=[
                {
                    "field": "version",
                    "code": "STALE",
                    "message": f"expected {expected}, current {actual}",
                }
            ],
        )


class InvalidReviewActionError(ServiceError):
    """The candidate is not in a state where this decision means anything."""

    def __init__(self, message: str, *, code: str = "INVALID_REVIEW_ACTION") -> None:
        super().__init__(status_code=409, code=code, message=message)
