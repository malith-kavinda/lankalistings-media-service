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
