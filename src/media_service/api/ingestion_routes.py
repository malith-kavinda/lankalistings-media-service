"""Batch ingestion endpoints.

Every handler here is a plain `def`, not `async def`, and that is deliberate. The work behind them
is blocking all the way down -- a synchronous database driver, file reads, Pillow -- so FastAPI runs
them in its threadpool. Declaring them `async` would put that blocking work on the event loop and
stall every other request in the process while one upload is hashed.

Upload answers **202**, never 200. The batch is durable when the response is written, but nothing
has been extracted yet: OCR and, later, a provider call take tens of seconds per image, and holding
an HTTP request open across them means a client timeout silently becomes a double-billed retry. The
`Location` header points at the progress resource that replaces polling the request itself.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Header, Query, Request, Response, UploadFile, status

from media_service.api.deps import Operator
from media_service.api.errors import ServiceError, correlation_id_from
from media_service.api.ingestion_schemas import BatchResponse, ItemResponse, RetryResponse
from media_service.api.schemas import SuccessEnvelope
from media_service.services.assets import PURPOSES, AssetService
from media_service.services.ingestion import IngestionService
from media_service.services.uploads import IncomingFile

router = APIRouter(prefix="/api/v1", tags=["ingestion"])

RETRY_MODES = ("resume", "reprocess")


@router.post(
    "/ingestion-batches",
    response_model=SuccessEnvelope,
    status_code=status.HTTP_202_ACCEPTED,
)
def create_ingestion_batch(
    request: Request,
    response: Response,
    operator: Operator,
    images: Annotated[list[UploadFile], File()],
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    correlation_id = correlation_id_from(request)

    batch = service.create_batch(
        [
            IncomingFile(
                filename=image.filename or "upload",
                declared_content_type=image.content_type,
                # The spooled file, not its contents: Starlette has already written the body to
                # disk, and reading it back whole would hold the entire batch in memory.
                stream=image.file,
            )
            for image in images
        ],
        created_by=operator,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
    )

    response.headers["Location"] = f"{request.url.path}/{batch.id}"
    # A replayed request did not create anything, so it is not 202 Accepted a second time.
    if batch.replayed:
        response.status_code = status.HTTP_200_OK
    return _envelope(BatchResponse.of(batch))


@router.get("/ingestion-batches", response_model=SuccessEnvelope)
def list_ingestion_batches(
    request: Request,
    operator: Operator,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    batches = service.list_batches(limit=limit, offset=offset)
    return {
        "data": [BatchResponse.of(batch).model_dump(mode="json") for batch in batches],
        "error": None,
    }


@router.get("/ingestion-batches/{batch_id}", response_model=SuccessEnvelope)
def get_ingestion_batch(request: Request, operator: Operator, batch_id: str) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    return _envelope(BatchResponse.of(service.get_batch(batch_id)))


@router.post("/ingestion-batches/{batch_id}/retry", response_model=SuccessEnvelope)
def retry_ingestion_batch(
    request: Request, operator: Operator, batch_id: str
) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    outcome = service.retry_batch(
        batch_id, actor_id=operator, correlation_id=correlation_id_from(request)
    )
    return _envelope(
        RetryResponse(
            mode=outcome.mode, items=[ItemResponse.of(item) for item in outcome.items]
        )
    )


@router.get("/ingestion-items/{item_id}", response_model=SuccessEnvelope)
def get_ingestion_item(request: Request, operator: Operator, item_id: str) -> dict[str, object]:
    service: IngestionService = request.app.state.ingestion_service
    return _envelope(ItemResponse.of(service.get_item(item_id)))


@router.post("/ingestion-items/{item_id}/retry", response_model=SuccessEnvelope)
def retry_ingestion_item(
    request: Request,
    operator: Operator,
    item_id: str,
    mode: Annotated[str, Query()] = "resume",
    force: Annotated[bool, Query()] = False,
) -> dict[str, object]:
    if mode not in RETRY_MODES:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "mode",
                    "code": "INVALID_VALUE",
                    "message": f"mode must be one of {', '.join(RETRY_MODES)}.",
                }
            ],
        )

    service: IngestionService = request.app.state.ingestion_service
    outcome = service.retry_item(
        item_id,
        mode=mode,
        actor_id=operator,
        force=force,
        correlation_id=correlation_id_from(request),
    )
    return _envelope(
        RetryResponse(
            mode=outcome.mode, items=[ItemResponse.of(item) for item in outcome.items]
        )
    )


@router.get("/media/assets/{asset_id}/{purpose}")
def get_asset_bytes(
    request: Request, operator: Operator, asset_id: str, purpose: str
) -> Response:
    """Serve stored bytes. Operator-only: these are the source scans, not public media."""
    if purpose not in PURPOSES:
        raise ServiceError(
            status_code=404,
            code="ASSET_NOT_FOUND",
            message=f"{purpose!r} is not a stored representation.",
        )

    service: AssetService = request.app.state.asset_service
    asset = service.read(asset_id, purpose)

    if request.headers.get("if-none-match") == asset.etag:
        # The bytes are addressed by their own digest, so an unchanged ETag is proof, not a hint.
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers={"ETag": asset.etag})

    return Response(
        content=asset.content,
        media_type=asset.content_type,
        headers={
            "ETag": asset.etag,
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'inline; filename="{asset.filename}"',
        },
    )


def _envelope(payload: object) -> dict[str, object]:
    return {"data": payload.model_dump(mode="json"), "error": None}  # type: ignore[attr-defined]
