from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from media_service.api.deps import Operator
from media_service.api.errors import ReviewRequiredError, ServiceError
from media_service.api.schemas import (
    AdvertisementResponse,
    AdvertisementUpdateRequest,
    ExtractionResponse,
    NewspaperExtractionResponse,
    OcrMetadataResponse,
    SuccessEnvelope,
)
from media_service.domain.models import AdvertisementStatus
from media_service.services.advertisements import AdvertisementService
from media_service.services.media import MediaExtractionService
from media_service.services.review import ReviewService

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Report whether this instance can actually read a page.

    Names the configured provider, because "OCR is available" means something different for each
    one and an operator looking at a degraded deployment needs to know which is in use. The reason
    string is written to be safe here: this endpoint is unauthenticated, so it says
    "GEMINI_API_KEY is not set", never a value.

    Availability is cached with a short TTL inside the provider. This used to spawn a Tesseract
    subprocess on every request.
    """
    provider = request.app.state.ocr_provider
    return {
        "data": {
            "service": "media-service",
            "status": "ok",
            "ocr_provider": provider.name,
            "ocr_available": provider.is_available(),
            "ocr_unavailable_reason": provider.availability_reason(),
        },
        "error": None,
    }


@router.post("/api/v1/media/extract", response_model=SuccessEnvelope)
async def extract_text_from_image(
    request: Request,
    operator: Operator,
    file: Annotated[UploadFile, File()],
) -> dict[str, object]:
    content_type = file.content_type or "application/octet-stream"
    image_bytes = await file.read()
    max_upload_bytes: int = request.app.state.settings.max_upload_bytes

    if not image_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "file",
                    "code": "REQUIRED",
                    "message": "Uploaded image is required.",
                }
            ],
        )

    if len(image_bytes) > max_upload_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "file",
                    "code": "MAX_SIZE",
                    "message": f"Uploaded image must be {max_upload_bytes} bytes or smaller.",
                }
            ],
        )

    service: MediaExtractionService = request.app.state.media_service
    asset, extraction, ocr_output = service.extract(
        image_bytes=image_bytes,
        content_type=content_type,
    )

    response = ExtractionResponse(
        extraction_id=extraction.id,
        asset=asset,
        detected_text=ocr_output.text,
        metadata=OcrMetadataResponse(
            engine=ocr_output.engine,
            model_version=ocr_output.model_version,
            language=ocr_output.language,
            confidence=ocr_output.confidence,
            processing_ms=ocr_output.processing_ms,
        ),
    )
    return {"data": response.model_dump(mode="json"), "error": None}


@router.post("/api/v1/advertisements", response_model=SuccessEnvelope)
async def create_advertisement(
    request: Request,
    operator: Operator,
    title: Annotated[str, Form()],
    price: Annotated[str, Form()],
    category: Annotated[str, Form()],
    location: Annotated[str, Form()],
    description: Annotated[str, Form()],
    image: Annotated[UploadFile, File()],
) -> dict[str, object]:
    image_bytes = await image.read()
    max_upload_bytes: int = request.app.state.settings.max_upload_bytes

    if not image_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "image",
                    "code": "REQUIRED",
                    "message": "Advertisement image is required.",
                }
            ],
        )

    if len(image_bytes) > max_upload_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "image",
                    "code": "MAX_SIZE",
                    "message": f"Uploaded image must be {max_upload_bytes} bytes or smaller.",
                }
            ],
        )

    service: AdvertisementService = request.app.state.advertisement_service
    advertisement = service.create(
        title=title,
        price=price,
        category=category,
        location=location,
        description=description,
        image_bytes=image_bytes,
        content_type=image.content_type or "application/octet-stream",
    )
    response = AdvertisementResponse.model_validate(advertisement)
    return {"data": response.model_dump(mode="json"), "error": None}


@router.post("/api/v1/newspaper-articles/extract", response_model=SuccessEnvelope)
async def extract_newspaper_article(
    request: Request,
    operator: Operator,
    image: Annotated[UploadFile, File()],
) -> dict[str, object]:
    image_bytes = await image.read()
    max_upload_bytes: int = request.app.state.settings.max_upload_bytes

    if not image_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "image",
                    "code": "REQUIRED",
                    "message": "Newspaper article image is required.",
                }
            ],
        )

    if len(image_bytes) > max_upload_bytes:
        raise ServiceError(
            status_code=422,
            code="VALIDATION_FAILED",
            message="Request validation failed.",
            details=[
                {
                    "field": "image",
                    "code": "MAX_SIZE",
                    "message": f"Uploaded image must be {max_upload_bytes} bytes or smaller.",
                }
            ],
        )

    service: AdvertisementService = request.app.state.advertisement_service
    advertisement, detected_text = service.extract_newspaper_article(
        image_bytes=image_bytes,
        content_type=image.content_type or "application/octet-stream",
    )
    response = NewspaperExtractionResponse(
        detected_text=detected_text,
        advertisement=AdvertisementResponse.model_validate(advertisement),
    )
    return {"data": response.model_dump(mode="json"), "error": None}


@router.get("/api/v1/advertisements", response_model=SuccessEnvelope)
async def list_advertisements(request: Request) -> dict[str, object]:
    service: AdvertisementService = request.app.state.advertisement_service
    advertisements = [
        AdvertisementResponse.model_validate(advertisement).model_dump(mode="json")
        for advertisement in service.list(status=AdvertisementStatus.ACTIVE)
    ]
    return {"data": advertisements, "error": None}


@router.get("/api/v1/legacy/advertisements/review", response_model=SuccessEnvelope)
async def list_legacy_review_advertisements(
    request: Request, operator: Operator
) -> dict[str, object]:
    """The prototype's review list, kept for the compatibility endpoint above (PRD 13.3).

    Moved off `/api/v1/advertisements/review`, which PRD 13.2 gives to the real review queue. The
    two read different stores -- this one the prototype's repository, that one the ingestion
    schema -- so serving both from one path would mean answering with whichever happened to be
    registered first.
    """
    service: AdvertisementService = request.app.state.advertisement_service
    advertisements = [
        AdvertisementResponse.model_validate(advertisement).model_dump(mode="json")
        for advertisement in service.list(status=AdvertisementStatus.PENDING_REVIEW)
    ]
    return {"data": advertisements, "error": None}


@router.patch(
    "/api/v1/legacy/advertisements/{advertisement_id}", response_model=SuccessEnvelope
)
async def update_legacy_advertisement(
    request: Request,
    operator: Operator,
    advertisement_id: str,
    payload: AdvertisementUpdateRequest,
) -> dict[str, object]:
    _refuse_pipeline_candidate(request, advertisement_id)
    service: AdvertisementService = request.app.state.advertisement_service
    advertisement = service.update(
        advertisement_id=advertisement_id,
        title=payload.title,
        price=payload.price,
        category=payload.category,
        location=payload.location,
        description=payload.description,
    )
    response = AdvertisementResponse.model_validate(advertisement)
    return {"data": response.model_dump(mode="json"), "error": None}


@router.post(
    "/api/v1/legacy/advertisements/{advertisement_id}/approve", response_model=SuccessEnvelope
)
async def approve_legacy_advertisement(
    request: Request, operator: Operator, advertisement_id: str
) -> dict[str, object]:
    _refuse_pipeline_candidate(request, advertisement_id)
    service: AdvertisementService = request.app.state.advertisement_service
    advertisement = service.approve(advertisement_id)
    response = AdvertisementResponse.model_validate(advertisement)
    return {"data": response.model_dump(mode="json"), "error": None}


@router.get("/api/v1/media/extractions/{extraction_id}", response_model=SuccessEnvelope)
async def get_extraction(request: Request, extraction_id: str) -> dict[str, object]:
    service: MediaExtractionService = request.app.state.media_service
    asset, extraction = service.get_extraction(extraction_id)
    response = ExtractionResponse(
        extraction_id=extraction.id,
        asset=asset,
        detected_text=extraction.raw_text,
        metadata=OcrMetadataResponse(
            engine=extraction.engine,
            model_version=extraction.model_version,
            language=extraction.language,
            confidence=extraction.confidence,
            processing_ms=extraction.processing_ms,
        ),
    )
    return {"data": response.model_dump(mode="json"), "error": None}


def _refuse_pipeline_candidate(request: Request, advertisement_id: str) -> None:
    """Keep the prototype's endpoints away from anything the pipeline produced.

    The two stores are the same table once MEDIA_REPOSITORY=sql, which is the migration path, so
    without this the old approve route is a second way to publish a candidate -- one with no field
    validation, no version check, no row lock and no audit event.
    """
    review_service: ReviewService = request.app.state.review_service
    if review_service.is_pipeline_candidate(advertisement_id):
        raise ReviewRequiredError(advertisement_id)
