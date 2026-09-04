from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile

from media_service.api.errors import ServiceError
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
from media_service.services.ocr import OcrEngine

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    ocr_engine: OcrEngine = request.app.state.ocr_engine
    return {
        "data": {
            "service": "media-service",
            "status": "ok",
            "ocr_available": ocr_engine.is_available(),
            "ocr_unavailable_reason": ocr_engine.availability_reason(),
        },
        "error": None,
    }


@router.post("/api/v1/media/extract", response_model=SuccessEnvelope)
async def extract_text_from_image(
    request: Request,
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


@router.get("/api/v1/advertisements/review", response_model=SuccessEnvelope)
async def list_review_advertisements(request: Request) -> dict[str, object]:
    service: AdvertisementService = request.app.state.advertisement_service
    advertisements = [
        AdvertisementResponse.model_validate(advertisement).model_dump(mode="json")
        for advertisement in service.list(status=AdvertisementStatus.PENDING_REVIEW)
    ]
    return {"data": advertisements, "error": None}


@router.patch("/api/v1/advertisements/{advertisement_id}", response_model=SuccessEnvelope)
async def update_advertisement(
    request: Request,
    advertisement_id: str,
    payload: AdvertisementUpdateRequest,
) -> dict[str, object]:
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


@router.post("/api/v1/advertisements/{advertisement_id}/approve", response_model=SuccessEnvelope)
async def approve_advertisement(request: Request, advertisement_id: str) -> dict[str, object]:
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
