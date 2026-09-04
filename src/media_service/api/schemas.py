from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    field: str | None = None
    code: str
    message: str


class ApiError(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
    correlation_id: str


class ErrorEnvelope(BaseModel):
    data: None = None
    error: ApiError


class MediaAssetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    content_type: str
    byte_size: int
    checksum: str
    width: int | None
    height: int | None
    created_at: datetime


class OcrMetadataResponse(BaseModel):
    engine: str
    model_version: str
    language: str
    confidence: str
    processing_ms: int


class ExtractionResponse(BaseModel):
    extraction_id: str
    asset: MediaAssetResponse
    detected_text: str
    metadata: OcrMetadataResponse


class AdvertisementResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    price: str
    category: str
    location: str
    description: str
    image_url: str
    status: str
    created_at: datetime
    source_text: str
    extraction_confidence: str


class AdvertisementUpdateRequest(BaseModel):
    title: str
    price: str
    category: str
    location: str
    description: str


class NewspaperExtractionResponse(BaseModel):
    detected_text: str
    advertisement: AdvertisementResponse


class SuccessEnvelope(BaseModel):
    data: Any
    error: None = None
