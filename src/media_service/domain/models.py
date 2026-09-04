from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ExtractionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class AdvertisementStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"


@dataclass(frozen=True)
class MediaAsset:
    id: str
    content_type: str
    byte_size: int
    checksum: str
    width: int | None
    height: int | None
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        asset_id: str,
        content_type: str,
        byte_size: int,
        checksum: str,
        width: int | None,
        height: int | None,
    ) -> "MediaAsset":
        return cls(
            id=asset_id,
            content_type=content_type,
            byte_size=byte_size,
            checksum=checksum,
            width=width,
            height=height,
            created_at=datetime.now(UTC),
        )


@dataclass(frozen=True)
class ExtractionResult:
    id: str
    asset_id: str
    status: ExtractionStatus
    raw_text: str
    engine: str
    model_version: str
    language: str
    confidence: str
    processing_ms: int
    created_at: datetime


@dataclass(frozen=True)
class Advertisement:
    id: str
    title: str
    price: str
    category: str
    location: str
    description: str
    image_url: str
    status: AdvertisementStatus
    created_at: datetime
    source_text: str = ""
    extraction_confidence: str = "manual"

    @classmethod
    def create(
        cls,
        *,
        advertisement_id: str,
        title: str,
        price: str,
        category: str,
        location: str,
        description: str,
        image_url: str,
        status: AdvertisementStatus = AdvertisementStatus.ACTIVE,
        source_text: str = "",
        extraction_confidence: str = "manual",
    ) -> "Advertisement":
        return cls(
            id=advertisement_id,
            title=title,
            price=price,
            category=category,
            location=location,
            description=description,
            image_url=image_url,
            status=status,
            created_at=datetime.now(UTC),
            source_text=source_text,
            extraction_confidence=extraction_confidence,
        )
