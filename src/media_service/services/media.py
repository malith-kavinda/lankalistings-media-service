from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from media_service.api.errors import ServiceError
from media_service.domain.models import ExtractionResult, ExtractionStatus, MediaAsset
from media_service.domain.repository import MediaRepository
from media_service.services.ocr import OcrEngine, OcrOutput


class MediaExtractionService:
    def __init__(self, *, repository: MediaRepository, ocr_engine: OcrEngine) -> None:
        self._repository = repository
        self._ocr_engine = ocr_engine

    def extract(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
    ) -> tuple[MediaAsset, ExtractionResult, OcrOutput]:
        ocr_output = self._ocr_engine.extract_text(image_bytes, content_type=content_type)
        asset = MediaAsset.create(
            asset_id=str(uuid4()),
            content_type=content_type,
            byte_size=len(image_bytes),
            checksum=sha256(image_bytes).hexdigest(),
            width=ocr_output.width,
            height=ocr_output.height,
        )
        extraction = ExtractionResult(
            id=str(uuid4()),
            asset_id=asset.id,
            status=ExtractionStatus.COMPLETED,
            raw_text=ocr_output.text,
            engine=ocr_output.engine,
            model_version=ocr_output.model_version,
            language=ocr_output.language,
            confidence=ocr_output.confidence,
            processing_ms=ocr_output.processing_ms,
            created_at=datetime.now(UTC),
        )
        self._repository.save_asset(asset)
        self._repository.save_extraction(extraction)
        return asset, extraction, ocr_output

    def get_extraction(self, extraction_id: str) -> tuple[MediaAsset, ExtractionResult]:
        extraction = self._repository.get_extraction(extraction_id)
        if extraction is None:
            raise ServiceError(
                status_code=404,
                code="EXTRACTION_NOT_FOUND",
                message="Extraction result was not found.",
                details=[
                    {
                        "field": "extraction_id",
                        "code": "NOT_FOUND",
                        "message": "No extraction result exists for this id.",
                    }
                ],
            )

        asset = self._repository.get_asset(extraction.asset_id)
        if asset is None:
            raise ServiceError(
                status_code=500,
                code="INTERNAL_ERROR",
                message="Extraction result is missing its media asset metadata.",
            )

        return asset, extraction
