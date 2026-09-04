from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from media_service.api.errors import ServiceError, service_error_handler, validation_error_handler
from media_service.api.routes import router
from media_service.config import Settings, get_settings
from media_service.domain.repository import JsonMediaRepository, MediaRepository
from media_service.services.advertisements import AdvertisementService
from media_service.services.media import MediaExtractionService
from media_service.services.ocr import OcrEngine, TesseractOcrEngine


def create_app(
    *,
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    repository: MediaRepository | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_ocr_engine = ocr_engine or TesseractOcrEngine(
        tesseract_cmd=resolved_settings.tesseract_cmd,
        language=resolved_settings.tesseract_lang,
        tessdata_dir=resolved_settings.tesseract_data_dir,
    )
    resolved_repository = repository or JsonMediaRepository(resolved_settings.metadata_path)

    app = FastAPI(
        title="LankaListings Media Service",
        version="0.1.0",
        description="Image validation and OCR extraction service for uploaded listing media.",
    )
    app.state.settings = resolved_settings
    app.state.ocr_engine = resolved_ocr_engine
    app.state.media_service = MediaExtractionService(
        repository=resolved_repository,
        ocr_engine=resolved_ocr_engine,
    )
    app.state.advertisement_service = AdvertisementService(
        repository=resolved_repository,
        ocr_engine=resolved_ocr_engine,
    )

    if resolved_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_exception_handler(ServiceError, service_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    return app


app = create_app()
