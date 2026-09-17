"""Application assembly.

Everything is constructed here and handed down; nothing reaches for a global. That is what lets a
test replace the OCR engine, the dispatcher, or the whole unit-of-work factory by passing an
argument, and it is why `create_app` keeps accepting the prototype's `repository` and `ocr_engine`
parameters unchanged.

Building the app touches no database. `create_engine` opens no connection, so an instance can be
constructed -- and the prototype's endpoints exercised -- with PostgreSQL absent. The worker is the
only part that needs the database at startup, and it starts from the lifespan hook rather than from
the constructor.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from media_service.api.errors import ServiceError, service_error_handler, validation_error_handler
from media_service.api.ingestion_routes import router as ingestion_router
from media_service.api.routes import router
from media_service.config import Settings, get_settings
from media_service.db.engine import build_session_factory, cached_engine
from media_service.db.repositories.legacy import SqlAlchemyMediaRepository
from media_service.db.uow import UnitOfWorkFactory
from media_service.domain.listings import LocalListingGateway
from media_service.domain.repository import JsonMediaRepository, MediaRepository
from media_service.jobs.dispatchers import InlineDispatcher, ManualDispatcher
from media_service.jobs.local_pool import LocalPoolDispatcher
from media_service.jobs.protocol import JobDispatcher, NullDispatcher
from media_service.jobs.runner import ItemRunner
from media_service.llm.registry import build_extraction_service
from media_service.ocr.compat import as_legacy_engine, as_provider
from media_service.ocr.preprocess import ImagePreprocessor
from media_service.ocr.registry import build_provider, preprocess_settings
from media_service.services.advertisements import AdvertisementService
from media_service.services.assets import AssetService
from media_service.services.ingestion import IngestionService
from media_service.services.media import MediaExtractionService
from media_service.services.ocr import OcrEngine
from media_service.storage import FilesystemAssetStore


def create_app(
    *,
    settings: Settings | None = None,
    ocr_engine: OcrEngine | None = None,
    repository: MediaRepository | None = None,
    unit_of_work: UnitOfWorkFactory | None = None,
    dispatcher: JobDispatcher | None = None,
    store: FilesystemAssetStore | None = None,
    extraction: object | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_settings.validate_startup()

    # The provider is the real reader. The prototype's endpoints get it wrapped back into the
    # older `OcrEngine` shape, so they keep working unchanged while the structured pipeline and the
    # single-image path read pages the same way.
    resolved_provider = (
        as_provider(ocr_engine) if ocr_engine is not None else build_provider(resolved_settings)
    )
    resolved_ocr_engine = ocr_engine or as_legacy_engine(resolved_provider)
    resolved_store = store or FilesystemAssetStore(resolved_settings.storage_root)
    resolved_unit_of_work = unit_of_work or UnitOfWorkFactory(
        build_session_factory(cached_engine(resolved_settings.database_url))
    )
    resolved_repository = repository or build_repository(
        resolved_settings, unit_of_work=resolved_unit_of_work, store=resolved_store
    )

    # Built before the runner so a refused provider -- a heuristic one in production, say --
    # fails here, on deploy, rather than on a worker thread after a page has been uploaded.
    resolved_extraction = extraction or build_extraction_service(resolved_settings)

    runner = ItemRunner(
        unit_of_work=resolved_unit_of_work,
        store=resolved_store,
        settings=resolved_settings,
        preprocess=ImagePreprocessor(preprocess_settings(resolved_settings)),
        ocr=resolved_provider,
        extraction=resolved_extraction,
        gateway=LocalListingGateway(),
    )
    resolved_dispatcher = dispatcher or build_dispatcher(
        resolved_settings, unit_of_work=resolved_unit_of_work, runner=runner
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved_dispatcher.start()
        try:
            yield
        finally:
            # Draining rather than dropping: an item abandoned here would wait out its whole lease
            # before another worker could take it.
            resolved_dispatcher.stop()

    app = FastAPI(
        title="LankaListings Media Service",
        version="0.1.0",
        description="Image ingestion, OCR extraction, and review candidates for listing media.",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.ocr_engine = resolved_ocr_engine
    app.state.ocr_provider = resolved_provider
    app.state.extraction_service = resolved_extraction
    app.state.unit_of_work = resolved_unit_of_work
    app.state.asset_store = resolved_store
    app.state.dispatcher = resolved_dispatcher
    app.state.item_runner = runner
    app.state.media_service = MediaExtractionService(
        repository=resolved_repository,
        ocr_engine=resolved_ocr_engine,
    )
    app.state.advertisement_service = AdvertisementService(
        repository=resolved_repository,
        ocr_engine=resolved_ocr_engine,
    )
    app.state.ingestion_service = IngestionService(
        unit_of_work=resolved_unit_of_work,
        store=resolved_store,
        settings=resolved_settings,
        dispatcher=resolved_dispatcher,
    )
    app.state.asset_service = AssetService(
        unit_of_work=resolved_unit_of_work, store=resolved_store
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
    app.include_router(ingestion_router)
    return app


def build_repository(
    settings: Settings, *, unit_of_work: UnitOfWorkFactory, store: FilesystemAssetStore
) -> MediaRepository:
    """Where the prototype's endpoints keep their data.

    `json` is the prototype's own file and stays the default until the import has been run;
    `sql` puts the same endpoints on PostgreSQL. The JSON file is never written by the `sql` path
    and never deleted, so rolling back is this variable and nothing else.
    """
    if settings.media_repository == "sql":
        return SqlAlchemyMediaRepository(
            unit_of_work=unit_of_work,
            store=store,
            public_image_url_mode=settings.public_image_url_mode,
        )
    return JsonMediaRepository(settings.metadata_path)


def build_dispatcher(
    settings: Settings, *, unit_of_work: UnitOfWorkFactory, runner: ItemRunner
) -> JobDispatcher:
    """Pick the worker for this deployment.

    An unknown mode is impossible here -- `validate_startup` already refused it -- so there is no
    silent fallback to a default that would leave uploads sitting in the queue unprocessed.
    """
    if settings.job_dispatch_mode == "local_pool":
        return LocalPoolDispatcher(
            unit_of_work=unit_of_work, runner=runner, settings=settings
        )
    if settings.job_dispatch_mode == "inline":
        return InlineDispatcher(
            unit_of_work=unit_of_work, runner=runner, lease_seconds=settings.lease_seconds
        )
    if settings.job_dispatch_mode == "manual":
        return ManualDispatcher(
            unit_of_work=unit_of_work, runner=runner, lease_seconds=settings.lease_seconds
        )
    return NullDispatcher()


app = create_app()
