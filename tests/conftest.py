"""Shared test fixtures.

Database tests run against a real PostgreSQL, the same engine used in development and production.
The schema is created once per session from the Alembic migrations rather than from `create_all`,
so the migrations themselves are exercised by every run: a model change that nobody wrote a
migration for fails here instead of at deployment.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from media_service.config import DEFAULT_DATABASE_URL, reset_settings_cache
from media_service.db.base import Base
from media_service.db.engine import build_engine, build_session_factory


def _with_database(url: str, database: str) -> str:
    """Swap only the database name.

    Not `str.replace`: the database name also appears as the username in the default URL, so a naive
    replace silently changes who is connecting.
    """
    base, separator, _ = url.rpartition("/")
    return f"{base}{separator}{database}" if separator else url


DEFAULT_TEST_DATABASE_URL = _with_database(DEFAULT_DATABASE_URL, "lankalistings_test")

SKIP_REASON = (
    "PostgreSQL is not reachable at {url}.\n"
    "Start it with:  docker compose up -d postgres\n"
    "Or point TEST_DATABASE_URL at another instance."
)


def test_database_url() -> str:
    return os.getenv("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL)


@pytest.fixture(scope="session")
def database_url() -> str:
    return test_database_url()


@pytest.fixture(scope="session")
def engine(database_url: str) -> Iterator[Engine]:
    """A session-wide engine with the schema migrated to head.

    Skips rather than errors when the database is unreachable, so the non-database tests still run
    for someone who has not started Docker -- but says exactly how to fix it.
    """
    candidate = build_engine(database_url, pool_size=5)
    try:
        with candidate.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - any connection failure means "no database"
        candidate.dispose()
        pytest.skip(SKIP_REASON.format(url=database_url) + f"\nUnderlying error: {exc}")

    _migrate_to_head(candidate, database_url)
    yield candidate
    candidate.dispose()


def _migrate_to_head(engine: Engine, url: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")


@pytest.fixture(autouse=True)
def clean_database(request: pytest.FixtureRequest) -> Iterator[None]:
    """Empty every table between tests.

    Truncation rather than transaction-rollback isolation: the worker pool opens its own sessions on
    its own connections, so work committed by a worker would be invisible to a test holding an
    uncommitted transaction.
    """
    if "engine" not in request.fixturenames:
        yield
        return

    engine: Engine = request.getfixturevalue("engine")
    _truncate_all(engine)
    yield
    _truncate_all(engine)


def _truncate_all(engine: Engine) -> None:
    tables = ", ".join(f'public."{table.name}"' for table in Base.metadata.sorted_tables)
    if not tables:
        return
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def session_factory(engine: Engine):  # type: ignore[no-untyped-def]
    return build_session_factory(engine)


@pytest.fixture
def session(session_factory) -> Iterator[Session]:  # type: ignore[no-untyped-def]
    with session_factory() as active:
        yield active


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep ambient configuration out of the tests.

    Without this, a developer who has exported LLM_PROVIDER or TESSERACT_LANG in their shell gets
    different test behaviour from CI, and the difference is invisible.
    """
    for name in (
        "MEDIA_SERVICE_ENV",
        "OCR_PROVIDER",
        "LLM_PROVIDER",
        "OPERATOR_AUTH_MODE",
        "OPERATOR_API_TOKEN",
        "PUBLIC_IMAGE_URL_MODE",
        "ADVERTISEMENT_STATUS_WIRE",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def asset_store(tmp_path):  # type: ignore[no-untyped-def]
    from media_service.storage import FilesystemAssetStore

    return FilesystemAssetStore(tmp_path / "media")


@pytest.fixture
def unit_of_work(session_factory):  # type: ignore[no-untyped-def]
    from media_service.db.uow import UnitOfWorkFactory

    return UnitOfWorkFactory(session_factory)


@pytest.fixture
def dispatcher():  # type: ignore[no-untyped-def]
    """Records what the service asked for without running anything."""
    from tests.support import RecordingDispatcher

    return RecordingDispatcher()


@pytest.fixture
def ingestion_settings():  # type: ignore[no-untyped-def]
    from media_service.config import Settings

    return Settings()


@pytest.fixture
def ingestion_service(unit_of_work, asset_store, ingestion_settings, dispatcher):  # type: ignore[no-untyped-def]
    from media_service.services.ingestion import IngestionService

    return IngestionService(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=ingestion_settings,
        dispatcher=dispatcher,
    )


@pytest.fixture
def pipeline(unit_of_work, asset_store, ingestion_settings):  # type: ignore[no-untyped-def]
    from tests.support import build_pipeline

    return build_pipeline(
        unit_of_work=unit_of_work, store=asset_store, settings=ingestion_settings
    )


@pytest.fixture
def api_settings(tmp_path, database_url):  # type: ignore[no-untyped-def]
    """Settings for an app under test: the test database, a temp store, a manual worker."""
    from media_service.config import Settings

    return Settings(
        database_url=database_url,
        storage_root=tmp_path / "media",
        metadata_path=tmp_path / "metadata.json",
        job_dispatch_mode="manual",
    )


@pytest.fixture
def api_client(api_settings, unit_of_work, asset_store):  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from media_service.main import create_app
    from tests.support import StubOcrEngine

    app = create_app(
        settings=api_settings,
        ocr_engine=StubOcrEngine(),
        unit_of_work=unit_of_work,
        store=asset_store,
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def no_live_provider_calls(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Make a real provider call impossible unless the test asks for one (AC-016).

    A `fake` provider is a convention, and conventions get edited. This is the guarantee: every
    outbound HTTP send raises unless the test carries the `live` marker, so a provider that
    accidentally reaches the network fails loudly in CI instead of quietly spending money.
    """
    if request.node.get_closest_marker("live"):
        yield
        return

    import httpx

    def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "This test attempted a real HTTP request. Mark it with @pytest.mark.live if that is "
            "intended, or use a fake provider."
        )

    # The *network* transport, not `Client.send`. Starlette's TestClient is itself an httpx client
    # talking to the app in-process through its own transport, so patching `send` would block every
    # API test in the suite while proving nothing about outbound calls.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    yield


@pytest.fixture(autouse=True)
def isolated_provider_environment(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Keep a developer's exported credentials out of the tests.

    Without this, someone with LLM_PROVIDER=anthropic in their shell gets different behaviour from
    CI and no indication why.
    """
    for name in (
        "LLM_PROVIDER",
        "LLM_FAKE_MODE",
        "LLM_FAKE_FIXTURE_DIR",
        "LLM_STRUCTURED_MODE",
        "LLM_PROMPT_VERSION",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def extraction_service():  # type: ignore[no-untyped-def]
    from media_service.config import Settings
    from media_service.llm.registry import build_extraction_service

    return build_extraction_service(Settings())
