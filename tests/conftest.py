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
