"""Alembic environment.

The database URL comes from the application settings, not from alembic.ini, so migrations and the
running service can never disagree about which database they are pointed at.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context

from media_service.config import get_settings

# Importing the tables module is what populates Base.metadata. Without it autogenerate sees an empty
# schema and helpfully offers to drop every table.
from media_service.db import tables  # noqa: F401
from media_service.db.base import Base
from media_service.db.engine import build_engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        engine = build_engine(_database_url(), pool_size=1)
        with engine.connect() as connection:
            _run(connection)
        engine.dispose()
    else:
        _run(connectable)


def _run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
