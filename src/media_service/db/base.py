"""Declarative base, naming conventions, and portable column types."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, MetaData, Text, TypeDecorator
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase

# Explicit constraint names. Without these, Alembic emits migrations that reference constraints it
# cannot name, and a later `DROP CONSTRAINT` has nothing to target.
#
# PostgreSQL truncates identifiers at 63 bytes, so any constraint spanning several columns must be
# given a short explicit name at the call site rather than relying on `column_0_N_name`. A test
# walks the metadata and fails if a generated name would exceed the limit.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s__%(column_0_N_name)s",
    "uq": "uq_%(table_name)s__%(column_0_N_name)s",
    "ck": "ck_%(table_name)s__%(constraint_name)s",
    "fk": "fk_%(table_name)s__%(column_0_N_name)s__%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MAX_IDENTIFIER_LENGTH = 63


class TZDateTime(TypeDecorator[datetime]):
    """A timestamp that refuses to store a naive datetime.

    PostgreSQL raises when a naive value is compared with a `timestamptz`, and the failure
    surfaces far from the code that produced it. Rejecting at the persistence boundary means the
    traceback points at the writer instead.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "Naive datetime rejected at the persistence boundary. Use datetime.now(UTC)."
            )
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        return value if value.tzinfo else value.replace(tzinfo=UTC)


# Plain `json` has no equality operator in PostgreSQL, so DISTINCT, GROUP BY, and UNION over such a
# column fail at runtime. JSONB has one, and can be indexed.
JsonDoc = postgresql.JSONB


def utcnow() -> datetime:
    """The single clock for column defaults.

    Deliberately not `server_default=func.now()`: that returns the server's timezone rather than
    UTC, putting row timestamps on a different clock from the application's.
    """
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    type_annotation_map = {
        datetime: TZDateTime,
        dict[str, Any]: JsonDoc,
        list[Any]: JsonDoc,
        str: Text,
    }
