"""Engine, session factory, and the savepoint helper."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker


def build_engine(url: str, *, echo: bool = False, pool_size: int = 10) -> Engine:
    return create_engine(
        url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=5,
        # The worker holds connections idle across OCR and provider calls, which can run for tens of
        # seconds. Without pre-ping, a container restart or an idle timeout surfaces as a failure in
        # the middle of a transaction rather than as a transparent reconnect.
        pool_pre_ping=True,
        future=True,
    )


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        # The worker reads attributes after committing a stage. With the default, every such read
        # re-queries through a session that may already be closed.
        expire_on_commit=False,
        autoflush=False,
        future=True,
    )


@lru_cache(maxsize=8)
def cached_engine(url: str) -> Engine:
    return build_engine(url)


@contextmanager
def savepoint(session: Session) -> Iterator[None]:
    """Run a statement that may violate a unique constraint, without killing the transaction.

    PostgreSQL aborts the *entire* transaction on a failed statement: every subsequent statement
    then fails with "current transaction is aborted" until rollback. A bare `except
    IntegrityError: pass` therefore leaves a session that looks alive and silently rejects
    everything after it.

    Wrapping the insert in a SAVEPOINT confines the damage, so the caller can catch IntegrityError
    and re-select the winning row:

        try:
            with savepoint(session):
                session.add(asset)
                session.flush()
        except IntegrityError:
            asset = session.scalar(select(MediaAsset).where(...))
    """
    nested = session.begin_nested()
    try:
        yield
    except IntegrityError:
        nested.rollback()
        raise
    else:
        nested.commit()
