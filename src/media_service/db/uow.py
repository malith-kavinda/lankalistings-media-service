"""Unit of work: one session, one transaction, one thread.

The worker has no request scope, so FastAPI's `Depends` cannot own session lifetime for it. This
context manager does instead, and the rule it enforces is short: *the session lives exactly as long
as the `with` block, on the thread that opened it.* A session handed across threads is the classic
silent-corruption bug in a design like this one -- two threads interleaving statements on one
connection -- and it is prevented by never having a session that outlives its block.

Commit is explicit. `__exit__` rolls back unless `commit()` was called, so an exception on any path
leaves nothing half-written, and a caller that simply forgets is treated as a failure rather than
as a silent partial write.

One unit of work per **stage boundary**, never one per item. An item takes 10-60 seconds end to
end; a transaction held open across Tesseract and a provider call pins a connection, holds row
locks the reaper and the progress endpoint want, and shows up as `idle in transaction`. The long
calls run with no transaction open at all.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType

from sqlalchemy.orm import Session, sessionmaker

from media_service.db.repositories.artifacts import ArtifactRepository
from media_service.db.repositories.assets import AssetRepository
from media_service.db.repositories.batches import BatchRepository
from media_service.db.repositories.candidates import CandidateRepository
from media_service.db.repositories.events import ReviewEventRepository
from media_service.db.repositories.items import ItemRepository


class UnitOfWork:
    """Repositories bound to one session, committed or rolled back as a whole."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.batches = BatchRepository(session)
        self.items = ItemRepository(session)
        self.assets = AssetRepository(session)
        self.artifacts = ArtifactRepository(session)
        self.candidates = CandidateRepository(session)
        self.events = ReviewEventRepository(session)
        self._committed = False

    def commit(self) -> None:
        self.session.commit()
        self._committed = True

    def rollback(self) -> None:
        self.session.rollback()

    def flush(self) -> None:
        self.session.flush()

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None or not self._committed:
            self.session.rollback()
        self.session.close()


class UnitOfWorkFactory:
    """Opens units of work from a session factory.

    Held by the app and by the worker. Both need to start a unit of work without knowing how the
    engine was built, and tests need to substitute one that points at a different database.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def __call__(self) -> Iterator[UnitOfWork]:
        unit = UnitOfWork(self._session_factory())
        with unit:
            yield unit
