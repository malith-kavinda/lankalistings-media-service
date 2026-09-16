"""The dispatch seam.

`enqueue_*` is a **latency hint, never the source of truth**. Every dispatcher is paired with a
poller that finds claimable rows on its own, so a lost notification costs a poll interval and
nothing else. That single property is what removes any need for a transactional outbox here, and it
is why the notification may safely be sent outside the transaction that created the work.

It must be sent *after* commit, though. Publishing before commit is the dual-write bug in its
classic form: the worker claims a row the publisher has not committed, finds nothing, and the work
waits for a poll that was supposed to be the fallback.

Delivery is **at-least-once dispatch**, and the database enforces artifact uniqueness per
generation, so the pair gives **effectively-once artifacts**. Nothing here needs to be made
exactly-once; if that looks like a bug later, it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class JobDispatcher(Protocol):
    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        """Hint that one item is claimable now."""

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        """Hint that a whole batch is claimable now."""

    def start(self) -> None: ...

    def stop(self, *, timeout: float = 30.0) -> None: ...


class NullDispatcher:
    """Accepts hints and does nothing.

    For read-only deployments and for tests that exercise the API without running any pipeline.
    """

    def enqueue_item(self, item_id: str, *, correlation_id: str | None = None) -> None:
        return None

    def enqueue_batch(
        self, batch_id: str, item_ids: Sequence[str], *, correlation_id: str | None = None
    ) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self, *, timeout: float = 30.0) -> None:
        return None
