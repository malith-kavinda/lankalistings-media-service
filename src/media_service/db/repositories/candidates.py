"""Candidate provenance -- where retry safety is enforced.

The uniqueness that makes a retry harmless lives on this table rather than on `advertisements`,
because the service performing the retry must own the constraint, and the advertisement row is the
part that later moves to the listing service.

`(item, generation, candidate_index)` is unique, so re-running a generation cannot produce a second
copy of its candidates (AC-006), and `(llm_run, candidate_index)` is unique, so replaying one run
cannot either (FR-CAN-004). A reviewer's hand-made candidate carries a NULL index, and NULLs are
distinct in a unique index, so any number of them can coexist for the same item.
"""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from media_service.db.base import utcnow
from media_service.db.engine import savepoint
from media_service.db.tables import AdvertisementProvenance

# A candidate a reviewer has acted on. Superseding one would discard a human decision, so a
# reprocess that would touch it is refused instead.
DECIDED_STATES = ("linked", "discarded")
UNDECIDED_STATE = "pending_publish"


class CandidateRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, provenance_id: str) -> AdvertisementProvenance | None:
        return self._session.get(AdvertisementProvenance, provenance_id)

    def list_for_item(
        self, item_id: str, *, generation: int | None = None
    ) -> list[AdvertisementProvenance]:
        statement = select(AdvertisementProvenance).where(
            AdvertisementProvenance.ingestion_item_id == item_id
        )
        if generation is not None:
            statement = statement.where(AdvertisementProvenance.generation == generation)
        return list(
            self._session.scalars(statement.order_by(AdvertisementProvenance.candidate_index))
        )

    def list_for_batch(self, batch_id: str) -> list[AdvertisementProvenance]:
        return list(
            self._session.scalars(
                select(AdvertisementProvenance)
                .where(AdvertisementProvenance.ingestion_batch_id == batch_id)
                .order_by(AdvertisementProvenance.id)
            )
        )

    def count_for(self, item_id: str, *, generation: int) -> int:
        return (
            self._session.scalar(
                select(func.count())
                .select_from(AdvertisementProvenance)
                .where(
                    AdvertisementProvenance.ingestion_item_id == item_id,
                    AdvertisementProvenance.generation == generation,
                )
            )
            or 0
        )

    def has_decided_candidate(self, item_id: str) -> bool:
        """Whether a reviewer has already accepted or rejected anything from this item."""
        return (
            self._session.scalar(
                select(func.count())
                .select_from(AdvertisementProvenance)
                .where(
                    AdvertisementProvenance.ingestion_item_id == item_id,
                    AdvertisementProvenance.candidate_state.in_(DECIDED_STATES),
                )
            )
            or 0
        ) > 0

    def record(
        self, provenance: AdvertisementProvenance
    ) -> tuple[AdvertisementProvenance, bool]:
        """Insert a candidate, or return the one a concurrent writer already inserted for it."""
        try:
            with savepoint(self._session):
                self._session.add(provenance)
                self._session.flush()
        except IntegrityError:
            existing = self._session.scalar(
                select(AdvertisementProvenance).where(
                    AdvertisementProvenance.ingestion_item_id == provenance.ingestion_item_id,
                    AdvertisementProvenance.generation == provenance.generation,
                    AdvertisementProvenance.candidate_index == provenance.candidate_index,
                )
            )
            if existing is None:  # pragma: no cover - a different constraint fired
                raise
            return existing, False
        return provenance, True

    def supersede_undecided(self, item_id: str, *, new_generation: int) -> int:
        """Retire candidates from earlier generations that nobody has ruled on.

        Only undecided ones. An approved or rejected candidate represents a decision a person made,
        and a reprocess is refused before it reaches here rather than quietly overwriting it.
        """
        result = self._session.execute(
            update(AdvertisementProvenance)
            .where(
                AdvertisementProvenance.ingestion_item_id == item_id,
                AdvertisementProvenance.generation < new_generation,
                AdvertisementProvenance.candidate_state == UNDECIDED_STATE,
            )
            .values(
                candidate_state="superseded",
                superseded_by_generation=new_generation,
                superseded_at=utcnow(),
            )
            .execution_options(synchronize_session=False)
        )
        return result.rowcount
