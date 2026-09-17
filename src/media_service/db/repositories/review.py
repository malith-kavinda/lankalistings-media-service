"""The review queue: finding candidates a person still has to rule on.

A queue query, not a listing. The difference shows in what it filters by -- batch, warning code,
confidence band -- because those are the questions a moderator working through a morning's scans
actually asks: *what came in from this batch*, *what did the pipeline flag*, *what is it least sure
about*. Sorting by confidence ascending by default follows from the same idea: the candidates most
likely to need a human are the ones a human should see first.

The join between `advertisement_provenance` and `advertisements` is on a plain indexed string, not a
foreign key, because the advertisement is the shim that later moves to the listing service. That is
also why the filters are split the way they are: batch, item and warning live on provenance, which
this service keeps; status, category and confidence live on the advertisement, which it does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from media_service.db.tables import Advertisement, AdvertisementProvenance, IngestionItem

# Ordering the queue by ascending confidence puts the least certain candidates first. A moderator
# working top-down then spends their attention where it changes the most outcomes.
DEFAULT_LIMIT = 25
MAX_LIMIT = 100


@dataclass(frozen=True, slots=True)
class ReviewFilters:
    batch_id: str | None = None
    item_id: str | None = None
    status: str | None = None
    category: str | None = None
    warning: str | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    # Undecided by default: the queue is work outstanding, not an archive.
    candidate_states: tuple[str, ...] = ("pending_publish",)
    query: str | None = None


@dataclass(frozen=True, slots=True)
class ReviewPage:
    rows: list[tuple[AdvertisementProvenance, Advertisement]]
    total: int
    limit: int
    offset: int


class ReviewRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def search(
        self,
        filters: ReviewFilters | None = None,
        *,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> ReviewPage:
        filters = filters or ReviewFilters()
        limit = max(1, min(limit, MAX_LIMIT))

        statement = self._filtered(
            select(AdvertisementProvenance, Advertisement).join(
                Advertisement, Advertisement.id == AdvertisementProvenance.advertisement_id
            ),
            filters,
        )

        total = (
            self._session.scalar(
                self._filtered(
                    select(func.count())
                    .select_from(AdvertisementProvenance)
                    .join(
                        Advertisement,
                        Advertisement.id == AdvertisementProvenance.advertisement_id,
                    ),
                    filters,
                )
            )
            or 0
        )

        rows = self._session.execute(
            statement.order_by(
                # NULLS LAST: a candidate with no confidence is not the most uncertain one, it is
                # an unmeasured one, and it should not monopolise the top of the queue.
                Advertisement.confidence_overall.asc().nullslast(),
                AdvertisementProvenance.id,
            )
            .limit(limit)
            .offset(offset)
        ).all()

        return ReviewPage(
            rows=[(provenance, advertisement) for provenance, advertisement in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def detail(
        self, advertisement_id: str
    ) -> tuple[AdvertisementProvenance, Advertisement] | None:
        return self._one(advertisement_id, lock=False)

    def lock(self, advertisement_id: str) -> tuple[AdvertisementProvenance, Advertisement] | None:
        """Read a candidate for a decision, holding the row until the transaction ends.

        The version column alone cannot make approve-then-approve safe: two requests can both read
        version 3, both find the candidate undecided, and both proceed. The row lock is what turns
        that into one winner and one caller who sees the state the first one left.
        """
        return self._one(advertisement_id, lock=True)

    def _one(
        self, advertisement_id: str, *, lock: bool
    ) -> tuple[AdvertisementProvenance, Advertisement] | None:
        statement = (
            select(AdvertisementProvenance, Advertisement)
            .join(Advertisement, Advertisement.id == AdvertisementProvenance.advertisement_id)
            .where(Advertisement.id == advertisement_id)
        )
        if lock:
            # Only the two rows this statement selects. A blanket FOR UPDATE would also lock rows
            # joined for reading, which here is nothing, but the narrower form says so explicitly.
            statement = statement.with_for_update(of=[Advertisement, AdvertisementProvenance])
        row = self._session.execute(statement).first()
        return (row[0], row[1]) if row else None

    def siblings(self, item_id: str, *, generation: int) -> list[str]:
        """Every candidate from the same image, in order.

        What makes "Ad 2 of 5 from page-3.png" possible, and what the previous/next controls walk.
        """
        return list(
            self._session.scalars(
                select(AdvertisementProvenance.advertisement_id)
                .where(
                    AdvertisementProvenance.ingestion_item_id == item_id,
                    AdvertisementProvenance.generation == generation,
                    AdvertisementProvenance.advertisement_id.is_not(None),
                )
                .order_by(AdvertisementProvenance.candidate_index)
            )
        )

    def item_of(self, item_id: str) -> IngestionItem | None:
        return self._session.get(IngestionItem, item_id)

    def counts_by_status(self, *, batch_id: str | None = None) -> dict[str, int]:
        statement = (
            select(Advertisement.status, func.count())
            .select_from(AdvertisementProvenance)
            .join(Advertisement, Advertisement.id == AdvertisementProvenance.advertisement_id)
            .group_by(Advertisement.status)
        )
        if batch_id is not None:
            statement = statement.where(AdvertisementProvenance.ingestion_batch_id == batch_id)
        return {status: count for status, count in self._session.execute(statement).all()}

    # -- filtering -----------------------------------------------------------------------------

    @staticmethod
    def _filtered(statement: Select[Any], filters: ReviewFilters) -> Select[Any]:
        if filters.candidate_states:
            statement = statement.where(
                AdvertisementProvenance.candidate_state.in_(filters.candidate_states)
            )
        if filters.batch_id:
            statement = statement.where(
                AdvertisementProvenance.ingestion_batch_id == filters.batch_id
            )
        if filters.item_id:
            statement = statement.where(
                AdvertisementProvenance.ingestion_item_id == filters.item_id
            )
        if filters.status:
            statement = statement.where(Advertisement.status == filters.status)
        if filters.category:
            statement = statement.where(Advertisement.category == filters.category)
        if filters.warning:
            # A PostgreSQL array containment check, which the GIN index on `warning_codes` serves.
            # Matching either side because a warning can be raised about the page or about the
            # candidate, and a moderator filtering for one means "show me anything flagged that
            # way".
            statement = statement.where(
                or_(
                    AdvertisementProvenance.warning_codes.any(filters.warning),
                    Advertisement.warning_codes.any(filters.warning),
                )
            )
        if filters.min_confidence is not None:
            statement = statement.where(
                Advertisement.confidence_overall >= filters.min_confidence
            )
        if filters.max_confidence is not None:
            statement = statement.where(
                Advertisement.confidence_overall <= filters.max_confidence
            )
        if filters.query:
            pattern = f"%{filters.query}%"
            statement = statement.where(
                or_(
                    Advertisement.title.ilike(pattern),
                    Advertisement.description.ilike(pattern),
                    Advertisement.location.ilike(pattern),
                )
            )
        return statement
