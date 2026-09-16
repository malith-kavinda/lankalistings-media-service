"""Row builders for database tests.

Every factory fills only what the schema requires and leaves the rest at its default, so a test
reads as the one thing it is varying rather than as a wall of boilerplate that hides it.
"""

from __future__ import annotations

from hashlib import sha256

from media_service.db.tables import IngestionBatch, IngestionItem, MediaAsset
from media_service.domain.ids import new_asset_id, new_batch_id, new_item_id
from media_service.storage import original_key


def make_asset(*, content: str = "bytes", content_type: str = "image/png") -> MediaAsset:
    checksum = sha256(content.encode()).hexdigest()
    return MediaAsset(
        id=new_asset_id(),
        storage_key=original_key(checksum, content_type),
        content_type=content_type,
        byte_size=max(len(content), 1),
        checksum_sha256=checksum,
        width=100,
        height=50,
    )


def make_batch(*, created_by: str = "operator-1", total_items: int = 0) -> IngestionBatch:
    return IngestionBatch(
        id=new_batch_id(),
        created_by=created_by,
        total_items=total_items,
    )


def make_item(
    *,
    batch: IngestionBatch,
    asset: MediaAsset,
    item_index: int = 0,
    filename: str = "page-1.png",
) -> IngestionItem:
    return IngestionItem(
        id=new_item_id(),
        batch_id=batch.id,
        item_index=item_index,
        source_asset_id=asset.id,
        original_filename=filename,
        declared_content_type=asset.content_type,
    )


def seed_batch_with_items(session, *, count: int = 1, created_by: str = "operator-1"):  # type: ignore[no-untyped-def]
    """A committed batch with `count` queued items, the starting point for most queue tests.

    The flush between the parents and the items is load-bearing. None of the tables declare ORM
    relationships -- cross-references are plain indexed strings by design -- so SQLAlchemy has no
    mapper dependency to sort a single flush by, and the items would be inserted before the rows
    their foreign keys point at. Production code orders its writes for the same reason.
    """
    batch = make_batch(created_by=created_by, total_items=count)
    session.add(batch)
    assets = [
        make_asset(content=f"image-{created_by}-{batch.id}-{index}") for index in range(count)
    ]
    session.add_all(assets)
    session.flush()

    items = [
        make_item(batch=batch, asset=asset, item_index=index, filename=f"page-{index}.png")
        for index, asset in enumerate(assets)
    ]
    session.add_all(items)
    session.commit()
    return batch, items
