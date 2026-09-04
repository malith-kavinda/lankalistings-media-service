import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from media_service.domain.models import (
    Advertisement,
    AdvertisementStatus,
    ExtractionResult,
    ExtractionStatus,
    MediaAsset,
)


class MediaRepository(Protocol):
    def save_asset(self, asset: MediaAsset) -> None: ...

    def save_extraction(self, extraction: ExtractionResult) -> None: ...

    def get_asset(self, asset_id: str) -> MediaAsset | None: ...

    def get_extraction(self, extraction_id: str) -> ExtractionResult | None: ...

    def save_advertisement(self, advertisement: Advertisement) -> None: ...

    def get_advertisement(self, advertisement_id: str) -> Advertisement | None: ...

    def list_advertisements(self) -> list[Advertisement]: ...


class InMemoryMediaRepository:
    def __init__(self) -> None:
        self._assets: dict[str, MediaAsset] = {}
        self._extractions: dict[str, ExtractionResult] = {}
        self._advertisements: dict[str, Advertisement] = {}

    def save_asset(self, asset: MediaAsset) -> None:
        self._assets[asset.id] = asset

    def save_extraction(self, extraction: ExtractionResult) -> None:
        self._extractions[extraction.id] = extraction

    def get_asset(self, asset_id: str) -> MediaAsset | None:
        return self._assets.get(asset_id)

    def get_extraction(self, extraction_id: str) -> ExtractionResult | None:
        return self._extractions.get(extraction_id)

    def save_advertisement(self, advertisement: Advertisement) -> None:
        self._advertisements[advertisement.id] = advertisement

    def get_advertisement(self, advertisement_id: str) -> Advertisement | None:
        return self._advertisements.get(advertisement_id)

    def list_advertisements(self) -> list[Advertisement]:
        return sorted(
            self._advertisements.values(),
            key=lambda advertisement: advertisement.created_at,
            reverse=True,
        )


class JsonMediaRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._assets: dict[str, MediaAsset] = {}
        self._extractions: dict[str, ExtractionResult] = {}
        self._advertisements: dict[str, Advertisement] = {}
        self._load()

    def save_asset(self, asset: MediaAsset) -> None:
        self._assets[asset.id] = asset
        self._flush()

    def save_extraction(self, extraction: ExtractionResult) -> None:
        self._extractions[extraction.id] = extraction
        self._flush()

    def get_asset(self, asset_id: str) -> MediaAsset | None:
        return self._assets.get(asset_id)

    def get_extraction(self, extraction_id: str) -> ExtractionResult | None:
        return self._extractions.get(extraction_id)

    def save_advertisement(self, advertisement: Advertisement) -> None:
        self._advertisements[advertisement.id] = advertisement
        self._flush()

    def get_advertisement(self, advertisement_id: str) -> Advertisement | None:
        return self._advertisements.get(advertisement_id)

    def list_advertisements(self) -> list[Advertisement]:
        return sorted(
            self._advertisements.values(),
            key=lambda advertisement: advertisement.created_at,
            reverse=True,
        )

    def _load(self) -> None:
        if not self._path.exists():
            return

        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._assets = {
            asset_id: self._asset_from_dict(asset)
            for asset_id, asset in data.get("assets", {}).items()
        }
        self._extractions = {
            extraction_id: self._extraction_from_dict(extraction)
            for extraction_id, extraction in data.get("extractions", {}).items()
        }
        self._advertisements = {
            advertisement_id: self._advertisement_from_dict(advertisement)
            for advertisement_id, advertisement in data.get("advertisements", {}).items()
        }

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "assets": {
                asset_id: self._serialize(asset) for asset_id, asset in self._assets.items()
            },
            "extractions": {
                extraction_id: self._serialize(extraction)
                for extraction_id, extraction in self._extractions.items()
            },
            "advertisements": {
                advertisement_id: self._serialize(advertisement)
                for advertisement_id, advertisement in self._advertisements.items()
            },
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def _serialize(value: MediaAsset | ExtractionResult | Advertisement) -> dict[str, object]:
        data = asdict(value)
        data["created_at"] = value.created_at.isoformat()
        if isinstance(value, ExtractionResult):
            data["status"] = value.status.value
        if isinstance(value, Advertisement):
            data["status"] = value.status.value
        return data

    @staticmethod
    def _asset_from_dict(data: dict[str, object]) -> MediaAsset:
        return MediaAsset(
            id=str(data["id"]),
            content_type=str(data["content_type"]),
            byte_size=int(data["byte_size"]),
            checksum=str(data["checksum"]),
            width=int(data["width"]) if data.get("width") is not None else None,
            height=int(data["height"]) if data.get("height") is not None else None,
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )

    @staticmethod
    def _extraction_from_dict(data: dict[str, object]) -> ExtractionResult:
        return ExtractionResult(
            id=str(data["id"]),
            asset_id=str(data["asset_id"]),
            status=ExtractionStatus(str(data["status"])),
            raw_text=str(data["raw_text"]),
            engine=str(data["engine"]),
            model_version=str(data["model_version"]),
            language=str(data["language"]),
            confidence=str(data["confidence"]),
            processing_ms=int(data["processing_ms"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
        )

    @staticmethod
    def _advertisement_from_dict(data: dict[str, object]) -> Advertisement:
        return Advertisement(
            id=str(data["id"]),
            title=str(data["title"]),
            price=str(data["price"]),
            category=str(data["category"]),
            location=str(data["location"]),
            description=str(data["description"]),
            image_url=str(data["image_url"]),
            status=AdvertisementStatus(str(data["status"])),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            source_text=str(data.get("source_text", "")),
            extraction_confidence=str(data.get("extraction_confidence", "manual")),
        )
