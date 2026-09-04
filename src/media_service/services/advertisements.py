import base64
import re
from dataclasses import replace
from io import BytesIO
from uuid import uuid4

from PIL import Image, UnidentifiedImageError

from media_service.api.errors import ServiceError
from media_service.domain.models import Advertisement, AdvertisementStatus
from media_service.domain.repository import MediaRepository
from media_service.services.ocr import SUPPORTED_IMAGE_TYPES, OcrEngine

CATEGORIES = ("Vehicles", "Property", "Electronics", "Jobs", "Home", "Land")
CATEGORY_KEYWORDS = {
    "Vehicles": ("vehicle", "car", "van", "bike", "prius", "toyota", "honda", "වාහන"),
    "Property": ("house", "land", "rent", "apartment", "property", "නිවස", "ඉඩම"),
    "Electronics": ("phone", "iphone", "laptop", "tv", "electronics", "දුරකථන"),
    "Jobs": ("job", "vacancy", "executive", "driver", "රැකියා", "ඇබෑර්තු"),
}
LOCATION_KEYWORDS = (
    "Colombo",
    "Kandy",
    "Galle",
    "Matara",
    "Jaffna",
    "Kurunegala",
    "Negombo",
    "Gampaha",
)
PRICE_PATTERN = re.compile(r"(?:rs\.?|රු\.?)\s*[\d,]+", re.IGNORECASE)
LOCATION_PATTERN = re.compile(
    r"\b(?:Colombo|Kandy|Galle|Matara|Jaffna|Kurunegala|Negombo|Gampaha)\s*\d{0,2}\b",
    re.IGNORECASE,
)


class AdvertisementService:
    def __init__(self, *, repository: MediaRepository, ocr_engine: OcrEngine | None = None) -> None:
        self._repository = repository
        self._ocr_engine = ocr_engine

    def create(
        self,
        *,
        title: str,
        price: str,
        category: str,
        location: str,
        description: str,
        image_bytes: bytes,
        content_type: str,
    ) -> Advertisement:
        self._validate_advertisement_fields(
            title=title,
            price=price,
            category=category,
            location=location,
            description=description,
        )
        self._validate_image(image_bytes=image_bytes, content_type=content_type)

        advertisement = Advertisement.create(
            advertisement_id=str(uuid4()),
            title=title.strip(),
            price=price.strip(),
            category=category.strip(),
            location=location.strip(),
            description=description.strip(),
            image_url=self._image_data_url(image_bytes=image_bytes, content_type=content_type),
        )
        self._repository.save_advertisement(advertisement)
        return advertisement

    def extract_newspaper_article(
        self,
        *,
        image_bytes: bytes,
        content_type: str,
    ) -> tuple[Advertisement, str]:
        if self._ocr_engine is None:
            raise ServiceError(
                status_code=503,
                code="OCR_UNAVAILABLE",
                message="OCR is not configured for advertisement extraction.",
            )

        self._validate_image(image_bytes=image_bytes, content_type=content_type)
        ocr_output = self._ocr_engine.extract_text(image_bytes, content_type=content_type)
        fields = self._extract_advertisement_fields(ocr_output.text)
        advertisement = Advertisement.create(
            advertisement_id=str(uuid4()),
            title=fields["title"],
            price=fields["price"],
            category=fields["category"],
            location=fields["location"],
            description=fields["description"],
            image_url=self._image_data_url(image_bytes=image_bytes, content_type=content_type),
            status=AdvertisementStatus.PENDING_REVIEW,
            source_text=ocr_output.text,
            extraction_confidence=ocr_output.confidence,
        )
        self._repository.save_advertisement(advertisement)
        return advertisement, ocr_output.text

    def list(self, *, status: AdvertisementStatus | None = None) -> list[Advertisement]:
        advertisements = self._repository.list_advertisements()
        if status is None:
            return advertisements
        return [
            advertisement
            for advertisement in advertisements
            if advertisement.status == status
        ]

    def update(
        self,
        *,
        advertisement_id: str,
        title: str,
        price: str,
        category: str,
        location: str,
        description: str,
    ) -> Advertisement:
        advertisement = self._get_existing_advertisement(advertisement_id)
        self._validate_advertisement_fields(
            title=title,
            price=price,
            category=category,
            location=location,
            description=description,
        )
        updated = replace(
            advertisement,
            title=title.strip(),
            price=price.strip(),
            category=category.strip(),
            location=location.strip(),
            description=description.strip(),
        )
        self._repository.save_advertisement(updated)
        return updated

    def approve(self, advertisement_id: str) -> Advertisement:
        advertisement = self._get_existing_advertisement(advertisement_id)
        approved = replace(advertisement, status=AdvertisementStatus.ACTIVE)
        self._repository.save_advertisement(approved)
        return approved

    def _get_existing_advertisement(self, advertisement_id: str) -> Advertisement:
        advertisement = self._repository.get_advertisement(advertisement_id)
        if advertisement is None:
            raise ServiceError(
                status_code=404,
                code="ADVERTISEMENT_NOT_FOUND",
                message="Advertisement was not found.",
                details=[
                    {
                        "field": "advertisement_id",
                        "code": "NOT_FOUND",
                        "message": "No advertisement exists for this id.",
                    }
                ],
            )
        return advertisement

    @staticmethod
    def _extract_advertisement_fields(text: str) -> dict[str, str]:
        lines = [line.strip(" -:") for line in text.splitlines() if line.strip(" -:")]
        text_for_matching = " ".join(lines)
        price_match = PRICE_PATTERN.search(text_for_matching)
        price = price_match.group(0).replace("Rs ", "Rs. ") if price_match else "Price pending"
        title = next(
            (line for line in lines if not PRICE_PATTERN.search(line) and len(line) > 3),
            "Newspaper advertisement",
        )
        location_match = LOCATION_PATTERN.search(text_for_matching)
        location = (
            location_match.group(0).strip()
            if location_match
            else next(
                (
                    location
                    for location in LOCATION_KEYWORDS
                    if location.lower() in text_for_matching.lower()
                ),
                "Location pending",
            )
        )
        category = AdvertisementService._detect_category(text_for_matching)
        description = text_for_matching or "OCR text pending reviewer verification."

        return {
            "title": title[:120],
            "price": price,
            "category": category,
            "location": location,
            "description": description,
        }

    @staticmethod
    def _detect_category(text: str) -> str:
        normalized = text.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword.lower() in normalized for keyword in keywords):
                return category
        return "Home"

    @staticmethod
    def _validate_advertisement_fields(
        *,
        title: str,
        price: str,
        category: str,
        location: str,
        description: str,
    ) -> None:
        AdvertisementService._validate_text("title", title)
        AdvertisementService._validate_text("price", price)
        AdvertisementService._validate_text("category", category)
        AdvertisementService._validate_text("location", location)
        AdvertisementService._validate_text("description", description)

    @staticmethod
    def _validate_text(field: str, value: str) -> None:
        if not value.strip():
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": field,
                        "code": "REQUIRED",
                        "message": f"{field.replace('_', ' ').title()} is required.",
                    }
                ],
            )

    @staticmethod
    def _validate_image(*, image_bytes: bytes, content_type: str) -> None:
        if content_type not in SUPPORTED_IMAGE_TYPES:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "image",
                        "code": "UNSUPPORTED_CONTENT_TYPE",
                        "message": "Only JPEG, PNG, WebP, TIFF, and BMP images are supported.",
                    }
                ],
            )

        try:
            Image.open(BytesIO(image_bytes)).verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ServiceError(
                status_code=422,
                code="VALIDATION_FAILED",
                message="Request validation failed.",
                details=[
                    {
                        "field": "image",
                        "code": "INVALID_IMAGE",
                        "message": "Uploaded file could not be decoded as an image.",
                    }
                ],
            ) from exc

    @staticmethod
    def _image_data_url(*, image_bytes: bytes, content_type: str) -> str:
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{content_type};base64,{encoded_image}"
