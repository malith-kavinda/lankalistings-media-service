from functools import lru_cache
from os import getenv
from pathlib import Path

from pydantic import BaseModel, Field

WINDOWS_TESSERACT_PATHS = (
    Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
    Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_TESSDATA_DIR = PROJECT_ROOT / ".tessdata"


class Settings(BaseModel):
    service_name: str = "media-service"
    api_prefix: str = "/api/v1"
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    tesseract_cmd: str | None = None
    tesseract_lang: str = "eng"
    tesseract_data_dir: Path | None = None
    metadata_path: Path = Path(".data/media_metadata.json")
    cors_origins: list[str] = Field(default_factory=list)


def _detect_tesseract_cmd() -> str | None:
    explicit_path = getenv("TESSERACT_CMD")
    if explicit_path:
        return explicit_path

    for path in WINDOWS_TESSERACT_PATHS:
        if path.exists():
            return str(path)

    return None


def _detect_tessdata_dir() -> Path | None:
    explicit_path = getenv("TESSERACT_TESSDATA_DIR")
    if explicit_path:
        return Path(explicit_path)

    if LOCAL_TESSDATA_DIR.exists():
        return LOCAL_TESSDATA_DIR

    return None


@lru_cache
def get_settings() -> Settings:
    max_upload_bytes = int(getenv("MEDIA_SERVICE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    cors_origins = [
        origin.strip()
        for origin in getenv(
            "MEDIA_SERVICE_CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    return Settings(
        max_upload_bytes=max_upload_bytes,
        tesseract_cmd=_detect_tesseract_cmd(),
        tesseract_lang=getenv("TESSERACT_LANG", "sin+eng"),
        tesseract_data_dir=_detect_tessdata_dir(),
        metadata_path=Path(getenv("MEDIA_SERVICE_METADATA_PATH", ".data/media_metadata.json")),
        cors_origins=cors_origins,
    )
