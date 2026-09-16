"""Service configuration.

`Settings` stays a plain Pydantic model that can be constructed directly, because every existing
test
does exactly that. Environment reading lives in `get_settings()`.

The model's defaults and the environment defaults are the same values. They were not:
`tesseract_lang` defaulted to "eng" on the model while `get_settings()` read "sin+eng" from the
environment, so any test constructing `Settings(...)` silently ran English-only OCR and nothing
caught it.
"""

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

DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://lankalistings:lankalistings@localhost:5434/lankalistings"
)
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"
)

# PRD 15.1. The total is binding and is checked first: 25 images at the per-image maximum would be
# 250 MiB, which the total forbids.
# Closed vocabularies for the switches. A typo in any of these is a configuration error the
# service must die on rather than silently fall back from -- falling back from an auth mode would
# fail open.
OPERATOR_AUTH_MODES = ("none", "static_token")
JOB_DISPATCH_MODES = ("local_pool", "inline", "manual", "none")
MEDIA_REPOSITORIES = ("json", "sql")

DEFAULT_MAX_IMAGES_PER_BATCH = 25
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_BATCH_BYTES = 100 * 1024 * 1024


def _env_bool(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}.") from exc


class Settings(BaseModel):
    service_name: str = "media-service"
    api_prefix: str = "/api/v1"
    environment: str = "local"

    # OCR runtime
    max_upload_bytes: int = Field(default=DEFAULT_MAX_IMAGE_BYTES, gt=0)
    tesseract_cmd: str | None = None
    tesseract_lang: str = "sin+eng"
    tesseract_data_dir: Path | None = None

    # Batch limits (PRD 15.1)
    max_images_per_batch: int = Field(default=DEFAULT_MAX_IMAGES_PER_BATCH, gt=0)
    max_image_bytes: int = Field(default=DEFAULT_MAX_IMAGE_BYTES, gt=0)
    max_batch_bytes: int = Field(default=DEFAULT_MAX_BATCH_BYTES, gt=0)

    # Persistence
    database_url: str = DEFAULT_DATABASE_URL
    database_echo: bool = False
    metadata_path: Path = Path(".data/media_metadata.json")
    storage_root: Path = Path(".data/media")

    # Worker and queue
    job_dispatch_mode: str = "local_pool"
    worker_concurrency: int = Field(default=2, gt=0)
    lease_seconds: int = Field(default=300, gt=0)
    heartbeat_seconds: int = Field(default=30, gt=0)
    reaper_interval_ms: int = Field(default=15_000, gt=0)
    poll_interval_ms: int = Field(default=2_000, gt=0)
    max_item_attempts: int = Field(default=3, gt=0)
    worker_single_instance: bool = True

    # Access control
    operator_auth_mode: str = "none"
    operator_api_token: str | None = None

    # Wire compatibility
    media_repository: str = "json"
    public_image_url_mode: str = "data_url"
    advertisement_status_wire: str = "legacy"

    cors_origins: list[str] = Field(default_factory=list)

    @property
    def is_local_or_test(self) -> bool:
        return self.environment in {"local", "test"}

    def validate_startup(self) -> None:
        """Refuse to boot on a configuration that would fail later, or fail open.

        Checked at startup rather than at first use: a service that accepts uploads for an hour and
        then discovers its auth mode is a typo has already let them through.
        """
        if self.operator_auth_mode not in OPERATOR_AUTH_MODES:
            raise ValueError(
                f"OPERATOR_AUTH_MODE={self.operator_auth_mode!r} is not one of "
                f"{', '.join(sorted(OPERATOR_AUTH_MODES))}."
            )
        if self.operator_auth_mode == "static_token" and not self.operator_api_token:
            raise ValueError(
                "OPERATOR_AUTH_MODE=static_token requires OPERATOR_API_TOKEN to be set."
            )
        if self.job_dispatch_mode not in JOB_DISPATCH_MODES:
            raise ValueError(
                f"JOB_DISPATCH_MODE={self.job_dispatch_mode!r} is not one of "
                f"{', '.join(sorted(JOB_DISPATCH_MODES))}."
            )
        if self.media_repository not in MEDIA_REPOSITORIES:
            raise ValueError(
                f"MEDIA_REPOSITORY={self.media_repository!r} is not one of "
                f"{', '.join(sorted(MEDIA_REPOSITORIES))}."
            )
        if self.operator_auth_mode == "none" and not self.is_local_or_test:
            raise ValueError(
                "OPERATOR_AUTH_MODE=none is only allowed in local and test environments."
            )


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
    cors_origins = [
        origin.strip()
        for origin in getenv("MEDIA_SERVICE_CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
        if origin.strip()
    ]
    return Settings(
        environment=getenv("MEDIA_SERVICE_ENV", "local"),
        max_upload_bytes=_env_int("MEDIA_SERVICE_MAX_UPLOAD_BYTES", DEFAULT_MAX_IMAGE_BYTES),
        tesseract_cmd=_detect_tesseract_cmd(),
        tesseract_lang=getenv("TESSERACT_LANG", "sin+eng"),
        tesseract_data_dir=_detect_tessdata_dir(),
        max_images_per_batch=_env_int("MAX_IMAGES_PER_BATCH", DEFAULT_MAX_IMAGES_PER_BATCH),
        max_image_bytes=_env_int("MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES),
        max_batch_bytes=_env_int("MAX_BATCH_BYTES", DEFAULT_MAX_BATCH_BYTES),
        database_url=getenv("DATABASE_URL", DEFAULT_DATABASE_URL),
        database_echo=_env_bool("DATABASE_ECHO", False),
        metadata_path=Path(getenv("MEDIA_SERVICE_METADATA_PATH", ".data/media_metadata.json")),
        storage_root=Path(getenv("MEDIA_STORAGE_ROOT", ".data/media")),
        job_dispatch_mode=getenv("JOB_DISPATCH_MODE", "local_pool"),
        worker_concurrency=_env_int("MAX_WORKER_CONCURRENCY", 2),
        lease_seconds=_env_int("LEASE_SECONDS", 300),
        heartbeat_seconds=_env_int("HEARTBEAT_SECONDS", 30),
        reaper_interval_ms=_env_int("REAPER_INTERVAL_MS", 15_000),
        poll_interval_ms=_env_int("POLL_INTERVAL_MS", 2_000),
        max_item_attempts=_env_int("MAX_ITEM_ATTEMPTS", 3),
        worker_single_instance=_env_bool("WORKER_SINGLE_INSTANCE", True),
        operator_auth_mode=getenv("OPERATOR_AUTH_MODE", "none"),
        operator_api_token=getenv("OPERATOR_API_TOKEN"),
        media_repository=getenv("MEDIA_REPOSITORY", "json"),
        public_image_url_mode=getenv("PUBLIC_IMAGE_URL_MODE", "data_url"),
        advertisement_status_wire=getenv("ADVERTISEMENT_STATUS_WIRE", "legacy"),
        cors_origins=cors_origins,
    )


def reset_settings_cache() -> None:
    """Drop the cached settings so a test's monkeypatched environment is visible."""
    get_settings.cache_clear()
