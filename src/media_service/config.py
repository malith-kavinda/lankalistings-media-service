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

from pydantic import BaseModel, Field, SecretStr

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
# `legacy` renames the stored `pending` to `pending_review` on the wire (PRD 10.3); `canonical`
# publishes the stored vocabulary unchanged. Validated because a typo would silently pick
# `canonical` and break the portal's only status check with nothing in the logs.
ADVERTISEMENT_STATUS_WIRES = ("legacy", "canonical")
OCR_PROVIDERS = ("tesseract", "paddle_tesseract", "vision_llm")
LLM_PROVIDERS = ("openai_compatible", "gemini", "anthropic", "fake", "rule_based")
LLM_STRUCTURED_MODES = ("auto", "json_schema", "json_object")
LLM_FAKE_MODES = ("rule_based", "empty", "always_invalid", "fixture")

DEFAULT_MAX_IMAGES_PER_BATCH = 25
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_BATCH_BYTES = 100 * 1024 * 1024

# PRD 15.3. An explicit cap rather than Pillow's default, which only *warns* between its limit and
# twice its limit -- a band where a highly compressible image passes validation under the byte cap
# and is then decoded in full by the worker.
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


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


def _env_float(name: str, default: float) -> float:
    raw = getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from exc


def _env_optional_int(name: str) -> int | None:
    """An unset value and an empty value both mean "off", not zero."""
    raw = getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or empty, got {raw!r}.") from exc


def _env_path(name: str) -> Path | None:
    raw = getenv(name)
    return Path(raw) if raw and raw.strip() else None


def _env_secret(name: str) -> SecretStr | None:
    raw = getenv(name)
    return SecretStr(raw) if raw and raw.strip() else None


def _env_optional_bool(name: str) -> bool | None:
    """Unset means "use the environment's answer", which is not the same as False."""
    raw = getenv(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


class Settings(BaseModel):
    service_name: str = "media-service"
    api_prefix: str = "/api/v1"
    environment: str = "local"

    # OCR runtime
    max_upload_bytes: int = Field(default=DEFAULT_MAX_IMAGE_BYTES, gt=0)
    # Detected rather than defaulted to None, so `Settings()` and `get_settings()` agree about
    # where Tesseract is. A model default that disagreed with the environment default is the bug
    # this module's docstring already records once; it would be worse here, because the symptom is
    # "OCR is unavailable" in tests and nowhere else.
    tesseract_cmd: str | None = Field(default_factory=_detect_tesseract_cmd)
    tesseract_data_dir: Path | None = Field(default_factory=_detect_tessdata_dir)

    # OCR selection and shared policy. `ocr_languages` is the single source of truth for what the
    # engine is asked to read; there is no second language setting to disagree with it.
    ocr_provider: str = "tesseract"
    ocr_languages: str = "sin+eng"
    ocr_concurrency: int = Field(default=2, gt=0)
    ocr_timeout_seconds: float = Field(default=60.0, gt=0)
    ocr_empty_text_min_chars: int = Field(default=8, ge=0)
    ocr_low_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    ocr_preprocess_version: str = "preprocess/v1"

    # Preprocessing. Orientation is always applied; the rest default off because they were measured
    # against the regression corpus and only orientation and greyscale left it unchanged --
    # denoising and thresholding both lowered mean confidence on clean scans.
    ocr_preprocess_grayscale: bool = True
    ocr_preprocess_autocontrast: bool = False
    ocr_preprocess_denoise: bool = False
    ocr_preprocess_threshold: int | None = None
    ocr_preprocess_deskew: bool = False

    # Tesseract. 3/3 is automatic page segmentation with the LSTM engine, the only one with a
    # Sinhala model.
    ocr_tesseract_psm: int = Field(default=3, ge=0, le=13)
    ocr_tesseract_oem: int = Field(default=3, ge=0, le=3)
    # 6 is "one uniform block of text", which is what a detected region is. Asking for full page
    # segmentation inside a crop makes Tesseract hunt for columns that are not there.
    ocr_tesseract_region_psm: int = Field(default=6, ge=0, le=13)

    # Paddle hybrid: detection only. PaddleOCR has no Sinhala recognition model, so Tesseract does
    # every recognition and Paddle only proposes regions.
    ocr_paddle_model_dir: Path | None = None
    ocr_paddle_device: str = "cpu"
    ocr_paddle_box_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    ocr_paddle_merge_iou: float = Field(default=0.1, ge=0.0, le=1.0)
    ocr_paddle_max_regions: int = Field(default=40, gt=0)
    ocr_paddle_fallback_to_tesseract: bool = True

    # LLM selection and shared policy.
    llm_provider: str = "fake"
    llm_prompt_version: str = "v1"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_output_tokens: int = Field(default=8000, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    # Two independent budgets (PRD 11.6): transport attempts and schema repairs. A retry must never
    # consume a repair, and a repair must never consume a retry.
    llm_max_attempts: int = Field(default=3, gt=0)
    llm_max_repairs: int = Field(default=1, ge=0)
    llm_backoff_base_seconds: float = Field(default=1.0, gt=0)
    llm_backoff_max_seconds: float = Field(default=20.0, gt=0)
    llm_total_deadline_seconds: float = Field(default=240.0, gt=0)
    llm_concurrency: int = Field(default=4, gt=0)
    llm_structured_mode: str = "auto"
    llm_max_ocr_chars: int = Field(default=24000, gt=0)
    llm_max_candidates_per_image: int = Field(default=20, gt=0)
    llm_fake_mode: str = "rule_based"
    llm_fake_fixture_dir: Path | None = None

    # Per provider. Keys are SecretStr so a stray repr, log line, or error body cannot print one.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4o-mini"
    openai_auth_header: str = "Authorization"

    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_thinking_budget: int | None = None

    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_version: str = "2023-06-01"
    anthropic_tool_name: str = "emit_advertisements"

    # Deliberate escape hatch for a local demo that wants a heuristic provider. Defaults to the
    # environment answer, so nobody has to set it to get the safe behaviour.
    allow_fake_providers_override: bool | None = None

    # Batch limits (PRD 15.1)
    max_images_per_batch: int = Field(default=DEFAULT_MAX_IMAGES_PER_BATCH, gt=0)
    max_image_bytes: int = Field(default=DEFAULT_MAX_IMAGE_BYTES, gt=0)
    max_batch_bytes: int = Field(default=DEFAULT_MAX_BATCH_BYTES, gt=0)
    max_image_pixels: int = Field(default=DEFAULT_MAX_IMAGE_PIXELS, gt=0)

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

    @property
    def allow_fake_providers(self) -> bool:
        """Whether a provider that invents advertisements without a model may be constructed.

        Defaults to the environment rather than to a flag, so the safe answer needs no
        configuration and the unsafe one has to be written down (PRD 11.6).
        """
        if self.allow_fake_providers_override is not None:
            return self.allow_fake_providers_override
        return self.is_local_or_test

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
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER={self.llm_provider!r} is not one of "
                f"{', '.join(sorted(LLM_PROVIDERS))}."
            )
        if self.llm_structured_mode not in LLM_STRUCTURED_MODES:
            raise ValueError(
                f"LLM_STRUCTURED_MODE={self.llm_structured_mode!r} is not one of "
                f"{', '.join(sorted(LLM_STRUCTURED_MODES))}."
            )
        if self.llm_fake_mode not in LLM_FAKE_MODES:
            raise ValueError(
                f"LLM_FAKE_MODE={self.llm_fake_mode!r} is not one of "
                f"{', '.join(sorted(LLM_FAKE_MODES))}."
            )
        if self.ocr_provider not in OCR_PROVIDERS:
            # Dies at startup with the valid list rather than silently falling back to Tesseract.
            # A typo that quietly selects a different engine changes every extraction the service
            # produces, and nothing in the output would say so.
            raise ValueError(
                f"OCR_PROVIDER={self.ocr_provider!r} is not one of "
                f"{', '.join(sorted(OCR_PROVIDERS))}."
            )
        if self.advertisement_status_wire not in ADVERTISEMENT_STATUS_WIRES:
            raise ValueError(
                f"ADVERTISEMENT_STATUS_WIRE={self.advertisement_status_wire!r} is not one of "
                f"{', '.join(sorted(ADVERTISEMENT_STATUS_WIRES))}."
            )
        if self.operator_auth_mode == "none" and not self.is_local_or_test:
            raise ValueError(
                "OPERATOR_AUTH_MODE=none is only allowed in local and test environments."
            )


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
        ocr_provider=getenv("OCR_PROVIDER", "tesseract"),
        # TESSERACT_LANG is the name the prototype documented; OCR_LANGUAGES is the one the
        # provider layer uses. Reading both keeps existing deployments working.
        ocr_languages=getenv("OCR_LANGUAGES", getenv("TESSERACT_LANG", "sin+eng")),
        ocr_concurrency=_env_int("OCR_CONCURRENCY", 2),
        ocr_timeout_seconds=_env_float("OCR_TIMEOUT_SECONDS", 60.0),
        ocr_empty_text_min_chars=_env_int("OCR_EMPTY_TEXT_MIN_CHARS", 8),
        ocr_low_confidence_threshold=_env_float("OCR_LOW_CONFIDENCE_THRESHOLD", 0.60),
        ocr_preprocess_version=getenv("OCR_PREPROCESS_VERSION", "preprocess/v1"),
        ocr_preprocess_grayscale=_env_bool("OCR_PREPROCESS_GRAYSCALE", True),
        ocr_preprocess_autocontrast=_env_bool("OCR_PREPROCESS_AUTOCONTRAST", False),
        ocr_preprocess_denoise=_env_bool("OCR_PREPROCESS_DENOISE", False),
        ocr_preprocess_threshold=_env_optional_int("OCR_PREPROCESS_THRESHOLD"),
        ocr_preprocess_deskew=_env_bool("OCR_PREPROCESS_DESKEW", False),
        ocr_tesseract_psm=_env_int("OCR_TESSERACT_PSM", 3),
        ocr_tesseract_oem=_env_int("OCR_TESSERACT_OEM", 3),
        ocr_tesseract_region_psm=_env_int("OCR_TESSERACT_REGION_PSM", 6),
        ocr_paddle_model_dir=_env_path("OCR_PADDLE_MODEL_DIR"),
        ocr_paddle_device=getenv("OCR_PADDLE_DEVICE", "cpu"),
        ocr_paddle_box_thresh=_env_float("OCR_PADDLE_BOX_THRESH", 0.5),
        ocr_paddle_merge_iou=_env_float("OCR_PADDLE_MERGE_IOU", 0.1),
        ocr_paddle_max_regions=_env_int("OCR_PADDLE_MAX_REGIONS", 40),
        ocr_paddle_fallback_to_tesseract=_env_bool("OCR_PADDLE_FALLBACK_TO_TESSERACT", True),
        llm_provider=getenv("LLM_PROVIDER", "fake"),
        llm_prompt_version=getenv("LLM_PROMPT_VERSION", "v1"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 8000),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 3),
        llm_max_repairs=_env_int("LLM_MAX_REPAIRS", 1),
        llm_backoff_base_seconds=_env_float("LLM_BACKOFF_BASE_SECONDS", 1.0),
        llm_backoff_max_seconds=_env_float("LLM_BACKOFF_MAX_SECONDS", 20.0),
        llm_total_deadline_seconds=_env_float("LLM_TOTAL_DEADLINE_SECONDS", 240.0),
        llm_concurrency=_env_int("LLM_CONCURRENCY", 4),
        llm_structured_mode=getenv("LLM_STRUCTURED_MODE", "auto"),
        llm_max_ocr_chars=_env_int("LLM_MAX_OCR_CHARS", 24000),
        llm_max_candidates_per_image=_env_int("LLM_MAX_CANDIDATES_PER_IMAGE", 20),
        llm_fake_mode=getenv("LLM_FAKE_MODE", "rule_based"),
        llm_fake_fixture_dir=_env_path("LLM_FAKE_FIXTURE_DIR"),
        openai_base_url=getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_api_key=_env_secret("OPENAI_API_KEY"),
        openai_model=getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_auth_header=getenv("OPENAI_AUTH_HEADER", "Authorization"),
        gemini_base_url=getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
        gemini_api_key=_env_secret("GEMINI_API_KEY"),
        gemini_model=getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_thinking_budget=_env_optional_int("GEMINI_THINKING_BUDGET"),
        anthropic_base_url=getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        anthropic_api_key=_env_secret("ANTHROPIC_API_KEY"),
        anthropic_model=getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        anthropic_version=getenv("ANTHROPIC_VERSION", "2023-06-01"),
        anthropic_tool_name=getenv("ANTHROPIC_TOOL_NAME", "emit_advertisements"),
        allow_fake_providers_override=_env_optional_bool("MEDIA_SERVICE_ALLOW_FAKE_PROVIDERS"),
        max_images_per_batch=_env_int("MAX_IMAGES_PER_BATCH", DEFAULT_MAX_IMAGES_PER_BATCH),
        max_image_bytes=_env_int("MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES),
        max_batch_bytes=_env_int("MAX_BATCH_BYTES", DEFAULT_MAX_BATCH_BYTES),
        max_image_pixels=_env_int("MAX_IMAGE_PIXELS", DEFAULT_MAX_IMAGE_PIXELS),
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
