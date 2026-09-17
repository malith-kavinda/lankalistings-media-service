"""Shared transport for the three real providers.

Everything here is protocol-agnostic: connection handling, timeouts, and the HTTP status codes that
mean the same thing everywhere. What each provider calls a rate limit in its *body* is its own
business and stays in its own adapter.

Sync `httpx.Client`, not async. LLM calls happen only on a worker thread and never inside a
request -- that is the point of answering `202` -- so an async client would buy nothing and would
need a bridge back to the thread pool.

The client is built lazily and reused. Constructing one per call would discard the connection pool
between attempts, which on a retry is exactly when the handshake cost is least welcome.

**A key never reaches a URL.** Gemini accepts `?key=`, and using it would write the credential into
every proxy log and error report along the path. All three adapters send credentials as headers.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Final

import httpx
from pydantic import SecretStr

from media_service.llm.types import (
    AUTH_FAILED,
    BAD_REQUEST,
    MODEL_UNAVAILABLE,
    RATE_LIMITED,
    SERVER_ERROR,
    TIMEOUT,
    TRANSPORT_ERROR,
    LlmProviderError,
)

# Bodies are truncated before they are recorded or raised. A provider error can echo the request,
# and the request contains OCR text (PRD 15.4).
MAX_DETAIL_CHARS: Final = 400


class HttpLlmProvider:
    """Connection handling and status classification. Subclasses own the protocol."""

    provider_name = "http"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr | None,
        model: str,
        timeout_seconds: float = 60.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._extra_headers = extra_headers or {}
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.provider_name

    @property
    def model(self) -> str:
        return self._model

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> str | None:
        """Names the missing variable, never its value -- this reaches `/health` unauthenticated."""
        if self._api_key is None or not self._api_key.get_secret_value():
            return f"{self.key_variable} is not set."
        if not self._model:
            return f"{self.model_variable} is not set."
        return None

    @property
    def key_variable(self) -> str:  # pragma: no cover - overridden
        return "API_KEY"

    @property
    def model_variable(self) -> str:  # pragma: no cover - overridden
        return "MODEL"

    # -- transport -----------------------------------------------------------------------------

    def _http(self) -> httpx.Client:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def post(
        self, url: str, *, payload: dict[str, Any], headers: dict[str, str]
    ) -> tuple[dict[str, Any], int]:
        """POST JSON and return the decoded body with the elapsed milliseconds."""
        started = time.monotonic()
        try:
            response = self._http().post(
                url, json=payload, headers={**self._extra_headers, **headers}
            )
        except httpx.TimeoutException as exc:
            raise LlmProviderError(TIMEOUT, f"{self.provider_name} timed out.") from exc
        except httpx.HTTPError as exc:
            raise LlmProviderError(
                TRANSPORT_ERROR, f"{self.provider_name} could not be reached: {type(exc).__name__}."
            ) from exc

        elapsed_ms = int((time.monotonic() - started) * 1000)
        if response.status_code >= 400:
            raise self.classify(response)

        try:
            return response.json(), elapsed_ms
        except ValueError as exc:
            raise LlmProviderError(
                SERVER_ERROR, f"{self.provider_name} returned a body that is not JSON."
            ) from exc

    def classify(self, response: httpx.Response) -> LlmProviderError:
        """Map an HTTP status onto the shared vocabulary.

        Subclasses override to read a provider-specific body first -- Gemini's `RESOURCE_EXHAUSTED`,
        Anthropic's `overloaded_error` -- and fall back here for everything else.
        """
        status = response.status_code
        detail = f"{self.provider_name} returned {status}: {_body_excerpt(response)}"

        if status in (401, 403):
            return LlmProviderError(AUTH_FAILED, detail, status_code=status)
        if status == 404:
            return LlmProviderError(MODEL_UNAVAILABLE, detail, status_code=status)
        if status == 429:
            return LlmProviderError(
                RATE_LIMITED,
                detail,
                retry_after_seconds=_retry_after(response),
                status_code=status,
            )
        if status >= 500:
            return LlmProviderError(
                SERVER_ERROR,
                detail,
                retry_after_seconds=_retry_after(response),
                status_code=status,
            )
        return LlmProviderError(BAD_REQUEST, detail, status_code=status)


def _retry_after(response: httpx.Response) -> float | None:
    """Honour a provider's own backoff instruction when it gives one (PRD 11.6)."""
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return max(float(header), 0.0)
    except ValueError:
        # The HTTP-date form. Not worth parsing: the runner's own backoff is a fine substitute.
        return None


def _body_excerpt(response: httpx.Response) -> str:
    """A short, allow-listed excerpt -- never the whole body, which can echo the request."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:MAX_DETAIL_CHARS]

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            fields = {
                key: error[key] for key in ("type", "code", "status", "message") if key in error
            }
            return str(fields)[:MAX_DETAIL_CHARS]
        if isinstance(error, str):
            return error[:MAX_DETAIL_CHARS]
    return str(body)[:MAX_DETAIL_CHARS]
