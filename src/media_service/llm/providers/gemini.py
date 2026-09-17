"""Google Gemini.

Two things differ from the others and both matter.

**The key goes in the `x-goog-api-key` header, never `?key=`.** Gemini accepts the query parameter,
and using it would write the credential into every proxy log, access log and error report along the
path -- including ones this service does not control.

**The schema must be the OpenAPI 3.0 subset** Gemini implements: uppercase type names, `nullable`
instead of a union with null, no `$ref`, and `propertyOrdering` to fix key order. `to_gemini` does
that transform; the model here never sees the schema Pydantic generates.

Gemini also reports rate limiting as `RESOURCE_EXHAUSTED` in the body, which is the kind of
provider-specific knowledge that belongs in an adapter and nowhere else.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx

from media_service.llm.dialects import to_gemini
from media_service.llm.providers.http import HttpLlmProvider, _body_excerpt, _retry_after
from media_service.llm.types import (
    CONTENT_BLOCKED,
    RATE_LIMITED,
    SERVER_ERROR,
    TRUNCATED_OUTPUT,
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

PROVIDER_NAME: Final = "gemini"

# Statuses Gemini reports in the body rather than only in the HTTP code.
BODY_STATUS_CODES: Final[dict[str, str]] = {
    "RESOURCE_EXHAUSTED": RATE_LIMITED,
    "UNAVAILABLE": SERVER_ERROR,
    "INTERNAL": SERVER_ERROR,
    "DEADLINE_EXCEEDED": SERVER_ERROR,
}


class GeminiProvider(HttpLlmProvider):
    provider_name = PROVIDER_NAME

    def __init__(self, *, thinking_budget: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thinking_budget = thinking_budget

    @property
    def key_variable(self) -> str:
        return "GEMINI_API_KEY"

    @property
    def model_variable(self) -> str:
        return "GEMINI_MODEL"

    def complete(self, request: LlmRequest) -> LlmResponse:
        url = f"{self._base_url}/v1beta/models/{self._model}:generateContent"
        body, elapsed_ms = self.post(url, payload=self._payload(request), headers=self._headers())
        return self._read(body, elapsed_ms)

    # -- request -------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        secret = self._api_key.get_secret_value() if self._api_key else ""
        return {"x-goog-api-key": secret}

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": request.user}]},
        ]
        if request.is_repair:
            prior = request.prior_response_text or ""
            contents.append({"role": "model", "parts": [{"text": prior}]})
            contents.append(
                {"role": "user", "parts": [{"text": request.repair_instruction or ""}]}
            )

        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
            "responseMimeType": "application/json",
            "responseSchema": to_gemini(request.json_schema),
        }
        if self._thinking_budget is not None:
            generation["thinkingConfig"] = {"thinkingBudget": self._thinking_budget}

        return {
            "contents": contents,
            # The system prompt is its own field here, not a message, which keeps the trust boundary
            # explicit: instructions and untrusted source never share a turn.
            "systemInstruction": {"parts": [{"text": request.system}]},
            "generationConfig": generation,
        }

    # -- response ------------------------------------------------------------------------------

    def classify(self, response: httpx.Response) -> LlmProviderError:
        try:
            body = response.json()
        except ValueError:
            return super().classify(response)

        status = (body.get("error") or {}).get("status") if isinstance(body, dict) else None
        code = BODY_STATUS_CODES.get(status or "")
        if code is None:
            return super().classify(response)
        return LlmProviderError(
            code,
            f"{PROVIDER_NAME} returned {status}: {_body_excerpt(response)}",
            retry_after_seconds=_retry_after(response),
            status_code=response.status_code,
        )

    def _read(self, body: dict[str, Any], elapsed_ms: int) -> LlmResponse:
        feedback = body.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LlmProviderError(
                CONTENT_BLOCKED,
                f"The prompt was blocked: {feedback.get('blockReason')}.",
                retryable=False,
            )

        candidates = body.get("candidates") or []
        if not candidates:
            raise LlmProviderError(UNPARSEABLE_RESPONSE, "The response carried no candidates.")

        candidate = candidates[0]
        reason = candidate.get("finishReason")
        if reason == "MAX_TOKENS":
            raise LlmProviderError(
                TRUNCATED_OUTPUT, "The response hit the output token limit.", retryable=True
            )
        if reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"):
            raise LlmProviderError(
                CONTENT_BLOCKED, f"The response was blocked: {reason}.", retryable=False
            )

        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text.strip():
            raise LlmProviderError(UNPARSEABLE_RESPONSE, "The response carried no text.")

        return LlmResponse(
            raw_text=text,
            payload=_parse(text),
            provider=PROVIDER_NAME,
            model=body.get("modelVersion") or self._model,
            usage=_usage(body.get("usageMetadata")),
            latency_ms=elapsed_ms,
            finish_reason=reason,
        )


def _parse(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _usage(usage: Any) -> LlmUsage:
    if not isinstance(usage, dict):
        return LlmUsage()
    return LlmUsage(
        input_tokens=usage.get("promptTokenCount"),
        output_tokens=usage.get("candidatesTokenCount"),
    )
