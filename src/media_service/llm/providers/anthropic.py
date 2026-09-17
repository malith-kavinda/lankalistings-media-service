"""Anthropic Claude.

There is no `response_format` here, so structure comes from a **forced tool call**: the schema is
declared as a tool and `tool_choice` requires the model to use it. The payoff is that
`tool_use.input` arrives as a **dict** -- no string is parsed, no code fence has to be stripped, and
the whole class of "the model wrapped its JSON in prose" simply does not arise.

Truncation matters more here and is easy to misread. A forced tool call cut off by `max_tokens`
looks like a schema failure -- the tool input is incomplete -- but repair cannot fix it. Treating it
as repairable would burn the item's single repair on something only a larger output budget solves.
`stop_reason` is therefore checked **before** anything looks at the payload.
"""

from __future__ import annotations

import json
from typing import Any, Final

import httpx

from media_service.llm.dialects import to_anthropic_tool
from media_service.llm.providers.http import HttpLlmProvider, _body_excerpt, _retry_after
from media_service.llm.types import (
    CONTENT_BLOCKED,
    SERVER_ERROR,
    TRUNCATED_OUTPUT,
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

PROVIDER_NAME: Final = "anthropic"
DEFAULT_VERSION: Final = "2023-06-01"
DEFAULT_TOOL_NAME: Final = "emit_advertisements"

TOOL_DESCRIPTION: Final = (
    "Return every independent classified advertisement found in the supplied OCR blocks."
)

# 529 is Anthropic's "overloaded". It is transient and is not in the 5xx range every client already
# retries, so it needs naming explicitly.
OVERLOADED_STATUS: Final = 529


class AnthropicProvider(HttpLlmProvider):
    provider_name = PROVIDER_NAME

    def __init__(
        self,
        *,
        version: str = DEFAULT_VERSION,
        tool_name: str = DEFAULT_TOOL_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._version = version
        self._tool_name = tool_name

    @property
    def key_variable(self) -> str:
        return "ANTHROPIC_API_KEY"

    @property
    def model_variable(self) -> str:
        return "ANTHROPIC_MODEL"

    def complete(self, request: LlmRequest) -> LlmResponse:
        body, elapsed_ms = self.post(
            f"{self._base_url}/v1/messages",
            payload=self._payload(request),
            headers=self._headers(),
        )
        return self._read(body, elapsed_ms)

    # -- request -------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        secret = self._api_key.get_secret_value() if self._api_key else ""
        return {"x-api-key": secret, "anthropic-version": self._version}

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.user}]
        if request.is_repair:
            messages.append({"role": "assistant", "content": request.prior_response_text or ""})
            messages.append({"role": "user", "content": request.repair_instruction or ""})

        tool = to_anthropic_tool(
            request.json_schema, name=self._tool_name, description=TOOL_DESCRIPTION
        )
        return {
            "model": self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            # The system prompt is its own field, never a message, so instructions and untrusted
            # source never share a turn.
            "system": request.system,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": self._tool_name},
        }

    # -- response ------------------------------------------------------------------------------

    def classify(self, response: httpx.Response) -> LlmProviderError:
        if response.status_code == OVERLOADED_STATUS:
            return LlmProviderError(
                SERVER_ERROR,
                f"{PROVIDER_NAME} is overloaded: {_body_excerpt(response)}",
                retryable=True,
                retry_after_seconds=_retry_after(response),
                status_code=OVERLOADED_STATUS,
            )
        return super().classify(response)

    def _read(self, body: dict[str, Any], elapsed_ms: int) -> LlmResponse:
        stop_reason = body.get("stop_reason")
        # Checked first, deliberately: a truncated tool call looks like a schema failure and is not.
        if stop_reason == "max_tokens":
            raise LlmProviderError(
                TRUNCATED_OUTPUT, "The response hit the output token limit.", retryable=True
            )
        if stop_reason == "refusal":
            raise LlmProviderError(
                CONTENT_BLOCKED, "The model refused to answer.", retryable=False
            )

        blocks = body.get("content") or []
        payload: dict[str, Any] | None = None
        text_parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("name") == self._tool_name:
                candidate = block.get("input")
                if isinstance(candidate, dict):
                    payload = candidate
            elif block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))

        if payload is None:
            raise LlmProviderError(
                UNPARSEABLE_RESPONSE,
                f"The response contained no {self._tool_name} tool call.",
            )

        return LlmResponse(
            # The tool input is already structured; this text exists only so a repair turn has
            # something to replay and so the run record holds what was returned.
            raw_text=json.dumps(payload, ensure_ascii=False),
            payload=payload,
            provider=PROVIDER_NAME,
            model=body.get("model") or self._model,
            usage=_usage(body.get("usage")),
            latency_ms=elapsed_ms,
            finish_reason=stop_reason,
        )


def _usage(usage: Any) -> LlmUsage:
    if not isinstance(usage, dict):
        return LlmUsage()
    return LlmUsage(
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )
