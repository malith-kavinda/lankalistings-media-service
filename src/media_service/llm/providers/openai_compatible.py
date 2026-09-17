"""OpenAI and everything that speaks its chat-completions protocol.

One adapter covers OpenAI, OpenRouter, Groq, Together, vLLM and LM Studio, because they share a wire
format. What they do *not* share is structured-output support, which is why `LLM_STRUCTURED_MODE`
exists:

* `json_schema` — strict structured output. The schema is enforced by the provider, and
  `to_openai_strict` has already stripped the keywords strict mode refuses.
* `json_object` — the provider only promises valid JSON, not a shape. Application validation does
  the rest, which it would anyway.
* `auto` — pick by base URL. Hosts known not to implement strict schema mode degrade rather than
  failing every request with a 400 that names a parameter the operator did not choose.

Whichever mode runs, Tier 1 validation still executes on the response. Provider enforcement is never
treated as sufficient: some modes have none, some cannot express numeric bounds, and any response
can be truncated mid-object (PRD 11.5).
"""

from __future__ import annotations

import json
from typing import Any, Final

from media_service.llm.dialects import to_openai_strict
from media_service.llm.providers.http import HttpLlmProvider
from media_service.llm.types import (
    CONTENT_BLOCKED,
    TRUNCATED_OUTPUT,
    UNPARSEABLE_RESPONSE,
    LlmProviderError,
    LlmRequest,
    LlmResponse,
    LlmUsage,
)

PROVIDER_NAME: Final = "openai_compatible"

STRUCTURED_MODES: Final = ("auto", "json_schema", "json_object")

# Hosts whose OpenAI-compatible surface does not implement strict `json_schema`. Degrading is
# better than a 400 on every request naming a parameter the operator never set.
NO_STRICT_SCHEMA_HOSTS: Final = (
    "groq.com",
    "together.xyz",
    "together.ai",
    "localhost",
    "127.0.0.1",
)


class OpenAiCompatibleProvider(HttpLlmProvider):
    provider_name = PROVIDER_NAME

    def __init__(
        self,
        *,
        structured_mode: str = "auto",
        auth_header: str = "Authorization",
        token_prefix: str = "Bearer ",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if structured_mode not in STRUCTURED_MODES:
            raise ValueError(
                f"LLM_STRUCTURED_MODE={structured_mode!r} is not one of "
                f"{', '.join(STRUCTURED_MODES)}."
            )
        self._structured_mode = structured_mode
        self._auth_header = auth_header
        self._token_prefix = token_prefix

    @property
    def key_variable(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def model_variable(self) -> str:
        return "OPENAI_MODEL"

    @property
    def structured_mode(self) -> str:
        """The mode that will actually be used, with `auto` already resolved."""
        if self._structured_mode != "auto":
            return self._structured_mode
        host = self._base_url.lower()
        return (
            "json_object"
            if any(marker in host for marker in NO_STRICT_SCHEMA_HOSTS)
            else "json_schema"
        )

    def complete(self, request: LlmRequest) -> LlmResponse:
        body, elapsed_ms = self.post(
            f"{self._base_url}/chat/completions",
            payload=self._payload(request),
            headers=self._headers(),
        )
        return self._read(body, elapsed_ms)

    # -- request -------------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        secret = self._api_key.get_secret_value() if self._api_key else ""
        return {self._auth_header: f"{self._token_prefix}{secret}".strip()}

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ]
        if request.is_repair:
            # The model's own answer is replayed so it corrects that answer rather than starting
            # again. The instruction carries field paths only, never values.
            messages.append({"role": "assistant", "content": request.prior_response_text or ""})
            messages.append({"role": "user", "content": request.repair_instruction or ""})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }

        if self.structured_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": to_openai_strict(request.json_schema),
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        return payload

    # -- response ------------------------------------------------------------------------------

    def _read(self, body: dict[str, Any], elapsed_ms: int) -> LlmResponse:
        choices = body.get("choices") or []
        if not choices:
            raise LlmProviderError(UNPARSEABLE_RESPONSE, "The response carried no choices.")

        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        content = message.get("content")

        if finish_reason == "length":
            # A transport failure, not a schema one: repair cannot lengthen a truncated object, and
            # treating it as repairable would spend the item's one repair for nothing (PRD 11.6).
            raise LlmProviderError(
                TRUNCATED_OUTPUT, "The response hit the output token limit.", retryable=True
            )
        if finish_reason == "content_filter":
            raise LlmProviderError(
                CONTENT_BLOCKED, "The provider blocked this content.", retryable=False
            )
        if message.get("refusal"):
            raise LlmProviderError(
                CONTENT_BLOCKED, "The model refused to answer.", retryable=False
            )
        if not isinstance(content, str) or not content.strip():
            raise LlmProviderError(UNPARSEABLE_RESPONSE, "The response carried no content.")

        return LlmResponse(
            raw_text=content,
            payload=_parse(content),
            provider=PROVIDER_NAME,
            model=body.get("model") or self._model,
            usage=_usage(body.get("usage")),
            latency_ms=elapsed_ms,
            finish_reason=finish_reason,
        )


def _parse(content: str) -> dict[str, Any] | None:
    """Parse the content, tolerating a code fence.

    `json_object` mode promises valid JSON and mostly delivers it; some compatible servers still
    wrap it. Returning None rather than raising leaves the decision to the runner, which treats an
    unparseable body as a Tier 1 failure and may repair it.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.removesuffix("```").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _usage(usage: Any) -> LlmUsage:
    if not isinstance(usage, dict):
        return LlmUsage()
    return LlmUsage(
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )
