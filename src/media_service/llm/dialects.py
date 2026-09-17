"""One Pydantic model, three provider dialects.

Every provider enforces structure differently, and none of them accepts the schema Pydantic
generates unchanged. These functions derive each wire format from that one source, so the schema the
model is shown and the schema the application enforces can never disagree -- a hand-maintained copy
per provider is exactly how that drift starts.

What each provider needs, and why:

**OpenAI strict mode** requires every property to appear in `required` (optionality is carried by
a nullable type instead) and rejects validation keywords it cannot enforce -- `minimum`, `maximum`,
`maxLength`. So `to_openai_strict` strips the bounds *for the wire only*; Pydantic still enforces
them when the response comes back, which is the point of validating independently of the provider.

**Gemini** takes an OpenAPI 3.0 subset: uppercase type names, `nullable: true` rather than a union
with null, no `$ref` at all, and `propertyOrdering` to fix key order.

**Anthropic** has no `response_format`. Structure comes from a forced tool call, so the schema is
delivered as a tool's `input_schema`.

All three inline `$defs` completely. Pydantic emits references, support for them varies across the
OpenAI-compatible family in particular, and the schema is small enough that inlining costs nothing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

# Keywords OpenAI strict mode rejects. They are constraints it will not enforce, so it refuses the
# schema rather than silently ignoring them.
UNSUPPORTED_IN_STRICT: Final = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "format",
        "default",
        "multipleOf",
        "uniqueItems",
    }
)

# Everything Gemini's OpenAPI subset does not understand.
UNSUPPORTED_IN_GEMINI: Final = UNSUPPORTED_IN_STRICT | {"additionalProperties", "title", "const"}

GEMINI_TYPES: Final[dict[str, str]] = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve every `$ref` and drop `$defs`.

    The schema is a tree with no cycles -- an advertisement cannot contain an advertisement -- so a
    straightforward substitution terminates. A cycle would be a modelling error worth failing on
    rather than working around.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node

        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.removeprefix("#/$defs/")
            if name in seen:
                raise ValueError(f"The schema is recursive through $defs/{name}.")
            if name not in definitions:
                raise ValueError(f"The schema references unknown $defs/{name}.")
            # Sibling keys beside a $ref (a title, say) are dropped: the reference is the type.
            return resolve(definitions[name], seen | {name})

        return {key: resolve(value, seen) for key, value in node.items() if key != "$defs"}

    resolved = resolve(deepcopy(schema), frozenset())
    if isinstance(resolved, dict):
        resolved.pop("$defs", None)
    return resolved  # type: ignore[no-any-return]


def to_openai_strict(schema: dict[str, Any]) -> dict[str, Any]:
    """The `json_schema` payload for OpenAI-compatible strict structured output."""
    return _strict(inline_refs(schema))


def to_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """A `responseSchema` in Gemini's OpenAPI-3.0 subset."""
    return _gemini(inline_refs(schema))


def to_anthropic_tool(
    schema: dict[str, Any], *, name: str, description: str
) -> dict[str, Any]:
    """A forced-tool definition. The tool input *is* the payload; no text is ever parsed."""
    return {
        "name": name,
        "description": description,
        "input_schema": inline_refs(schema),
    }


# -- transforms --------------------------------------------------------------------------------


def _rewrite(node: Any, *, drop: frozenset[str], null_as_type: bool) -> Any:
    """Walk a schema, dropping keywords the target cannot accept.

    The recursion distinguishes **schema keywords from property names**, which is not optional
    bookkeeping: this schema has a field called `title`, and a traversal that filtered dictionary
    keys uniformly would delete it along with the JSON Schema `title` annotation. The advertisement
    would then have no title at all, and the model would faithfully omit one.
    """
    if isinstance(node, list):
        return [_rewrite(item, drop=drop, null_as_type=null_as_type) for item in node]
    if not isinstance(node, dict):
        return node

    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in drop:
            continue
        if key == "properties" and isinstance(value, dict):
            # Keys here are field names. They are copied verbatim and only their values are walked.
            result[key] = {
                name: _rewrite(sub, drop=drop, null_as_type=null_as_type)
                for name, sub in value.items()
            }
        else:
            result[key] = _rewrite(value, drop=drop, null_as_type=null_as_type)

    return _collapse_nullable(result, null_as_type=null_as_type)


def _strict(node: Any) -> Any:
    result = _rewrite(node, drop=UNSUPPORTED_IN_STRICT, null_as_type=True)
    return _require_every_property(result)


def _require_every_property(node: Any) -> Any:
    if isinstance(node, list):
        return [_require_every_property(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {
        key: (
            {name: _require_every_property(sub) for name, sub in value.items()}
            if key == "properties" and isinstance(value, dict)
            else _require_every_property(value)
        )
        for key, value in node.items()
    }
    if result.get("type") == "object" or "properties" in result:
        result["additionalProperties"] = False
        # Strict mode requires *every* property in `required`. Optionality is carried by the type
        # being nullable instead, which `_collapse_nullable` has already arranged.
        result["required"] = list(result.get("properties", {}))
    return result


def _gemini(node: Any) -> Any:
    result = _rewrite(node, drop=UNSUPPORTED_IN_GEMINI, null_as_type=False)
    return _gemini_types(result)


def _gemini_types(node: Any) -> Any:
    if isinstance(node, list):
        return [_gemini_types(item) for item in node]
    if not isinstance(node, dict):
        return node

    result = {
        key: (
            {name: _gemini_types(sub) for name, sub in value.items()}
            if key == "properties" and isinstance(value, dict)
            else _gemini_types(value)
        )
        for key, value in node.items()
    }

    kind = result.get("type")
    if isinstance(kind, str) and kind in GEMINI_TYPES:
        result["type"] = GEMINI_TYPES[kind]

    if "properties" in result:
        # Fixes the order keys are generated in. Without it the model chooses, and a diff between
        # two runs of the same page becomes unreadable.
        result["propertyOrdering"] = list(result["properties"])
    return result


def _collapse_nullable(node: dict[str, Any], *, null_as_type: bool) -> dict[str, Any]:
    """Turn Pydantic's `anyOf: [T, null]` into whichever nullable spelling the provider wants.

    Pydantic expresses an optional field as a union with null. OpenAI accepts a type array;
    Gemini accepts neither and wants a `nullable` flag beside the single type. Both need the union
    flattened first, because a one-of-two-branches union is not what either means by "optional".
    """
    branches = node.get("anyOf")
    if not isinstance(branches, list):
        return node

    nulls = [branch for branch in branches if _is_null_type(branch)]
    others = [branch for branch in branches if not _is_null_type(branch)]
    if not nulls or len(others) != 1:
        return node

    collapsed = {key: value for key, value in node.items() if key != "anyOf"}
    collapsed |= others[0]

    if null_as_type:
        kind = collapsed.get("type")
        if isinstance(kind, str):
            collapsed["type"] = [kind, "null"]
    else:
        collapsed["nullable"] = True
    return collapsed


def _is_null_type(branch: Any) -> bool:
    return isinstance(branch, dict) and branch.get("type") == "null"
