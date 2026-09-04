"""Versioned prompt templates.

Every extraction run records both the prompt version and a checksum of the rendered template files
(FR-LLM-004). The checksum exists because a version string alone is not enough: editing `system.txt`
without renaming the directory would silently invalidate the regression corpus and every historical
accuracy measurement, with nothing in the provenance record showing that anything changed.

Templates are read through `importlib.resources`, not by walking `__file__`, so they resolve
correctly when the package is installed rather than run from a source checkout.
"""

from dataclasses import dataclass
from functools import cache
from hashlib import sha256
from importlib.resources import files
from json import loads
from re import findall
from typing import Final

PROMPT_PACKAGE: Final = "media_service.llm.prompts"

# Placeholders use {{name}} rather than str.format braces because the templates contain literal JSON
# braces, and rather than a template engine because a prompt needs no control flow.
_PLACEHOLDER_PATTERN: Final = r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}"


class PromptError(RuntimeError):
    """A prompt template is missing, malformed, or rendered with the wrong values."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    version: str
    checksum: str
    schema_version: str
    system: str
    user: str
    repair: str
    max_ocr_chars: int
    required_placeholders: tuple[str, ...]
    repair_placeholders: tuple[str, ...]

    def render_user(self, values: dict[str, str]) -> str:
        return _render(self.user, values, where=f"{self.version}:user")

    def render_repair(self, values: dict[str, str]) -> str:
        return _render(self.repair, values, where=f"{self.version}:repair")


def _render(template: str, values: dict[str, str], *, where: str) -> str:
    required = set(findall(_PLACEHOLDER_PATTERN, template))
    supplied = set(values)

    missing = required - supplied
    if missing:
        raise PromptError(f"{where} is missing values for: {', '.join(sorted(missing))}.")

    unused = supplied - required
    if unused:
        # An unused value almost always means a placeholder was renamed in the template and the
        # caller was not updated, which would otherwise ship a prompt with an empty section.
        raise PromptError(f"{where} was given values it does not use: {', '.join(sorted(unused))}.")

    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{{" + name + "}}", value)

    leftover = findall(_PLACEHOLDER_PATTERN, rendered)
    if leftover:
        raise PromptError(f"{where} still contains placeholders after rendering: {leftover}.")
    return rendered


@cache
def load_prompt(family: str, version: str) -> PromptTemplate:
    """Load one versioned prompt family, for example `load_prompt("ad_extraction", "v1")`."""
    try:
        directory = files(PROMPT_PACKAGE).joinpath(family).joinpath(version)
        manifest = loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))
        system = directory.joinpath(manifest["system"]).read_text(encoding="utf-8")
        user = directory.joinpath(manifest["user"]).read_text(encoding="utf-8")
        repair = directory.joinpath(manifest["repair"]).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(f"Prompt {family}/{version} is missing a file: {exc}.") from exc
    except KeyError as exc:
        raise PromptError(f"Prompt {family}/{version} manifest is missing key {exc}.") from exc

    declared = tuple(manifest["required_placeholders"])
    actual = set(findall(_PLACEHOLDER_PATTERN, user))
    if set(declared) != actual:
        raise PromptError(
            f"Prompt {family}/{version} manifest declares {sorted(declared)} "
            f"but user.txt uses {sorted(actual)}."
        )

    return PromptTemplate(
        version=manifest["version"],
        checksum=checksum_of(system, user, repair),
        schema_version=manifest["schema_version"],
        system=system,
        user=user,
        repair=repair,
        max_ocr_chars=int(manifest["max_ocr_chars"]),
        required_placeholders=declared,
        repair_placeholders=tuple(manifest["repair_placeholders"]),
    )


def checksum_of(system: str, user: str, repair: str) -> str:
    digest = sha256()
    for part in (system, user, repair):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]
