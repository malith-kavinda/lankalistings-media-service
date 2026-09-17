"""The extraction stage: OCR blocks in, reviewed-ready candidates out.

This is where the pieces meet. It renders the blocks into the versioned prompt, drives the runner,
applies Tier 2 semantic validation to whatever survived Tier 1, and turns the result into candidate
drafts. Nothing here knows which provider ran.

Two properties are worth stating because they are easy to lose later.

**The same Tier 2 pass runs on a resume.** When an item resumes from a stored `validated_response`,
the candidates are rebuilt by re-running semantic validation over that response rather than by
trusting drafts recorded earlier. It is deterministic and free -- no provider call -- and it means a
tightened rule applies to work already done instead of only to new pages.

**The request hash is the provenance behind the cost guarantee.** It covers provider, model,
prompt version and checksum, schema version, catalog version, and the exact rendered input, and is
recorded on every run. Reuse itself is governed by the generation-scoped resume in the job runner --
a generation's inputs are fixed by construction -- so the hash is what makes the guarantee auditable
rather than what implements it (AC-006).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

from media_service.domain.categories import CATEGORY_CATALOG_VERSION, DEFAULT_CATALOG
from media_service.domain.listings import CandidateDraft
from media_service.llm.prompts.registry import PromptTemplate
from media_service.llm.rendering import render_blocks
from media_service.llm.runner import AttemptRecorder, LlmExtractionRunner, RunnerSettings
from media_service.llm.schema import SUPPORTED_SCHEMA_VERSIONS, AdExtractionEnvelope, json_schema
from media_service.llm.types import HEURISTIC_ORIGIN, HEURISTIC_PROVIDERS, MODEL_ORIGIN
from media_service.llm.validation import ValidatedCandidate, validate_extraction
from media_service.ocr.types import OcrResult

EXTRACTION_FAILED: Final = "LLM_EXTRACTION_FAILED"


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    """What the stage needs beyond the OCR result itself."""

    item_id: str
    generation: int = 1
    ocr_extraction_id: str | None = None
    correlation_id: str | None = None
    recorder: AttemptRecorder | None = None


@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    drafts: list[CandidateDraft]
    run_id: str | None = None
    warnings: tuple[str, ...] = ()
    failure_code: str | None = None
    failure_detail: str | None = None
    request_hash: str | None = None
    validated_response: dict[str, Any] | None = None

    @property
    def failed(self) -> bool:
        return self.failure_code is not None


class LlmExtractionService:
    def __init__(
        self,
        *,
        provider: Any,
        prompt: PromptTemplate,
        runner_settings: RunnerSettings | None = None,
        max_candidates: int = 20,
        max_ocr_chars: int | None = None,
        catalog_version: str = CATEGORY_CATALOG_VERSION,
    ) -> None:
        self._provider = provider
        self._prompt = prompt
        self._runner_settings = runner_settings or RunnerSettings()
        self._max_candidates = max_candidates
        self._max_ocr_chars = max_ocr_chars or prompt.max_ocr_chars
        self._catalog_version = catalog_version

    @property
    def provider_name(self) -> str:
        return getattr(self._provider, "name", "unknown")

    @property
    def model(self) -> str:
        return getattr(self._provider, "model", "unknown")

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    @property
    def prompt_checksum(self) -> str:
        return self._prompt.checksum

    @property
    def candidate_origin(self) -> str:
        """What produced these candidates, recorded on every advertisement.

        The whole point of the column: a reviewer, an audit, or an accuracy measurement has to be
        able to separate what a model said from what a regex guessed.
        """
        return (
            HEURISTIC_ORIGIN
            if self.provider_name in HEURISTIC_PROVIDERS
            else MODEL_ORIGIN
        )

    def is_available(self) -> bool:
        return self._provider.is_available()

    def availability_reason(self) -> str | None:
        return self._provider.availability_reason()

    # -- the stage ------------------------------------------------------------------------------

    def run(self, result: OcrResult, context: ExtractionContext) -> ExtractionOutput:
        rendered, render_warnings = render_blocks(result.blocks, max_chars=self._max_ocr_chars)
        user = self._prompt.render_user(
            {
                "ingestion_item_id": context.item_id,
                "category_catalog": DEFAULT_CATALOG.as_prompt_list(),
                "schema_version": self._prompt.schema_version,
                "ocr_blocks": rendered,
            }
        )
        request_hash = self.request_hash(rendered)

        runner = LlmExtractionRunner(
            provider=self._provider,
            prompt=self._prompt,
            settings=self._runner_settings,
            recorder=context.recorder,
        )
        outcome = runner.run(system=self._prompt.system, user=user, json_schema=json_schema())

        if not outcome.succeeded or outcome.envelope is None:
            return ExtractionOutput(
                drafts=[],
                warnings=render_warnings,
                failure_code=outcome.failure_code or EXTRACTION_FAILED,
                failure_detail=outcome.failure_detail,
                request_hash=request_hash,
            )

        drafts, semantic_warnings = self._drafts(outcome.envelope, result, render_warnings)
        return ExtractionOutput(
            drafts=drafts,
            warnings=semantic_warnings,
            request_hash=request_hash,
            validated_response=outcome.envelope.model_dump(mode="json"),
        )

    def rebuild(self, payload: dict[str, Any], result: OcrResult) -> ExtractionOutput:
        """Re-derive candidates from a stored response, without calling a provider.

        Tier 2 runs again here on purpose -- see the module docstring.
        """
        try:
            envelope = AdExtractionEnvelope.model_validate(payload)
        except Exception:  # noqa: BLE001 - a stored response that no longer validates
            return ExtractionOutput(
                drafts=[],
                failure_code="LLM_STORED_RESPONSE_INVALID",
                failure_detail="The stored response no longer matches the schema.",
            )
        drafts, warnings = self._drafts(envelope, result, ())
        return ExtractionOutput(drafts=drafts, warnings=warnings, validated_response=payload)

    def request_hash_for(self, result: OcrResult) -> str:
        """The hash for this OCR result, rendered exactly as the prompt would render it."""
        rendered, _ = render_blocks(result.blocks, max_chars=self._max_ocr_chars)
        return self.request_hash(rendered)

    def request_hash(self, rendered_blocks: str) -> str:
        """Identifies this exact request, so an unchanged retry can reuse its answer."""
        material = "|".join(
            [
                self.provider_name,
                self.model,
                self._prompt.version,
                self._prompt.checksum,
                self._prompt.schema_version,
                self._catalog_version,
                rendered_blocks,
            ]
        )
        return sha256(material.encode("utf-8")).hexdigest()

    # -- candidates -----------------------------------------------------------------------------

    def _drafts(
        self,
        envelope: AdExtractionEnvelope,
        result: OcrResult,
        render_warnings: tuple[str, ...],
    ) -> tuple[list[CandidateDraft], tuple[str, ...]]:
        outcome = validate_extraction(
            envelope,
            known_block_ids=frozenset(block.id for block in result.blocks),
            max_candidates=self._max_candidates,
            catalog=DEFAULT_CATALOG,
            supported_schema_versions=SUPPORTED_SCHEMA_VERSIONS,
        )
        page_warnings = tuple(dict.fromkeys((*render_warnings, *outcome.warning_codes)))
        return (
            [self._draft(candidate, page_warnings) for candidate in outcome.candidates],
            page_warnings,
        )

    def _draft(
        self, candidate: ValidatedCandidate, page_warnings: tuple[str, ...]
    ) -> CandidateDraft:
        advertisement = candidate.advertisement
        confidence = advertisement.confidence
        price = advertisement.price

        return CandidateDraft(
            index=candidate.index,
            title=advertisement.title or "",
            description=advertisement.description or "",
            category=candidate.category,
            location=advertisement.location or "",
            price=(price.raw if price and price.raw else ""),
            phones=candidate.phones,
            language=advertisement.language,
            confidence=confidence.overall,
            confidence_label=_label(confidence.overall),
            source_text="",
            source_block_ids=candidate.source_block_ids,
            field_confidence=_field_confidence(confidence),
            warnings=tuple(warning.message for warning in advertisement.warnings),
            warning_codes=tuple(dict.fromkeys((*candidate.warning_codes, *page_warnings))),
            extracted_values=advertisement.model_dump(mode="json"),
        )


def _field_confidence(confidence: Any) -> dict[str, float]:
    return {
        name: value
        for name, value in confidence.model_dump().items()
        if name != "overall" and value is not None
    }


def _label(value: float | None) -> str:
    from media_service.ocr.compat import confidence_to_word

    return confidence_to_word(value)
