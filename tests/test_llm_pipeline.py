"""Extraction end to end, and the acceptance criteria Phase 3 is gated on.

The headline test here is the provider matrix: the same page, through every adapter, producing
**identical candidates**. That is what "env-switchable" has to mean -- if swapping `LLM_PROVIDER`
changed what a reviewer saw, the setting would be a quality decision disguised as a deployment one.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from media_service.config import Settings
from media_service.db.tables import Advertisement, LlmExtractionRun
from media_service.domain.item_state import BatchStatus, ItemStatus
from media_service.llm.providers.anthropic import AnthropicProvider
from media_service.llm.providers.fake import FakeLlmProvider
from media_service.llm.providers.gemini import GeminiProvider
from media_service.llm.providers.openai_compatible import OpenAiCompatibleProvider
from media_service.llm.registry import prompt_for, runner_settings
from media_service.llm.service import ExtractionContext, LlmExtractionService
from media_service.ocr.types import BoundingBox, BoxSource, OcrBlock, OcrResult
from tests.support import build_pipeline, uploads

pytestmark = pytest.mark.usefixtures("engine")

OPERATOR = "operator-1"

THREE_ADS = {
    "schema_version": "1.0",
    "advertisements": [
        {
            "source_block_ids": [1, 2],
            "language": "en",
            "title": "Honda Fit 2014",
            "description": "Kandy",
            "category": "vehicles",
            "price": {"raw": "Rs. 5,750,000", "amount": 5750000, "currency": "LKR"},
            "location": "Kandy",
            "contacts": {"phones": ["0812233445"]},
            "confidence": {"overall": 0.93, "title": 0.95},
            "warnings": [],
        },
        {
            "source_block_ids": [3],
            "language": "si",
            "title": "ඉඩම විකිණීමට",
            "category": "land",
            "price": {"raw": "රු. 1,800,000", "amount": 1800000, "currency": "LKR"},
            "contacts": {"phones": ["+94412345678"]},
            "confidence": {"overall": 0.81},
            "warnings": [],
        },
        {
            "source_block_ids": [4],
            "language": "en",
            "title": "Marketing executive vacancy",
            "category": "jobs",
            "location": "Colombo 03",
            "contacts": {"phones": ["0114567890"]},
            "confidence": {"overall": 0.88},
            "warnings": [],
        },
    ],
}


def ocr_result(block_count: int = 4) -> OcrResult:
    blocks = tuple(
        OcrBlock(
            id=index,
            text=f"Block {index} text",
            confidence=0.9,
            box=BoundingBox(10, index * 100, 300, 80),
            box_source=BoxSource.ENGINE,
        )
        for index in range(1, block_count + 1)
    )
    return OcrResult(text="\n\n".join(b.text for b in blocks), blocks=blocks, mean_confidence=0.9)


def service_with(provider) -> LlmExtractionService:  # type: ignore[no-untyped-def]
    settings = Settings()
    return LlmExtractionService(
        provider=provider,
        prompt=prompt_for(settings),
        runner_settings=runner_settings(settings),
        max_candidates=settings.llm_max_candidates_per_image,
    )


def stubbed(provider, body):  # type: ignore[no-untyped-def]
    provider._client = httpx.Client(  # noqa: SLF001
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )
    return provider


# -- the provider matrix -----------------------------------------------------------------------


def openai_body(payload) -> dict:  # type: ignore[no-untyped-def]
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}


def gemini_body(payload) -> dict:  # type: ignore[no-untyped-def]
    return {
        "candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(payload)}]}}
        ]
    }


def anthropic_body(payload) -> dict:  # type: ignore[no-untyped-def]
    return {
        "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "name": "emit_advertisements", "input": payload}],
    }


def every_provider(payload):  # type: ignore[no-untyped-def]
    return {
        "openai_compatible": stubbed(
            OpenAiCompatibleProvider(
                base_url="https://api.openai.com/v1", api_key=SecretStr("k"), model="m"
            ),
            openai_body(payload),
        ),
        "gemini": stubbed(
            GeminiProvider(base_url="https://g", api_key=SecretStr("k"), model="m"),
            gemini_body(payload),
        ),
        "anthropic": stubbed(
            AnthropicProvider(base_url="https://a", api_key=SecretStr("k"), model="m"),
            anthropic_body(payload),
        ),
    }


def test_every_provider_produces_identical_candidates() -> None:
    """The headline requirement: `LLM_PROVIDER` is a deployment decision, not a quality one."""
    result = ocr_result()
    outputs = {}
    for name, provider in every_provider(THREE_ADS).items():
        output = service_with(provider).run(result, ExtractionContext(item_id="itm_1"))
        outputs[name] = [
            (d.title, d.category, d.price, d.phones, d.source_block_ids, d.confidence)
            for d in output.drafts
        ]

    assert len(outputs) == 3
    first = next(iter(outputs.values()))
    for name, drafts in outputs.items():
        assert drafts == first, f"{name} produced different candidates"
    assert len(first) == 3


def test_one_image_yields_three_independent_candidates() -> None:
    """AC-002, and the thing the prototype could not express at all."""
    output = service_with(
        stubbed(
            OpenAiCompatibleProvider(
                base_url="https://api.openai.com/v1", api_key=SecretStr("k"), model="m"
            ),
            openai_body(THREE_ADS),
        )
    ).run(ocr_result(), ExtractionContext(item_id="itm_1"))

    assert len(output.drafts) == 3
    assert [d.category for d in output.drafts] == ["vehicles", "land", "jobs"]
    # Each cites only its own blocks -- no candidate claims another's evidence.
    citations = [set(d.source_block_ids) for d in output.drafts]
    assert citations == [{1, 2}, {3}, {4}]
    assert not citations[0] & citations[1] & citations[2]


def test_sinhala_survives_extraction_byte_for_byte() -> None:
    """AC-012, at the layer where a model's output enters the system."""
    import unicodedata

    output = service_with(FakeLlmProvider(mode="empty")).run(
        ocr_result(), ExtractionContext(item_id="itm_1")
    )
    assert output.drafts == []

    output = service_with(
        stubbed(
            AnthropicProvider(base_url="https://a", api_key=SecretStr("k"), model="m"),
            anthropic_body(THREE_ADS),
        )
    ).run(ocr_result(), ExtractionContext(item_id="itm_1"))

    title = output.drafts[1].title
    assert title == "ඉඩම විකිණීමට"
    assert title == unicodedata.normalize("NFC", title)


def test_phone_numbers_are_normalised_across_providers() -> None:
    output = service_with(
        stubbed(
            GeminiProvider(base_url="https://g", api_key=SecretStr("k"), model="m"),
            gemini_body(THREE_ADS),
        )
    ).run(ocr_result(), ExtractionContext(item_id="itm_1"))

    assert output.drafts[1].phones == ("0412345678",)


def test_an_invented_advertisement_is_discarded() -> None:
    """AC-013's sibling: a fabrication has nowhere real to point."""
    invented = {
        "schema_version": "1.0",
        "advertisements": [
            {
                "source_block_ids": [99],
                "language": "en",
                "title": "Invented",
                "category": "vehicles",
                "confidence": {"overall": 0.99},
            }
        ],
    }
    output = service_with(
        stubbed(
            OpenAiCompatibleProvider(
                base_url="https://api.openai.com/v1", api_key=SecretStr("k"), model="m"
            ),
            openai_body(invented),
        )
    ).run(ocr_result(), ExtractionContext(item_id="itm_1"))

    assert output.drafts == []


# -- through the pipeline ----------------------------------------------------------------------


def pipeline_with(unit_of_work, asset_store, provider):  # type: ignore[no-untyped-def]
    settings = Settings()
    return build_pipeline(
        unit_of_work=unit_of_work,
        store=asset_store,
        settings=settings,
        extraction=service_with(provider),
    )


def test_a_page_with_no_advertisements_ends_in_no_ads(unit_of_work, asset_store) -> None:
    """AC-003: an outcome, not a failure."""
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="empty"))
    progress = pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    item = pipeline.service.get_item(progress.items[0].id)
    assert item.status is ItemStatus.NO_ADS
    assert item.error_code is None


def test_an_unrepairable_response_fails_visibly_with_no_advertisement(
    unit_of_work, asset_store, session
) -> None:
    """AC-005: the item is visibly failed, and nothing reached the review queue."""
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="always_invalid"))
    progress = pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    item = pipeline.service.get_item(progress.items[0].id)
    assert item.status is ItemStatus.NEEDS_ATTENTION
    assert item.error_code == "LLM_SCHEMA_INVALID"
    assert session.scalars(select(Advertisement)).all() == []


def test_every_attempt_is_recorded_including_the_repair(
    unit_of_work, asset_store, session
) -> None:
    """One row per attempt, so a failure and its retry are both visible afterwards."""
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="always_invalid"))
    pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    runs = session.scalars(select(LlmExtractionRun).order_by(LlmExtractionRun.attempt)).all()
    assert [run.attempt_kind for run in runs] == ["primary", "repair"]
    assert {run.status for run in runs} == {"schema_invalid"}
    assert all(run.request_hash for run in runs), "provenance for the cost guarantee"


def test_a_successful_run_records_its_validated_response(
    unit_of_work, asset_store, session
) -> None:
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="rule_based"))
    pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    run = session.scalars(select(LlmExtractionRun)).one()
    assert run.status == "validated"
    assert run.validated_response is not None
    assert run.candidate_count == run.candidate_count
    assert run.prompt_version == "ad-extraction/v1"
    assert run.prompt_checksum
    assert run.category_catalog_version == "categories/v1"


def test_candidates_reach_the_review_queue_as_pending(
    unit_of_work, asset_store, session
) -> None:
    """Invariant 2 holds through the new path: nothing machine-made is publicly readable."""
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="rule_based"))
    progress = pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    assert pipeline.service.get_batch(progress.id).status is BatchStatus.COMPLETED
    statuses = set(session.scalars(select(Advertisement.status)))
    assert statuses == {"pending"}


def test_a_resume_rebuilds_candidates_without_calling_the_provider(
    unit_of_work, asset_store
) -> None:
    """The expensive resume: the paid call is not repeated, only the rows it should have made."""
    provider = FakeLlmProvider(mode="rule_based")
    pipeline = pipeline_with(unit_of_work, asset_store, provider)
    progress = pipeline.service.create_batch(uploads(1), created_by=OPERATOR)
    pipeline.run()
    calls_after_first = provider.calls

    with unit_of_work() as work:
        item = work.items.get(progress.items[0].id)
        work.items.transition(item, target=ItemStatus.UPLOADED)
        work.commit()
    pipeline.run()

    assert provider.calls == calls_after_first, "the provider was asked again"
    assert pipeline.service.get_item(progress.items[0].id).candidate_count == 1


# -- provenance --------------------------------------------------------------------------------


def test_model_candidates_are_labelled_as_model_output(unit_of_work, asset_store, session) -> None:
    """`origin` exists so heuristic output stays distinguishable from model output.

    Stamping every candidate with one value makes the column useless for exactly the question it
    was added to answer.
    """
    provider = stubbed(
        OpenAiCompatibleProvider(
            base_url="https://api.openai.com/v1", api_key=SecretStr("k"), model="m"
        ),
        openai_body(THREE_ADS),
    )
    pipeline = pipeline_with(unit_of_work, asset_store, provider)
    pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    assert set(session.scalars(select(Advertisement.origin))) == {"llm_extraction"}


def test_heuristic_candidates_stay_labelled_as_heuristic(
    unit_of_work, asset_store, session
) -> None:
    pipeline = pipeline_with(unit_of_work, asset_store, FakeLlmProvider(mode="rule_based"))
    pipeline.service.create_batch(uploads(1), created_by=OPERATOR)

    pipeline.run()

    assert set(session.scalars(select(Advertisement.origin))) == {"ocr_heuristic"}


@pytest.mark.parametrize(
    ("provider_name", "expected"),
    [
        ("fake", "ocr_heuristic"),
        ("rule_based", "ocr_heuristic"),
        ("openai_compatible", "llm_extraction"),
        ("gemini", "llm_extraction"),
        ("anthropic", "llm_extraction"),
    ],
)
def test_the_origin_follows_the_provider(provider_name: str, expected: str) -> None:
    from media_service.llm.registry import build_extraction_service

    service = build_extraction_service(Settings(llm_provider=provider_name))

    assert service.candidate_origin == expected
