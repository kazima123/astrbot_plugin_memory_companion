"""Compression must not replace true memories with an unsupported summary."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.models import EntityRef, MemoryRecord
from .test_source_query import service  # noqa: F401

pytestmark = pytest.mark.asyncio


async def seed(service, **overrides):
    fields = dict(id="source-a", memory_type="assertion", content="用户今天打算饭后洗澡。",
                  subject=EntityRef(kind="user", id="user"), object=EntityRef.bot_self("bot"),
                  scope="private", session_id="qq:FriendMessage:user", lifecycle="stable_memory",
                  review_status="auto", reality_level="observed_utterance")
    fields.update(overrides)
    record = MemoryRecord(**fields)
    await service.store.insert_memory(record)
    return {"record": record, "score": .9}


def provider_for(service, summary, supported):
    provider = SimpleNamespace(text_chat=AsyncMock(side_effect=[
        SimpleNamespace(completion_text=summary),
        SimpleNamespace(completion_text=json.dumps({"supported": supported,
            "issues": [] if supported else ["单次安排不能提炼为习惯"]})),
    ]))
    service._summary_provider_attempts = AsyncMock(return_value=[
        {"provider": provider, "provider_id": "background", "source": "primary"}])
    service._schedule_memory_embedding = lambda *a: None
    return provider


async def test_unsupported_compression_keeps_original_available(service):
    original = await seed(service)
    provider_for(service, "用户一直习惯饭后洗澡。", False)
    result = await service._summarize_decay_group({"bucket": "one", "items": [original]})
    assert result["reason"] == "summary_failed"
    assert result["archived"] == 0 and not result["summary_id"]
    assert (await service.store.get_memory("source-a")).lifecycle == "stable_memory"
    assert service.store._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1


async def test_compression_archives_only_complete_reviewed_input(service):
    service.config.raw.setdefault("maintenance_decay", {})["memory_decay_summary_input_chars"] = 800
    a = await seed(service)
    b = await seed(service, id="source-b", content="独立重要事实。" * 400)
    provider = provider_for(service, "用户今天打算饭后洗澡。", True)
    result = await service._summarize_decay_group({"bucket": "one", "items": [a, b]})
    assert result["archived"] == 1
    assert (await service.store.get_memory("source-b")).lifecycle == "stable_memory"
    memory = await service.store.get_memory(result["summary_id"])
    assert memory.metadata["source_memory_ids"] == ["source-a"]
    review = provider.text_chat.call_args_list[1].kwargs["prompt"]
    assert '"source_kind":"memory"' in review.replace(" ", "")
    assert "source-b" not in review


async def test_pending_memory_cannot_be_promoted_by_compression(service):
    item = await seed(service, lifecycle="short_term_candidate", review_status="pending")
    provider = provider_for(service, "不应生成", True)
    result = await service._summarize_decay_group({"bucket": "one", "items": [item]})
    assert not result["summary_id"] and result["archived"] == 0
    provider.text_chat.assert_not_awaited()


async def test_maintenance_does_not_cold_archive_sources_that_never_fit_the_input(service):
    service.config.raw.setdefault("maintenance_decay", {}).update(
        memory_decay_summary_input_chars=800, memory_decay_archive_without_summary=True,
    )
    items = [await seed(service, id=f"long-{index}", content=f"独立事实 {index}。" * 200)
             for index in range(4)]
    service.store.list_decay_candidate_pool = AsyncMock(return_value=[item["record"] for item in items])
    service._decay_person_factors = AsyncMock(return_value={})
    service._decay_candidate = lambda record, **kwargs: {"record": record, "score": .9}
    service._split_decay_fade = lambda candidates: ([], candidates, 0)
    provider = provider_for(service, "根本不应调用模型。", True)

    result = await service._run_memory_decay()

    assert result["archived"] == 0 and result["summaries"] == 0, (
        "no input fitting the budget is not a providerless archive decision; all unread originals must survive"
    )
    for item in items:
        assert (await service.store.get_memory(item["record"].id)).lifecycle == "stable_memory"
    provider.text_chat.assert_not_awaited()
