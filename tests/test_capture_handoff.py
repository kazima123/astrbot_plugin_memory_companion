"""A received turn is one source; it must never stand in for the bot's reply."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import json

import pytest

from .test_source_query import service, ctx
from .test_source_capture_lifecycle_p1 import event

pytestmark = pytest.mark.asyncio


async def test_observer_and_request_share_one_original(service, ctx):
    ctx.message_text = "我先去处理工作，饭好了叫我"
    ctx.user_name = "小林"
    incoming = event(ctx.message_text, ctx.message_id)
    service.reply_chain.resolve = AsyncMock(return_value=[])
    service.portraits.capture_user_message = AsyncMock(return_value={})
    service._schedule_session_summary = lambda *args, **kwargs: None
    await service.handle_group_message(incoming)
    await service.capture.incoming(incoming, ctx)
    rows = service.store._conn.execute("SELECT * FROM timeline").fetchall()
    assert len(rows) == 1, "the observer and LLM hook must not turn one utterance into two evidence sources"
    assert rows[0]["content"] == ctx.message_text
    assert json.loads(rows[0]["metadata"])["sender_name"] == "小林"


async def test_incoming_receipt_does_not_suppress_standalone_reply(service, ctx):
    ctx.message_text = "饭好了叫我"
    incoming = event("饭好了叫我", ctx.message_id)
    await service.capture.incoming(incoming, ctx)
    service._schedule_session_summary = lambda *args, **kwargs: None
    await service.handle_llm_response(incoming, SimpleNamespace(role="assistant", completion_text="好，等饭好了我叫你。你先安心处理工作，我记得你的安排。"))
    rows = service.store._conn.execute("SELECT * FROM timeline WHERE event_type='bot_response'").fetchall()
    assert len(rows) == 1, "an incoming receipt proves nothing about outgoing capture"


async def test_native_delivery_records_only_returned_parts_once(service, ctx):
    incoming = event("晚饭好了叫我", ctx.message_id)
    incoming._memory_companion_delivery_owner = True
    await service.capture.incoming(incoming, ctx)
    service._schedule_session_summary = lambda *args, **kwargs: None
    await service.handle_llm_response(incoming, SimpleNamespace(role="assistant", completion_text="未发送的草稿"))
    assert service.store._conn.execute("SELECT count(*) FROM timeline WHERE event_type='bot_response'").fetchone()[0] == 0
    text = "饭煮好了，过来吃吧。"
    for _ in range(2):
        receipt = service.capture.sent(ctx, text, "turn-1:part-1")
        assert receipt["status"] == "committed"
    row = service.store._conn.execute("SELECT * FROM timeline WHERE event_type='bot_response'").fetchone()
    assert row["content"] == text
    assert row["subject_id"] == ctx.bot_id
    assert json.loads(row["metadata"])["delivery_evidence"] == "adapter_completed"
    assert service.store._conn.execute("SELECT count(*) FROM timeline WHERE event_type='bot_response'").fetchone()[0] == 1


async def test_sent_capture_respects_disabled_bot_capture(service, ctx):
    service.config.raw.setdefault("memory_capture", {})["capture_bot_responses"] = False
    result = service.capture.sent(ctx, "不应入库", "turn-2:part-1")
    assert result["status"] == "excluded"
    assert service.store._conn.execute("SELECT count(*) FROM timeline").fetchone()[0] == 0
