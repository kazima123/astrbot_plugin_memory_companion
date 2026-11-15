"""Internal companion prompts must not become user evidence or identity hints."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from .test_source_query import service  # noqa: F401


@pytest.mark.asyncio
@pytest.mark.parametrize("event", [
    SimpleNamespace(private_companion_proactive_framework=True, message_str="主动联系素材"),
    SimpleNamespace(message_str="Private Companion 主动消息：这不是用户消息"),
])
async def test_internal_generation_skips_identity_and_all_capture(service, event):
    service.identity.resolve_event_context = AsyncMock()
    service._absorb_relation_hints = AsyncMock()
    service.capture.incoming = AsyncMock()
    request = SimpleNamespace(prompt="内部生成任务", contexts=[], extra_user_content_parts=[])

    await service.handle_llm_request(event, request)

    service.identity.resolve_event_context.assert_not_awaited()
    service._absorb_relation_hints.assert_not_awaited()
    service.capture.incoming.assert_not_awaited()
    assert not hasattr(event, "_memory_companion_sent_observer")
    assert service.store._conn.execute("SELECT count(*) FROM identities").fetchone()[0] == 0
    assert service.store._conn.execute("SELECT count(*) FROM timeline").fetchone()[0] == 0
