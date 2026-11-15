"""Companion's extra recall path must respect the same memory decisions."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from core.companion_linkage import CompanionMemoryLinkage
from core.models import EntityRef, MemoryRecord
from .test_reply_continuity import service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def linkage(service):
    service._resolve_default_persona_id = AsyncMock(return_value="persona-a")
    return CompanionMemoryLinkage(SimpleNamespace(service=service))


def request(**scope_changes):
    return {"context": {"scope": {
        "kind": "private", "native_scope_id": "user-a", "session_id": "user-a",
        "platform": "qq", "bot_id": "logical-role-owner", "account_id": "qq-bot",
        **scope_changes,
    }}, "limit": 12}


async def add_memory(service, record_id, **changes):
    values = dict(
        id=record_id, memory_type="conversation_summary", content=f"已核实记忆 {record_id}",
        subject=EntityRef(kind="user", id="user-a"), object=EntityRef.bot_self(bot_id="qq-bot"),
        scope="private", session_id="qq:FriendMessage:user-a", platform="qq",
        owner_bot_id="qq-bot", visibility="private_pair", lifecycle="stable_memory",
        review_status="auto", metadata={"persona_id": "persona-a"},
    )
    values.update(changes)
    await service.store.insert_memory(MemoryRecord(**values))


async def test_native_account_identity_keeps_the_actual_pair_readable(service, linkage):
    await add_memory(service, "allowed")
    result = await linkage._read(request())
    assert [item["id"] for item in result["items"]] == ["allowed"], (
        "Companion's logical bot owner must not be confused with Memory's native QQ owner"
    )
    event = service._resolve_default_persona_id.call_args.args[0]
    assert event.unified_msg_origin == "qq:FriendMessage:user-a"


@pytest.mark.parametrize("changes", [
    {"memory_type": "core_memory", "metadata": {"core_enabled": False}},
    {"lifecycle": "raw_event"},
    {"memory_type": "assertion", "metadata": {"profile_state": "candidate"}},
    {"review_status": "pending", "lifecycle": "short_term_candidate"},
    {"metadata": {"persona_id": "persona-b"}},
    {"owner_bot_id": "another-account", "object": EntityRef.bot_self(bot_id="another-account")},
    {"platform": "other", "session_id": "other:FriendMessage:user-a"},
])
async def test_extra_reader_does_not_reintroduce_unusable_memory(service, linkage, changes):
    await add_memory(service, "allowed")
    await add_memory(service, "excluded", **changes)
    result = await linkage._read(request())
    assert [item["id"] for item in result["items"]] == ["allowed"], (
        "a disabled core block or unaccepted fact cannot become evidence via Companion's second reader"
    )


@pytest.mark.parametrize("section,key", [
    ("scope_control", "private_recall_enabled"),
    ("private_companion_bridge", "enabled"),
])
async def test_disabled_recall_is_not_bypassed_by_the_registered_reader(service, linkage, section, key):
    await add_memory(service, "allowed")
    service.config.raw.setdefault(section, {})[key] = False
    assert await linkage._read(request()) == {"items": []}


@pytest.mark.parametrize("changes", [{"kind": "group"}, {"account_id": ""}, {"session_id": "another-user"}])
async def test_reader_requires_the_actual_private_context(service, linkage, changes):
    await add_memory(service, "allowed")
    assert await linkage._read(request(**changes)) == {"items": []}


async def test_recall_keeps_the_qualification_at_the_end_or_omits_the_whole_record(service, linkage):
    text = "关于散步的梦境细节。" * 150 + "这些只发生在梦里，没有在现实中一起出门。"
    await add_memory(service, "whole", content=text)
    result = await linkage._read({**request(), "max_chars": 2400})
    assert result["items"][0]["content"] == text, "the end of a memory may carry its essential qualification"
    assert await linkage._read({**request(), "max_chars": 1200}) == {"items": []}
    assert await linkage._read({**request(), "max_chars": 0}) == {"items": []}


async def test_companion_hot_reload_replaces_the_registration_instead_of_reporting_a_stale_connection(linkage):
    previous = SimpleNamespace(closed=False, register=Mock(return_value="old-token"), unregister=Mock())
    host = SimpleNamespace(_private_companion_memory_providers=previous)
    linkage.plugin.context = host
    assert linkage.register()
    assert linkage.register()
    assert previous.register.call_count == 1

    replacement = SimpleNamespace(closed=False, register=Mock(return_value="new-token"), unregister=Mock())
    host._private_companion_memory_providers = replacement
    assert linkage.register(), "opening the existing bridge after reload must reconnect to the new registry"
    previous.unregister.assert_called_once_with("old-token")
    replacement.register.assert_called_once()
    assert linkage._registry is replacement and linkage.registered

    replacement.closed = True
    replacement.register.side_effect = ValueError("memory provider registry is closed")
    assert not linkage.register(), "a stopped host cannot remain reported as an active connection"
    replacement.unregister.assert_called_once_with("new-token")
    assert not linkage.registered
