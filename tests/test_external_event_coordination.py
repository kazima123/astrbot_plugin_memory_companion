"""Companion pushes retain delivery identity, scope, and the latest journal revision."""
import json
from unittest.mock import AsyncMock

import pytest

from core.models import EntityRef, SessionContext
from .test_source_query import service


pytestmark = pytest.mark.asyncio


async def test_failed_send_audit_rows_are_never_user_evidence(service):
    row = {"id": "failed", "event_type": "bot_draft", "subject_id": "bot-one", "content": "未发送草稿", "metadata": "{}"}
    assert service.summarizer.rows_for_prompt([row]) == []
    assert service._timeline_row_as_memory(SessionContext(), row) is None


async def test_confirmed_proactive_timeline_keeps_the_bot_as_speaker(service):
    ctx = SessionContext(bot_id="bot-one", user_id="user-one", scope="private")
    row = {"id": "sent", "event_type": "proactive_message", "subject_id": "bot-one", "content": "饭好了。", "metadata": "{}"}
    assert service._timeline_row_as_memory(ctx, row).subject.kind == "bot"


@pytest.fixture
def companion_record(service):
    service._resolve_default_persona_id = AsyncMock(return_value="persona-one")
    service._schedule_memory_embedding = lambda *args, **kwargs: None
    return {
        "memory_id": "contact-one",
        "content": "晚饭煮好了，过来一起吃吧。",
        "memory_type": "proactive_message",
        "subject": EntityRef(kind="bot", id="bot-one", name="小星", role="bot_self"),
        "object": EntityRef(kind="user", id="user-one", role="target_user"),
        "scope": "private",
        "session_id": "qq:FriendMessage:user-one",
        "platform": "qq",
        "reality_level": "bot_action",
        "source_plugin": "astrbot_plugin_private_companion",
        "occurred_at": "2026-11-14T12:30:00+00:00",
        "metadata": {"clean_visible_text": "晚饭煮好了，过来一起吃吧。"},
    }


async def test_retried_companion_push_is_one_memory_and_one_timeline_source(service, companion_record):
    first = await service.record_external_event(**companion_record)
    second = await service.record_external_event(**companion_record)
    assert first == second == "contact-one"
    assert service.store._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    rows = service.store._conn.execute("SELECT * FROM timeline").fetchall()
    assert len(rows) == 1, "a retried push must not become two independent observations of the same event"
    assert rows[0]["content"] == companion_record["metadata"]["clean_visible_text"]
    assert rows[0]["message_id"], "the bridge's stable memory identity must also deduplicate its timeline projection"


async def test_companion_push_keeps_occurrence_time_and_live_bot_persona(service, companion_record):
    memory_id = await service.record_external_event(**companion_record)
    record = await service.store.get_memory(memory_id)
    row = service.store._conn.execute("SELECT * FROM timeline").fetchone()
    metadata = json.loads(row["metadata"])
    assert record.occurred_at == row["occurred_at"] == companion_record["occurred_at"], (
        "retry or bridge arrival time must not move an evening event into another day"
    )
    assert record.owner_bot_id == row["bot_id"] == "bot-one"
    assert record.metadata["persona_id"] == metadata["persona_id"] == "persona-one"
    assert (row["subject_id"], row["object_id"], row["session_id"]) == (
        "bot-one", "user-one", "qq:FriendMessage:user-one",
    )
    service._resolve_default_persona_id.assert_awaited_once()
    assert service._resolve_default_persona_id.await_args.args[0].unified_msg_origin == companion_record["session_id"]


async def test_companion_bridge_respects_disabled_group_capture(service, companion_record):
    service.config.raw.setdefault("scope_control", {})["group_capture_enabled"] = False
    companion_record.update(
        scope="group", group_id="group-one", session_id="qq:GroupMessage:group-one",
        object=EntityRef(kind="group", id="group-one", role="target_group"),
    )
    assert await service.record_external_event(**companion_record) == ""
    assert service.store._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 0
    assert service.store._conn.execute("SELECT count(*) FROM timeline").fetchone()[0] == 0
    service._resolve_default_persona_id.assert_not_awaited()


async def test_revised_journal_replaces_recall_but_preserves_previous_text(service, companion_record):
    companion_record.update(
        memory_id="journal-one-r1", memory_type="persona_life", reality_level="persona_life",
        content="日记初稿：今天没有一起下棋。",
        metadata={"journal_entry_id": "journal-one", "journal_revision": 1, "event_type": "diary"},
    )
    old_id = await service.record_external_event(**companion_record)
    other_id = await service.record_external_event(**{
        **companion_record, "memory_id": "journal-another-r1", "content": "另一天的日记内容。",
        "metadata": {"journal_entry_id": "journal-another", "journal_revision": 1, "event_type": "diary"},
    })
    new_id = await service.record_external_event(**{
        **companion_record, "memory_id": "journal-one-r2", "content": "日记修订：今天一起下了井字棋，结果是平局。",
        "metadata": {"journal_entry_id": "journal-one", "journal_revision": 2, "event_type": "diary"},
    })
    old, new, other = [await service.store.get_memory(item) for item in (old_id, new_id, other_id)]
    assert old.lifecycle == "archived", "a corrected journal must not coexist with the old claim in active recall"
    assert old.content == companion_record["content"], "archival must preserve the reviewed historical version"
    assert new.lifecycle == other.lifecycle == "stable_memory"
    assert old.supersedes_id == new_id
    assert new.supersedes_id == "", "archive_memories links the retired version to its replacement"
    assert service.store._conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 3


async def test_late_retry_of_old_journal_cannot_reactivate_superseded_claim(service, companion_record):
    companion_record.update(
        memory_id="journal-late-r1", memory_type="persona_life", reality_level="persona_life",
        content="初稿误写成五子棋平局。",
        metadata={"journal_entry_id": "journal-late", "journal_revision": 1, "event_type": "diary"},
    )
    old_id = await service.record_external_event(**companion_record)
    new_id = await service.record_external_event(**{
        **companion_record, "memory_id": "journal-late-r2", "content": "更正为井字棋平局。",
        "metadata": {"journal_entry_id": "journal-late", "journal_revision": 2, "event_type": "diary"},
    })
    assert await service.record_external_event(**companion_record) == old_id
    assert (await service.store.get_memory(old_id)).lifecycle == "archived", (
        "a delayed retry must not revive a journal claim already corrected by a newer revision"
    )
    assert (await service.store.get_memory(new_id)).lifecycle == "stable_memory"


async def test_same_text_does_not_merge_distinct_journals_or_their_revisions(service, companion_record):
    companion_record.update(
        memory_type="persona_life", reality_level="persona_life", content="今天过得很平静，我很喜欢。",
    )
    ids = []
    for entry, revision in (("day-one", 1), ("day-two", 1), ("day-one", 2)):
        record_id = f"{entry}-r{revision}"
        ids.append(await service.record_external_event(**{
            **companion_record, "memory_id": record_id,
            "metadata": {"journal_entry_id": entry, "journal_revision": revision, "event_type": "diary"},
        }))
    assert ids == ["day-one-r1", "day-two-r1", "day-one-r2"], (
        "identical diary wording is not evidence that two dates or revisions are the same document"
    )
    old, unrelated, current = [await service.store.get_memory(record_id) for record_id in ids]
    assert old.lifecycle == "archived" and old.supersedes_id == current.id
    assert unrelated.lifecycle == current.lifecycle == "stable_memory"
    assert unrelated.metadata["journal_entry_id"] == "day-two"
    assert current.metadata["journal_revision"] == 2
