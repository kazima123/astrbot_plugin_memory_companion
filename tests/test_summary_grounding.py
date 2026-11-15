"""Lexically matching a quote must not authorize speaker or habit inventions."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from core.summarizer import MemorySummarizer, SummaryFormatError
from .test_source_query import service, ctx

@pytest.mark.asyncio
@pytest.mark.parametrize('claim,issue', [
    ('我夸用户最会心疼人', 'summary：原话由用户发出，不能归给 Bot'),
    ('用户习惯饭后洗澡', 'assertions：只有一次当晚安排，不支持习惯'),
    ('用户当天一直没有出现', 'summary：单向窗口不支持全天缺席'),
])
async def test_semantic_rejection_preserves_reason_and_role(claim, issue):
    s = MemorySummarizer()
    provider = SimpleNamespace(text_chat=AsyncMock(return_value=SimpleNamespace(
        completion_text=json.dumps({'supported':False,'issues':[issue]},ensure_ascii=False))))
    rows=[dict(id='one',message_id='native-1',event_type='user_message',subject_id='user',
               content='还是我家宝宝最会心疼人了',metadata='{}',occurred_at='2026-11-14T11:44:00+00:00')]
    usage=[]
    errors=await s.review_grounding(provider,payload={'summary':claim},rows=rows,provider_id='background',
                                   usage_recorder=lambda **kw:usage.append(kw))
    assert errors == [issue]
    prompt=provider.text_chat.call_args.kwargs['prompt']
    assert '"speaker_role":"user"' in prompt.replace(' ','')
    assert 'native-1' in prompt
    assert usage[0]['task']=='memory_summary_grounding'

@pytest.mark.asyncio
async def test_missing_verdict_does_not_mean_supported():
    s=MemorySummarizer()
    p=SimpleNamespace(text_chat=AsyncMock(return_value=SimpleNamespace(completion_text='{}')))
    with pytest.raises(SummaryFormatError):
        await s.review_grounding(p,payload={},rows=[],provider_id='background',usage_recorder=None)

def test_long_utterance_keeps_its_final_negation_and_separate_quote_context():
    summarizer = MemorySummarizer()
    text = "这是前面的背景。" * 110 + "但我并没有答应明天去。"
    rows = [dict(id="long", event_type="user_message", subject_id="user", content=text,
                 metadata=json.dumps({"reply_chain_preview": "Bot 曾问明天是否一起去",
                                      "reply_chain_message_ids": ["quoted-one"]}))]
    transcript, consumed = summarizer._transcript_lines_and_rows(rows)
    parsed = json.loads(transcript[0])
    assert parsed["content"].endswith("但我并没有答应明天去。")
    assert parsed["quoted_context"]["message_ids"] == ["quoted-one"]
    assert "明天是否一起去" not in parsed["content"]
    assert consumed == rows

def test_generated_reply_is_labeled_as_a_draft_in_summary_input():
    summarizer = MemorySummarizer()
    rows = [dict(id="draft", event_type="bot_response", subject_id="bot", content="我答应你。",
                 metadata=json.dumps({"delivery_evidence": "generated"}))]
    transcript, _ = summarizer._transcript_lines_and_rows(rows)
    assert json.loads(transcript[0])["delivery_evidence"] == "generated"
    assert "不代表已发送" in summarizer._build_prompt(rows, "private")

def test_text_and_voice_share_one_reply_turn_and_control_marker_is_not_a_fact():
    summarizer = MemorySummarizer()
    rows = [dict(id=f"part-{i}", event_type="bot_response", subject_id="bot",
                 message_id=f"sent:core-reply:one-turn:{i}", content=text, metadata="{}")
            for i, text in enumerate(["我们一会儿吃饭。", "我们一会儿吃饭。", "<<PRIVATE_COMPANION_SPLIT>>"])]
    transcript, consumed = summarizer._transcript_lines_and_rows(rows)
    assert len(consumed) == 2
    assert {json.loads(line)["reply_turn_id"] for line in transcript} == {"sent:core-reply:one-turn"}

@pytest.mark.asyncio
@pytest.mark.parametrize('supported',[True,False])
async def test_verdict_controls_stable_memory_and_derivatives_within_budget(service,ctx,supported):
    service.config.raw['memory_summary']={'max_retries':2,'min_events':1,'trigger_event_count':1}
    ref=await service.store.add_timeline_event(event_type='user_message',session_id=ctx.session_id,scope='private',
        subject_id=ctx.user_id,object_id=ctx.bot_id,content='还是我家宝宝最会心疼人了',
        metadata={'owner_bot_id':ctx.bot_id,'message_id':'one'})
    payload={'summary':'用户说我最会心疼人了，这句话让我觉得被关心。',
             'persona_summary':'用户夸我会心疼人，我觉得很暖。','canonical_summary':'用户夸 Bot 会心疼人。',
             'summary_refs':[ref],'key_facts':[],'_consumed_event_ids':[ref]}
    service.summarizer.summarize_with_provider=AsyncMock(return_value=payload)
    provider=SimpleNamespace(text_chat=AsyncMock(return_value=SimpleNamespace(completion_text=json.dumps(
        {'supported':supported,'issues':[] if supported else ['summary：说话人归属错误']}))))
    service._summary_provider_attempts=AsyncMock(return_value=[{'provider':provider,'provider_id':'background','source':'primary'}])
    service._schedule_memory_embedding=lambda *a,**k:None
    service._record_self_experience=AsyncMock()
    service._persist_assertions=AsyncMock()
    service._record_verified_group_bot_self_facts=AsyncMock()
    service._index_summary_knowledge_graph=AsyncMock()
    ident=await service.maybe_summarize_session(ctx,force=True)
    row=await service.store.get_memory(ident)
    assert row.lifecycle==('stable_memory' if supported else 'short_term_candidate')
    assert service._record_self_experience.await_count==int(supported)
    assert service._persist_assertions.await_count==int(supported)
    assert service.store._conn.execute('select count(*) from summary_batch_calls').fetchone()[0]==2
