"""Core blocks must contain full facts, never clipped subjective prose."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from .test_source_query import service

@pytest.mark.asyncio
async def test_diaries_and_conversation_are_not_always_on_facts(service):
    service.list_core_memory_blocks=AsyncMock(return_value=[])
    service.store.list_memories=AsyncMock(return_value=[SimpleNamespace(memory_type=t)
        for t in ('persona_life','conversation_summary','self_action','proactive_message')])
    service.save_core_memory_block=AsyncMock()
    result=await service.auto_distill_core_memory()
    assert result['created']==0
    service.save_core_memory_block.assert_not_awaited()

@pytest.mark.asyncio
async def test_complete_fact_tail_survives_distillation(service):
    text='明确的长期规则：'+'需保留的完整条件，'*18+'仅在用户明确请求时适用。'
    service.list_core_memory_blocks=AsyncMock(return_value=[])
    service.store.list_memories=AsyncMock(return_value=[SimpleNamespace(memory_type='rule',lifecycle='stable_memory',
        review_status='manual',salience=.9,scope='global',content=value) for value in (text,'另一条明确的规则。')])
    service.save_core_memory_block=AsyncMock(return_value={'ok':True})
    await service.auto_distill_core_memory()
    saved=service.save_core_memory_block.call_args.args[0]['content']
    assert text in saved and '…' not in saved
