"""生命周期测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from store.models import AccessContext, MemoryRecord, SearchFilters
from tests.helpers import build_stack, user_context


async def _seed_event(store, index: int, owner: str = "u1") -> str:
    record = MemoryRecord(
        content=f"上周和同事讨论项目 Alpha 的第 {index} 次进展。",
        summary=f"项目 Alpha 进展 {index}",
        memory_type="event",
        owner_scope="personal",
        owner_id=owner,
        source="user",
        importance=0.3,
        confidence=0.7,
    )
    return await store.create_memory(record)


def test_compress_archive_soft_delete_and_revive(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        ids = [await _seed_event(stack.store, i) for i in range(3)]
        compressed = await stack.lifecycle.compress(
            ids[:2], context=ctx, actor="user:u1", summary="项目 Alpha 前两次进展汇总"
        )
        assert compressed.startswith("mem_")
        new = await stack.store.get_memory(compressed)
        assert set(new["compressed_from"]) == set(ids[:2])
        for old_id in ids[:2]:
            old = await stack.store.get_memory(old_id)
            assert old["lifecycle"] == "compressed"
            assert old["compressed_into"] == compressed
            assert await stack.store.get_cold(old_id)
        # 压缩原文不在热索引中
        fts = await stack.store.search_fts("第 0 次进展", ctx, SearchFilters())
        assert old_id not in [item[0] for item in fts]

        assert await stack.lifecycle.archive(ids[2:], actor="user:u1", reason="test")
        archived = await stack.store.get_memory(ids[2])
        assert archived["lifecycle"] == "archived"
        revived = await stack.lifecycle.revive(ids[2], context=ctx, actor="user:u1")
        assert revived["warning"]
        assert (await stack.store.get_memory(ids[2]))["lifecycle"] == "active"

        assert await stack.lifecycle.soft_delete(
            ids[2], context=ctx, actor="user:u1", reason="test"
        )
        soft = await stack.store.get_memory(ids[2])
        assert soft["lifecycle"] == "soft_deleted"
        await stack.lifecycle.revive(ids[2], context=ctx, actor="user:u1")
        assert (await stack.store.get_memory(ids[2]))["lifecycle"] == "active"
        await stack.store.close()

    asyncio.run(scenario())


def test_hard_delete_requires_authorization_and_writes_audit(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context(is_admin=True)
        ids = [await _seed_event(stack.store, i) for i in range(2)]
        try:
            await stack.lifecycle.hard_delete(
                ids[0],
                context=ctx,
                actor="user:u1",
                reason="test",
                authorized=False,
            )
            raise AssertionError("不应允许未授权硬删除")
        except PermissionError:
            pass
        stack.settings.hard_delete_enabled = True
        ok = await stack.lifecycle.hard_delete(
            ids[0],
            context=ctx,
            actor="user:u1",
            reason="test",
            authorized=True,
        )
        assert ok
        assert await stack.store.get_memory(ids[0]) == {}
        audits = await stack.store.list_audit(limit=10)
        assert any(item["action"] == "memory.hard_delete" for item in audits)
        await stack.store.close()

    asyncio.run(scenario())


def test_promote_global_is_admin_only_and_visible_to_others(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        admin = AccessContext(user_id="u1", session_id="admin", is_admin=True)
        ordinary = AccessContext(user_id="u1", session_id="ordinary")
        report = await stack.writer.process_turns(
            [{"role": "user", "content": "记住：公共约定是每周一同步。"}],
            admin,
            explicit=True,
        )
        memory_id = report.created_ids[0]
        try:
            await stack.lifecycle.promote_global(
                memory_id, context=ordinary, actor="user:u1"
            )
            raise AssertionError("普通用户不应能提升全局记忆")
        except PermissionError:
            pass
        result = await stack.lifecycle.promote_global(
            memory_id, context=admin, actor="user:u1"
        )
        assert result["owner_scope"] == "global"
        bob = AccessContext(user_id="bob", session_id="bob")
        found = await stack.store.get_memories_by_ids([memory_id], bob)
        assert found and found[0]["owner_scope"] == "global"
        await stack.store.close()

    asyncio.run(scenario())


def test_pinned_memory_is_protected_from_archive(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        memory_id = await _seed_event(stack.store, 9)
        await stack.lifecycle.pin(memory_id, context=ctx, pinned=True, actor="user:u1")
        archived = await stack.lifecycle.archive(
            [memory_id], actor="maintenance", reason="auto_archive"
        )
        assert archived == []
        assert (await stack.store.get_memory(memory_id))["lifecycle"] == "active"
        await stack.store.close()

    asyncio.run(scenario())
