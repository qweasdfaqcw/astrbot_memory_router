"""存储层测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from store.models import AccessContext, MemoryRecord, SearchFilters, TagInput
from store.sqlite_store import SQLiteStore


def test_store_crud_fts_isolation_and_access(tmp_path: Path):
    async def scenario():
        store = SQLiteStore(tmp_path / "memory.db")
        store.open()
        await store.start()
        try:
            memory = MemoryRecord(
                content="我喜欢喝美式咖啡，不加糖",
                summary="喜欢美式咖啡",
                memory_type="preference",
                owner_scope="personal",
                owner_id="u1",
                source="user",
                confidence=0.8,
                importance=0.7,
            )
            memory_id = await store.create_memory(
                memory, [TagInput(namespace="topic", tag="咖啡")]
            )
            assert memory_id.startswith("mem_")
            assert await store.get_write_epoch() == 1

            fetched = await store.get_memory(memory_id)
            assert fetched["content"].startswith("我喜欢")
            assert (await store.get_tags_map([memory_id]))[memory_id]

            fts = await store.search_fts(
                "美式咖啡", AccessContext(user_id="u1"), SearchFilters()
            )
            assert memory_id in [item[0] for item in fts]

            other = await store.get_memories_by_ids(
                [memory_id], AccessContext(user_id="u2")
            )
            assert other == []

            await store.record_access(
                memory_id,
                "main_model",
                "search",
                "used_by_main_model",
                session_id="s1",
                trace_id="trc_test",
            )
            stats = (await store.get_access_stats([memory_id]))[memory_id]
            assert stats.count("used_by_main_model") == 1
            assert stats.last_access_at

            conflict_id = await store.add_conflict(
                memory_id, "mem_other", "negation"
            )
            assert conflict_id.startswith("cft_")
            conflicts = await store.get_conflicts_for([memory_id])
            assert conflicts[memory_id][0]["conflict_type"] == "negation"

            await store.add_audit(
                actor="user:u1",
                action="test",
                target=memory_id,
                detail={"note": "ok"},
            )
            audits = await store.list_audit(limit=5)
            assert audits and audits[0]["action"] == "test"
        finally:
            await store.close()

    asyncio.run(scenario())


def test_tag_merge_and_orphan_prune(tmp_path: Path):
    async def scenario():
        store = SQLiteStore(tmp_path / "memory.db")
        store.open()
        await store.start()
        try:
            memory = MemoryRecord(
                content="标签测试内容",
                summary="标签测试",
                memory_type="fact",
                owner_scope="personal",
                owner_id="u1",
            )
            memory_id = await store.create_memory(
                memory,
                [
                    TagInput(namespace="free", tag="旧标签"),
                    TagInput(namespace="free", tag="新标签"),
                ],
            )
            affected = await store.merge_tags("free:旧标签", "free:新标签")
            assert affected == 1
            tags = (await store.get_tags_map([memory_id]))[memory_id]
            assert "free:旧标签" not in tags
            assert "free:新标签" in tags
            # 没有记忆引用的标签会被标记 inactive。
            await store.prune_orphan_tags()
            stats = await store.tag_stats(limit=20)
            old = next(item for item in stats if item["tag"] == "旧标签")
            assert old["status"] in {"merged", "inactive"}
        finally:
            await store.close()

    asyncio.run(scenario())


def test_group_memory_not_visible_in_private(tmp_path: Path):
    async def scenario():
        store = SQLiteStore(tmp_path / "memory.db")
        store.open()
        await store.start()
        try:
            group_memory = MemoryRecord(
                content="群里的约定是周五开会",
                summary="周五开会",
                memory_type="task_state",
                owner_scope="group",
                owner_id="g1",
            )
            group_id = await store.create_memory(group_memory)
            private_context = AccessContext(user_id="u1", session_id="private")
            assert await store.get_memories_by_ids([group_id], private_context) == []
            group_context = AccessContext(
                user_id="u1", group_id="g1", session_id="group", is_group=True
            )
            assert (
                await store.get_memories_by_ids([group_id], group_context)
            )[0]["id"] == group_id
        finally:
            await store.close()

    asyncio.run(scenario())
