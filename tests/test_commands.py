"""用户命令测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.helpers import build_stack, user_context


def test_commands_list_keep_why_stats_and_forget(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        report = await stack.writer.process_turns(
            [{"role": "user", "content": "记住：我喜欢喝美式咖啡。"}],
            ctx,
            explicit=True,
        )
        memory_id = report.created_ids[0]
        listing = await stack.commands.handle(ctx, "list")
        assert memory_id in listing
        assert "咖啡" in listing

        kept = await stack.commands.handle(ctx, f"keep {memory_id}")
        assert "长期保留" in kept
        why = await stack.commands.handle(ctx, f"why {memory_id}")
        assert "留存" in why and memory_id in why
        stats = await stack.commands.handle(ctx, "stats")
        assert "记忆统计" in stats and "召回指标" in stats
        audit = await stack.commands.handle(ctx, "audit")
        assert "审计记录" in audit

        forgotten = await stack.commands.handle(ctx, f"forget {memory_id}")
        assert "软删除" in forgotten
        memory = await stack.store.get_memory(memory_id)
        assert memory["lifecycle"] == "soft_deleted"
        revived = await stack.commands.handle(ctx, f"revive {memory_id}")
        assert "已复活" in revived
        await stack.store.close()

    asyncio.run(scenario())


def test_command_correct_keeps_history(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        report = await stack.writer.process_turns(
            [{"role": "user", "content": "记住：我的生日是 3 月 5 日。"}],
            ctx,
            explicit=True,
        )
        old_id = report.created_ids[0]
        result = await stack.commands.handle(ctx, f"correct {old_id} 我的生日是 3 月 6 日。")
        assert "已纠正" in result
        old = await stack.store.get_memory(old_id)
        assert old["status"] == "superseded"
        stats = await stack.store.get_access_stats([old_id])
        assert stats[old_id].count("user_corrected") == 1
        await stack.store.close()

    asyncio.run(scenario())
