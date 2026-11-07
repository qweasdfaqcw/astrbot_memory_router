"""memory_access / memory_probe 测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.helpers import build_stack, user_context


def test_memory_access_defaults_to_summary_and_fetch_returns_content(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        report = await stack.writer.process_turns(
            [{"role": "user", "content": "记住：我喜欢喝美式咖啡。"}],
            ctx,
            explicit=True,
        )
        memory_id = report.created_ids[0]
        result = await stack.tools.memory_access(
            ctx, caller="main_model", mode="search", query="咖啡"
        )
        assert result.data["items"]
        item = result.data["items"][0]
        assert item["memory_id"] == memory_id
        assert "content" not in item
        assert item["content_ref"] == f"memory://{memory_id}"
        assert "scores" in item
        assert result.data["explain"]["threshold"] <= 0.25

        fetched = await stack.tools.memory_access(
            ctx,
            caller="main_model",
            mode="fetch",
            filters={"memory_ids": [memory_id]},
        )
        assert fetched.data["items"][0]["content"]
        traces = await stack.store.get_recent_traces(ctx.session_id, limit=5)
        assert traces
        assert any(trace["used"] for trace in traces)
        await stack.store.close()

    asyncio.run(scenario())


def test_probe_returns_lightweight_structure(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        ctx = user_context()
        await stack.writer.process_turns(
            [{"role": "user", "content": "我计划下个月去杭州旅行。"}],
            ctx,
            explicit=True,
        )
        result = await stack.tools.memory_probe(ctx, query="杭州旅行")
        data = result.data
        assert data["exists"] is True
        assert data["candidates"]
        assert "suggested_queries" in data
        assert all("content" not in candidate for candidate in data["candidates"])
        await stack.store.close()

    asyncio.run(scenario())


def test_fetch_cannot_cross_user(tmp_path: Path):
    async def scenario():
        stack = await build_stack(tmp_path)
        alice = user_context("alice", "sa")
        bob = user_context("bob", "sb")
        report = await stack.writer.process_turns(
            [{"role": "user", "content": "记住：我的生日是 3 月 5 日。"}],
            alice,
            explicit=True,
        )
        result = await stack.tools.memory_access(
            bob,
            caller="user",
            mode="fetch",
            filters={"memory_ids": [report.created_ids[0]]},
            options={"require_user_auth": True},
        )
        assert result.data["items"] == []
        assert result.data["blocked"][0]["reason"] == "not_found_or_out_of_scope"
        await stack.store.close()

    asyncio.run(scenario())
