"""用户命令实现。

命令只做展示与治理；真正的数据权限仍由 store / lifecycle 再校验一次。
"""

from __future__ import annotations

from typing import Any

from config.settings import MemorySettings
from governance.audit import AuditLogger
from governance.policy import PermissionDenied, ensure_memory_access
from lifecycle.manager import LifecycleManager
from memory_tool.service import MemoryToolService
from stats.metrics import MetricsCollector
from stats.retention import compute_retention
from store.models import AccessContext, SearchFilters
from store.sqlite_store import SQLiteStore
from utils.text import truncate_text
from writer.sanitizer import mask_sensitive

_TYPE_LABELS = {
    "fact": "事实",
    "preference": "偏好",
    "event": "事件",
    "task_state": "任务状态",
    "relationship": "关系",
    "constraint": "约束",
}

_HELP = """记忆路由命令：
  /memory list [主题] [--type 类型] [--limit N] [--all]
  /memory search <查询>
  /memory forget <id> [--hard]
  /memory keep <id> [--off]
  /memory why <id>
  /memory confirm <id>
  /memory correct <id> <新内容>
  /memory revive <id>
  /memory audit [N]
  /memory stats
  /memory compress <id1,id2,...>   （管理员）
  /memory maintain                  （管理员）
  /memory globalize <id>            （管理员，提升为全局共享记忆）
  /memory tagmerge <源标签> <目标标签>  （管理员，标签命名空间:值）
  /memory tags                      （管理员，查看标签使用情况）
"""


class MemoryCommandService:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        lifecycle: LifecycleManager,
        tools: MemoryToolService,
        metrics: MetricsCollector,
        audit: AuditLogger,
    ):
        self.store = store
        self.settings = settings
        self.lifecycle = lifecycle
        self.tools = tools
        self.metrics = metrics
        self.audit = audit

    async def handle(self, context: AccessContext, raw_args: str) -> str:
        args = (raw_args or "").strip()
        if not args:
            return _HELP
        action, _, rest = args.partition(" ")
        action = action.lower()
        rest = rest.strip()
        aliases = {
            "列表": "list", "查看": "list", "搜索": "search", "忘记": "forget",
            "删除": "forget", "保留": "keep", "为什么": "why", "审计": "audit",
            "统计": "stats", "确认": "confirm", "纠正": "correct", "复活": "revive",
            "压缩": "compress", "维护": "maintain", "帮助": "help",
            "全局": "globalize", "共享": "globalize", "标签": "tags",
            "标签合并": "tagmerge",
        }
        action = aliases.get(action, action)
        try:
            if action == "list":
                return await self._list(context, rest)
            if action == "search":
                return await self._search(context, rest)
            if action == "forget":
                return await self._forget(context, rest)
            if action == "keep":
                return await self._keep(context, rest)
            if action == "why":
                return await self._why(context, rest)
            if action == "confirm":
                return await self._confirm(context, rest)
            if action == "correct":
                return await self._correct(context, rest)
            if action == "revive":
                return await self._revive(context, rest)
            if action == "audit":
                return await self._audit(context, rest)
            if action == "stats":
                return await self._stats(context)
            if action == "compress":
                return await self._compress(context, rest)
            if action == "maintain":
                return await self._maintain(context)
            if action == "globalize":
                return await self._globalize(context, rest)
            if action == "tagmerge":
                return await self._tagmerge(context, rest)
            if action == "tags":
                return await self._tags(context)
            if action in {"help", "?"}:
                return _HELP
            return f"未知子命令：{action}\n\n{_HELP}"
        except PermissionDenied as exc:
            return f"权限不足：{exc}"
        except ValueError as exc:
            return f"参数错误：{exc}"
        except Exception as exc:  # noqa: BLE001 - 命令失败不应崩插件
            self.metrics.observe_error()
            return f"命令执行失败：{mask_sensitive(str(exc))[:200]}"

    # ------------------------------------------------------------------
    def _actor(self, context: AccessContext) -> str:
        return f"user:{context.user_id}"

    async def _list(self, context: AccessContext, rest: str) -> str:
        topic = ""
        memory_type = ""
        limit = 20
        include_all = False
        parts = rest.split()
        index = 0
        while index < len(parts):
            part = parts[index]
            if part == "--type" and index + 1 < len(parts):
                memory_type = parts[index + 1]
                index += 2
                continue
            if part == "--limit" and index + 1 < len(parts):
                try:
                    limit = max(1, min(50, int(parts[index + 1])))
                except ValueError:
                    pass
                index += 2
                continue
            if part == "--all":
                include_all = True
                index += 1
                continue
            if not topic:
                topic = part
            index += 1
        filters = SearchFilters.from_mapping(
            {
                "topic": topic,
                "sensitivity_max": "normal" if context.is_group else "secret",
                "lifecycle": ["active", "compressed", "archived", "soft_deleted"]
                if include_all else ["active"],
            }
        )
        memories = await self.store.list_memories(
            context, filters, limit=limit, include_inactive=include_all
        )
        if memory_type:
            memories = [
                item for item in memories
                if str(item.get("memory_type")) == memory_type
            ]
        if not memories:
            return "没有找到符合条件的记忆。"
        groups: dict[str, list[dict[str, Any]]] = {}
        for memory in memories:
            groups.setdefault(str(memory.get("memory_type") or "fact"), []).append(memory)
        lines = [f"长期记忆（{len(memories)} 条）"]
        for type_name, items in groups.items():
            lines.append(f"\n[{_TYPE_LABELS.get(type_name, type_name)}]")
            for memory in items:
                tags = await self.store.get_tags_map([str(memory["id"])])
                if str(memory.get("sensitivity") or "normal") != "normal":
                    tag_text = "[敏感标签已隐藏]"
                else:
                    tag_text = " ".join(tags.get(str(memory["id"]), [])[:4])
                pin = "📌 " if memory.get("pinned") else ""
                summary = self._safe_summary(memory, private=not context.is_group)
                lines.append(
                    f"{pin}{memory['id']}  {summary}\n"
                    f"    置信 {float(memory.get('confidence') or 0):.2f} · "
                    f"生命周期 {memory.get('lifecycle')} · {tag_text}"
                )
        return truncate_text("\n".join(lines), 3500)

    async def _search(self, context: AccessContext, rest: str) -> str:
        if not rest:
            return "用法：/memory search <查询>"
        result = await self.tools.memory_access(
            context,
            caller="user",
            mode="search",
            query=rest,
            filters={"sensitivity_max": "normal" if context.is_group else "secret"},
            budget={"max_items": 8, "max_tokens": 900, "max_latency_ms": 3000},
            options={
                "require_user_auth": not context.is_group,
                "include_scores": True,
                "dedup": True,
            },
        )
        data = result.data
        items = data.get("items") or []
        if not items:
            reason = (data.get("explain") or {}).get("error") or "没有匹配记忆"
            return f"没有找到相关记忆（{reason}）。"
        lines = [f"记忆搜索：{rest}"]
        for item in items:
            sensitivity = (item.get("scores") or {}).get("sensitivity", "normal")
            summary = str(item.get("summary") or "")
            if sensitivity != "normal":
                summary = "[敏感记忆，摘要已隐藏]"
            gate = (item.get("scores") or {}).get("gate", 0)
            lines.append(
                f"{item.get('memory_id')}  {summary}\n"
                f"    来源 {item.get('provenance', {}).get('source')} · "
                f"分数 {gate:.2f} · {item.get('lifecycle')}"
            )
        lines.append(f"\n{data.get('explain', {}).get('score_formula', '')}")
        return truncate_text("\n".join(lines), 3500)

    async def _forget(self, context: AccessContext, rest: str) -> str:
        parts = rest.split()
        if not parts:
            return "用法：/memory forget <id> [--hard]"
        memory_id = parts[0]
        hard = "--hard" in parts[1:]
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return "没有找到该记忆。"
        ensure_memory_access(context, memory)
        if hard:
            if not context.is_admin:
                return "硬删除需要管理员权限。"
            if not self.settings.hard_delete_enabled:
                return "硬删除未启用（hard_delete_enabled=false）。"
            ok = await self.lifecycle.hard_delete(
                memory_id,
                context=context,
                actor=self._actor(context),
                reason="user_hard_forget",
                authorized=True,
            )
            return "已硬删除并写入审计。" if ok else "硬删除失败。"
        ok = await self.lifecycle.soft_delete(
            memory_id,
            context=context,
            actor=self._actor(context),
            reason="user_forget",
        )
        if ok:
            await self._feedback(context, memory_id, "user_forget")
        return (
            "已软删除，可用 /memory revive 恢复。"
            if ok else "删除失败。"
        )

    async def _keep(self, context: AccessContext, rest: str) -> str:
        parts = rest.split()
        if not parts:
            return "用法：/memory keep <id> [--off]"
        memory_id = parts[0]
        pinned = "--off" not in parts[1:]
        ok = await self.lifecycle.pin(
            memory_id,
            context=context,
            pinned=pinned,
            actor=self._actor(context),
        )
        if not ok:
            return "未找到或无权操作该记忆。"
        return "已标记长期保留，自动流程不会删除。" if pinned else "已取消长期保留。"

    async def _why(self, context: AccessContext, rest: str) -> str:
        memory_id = rest.split()[0] if rest.split() else ""
        if not memory_id:
            return "用法：/memory why <id>"
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return "没有找到该记忆。"
        ensure_memory_access(context, memory)
        explanation = await self.lifecycle.explain(memory)
        trace = await self.store.find_trace_for_memory(
            memory_id, context.session_id, limit=100
        )
        lines = [
            f"记忆 {memory_id}",
            f"摘要：{self._safe_summary(memory, private=not context.is_group)}",
            f"生命周期：{memory.get('lifecycle')} / 状态：{memory.get('status')}",
            f"留存：{explanation.get('retention_explain')}",
            f"原因：{explanation.get('reason')}",
        ]
        if memory.get("compressed_from"):
            lines.append(
                "压缩来源：" + ", ".join(memory["compressed_from"][:10])
            )
        if memory.get("compressed_into"):
            lines.append(f"已压缩为：{memory['compressed_into']}")
        if trace:
            candidate = next(
                (
                    item for item in trace.get("candidates", [])
                    if isinstance(item, dict) and item.get("memory_id") == memory_id
                ),
                {},
            )
            lines.append(
                f"最近召回：trace={trace.get('trace_id')} "
                f"query={truncate_text(str(trace.get('query') or ''), 80)} "
                f"候选分={candidate.get('relevance', 'n/a')} "
                f"注入={memory_id in (trace.get('injected_ids') or [])}"
            )
        else:
            lines.append("最近召回：当前会话没有该记忆的召回记录。")
        access = await self.store.aggregate_access([memory_id])
        lines.append(f"访问统计：{access.get(memory_id, {})}")
        conflicts = explanation.get("conflicts") or []
        if conflicts:
            lines.append(f"冲突记录：{len(conflicts)} 条（未自动裁决真伪）")
        return truncate_text("\n".join(lines), 3500)

    async def _confirm(self, context: AccessContext, rest: str) -> str:
        memory_id = rest.split()[0] if rest.split() else ""
        if not memory_id:
            return "用法：/memory confirm <id>"
        result = await self.lifecycle.confirm(
            memory_id, context=context, actor=self._actor(context)
        )
        if not result:
            return "未找到或无权操作该记忆。"
        await self._feedback(context, memory_id, "user_confirmed")
        return f"已确认记忆 {memory_id}，置信度提升到 {result['confidence']:.2f}。"

    async def _correct(self, context: AccessContext, rest: str) -> str:
        memory_id, _, new_text = rest.partition(" ")
        if not memory_id or not new_text.strip():
            return "用法：/memory correct <id> <新内容>"
        result = await self.lifecycle.correct(
            memory_id,
            new_text,
            context=context,
            actor=self._actor(context),
        )
        if not result:
            return "未找到或无权操作该记忆。"
        await self._feedback(context, memory_id, "user_corrected")
        return (
            f"已纠正：旧记忆 {memory_id} 标记为 superseded，"
            f"新记忆 {result['new_memory_id']}。"
        )

    async def _revive(self, context: AccessContext, rest: str) -> str:
        memory_id = rest.split()[0] if rest.split() else ""
        if not memory_id:
            return "用法：/memory revive <id>"
        result = await self.lifecycle.revive(
            memory_id, context=context, actor=self._actor(context)
        )
        if not result:
            return "未找到或无权操作该记忆。"
        if result.get("warning") and not result.get("confidence"):
            return result["warning"]
        await self._feedback(context, memory_id, "revived")
        return (
            f"已复活记忆 {memory_id}，置信度 {result.get('confidence')}。\n"
            f"提示：{result.get('warning')}"
        )

    async def _audit(self, context: AccessContext, rest: str) -> str:
        try:
            limit = max(1, min(100, int(rest.strip() or "20")))
        except ValueError:
            limit = 20
        actor = "" if context.is_admin else self._actor(context)
        rows = await self.store.list_audit(actor=actor, limit=limit)
        if not rows:
            return "没有审计记录。"
        lines = ["审计记录："]
        for row in rows:
            lines.append(
                f"{row.get('created_at')} {row.get('action')} "
                f"target={row.get('target')} actor={row.get('actor')}\n"
                f"  {truncate_text(str(row.get('detail')), 300)}"
            )
        return truncate_text("\n".join(lines), 4000)

    async def _stats(self, context: AccessContext) -> str:
        overview = await self.store.stats_overview(context)
        metrics = self.metrics.snapshot()
        memories = await self.store.list_memories(
            context, SearchFilters.from_mapping({"lifecycle": ["active"]}), limit=200
        )
        stats = await self.store.get_access_stats([str(item["id"]) for item in memories])
        scores = []
        for memory in memories:
            scores.append(compute_retention(memory, stats.get(str(memory["id"]))).score)
        average = sum(scores) / len(scores) if scores else 0.0
        access_totals = await self.store.access_event_totals(context)
        corrections = access_totals.get("user_corrected", 0)
        used = access_totals.get("used_by_main_model", 0)
        injected = access_totals.get("injected", 0)
        correction_rate = corrections / max(1, used + injected)
        lines = [
            "记忆统计",
            f"可见记忆：{overview.get('total', 0)} 条",
            f"按类型：{overview.get('by_type', {})}",
            f"按生命周期：{overview.get('by_lifecycle', {})}",
            f"按状态：{overview.get('by_status', {})}",
            f"平均留存分（前200条）：{average:.3f}",
            f"召回指标：{metrics}",
            f"用户纠错率（纠正/使用+注入）：{correction_rate:.3f}",
        ]
        if context.is_admin:
            actions = await self.store.audit_action_counts(
                ["memory.archive", "memory.revive", "memory.hard_delete"]
            )
            archived = actions.get("memory.archive", 0)
            revived = actions.get("memory.revive", 0)
            false_kill_proxy = revived / max(1, archived)
            lines.append(
                f"遗忘误杀代理（复活/归档）：{false_kill_proxy:.3f} "
                f"(archive={archived}, revive={revived})"
            )
        return truncate_text("\n".join(lines), 4000)

    async def _compress(self, context: AccessContext, rest: str) -> str:
        if not context.is_admin:
            return "压缩需要管理员权限。"
        ids = [item.strip() for item in rest.split(",") if item.strip()]
        if len(ids) < 2:
            return "用法：/memory compress <id1,id2,...>（至少 2 条）"
        new_id = await self.lifecycle.compress(
            ids, context=context, actor=self._actor(context)
        )
        return f"已压缩为 {new_id}，原文保留在冷存储，可通过 compressed_from 回溯。"

    async def _maintain(self, context: AccessContext) -> str:
        if not context.is_admin:
            return "维护命令需要管理员权限。"
        report = await self.lifecycle.run_maintenance()
        return (
            f"维护完成：留存更新 {report.retention_updated} 条，"
            f"归档 {len(report.archived)} 条，压缩 {len(report.compressed)} 组，"
            f"清理标签 {report.tags_pruned} 个，错误 {len(report.errors)} 个。"
            + ("\n" + "\n".join(report.errors[:5]) if report.errors else "")
        )

    async def _globalize(self, context: AccessContext, rest: str) -> str:
        if not context.is_admin:
            return "全局共享需要管理员权限。"
        memory_id = rest.split()[0] if rest.split() else ""
        if not memory_id:
            return "用法：/memory globalize <id>"
        result = await self.lifecycle.promote_global(
            memory_id, context=context, actor=self._actor(context)
        )
        if not result:
            return "未找到或无权操作该记忆。"
        return f"已提升为全局共享记忆：{memory_id}（已审计）。"

    async def _tagmerge(self, context: AccessContext, rest: str) -> str:
        if not context.is_admin:
            return "标签合并需要管理员权限。"
        parts = rest.split()
        if len(parts) != 2:
            return "用法：/memory tagmerge <namespace:源标签> <namespace:目标标签>"
        count = await self.store.merge_tags(parts[0], parts[1])
        await self.audit.log(
            actor=self._actor(context),
            action="memory.tag_merge",
            target=parts[0],
            detail={"target": parts[1], "affected": count},
        )
        return f"已合并标签 {parts[0]} -> {parts[1]}，影响 {count} 条记忆。"

    async def _tags(self, context: AccessContext) -> str:
        if not context.is_admin:
            return "标签统计需要管理员权限。"
        rows = await self.store.tag_stats(limit=30)
        if not rows:
            return "没有标签。"
        lines = ["标签使用情况："]
        for row in rows:
            lines.append(
                f"{row['namespace']}:{row['tag']}  usage={row['usage']} "
                f"status={row['status']} version={row['version']}"
            )
        return truncate_text("\n".join(lines), 3500)

    async def _feedback(
        self, context: AccessContext, memory_id: str, feedback: str
    ) -> None:
        """把用户反馈回写到最近一次包含该记忆的召回 trace。"""
        try:
            trace = await self.store.find_trace_for_memory(
                memory_id, context.session_id, limit=20
            )
            trace_id = str(trace.get("trace_id") or "")
            if trace_id:
                await self.store.update_trace_feedback(trace_id, feedback)
        except Exception:  # noqa: BLE001 - 反馈不应影响主命令结果
            return

    @staticmethod
    def _safe_summary(memory: dict[str, Any], *, private: bool) -> str:
        summary = str(memory.get("summary") or memory.get("content") or "")
        sensitivity = str(memory.get("sensitivity") or "normal")
        if sensitivity != "normal":
            return "[敏感记忆，摘要已隐藏]"
        return truncate_text(mask_sensitive(summary), 160 if private else 100)
