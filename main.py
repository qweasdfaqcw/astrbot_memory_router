"""astrbot_memory_router 插件入口。

只在这里注册 AstrBot Handler / Tool，业务逻辑全部委托给模块。所有持久化
数据位于 AstrBot ``data/plugin_data/astrbot_memory_router/``，不写插件目录。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

# AstrBot 可能把插件目录重命名后以独立模块方式加载，子模块不在 sys.path 中。
# 在导入本项目模块前把插件根目录放到最前面；所有内部模块使用绝对导入，
# 兼容 AstrBot 热重载与目录重命名。
_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from adapter.astrbot_adapter import (
    append_dynamic_context,
    build_access_context,
    get_event_text,
    is_command_text,
    resolve_data_directory,
)
from adapter.llm_adapter import LLMAdapter
from commands.memory_commands import MemoryCommandService
from config.settings import MemorySettings
from gate.gate import Gate
from governance.audit import AuditLogger
from governance.rate_limit import RateLimiter
from ingress.buffer import SessionBuffer
from ingress.session_state import SessionStateStore
from injector.injector import MemoryInjector
from lifecycle.manager import LifecycleManager
from memory_tool.service import MemoryToolService
from retriever.query_rewrite import QueryRewriter
from retriever.retriever import MemoryRetriever
from router.reranker import LLMReranker, NoopReranker, Reranker
from router.router import MemoryRouter
from stats.metrics import MetricsCollector
from store.sqlite_store import SQLiteStore
from store.vector_backend import build_vector_backend
from utils.json_utils import dump
from utils.text import normalize_text
from writer.extractor import HybridExtractor
from writer.pipeline import MemoryWriter
from writer.sanitizer import parse_memory_directive

PLUGIN_NAME = "astrbot_memory_router"


class MemoryRouterPlugin(Star):
    """AstrBot 长期记忆路由插件。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        # Star 基类可能已经由 AstrBot 注入 name；不要覆盖它。
        self.settings = MemorySettings.from_mapping(config or {})
        self._fatal = ""
        self._runtime_started = False
        self._shutting_down = False
        self._startup_lock = asyncio.Lock()
        self._extract_locks: dict[str, asyncio.Lock] = {}
        self._extract_tasks: dict[str, asyncio.Task[Any]] = {}
        self._extract_pending: set[str] = set()
        self._maintenance_task: asyncio.Task[Any] | None = None
        self.buffer = SessionBuffer(
            max_turns=self.settings.buffer_max_turns,
            ttl_seconds=self.settings.buffer_ttl_seconds,
        )
        self.sessions = SessionStateStore()
        self.metrics = MetricsCollector()
        self.rate_limiter = RateLimiter(
            per_minute=self.settings.rate_limit_per_minute,
            tokens_per_hour=self.settings.rate_limit_tokens_per_hour,
        )
        self.store: SQLiteStore | None = None
        self.usage_rules = ""
        self.writer: MemoryWriter | None = None
        self.router: MemoryRouter | None = None
        self.tools: MemoryToolService | None = None
        self.lifecycle: LifecycleManager | None = None
        self.commands: MemoryCommandService | None = None
        self.audit: AuditLogger | None = None
        try:
            data_dir = resolve_data_directory(self.settings.data_dir)
            self.data_dir = data_dir
            db_path = data_dir / self.settings.db_filename
            self.store = SQLiteStore(db_path)
            self.store.open()
            self._build_services()
            logger.info(
                "%s 已初始化: db=%s vector=%s",
                PLUGIN_NAME,
                db_path,
                self.settings.vector_backend,
            )
        except Exception as exc:  # noqa: BLE001 - 插件加载失败不应拖垮 AstrBot
            self._fatal = f"{type(exc).__name__}: {exc}"
            self.data_dir = None
            logger.exception("记忆路由初始化失败: %s", self._fatal)

    def _build_services(self) -> None:
        assert self.store is not None
        self.settings.db_filename = self.settings.db_filename or "memory_router.sqlite3"
        llm = LLMAdapter(self.context, self.settings.small_model_provider_id)
        self.usage_rules = llm.usage_rules()
        extractor = HybridExtractor(
            self.settings, llm, enabled=self.settings.extract_enabled
        )
        vector = build_vector_backend(
            self.settings.vector_backend, self.store, self.settings.embedding_dim
        )
        self.writer = MemoryWriter(self.store, self.settings, extractor, vector)
        rewriter = QueryRewriter(
            llm, use_llm=self.settings.query_rewrite_with_llm
        )
        retriever = MemoryRetriever(self.store, self.settings, vector, rewriter)
        reranker: Reranker
        if self.settings.llm_rerank_enabled:
            reranker = LLMReranker(llm, self.settings)
        else:
            reranker = NoopReranker()
        gate = Gate(self.settings)
        injector = MemoryInjector()
        self.router = MemoryRouter(
            self.store,
            self.settings,
            retriever,
            reranker,
            gate,
            injector,
            self.metrics,
            self.rate_limiter,
        )
        self.tools = MemoryToolService(
            self.store,
            self.settings,
            retriever,
            reranker,
            gate,
            self.metrics,
            self.rate_limiter,
        )
        self.audit = AuditLogger(self.store)
        self.lifecycle = LifecycleManager(
            self.store, self.settings, self.audit, vector
        )
        self.commands = MemoryCommandService(
            self.store,
            self.settings,
            self.lifecycle,
            self.tools,
            self.metrics,
            self.audit,
        )

    async def _ensure_runtime(self) -> None:
        if self._fatal:
            return
        if self._runtime_started:
            return
        async with self._startup_lock:
            if self._runtime_started or self._fatal:
                return
            assert self.store is not None
            await self.store.start()
            self._runtime_started = True
            if self.settings.auto_maintenance and self._maintenance_task is None:
                self._maintenance_task = asyncio.create_task(
                    self._maintenance_loop(), name="memory-router-maintenance"
                )

    async def _maintenance_loop(self) -> None:
        interval = max(60, self.settings.maintenance_interval_seconds)
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.lifecycle:
                    continue
                report = await self.lifecycle.run_maintenance()
                logger.info(
                    "记忆维护: retention=%s archived=%s compressed=%s errors=%s",
                    report.retention_updated,
                    len(report.archived),
                    len(report.compressed),
                    len(report.errors),
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("记忆维护任务失败")

    async def _ensure_session_runtime(self) -> None:
        if not self._fatal:
            await self._ensure_runtime()

    def _admin_ids(self) -> list[str]:
        return list(self.settings.admin_user_ids)

    def _context_for(self, event: AstrMessageEvent):
        return build_access_context(
            event,
            admin_ids=self._admin_ids(),
            allow_personal_in_group=self.settings.allow_personal_in_group,
        )

    # ------------------------------------------------------------------
    # 消息接入 / 写通路
    # ------------------------------------------------------------------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message_capture(self, event: AstrMessageEvent):
        """接收消息并维护短期缓冲；满足条件时异步抽取长期记忆。"""
        if self._fatal or not self.settings.enabled:
            return
        try:
            await self._ensure_session_runtime()
            text = get_event_text(event)
            if not text or is_command_text(text):
                return
            context = self._context_for(event)
            directive = parse_memory_directive(text)
            suppressed = directive.action == "suppress"
            turn = self.buffer.add_turn(
                context.session_id, "user", text, suppressed=suppressed
            )
            if turn is None:
                return
            if suppressed:
                if self.audit:
                    await self.audit.log(
                        actor=f"user:{context.user_id}",
                        action="memory.write_suppressed",
                        target=context.session_id,
                        detail={"reason": "explicit_no_remember"},
                    )
                return
            if directive.action == "delete":
                # 纯文本里的“删掉记忆”不自动执行；必须走 /memory forget，防误删。
                if self.audit:
                    await self.audit.log(
                        actor=f"user:{context.user_id}",
                        action="memory.delete_request_requires_command",
                        target=context.session_id,
                        detail={"summary": "用户要求删除记忆，已提示使用 /memory forget"},
                    )
                return
            if directive.action == "remember" and self.settings.explicit_remember_enabled:
                await self._run_explicit_extraction(context, event)
                return
            pending = self.buffer.pending_turns(context.session_id)
            char_count = sum(len(item.content) for item in pending)
            if (
                len(pending) >= self.settings.extraction_every_n_messages
                or char_count >= self.settings.extraction_min_chars * 2
            ):
                self._schedule_extraction(context, event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.metrics.observe_error()
            logger.exception("消息接入失败")

    async def _turns_for_extraction(
        self,
        context: Any,
        event: AstrMessageEvent,
        *,
        buffer_limit: int,
        pending_only: bool = False,
    ) -> list[dict[str, str]]:
        buffer_turns = (
            self.buffer.pending_turns(context.session_id)
            if pending_only
            else self.buffer.turns(context.session_id, limit=buffer_limit)
        )
        # 只把本会话的未抽取缓冲交给抽取器；AstrBot 历史可能包含已经处理过的
        # 助手回复，重复送入会造成反复抽取。conversation_manager 适配保留给
        # 未来做“只读上下文、不作为抽取源”的增强。
        return [
            {"role": turn.role, "content": turn.content}
            for turn in buffer_turns[-buffer_limit:]
        ]

    async def _run_explicit_extraction(
        self, context: Any, event: AstrMessageEvent
    ) -> None:
        if not self.writer:
            return
        # 显式“记住”只同步处理当前消息，避免把历史会话重新抽取一遍。
        current_turns = self.buffer.turns(context.session_id, limit=1)
        if not current_turns:
            return
        turns = [
            {"role": turn.role, "content": turn.content}
            for turn in current_turns
        ]
        lock = self._extract_locks.setdefault(context.session_id, asyncio.Lock())
        async with lock:
            try:
                report = await self.writer.process_turns(
                    turns, context, event=event, explicit=True
                )
                self.buffer.mark_extracted(
                    context.session_id, [item.id for item in current_turns]
                )
                self.metrics.observe_write(
                    created=len(report.created_ids),
                    merged=len(report.merged_ids),
                    conflicts=len(report.conflict_ids),
                    degraded=report.degraded,
                )
                logger.info(
                    "显式记忆写入: created=%s merged=%s conflicts=%s epoch=%s degraded=%s",
                    len(report.created_ids),
                    len(report.merged_ids),
                    len(report.conflict_ids),
                    report.write_epoch,
                    report.degraded,
                )
            except Exception:  # noqa: BLE001
                self.metrics.observe_error()
                logger.exception("显式记忆写入失败")

    def _schedule_extraction(self, context: Any, event: AstrMessageEvent) -> None:
        session_id = context.session_id
        existing = self._extract_tasks.get(session_id)
        if existing and not existing.done():
            self._extract_pending.add(session_id)
            return
        task = asyncio.create_task(
            self._extraction_worker(session_id, context, event),
            name=f"memory-extract-{session_id[:24]}",
        )
        self._extract_tasks[session_id] = task

        def _done(_: asyncio.Task[Any]) -> None:
            self._extract_tasks.pop(session_id, None)
            if self._shutting_down:
                return
            if session_id in self._extract_pending:
                self._extract_pending.discard(session_id)
                pending = self.buffer.pending_turns(session_id)
                if pending:
                    self._schedule_extraction(context, event)

        task.add_done_callback(_done)

    async def _extraction_worker(
        self, session_id: str, context: Any, event: AstrMessageEvent
    ) -> None:
        try:
            if self.settings.extraction_debounce_seconds > 0:
                await asyncio.sleep(self.settings.extraction_debounce_seconds)
            if not self.writer:
                return
            lock = self._extract_locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                pending = self.buffer.pending_turns(session_id)
                if not pending:
                    return
                turns = await self._turns_for_extraction(
                    context, event, buffer_limit=10, pending_only=True
                )
                if not turns:
                    return
                report = await self.writer.process_turns(
                    turns, context, event=event, explicit=False
                )
                self.buffer.mark_extracted(session_id, [item.id for item in pending])
                self.metrics.observe_write(
                    created=len(report.created_ids),
                    merged=len(report.merged_ids),
                    conflicts=len(report.conflict_ids),
                    degraded=report.degraded,
                )
                logger.info(
                    "后台记忆写入: created=%s merged=%s degraded=%s epoch=%s",
                    len(report.created_ids),
                    len(report.merged_ids),
                    report.degraded,
                    report.write_epoch,
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.metrics.observe_error()
            logger.exception("后台记忆抽取失败")

    # ------------------------------------------------------------------
    # 读通路：LLM 请求前预取注入
    # ------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        """在主模型请求前检索少量记忆，作为不可执行的历史资料注入。"""
        if (
            self._fatal
            or not self.settings.enabled
            or not self.settings.retrieval_enabled
            or not self.settings.injection_enabled
            or not self.router
        ):
            return
        try:
            text = get_event_text(event)
            if not text or is_command_text(text):
                return
            await self._ensure_session_runtime()
            # 稳定的工具触发规则放在 system_prompt；每轮变化的记忆摘要放在
            # extra_user_content_parts，避免把动态内容塞进系统提示破坏缓存。
            if self.usage_rules:
                system_prompt = getattr(req, "system_prompt", None)
                if isinstance(system_prompt, str) and "长期记忆工具使用规则" not in system_prompt:
                    req.system_prompt = (
                        system_prompt.rstrip()
                        + "\n\n"
                        + self.usage_rules
                    )
            context = self._context_for(event)
            state = self.sessions.get(context.session_id)
            recent = self.buffer.turns(context.session_id, limit=6)
            result = await self.router.prefetch(
                context,
                text,
                recent_turns=recent,
                event=event,
                session_state=state,
            )
            if result.injection.empty:
                return
            location = append_dynamic_context(req, result.injection.text)
            if location == "none":
                logger.warning("无法把记忆注入到 LLM 请求，位置不可用")
                return
            state.last_injected_text = result.injection.text
            logger.info(
                "记忆注入: trace=%s items=%s tokens=%s candidates=%s location=%s",
                result.trace_id,
                result.injection.item_count,
                result.injection.estimated_tokens,
                len(result.retrieval.candidates) if result.retrieval else 0,
                location,
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 注入失败不能阻塞对话
            self.metrics.observe_error()
            logger.exception("记忆注入失败")
            return

    # ------------------------------------------------------------------
    # 主模型工具
    # ------------------------------------------------------------------
    @staticmethod
    def _tool_result(text: str):
        """构造 AstrBot 支持的工具结果。

        当前 AstrBot 的本地工具执行器接受 str 或 mcp.types.CallToolResult。
        这里优先返回 CallToolResult；旧版本 / 本地测试环境无 mcp 时回退为 str。
        """
        try:
            from mcp.types import CallToolResult, TextContent

            return CallToolResult(
                content=[TextContent(type="text", text=text)]
            )
        except Exception:  # noqa: BLE001
            return text

    @filter.llm_tool(name="memory_access")
    async def memory_access_tool(
        self,
        event: AstrMessageEvent,
        caller: str = "main_model",
        mode: str = "search",
        query: str = "",
        filters: dict | None = None,
        budget: dict | None = None,
        options: dict | None = None,
    ):
        """长期记忆统一访问接口。默认只返回摘要与引用，全文用 mode=fetch 拉取。

        Args:
            caller(string): 调用者标识；工具调用固定为 main_model，服务端不会采信越权值
            mode(string): search 搜索摘要、fetch 按 memory_ids 拉全文、prefetch 预取、probe 轻量探测
            query(string): 当前查询；fetch 模式可传单个 memory_id
            filters(object): 过滤对象，支持 time_range、tags_any、tags_all、memory_ids、owner_scope、sensitivity_max 等
            budget(object): 预算对象，支持 max_tokens、max_items、max_latency_ms
            options(object): 选项对象，支持 include_scores、include_conflicts、require_user_auth、dedup、diversify
        """
        if self._fatal or not self.tools or not self.settings.enabled:
            return self._tool_result(
                dump({
                    "items": [],
                    "error": self._fatal or ("memory_disabled" if not self.settings.enabled else "plugin_not_ready"),
                })
            )
        try:
            await self._ensure_session_runtime()
            context = self._context_for(event)
            result = await self.tools.memory_access(
                context,
                caller="main_model",
                mode=mode,
                query=query,
                filters=filters,
                budget=budget,
                options=options,
            )
            return self._tool_result(
                json.dumps(result.data, ensure_ascii=False, default=str)
            )
        except Exception as exc:  # noqa: BLE001
            self.metrics.observe_error()
            logger.exception("memory_access 工具失败")
            return self._tool_result(
                dump({"items": [], "error": f"memory_access_failed:{exc}"})
            )

    @filter.llm_tool(name="memory_probe")
    async def memory_probe_tool(
        self,
        event: AstrMessageEvent,
        query: str,
        filters: dict | None = None,
    ):
        """轻量探测是否存在相关长期记忆，只返回候选摘要和建议查询，不返回全文。

        Args:
            query(string): 要探测的查询
            filters(object): 可选过滤对象，支持 owner_scope、tags_any、tags_all、sensitivity_max 等
        """
        if self._fatal or not self.tools or not self.settings.enabled:
            return self._tool_result(
                dump({
                    "exists": False,
                    "error": self._fatal or ("memory_disabled" if not self.settings.enabled else "plugin_not_ready"),
                })
            )
        try:
            await self._ensure_session_runtime()
            context = self._context_for(event)
            result = await self.tools.memory_probe(
                context, query=query, filters=filters
            )
            return self._tool_result(
                json.dumps(result.data, ensure_ascii=False, default=str)
            )
        except Exception as exc:  # noqa: BLE001
            self.metrics.observe_error()
            logger.exception("memory_probe 工具失败")
            return self._tool_result(
                dump({"exists": False, "error": f"memory_probe_failed:{exc}"})
            )

    # ------------------------------------------------------------------
    # 用户命令
    # ------------------------------------------------------------------
    @filter.command("memory", alias={"记忆"})
    async def memory_command(self, event: AstrMessageEvent):
        """长期记忆管理：list/search/forget/keep/why/audit/stats/confirm/correct/revive。"""
        if self._fatal or not self.commands or not self.settings.enabled:
            reason = self._fatal or ("memory_disabled" if not self.settings.enabled else "not_ready")
            yield event.plain_result(f"记忆插件未就绪：{reason}")
            return
        try:
            await self._ensure_session_runtime()
            text = normalize_text(get_event_text(event))
            args = self._command_args(text)
            context = self._context_for(event)
            response = await self.commands.handle(context, args)
            yield event.plain_result(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.metrics.observe_error()
            logger.exception("memory 命令失败")
            yield event.plain_result(f"命令执行失败：{exc}")

    @staticmethod
    def _command_args(text: str) -> str:
        raw = text.strip()
        if raw.startswith(("/", "／")):
            raw = raw[1:].strip()
        if not raw:
            return ""
        parts = raw.split(None, 1)
        first = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        # 兼容 /memory@bot 的群聊写法。
        if first.split("@", 1)[0] not in {"memory", "记忆"}:
            return raw
        return rest.strip()

    # ------------------------------------------------------------------
    # 卸载
    # ------------------------------------------------------------------
    async def terminate(self):
        """插件卸载/停用时关闭队列、任务与数据库。"""
        self._shutting_down = True
        tasks: list[asyncio.Task[Any]] = []
        if self._maintenance_task:
            self._maintenance_task.cancel()
            tasks.append(self._maintenance_task)
        for task in list(self._extract_tasks.values()):
            task.cancel()
            tasks.append(task)
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._extract_tasks.clear()
        if self.store:
            await self.store.close()
        logger.info("记忆路由已关闭")
