"""memory_access / memory_probe 的业务实现。

主模型只能拿到「摘要 + 来源 + 置信度 + 冲突 + 分数」；全文必须显式 fetch。
fetch 仍受 owner/敏感性/不可执行内容策略限制，且所有调用都会写访问统计与 trace。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Mapping

from config.settings import MemorySettings
from gate.gate import Gate, GateDecision, GateItem
from governance.policy import (
    PermissionDenied,
    can_read_sensitive,
    ensure_memory_access,
    filter_sensitivity_for_caller,
    normalize_filter_input,
    validate_filter_scope,
)
from governance.rate_limit import RateLimiter
from retriever.retriever import MemoryRetriever, RetrievedMemory
from router.reranker import Reranker
from stats.metrics import MetricsCollector
from store.models import AccessContext, SearchFilters, unique_strings
from store.sqlite_store import SQLiteStore
from utils.ids import new_trace_id
from utils.text import escape_markup, estimate_tokens, truncate_text
from utils.time_utils import now_iso
from writer.sanitizer import mask_sensitive

_ALLOWED_MODES = {"search", "fetch", "prefetch", "probe"}
_ALLOWED_CALLERS = {"main_model", "router_small", "system", "user"}


@dataclass(slots=True)
class ToolResult:
    data: dict[str, Any]
    trace_id: str = ""
    latency_ms: int = 0


class MemoryToolService:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        retriever: MemoryRetriever,
        reranker: Reranker,
        gate: Gate,
        metrics: MetricsCollector,
        rate_limiter: RateLimiter,
    ):
        self.store = store
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.gate = gate
        self.metrics = metrics
        self.rate_limiter = rate_limiter

    async def memory_access(
        self,
        context: AccessContext,
        *,
        caller: str = "main_model",
        mode: str = "search",
        query: str = "",
        filters: Any = None,
        budget: Any = None,
        options: Any = None,
    ) -> ToolResult:
        caller = caller if caller in _ALLOWED_CALLERS else "main_model"
        mode = mode if mode in _ALLOWED_MODES else "search"
        filters_map = normalize_filter_input(filters)
        options_map = normalize_filter_input(options)
        budget_map = normalize_filter_input(budget)
        raw_query = truncate_text(str(query or ""), 1000)
        if mode == "fetch":
            return await self._fetch(
                context,
                caller=caller,
                query=raw_query,
                filters_map=filters_map,
                budget_map=budget_map,
                options_map=options_map,
            )
        if mode == "probe":
            return await self._probe(
                context,
                caller=caller,
                query=raw_query,
                filters_map=filters_map,
                budget_map=budget_map,
                options_map=options_map,
            )
        return await self._search(
            context,
            caller=caller,
            mode=mode,
            query=raw_query,
            filters_map=filters_map,
            budget_map=budget_map,
            options_map=options_map,
        )

    async def memory_probe(
        self,
        context: AccessContext,
        *,
        query: str,
        filters: Any = None,
        budget: Any = None,
        options: Any = None,
    ) -> ToolResult:
        return await self.memory_access(
            context,
            caller="main_model",
            mode="probe",
            query=query,
            filters=filters,
            budget=budget,
            options=options,
        )

    # ------------------------------------------------------------------
    # search / prefetch
    # ------------------------------------------------------------------
    async def _search(
        self,
        context: AccessContext,
        *,
        caller: str,
        mode: str,
        query: str,
        filters_map: Mapping[str, Any],
        budget_map: Mapping[str, Any],
        options_map: Mapping[str, Any],
    ) -> ToolResult:
        started = time.monotonic()
        trace_id = new_trace_id()
        if not query.strip() and not filters_map.get("tags_any") and not filters_map.get("memory_ids"):
            return ToolResult(
                data={
                    "items": [],
                    "truncated": False,
                    "next_cursor": None,
                    "explain": {"error": "empty_query"},
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
            )
        try:
            filters = SearchFilters.from_mapping(filters_map)
            validate_filter_scope(context, filters)
            require_user_auth = bool(options_map.get("require_user_auth", False))
            filters = filter_sensitivity_for_caller(
                filters, caller, context, require_user_auth
            )
        except PermissionDenied as exc:
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "permission_denied", "detail": str(exc)},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        rate = self.rate_limiter.check(
            context.session_id, estimated_tokens=int(budget_map.get("max_tokens", 800) or 800)
        )
        if not rate.allowed and caller != "user":
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "rate_limited", "detail": rate.reason},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        timeout_s = max(0.2, min(8.0, float(budget_map.get("max_latency_ms", 2500) or 2500) / 1000.0))

        async def pipeline():
            retrieval = await self.retriever.retrieve(
                context,
                query,
                filters,
                recent_turns=(),
                limit=self.settings.candidate_limit,
                include_inactive="archived" in filters.lifecycle,
            )
            reranked = await self.reranker.rerank(
                retrieval.plan, retrieval.candidates, context, filters, event=None
            )
            explicit_options = dict(options_map)
            explicit_options.setdefault("explicit_search", True)
            explicit_options.setdefault(
                "require_user_auth", bool(options_map.get("require_user_auth", False))
            )
            decided = await self.gate.evaluate(
                reranked,
                context,
                filters,
                caller=caller,
                budget=budget_map,
                options=explicit_options,
            )
            return retrieval, reranked, decided

        try:
            retrieval, candidates, decision = await asyncio.wait_for(
                pipeline(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            self.metrics.observe_error()
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "timeout"}, "trace_id": trace_id},
                trace_id=trace_id,
            )
        latency = int((time.monotonic() - started) * 1000)
        items = [
            self._item_payload(item, options_map, include_content=False)
            for item in decision.selected
        ]
        deduped = self._dedup_items(items) if options_map.get("dedup", True) else items
        truncated = decision.truncated or len(deduped) < len(decision.selected)
        offset = filters.offset
        next_cursor = str(offset + len(deduped)) if truncated else None
        response = {
            "items": deduped,
            "truncated": truncated,
            "next_cursor": next_cursor,
            "explain": self._explain(
                retrieval.plan,
                decision,
                latency,
                channel_counts=retrieval.channel_counts,
            ),
            "trace_id": trace_id,
            "mode": mode,
        }
        self.metrics.observe_recall(
            latency_ms=latency,
            candidates=len(candidates),
            injected_items=len(deduped),
            injected_tokens=sum(item.get("estimated_tokens", 0) for item in deduped),
            blocked_sensitive=decision.blocked_sensitive,
            blocked_injection=decision.blocked_injection,
        )
        self.rate_limiter.consume(
            context.session_id,
            tokens=sum(item.get("estimated_tokens", 0) for item in deduped),
        )
        await self._record_recall(
            context,
            trace_id=trace_id,
            query=query,
            plan=retrieval.plan.normalized,
            candidates=candidates,
            decision=decision,
            caller=caller,
            mode=mode,
            latency=latency,
            injected_ids=[item.get("memory_id", "") for item in deduped],
            injected_text="\n".join(str(item.get("summary") or "") for item in deduped),
        )
        return ToolResult(
            data=response,
            trace_id=trace_id,
            latency_ms=latency,
        )

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------
    async def _fetch(
        self,
        context: AccessContext,
        *,
        caller: str,
        query: str,
        filters_map: Mapping[str, Any],
        budget_map: Mapping[str, Any],
        options_map: Mapping[str, Any],
    ) -> ToolResult:
        started = time.monotonic()
        trace_id = new_trace_id()
        try:
            filters = SearchFilters.from_mapping(filters_map)
            validate_filter_scope(context, filters)
        except PermissionDenied as exc:
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "permission_denied", "detail": str(exc)},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        ids = list(filters.memory_ids)
        if not ids and query.startswith("memory://"):
            ids = [query.strip()[len("memory://"):]]
        elif not ids and query.startswith("mem_"):
            ids = [query.strip()]
        if not ids:
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "fetch_requires_memory_ids"},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        rate = self.rate_limiter.check(
            context.session_id, estimated_tokens=400
        )
        if not rate.allowed and caller != "user":
            return ToolResult(
                data={"items": [], "truncated": False, "next_cursor": None,
                      "explain": {"error": "rate_limited", "detail": rate.reason},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        max_items = _clamp_int(budget_map.get("max_items", 5), 1, 20)
        fetch_token_budget = _clamp_int(
            budget_map.get("max_tokens", 4000), 80, 16000
        )
        fetch_used_tokens = 0
        ids = unique_strings(ids, limit=max_items + 20)
        require_user_auth = bool(options_map.get("require_user_auth", False))
        memories = await self.store.get_memories_by_ids(
            ids, context, filters, include_inactive=True
        )
        by_id = {str(item["id"]): item for item in memories}
        tags_map, conflicts_map, stats_map = await asyncio.gather(
            self.store.get_tags_map(ids),
            self.store.get_conflicts_for(ids),
            self.store.get_access_stats(ids),
        )
        items: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for memory_id in ids:
            memory = by_id.get(memory_id)
            if not memory:
                blocked.append({"memory_id": memory_id, "reason": "not_found_or_out_of_scope"})
                continue
            ensure_memory_access(context, memory)
            lifecycle = str(memory.get("lifecycle") or "active")
            if lifecycle == "soft_deleted" and caller != "user":
                blocked.append({"memory_id": memory_id, "reason": "soft_deleted"})
                continue
            sensitivity = str(memory.get("sensitivity") or "normal")
            if sensitivity != "normal" and not can_read_sensitive(
                context, caller, require_user_auth
            ):
                blocked.append({"memory_id": memory_id, "reason": "sensitive_requires_user_auth"})
                continue
            injectable = bool(memory.get("injectable", 1))
            content = str(memory.get("content") or "")
            if not injectable and caller in {"main_model", "router_small"}:
                # 不给主模型全文，避免执行其中的指令性文本。
                blocked.append(
                    {
                        "memory_id": memory_id,
                        "reason": "non_injectable_content",
                        "summary": truncate_text(str(memory.get("summary") or ""), 160),
                    }
                )
                continue
            content_tokens = estimate_tokens(content) + 24
            if fetch_used_tokens + content_tokens > fetch_token_budget:
                blocked.append({"memory_id": memory_id, "reason": "token_budget"})
                continue
            item = {
                "memory_id": memory_id,
                "memory_type": memory.get("memory_type"),
                "summary": truncate_text(str(memory.get("summary") or ""), 400),
                "content": truncate_text(escape_markup(content), 4000),
                "content_ref": f"memory://{memory_id}",
                "content_trust": "untrusted_history",
                "tags": tags_map.get(memory_id, []),
                "provenance": {
                    "source": memory.get("source"),
                    "source_ref": memory.get("source_ref"),
                    "confidence": round(float(memory.get("confidence") or 0.0), 4),
                    "created_at": memory.get("created_at"),
                    "updated_at": memory.get("updated_at"),
                },
                "scores": {
                    "relevance": 1.0,
                    "recency": 1.0,
                    "importance": round(float(memory.get("importance") or 0.0), 4),
                    "sensitivity": sensitivity,
                    "gate": 1.0,
                },
                "conflicts_with": [
                    {
                        "id": conflict.get("id"),
                        "type": conflict.get("conflict_type"),
                        "status": conflict.get("status"),
                    }
                    for conflict in conflicts_map.get(memory_id, [])[:5]
                ],
                "lifecycle": lifecycle,
                "access_stats": stats_map.get(memory_id).to_dict() if stats_map.get(memory_id) else {},
                "estimated_tokens": content_tokens,
            }
            fetch_used_tokens += content_tokens
            if lifecycle in {"archived", "compressed"}:
                item["warning"] = "归档/压缩记忆，可能过时；已降低置信度，请向用户确认"
            items.append(item)
            if len(items) >= max_items:
                break
        latency = int((time.monotonic() - started) * 1000)
        truncated = (
            len(ids) > len(items) + len(blocked)
            or any(item.get("reason") == "token_budget" for item in blocked)
        )
        response = {
            "items": items,
            "blocked": blocked,
            "truncated": truncated,
            "next_cursor": None,
            "explain": {
                "mode": "fetch",
                "requested": len(ids),
                "returned": len(items),
                "blocked": len(blocked),
                "latency_ms": latency,
            },
            "trace_id": trace_id,
        }
        self.rate_limiter.consume(context.session_id, tokens=fetch_used_tokens)
        # fetch 是主模型确实使用了某条记忆的强信号。
        entries = [
            {
                "memory_id": item["memory_id"],
                "caller": caller,
                "mode": "fetch",
                "event_type": "used_by_main_model",
                "session_id": context.session_id,
                "trace_id": trace_id,
                "meta": {"returned_chars": len(item.get("content", ""))},
            }
            for item in items
        ]
        if entries:
            await self.store.record_access_batch(entries)
        await self._mark_latest_trace_used(context.session_id, [item["memory_id"] for item in items])
        await self.store.record_trace(
            {
                "trace_id": trace_id,
                "session_id": context.session_id,
                "owner_scope": context.write_scope()[0],
                "owner_id": context.write_scope()[1],
                "query": mask_sensitive(query),
                "rewritten_query": "",
                "candidate_count": len(ids),
                "candidates": [{"memory_id": item["memory_id"], "mode": "fetch"} for item in items],
                "gate_decision": {"mode": "fetch", "blocked": blocked},
                "injected_text": mask_sensitive(
                    " ".join(item.get("summary", "") for item in items)
                )[:4000],
                "injected_ids": [item["memory_id"] for item in items],
                "used": 1,
                "used_ids": [item["memory_id"] for item in items],
                "latency_ms": latency,
                "created_at": now_iso(),
            }
        )
        return ToolResult(data=response, trace_id=trace_id, latency_ms=latency)

    # ------------------------------------------------------------------
    # probe
    # ------------------------------------------------------------------
    async def _probe(
        self,
        context: AccessContext,
        *,
        caller: str,
        query: str,
        filters_map: Mapping[str, Any],
        budget_map: Mapping[str, Any],
        options_map: Mapping[str, Any],
    ) -> ToolResult:
        started = time.monotonic()
        trace_id = new_trace_id()
        try:
            filters = SearchFilters.from_mapping(filters_map)
            validate_filter_scope(context, filters)
            filters = filter_sensitivity_for_caller(
                filters, caller, context, False
            )
        except PermissionDenied as exc:
            return ToolResult(
                data={"exists": False, "candidates": [], "suggested_queries": [],
                      "explain": {"error": "permission_denied", "detail": str(exc)},
                      "trace_id": trace_id},
                trace_id=trace_id,
            )
        rate = self.rate_limiter.check(context.session_id, estimated_tokens=320)
        if not rate.allowed and caller != "user":
            return ToolResult(
                data={
                    "exists": False,
                    "candidates": [],
                    "suggested_queries": [],
                    "explain": {"error": "rate_limited", "detail": rate.reason},
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
            )
        retrieval = await self.retriever.retrieve(
            context,
            query,
            filters,
            limit=8,
        )
        candidates = await self.reranker.rerank(
            retrieval.plan, retrieval.candidates, context, filters, event=None
        )
        decision = await self.gate.evaluate(
            candidates,
            context,
            filters,
            caller=caller,
            budget={"max_items": 3, "max_tokens": 320, **dict(budget_map)},
            options={"threshold": 0.3, "explicit_search": True, "require_user_auth": False},
        )
        selected = decision.selected[:3]
        latency = int((time.monotonic() - started) * 1000)
        response = {
            "exists": bool(selected),
            "candidates": [
                {
                    "memory_id": item.memory_id,
                    "summary": truncate_text(item.summary, 140),
                    "memory_type": item.candidate.memory.get("memory_type"),
                    "score": round(item.gate_score, 4),
                    "source": item.candidate.memory.get("source"),
                }
                for item in selected
            ],
            "suggested_queries": retrieval.plan.all_queries()[:3],
            "explain": {
                "considered": decision.considered,
                "threshold": decision.threshold,
                "latency_ms": latency,
                "blocked_sensitive": decision.blocked_sensitive,
            },
            "trace_id": trace_id,
        }
        self.metrics.observe_recall(
            latency_ms=latency,
            candidates=len(candidates),
            injected_items=0,
            injected_tokens=0,
            blocked_sensitive=decision.blocked_sensitive,
        )
        self.rate_limiter.consume(
            context.session_id,
            tokens=sum(item.estimated_tokens for item in selected),
        )
        await self.store.record_access_batch(
            [
                {
                    "memory_id": item.memory_id,
                    "caller": caller,
                    "mode": "probe",
                    "event_type": "candidate_recalled",
                    "session_id": context.session_id,
                    "trace_id": trace_id,
                    "meta": {"gate": round(item.gate_score, 4)},
                }
                for item in selected
            ]
        )
        await self.store.record_trace(
            {
                "trace_id": trace_id,
                "session_id": context.session_id,
                "owner_scope": context.write_scope()[0],
                "owner_id": context.write_scope()[1],
                "query": mask_sensitive(query),
                "rewritten_query": mask_sensitive(retrieval.plan.normalized),
                "candidate_count": len(retrieval.candidates),
                "candidates": [item.to_trace() for item in retrieval.candidates[:10]],
                "gate_decision": decision.to_trace(),
                "injected_text": "",
                "injected_ids": [],
                "used": 0,
                "latency_ms": latency,
                "created_at": now_iso(),
            }
        )
        return ToolResult(data=response, trace_id=trace_id, latency_ms=latency)

    # ------------------------------------------------------------------
    # 输出与 trace
    # ------------------------------------------------------------------
    def _item_payload(
        self,
        item: GateItem,
        options: Mapping[str, Any],
        *,
        include_content: bool,
    ) -> dict[str, Any]:
        memory = item.candidate.memory
        include_scores = bool(options.get("include_scores", True))
        include_provenance = bool(options.get("include_provenance", True))
        include_conflicts = bool(options.get("include_conflicts", True))
        return_summary = bool(options.get("return_summary", True))
        payload: dict[str, Any] = {
            "memory_id": item.memory_id,
            "content_ref": f"memory://{item.memory_id}",
            "memory_type": memory.get("memory_type"),
            "tags": item.candidate.tags[:20],
            "lifecycle": memory.get("lifecycle"),
            "access_stats": item.candidate.access_stats.to_dict(),
            "estimated_tokens": item.estimated_tokens,
        }
        if return_summary:
            payload["summary"] = truncate_text(item.summary, 400)
        if include_provenance:
            payload["provenance"] = {
                "source": memory.get("source"),
                "source_ref": memory.get("source_ref"),
                "confidence": round(float(memory.get("confidence") or 0.0), 4),
                "created_at": memory.get("created_at"),
                "updated_at": memory.get("updated_at"),
                "owner_scope": memory.get("owner_scope"),
            }
        if include_conflicts:
            payload["conflicts_with"] = [
                {
                    "id": conflict.get("id"),
                    "type": conflict.get("conflict_type"),
                    "status": conflict.get("status"),
                }
                for conflict in item.candidate.conflicts[:5]
            ]
        if include_scores:
            payload["scores"] = {
                "relevance": round(item.candidate.relevance, 4),
                "recency": round(item.components.get("recency", 0.0), 4),
                "importance": round(item.components.get("importance", 0.0), 4),
                "sensitivity": memory.get("sensitivity", "normal"),
                "gate": round(item.gate_score, 4),
                "components": {
                    key: round(value, 4) for key, value in item.components.items()
                },
            }
        if include_content:
            payload["content"] = truncate_text(
                escape_markup(str(memory.get("content") or "")), 4000
            )
        return payload

    @staticmethod
    def _dedup_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for item in items:
            summary = str(item.get("summary") or "")
            key = (summary.strip().lower() or str(item.get("memory_id") or ""))
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @staticmethod
    def _explain(
        plan: Any,
        decision: GateDecision,
        latency_ms: int,
        *,
        channel_counts: Mapping[str, int],
    ) -> dict[str, Any]:
        return {
            "query": plan.original,
            "rewritten": plan.normalized,
            "rewritten_by": plan.rewritten_by,
            "queries": plan.all_queries(),
            "channels": dict(channel_counts),
            "threshold": round(decision.threshold, 4),
            "candidate_count": decision.considered,
            "selected_count": len(decision.selected),
            "blocked_sensitive": decision.blocked_sensitive,
            "blocked_injection": decision.blocked_injection,
            "truncated": decision.truncated,
            "latency_ms": latency_ms,
            "score_formula": (
                "0.50*相关性 + 0.10*场景 + 0.10*任务 + 0.18*重要性/情绪 "
                "+ 0.12*来源置信；再乘时效/权限，减 token 成本与冲突惩罚"
            ),
        }

    async def _record_recall(
        self,
        context: AccessContext,
        *,
        trace_id: str,
        query: str,
        plan: str,
        candidates: list[RetrievedMemory],
        decision: GateDecision,
        caller: str,
        mode: str,
        latency: int,
        injected_ids: list[str],
        injected_text: str = "",
    ) -> None:
        entries: list[dict[str, Any]] = []
        for candidate in candidates:
            entries.append(
                {
                    "memory_id": candidate.memory_id,
                    "caller": caller,
                    "mode": mode,
                    "event_type": "candidate_recalled",
                    "session_id": context.session_id,
                    "trace_id": trace_id,
                    "meta": {"relevance": round(candidate.relevance, 4)},
                }
            )
        for memory_id in injected_ids:
            if not memory_id:
                continue
            entries.append(
                {
                    "memory_id": memory_id,
                    "caller": caller,
                    "mode": mode,
                    "event_type": "injected",
                    "session_id": context.session_id,
                    "trace_id": trace_id,
                    "meta": {},
                }
            )
        if entries:
            await self.store.record_access_batch(entries)
        await self.store.record_trace(
            {
                "trace_id": trace_id,
                "session_id": context.session_id,
                "owner_scope": context.write_scope()[0],
                "owner_id": context.write_scope()[1],
                "query": mask_sensitive(query),
                "rewritten_query": mask_sensitive(plan),
                "candidate_count": len(candidates),
                "candidates": [item.to_trace() for item in candidates[:30]],
                "gate_decision": decision.to_trace(),
                "injected_text": mask_sensitive(injected_text)[:8000],
                "injected_ids": injected_ids,
                "used": 0,
                "latency_ms": latency,
                "created_at": now_iso(),
            }
        )

    async def _mark_latest_trace_used(self, session_id: str, memory_ids: list[str]) -> None:
        if not memory_ids or not session_id:
            return
        try:
            traces = await self.store.get_recent_traces(session_id, limit=1)
            if traces:
                await self.store.mark_trace_used(
                    str(traces[0].get("trace_id") or ""), memory_ids
                )
        except Exception:  # noqa: BLE001
            return


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return minimum
