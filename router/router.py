"""召回路由：预取 → 精排 → 门控 → 注入 → 访问统计/trace。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from astrbot.api import logger

from config.settings import MemorySettings
from gate.gate import Gate, GateDecision
from governance.rate_limit import RateLimiter
from injector.injector import InjectionPayload, MemoryInjector
from retriever.query_rewrite import QueryPlan
from retriever.retriever import MemoryRetriever, RetrievalResult
from router.reranker import Reranker
from stats.metrics import MetricsCollector
from store.models import AccessContext, SearchFilters
from store.sqlite_store import SQLiteStore
from utils.ids import new_trace_id
from utils.time_utils import now_iso
from writer.sanitizer import mask_sensitive


@dataclass(slots=True)
class RouterResult:
    trace_id: str
    query: str
    plan: QueryPlan | None = None
    retrieval: RetrievalResult | None = None
    gate: GateDecision | None = None
    injection: InjectionPayload = field(default_factory=InjectionPayload)
    latency_ms: int = 0
    rate_limited: bool = False
    error: str = ""

    @property
    def empty(self) -> bool:
        return self.injection.empty


class MemoryRouter:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        retriever: MemoryRetriever,
        reranker: Reranker,
        gate: Gate,
        injector: MemoryInjector,
        metrics: MetricsCollector,
        rate_limiter: RateLimiter,
    ):
        self.store = store
        self.settings = settings
        self.retriever = retriever
        self.reranker = reranker
        self.gate = gate
        self.injector = injector
        self.metrics = metrics
        self.rate_limiter = rate_limiter

    async def prefetch(
        self,
        context: AccessContext,
        query: str,
        *,
        recent_turns: Iterable[Any] = (),
        event: Any = None,
        session_state: Any = None,
        budget: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
        caller: str = "router_small",
        record_trace: bool = True,
    ) -> RouterResult:
        trace_id = new_trace_id()
        started = time.monotonic()
        if not self.settings.retrieval_enabled:
            return RouterResult(trace_id=trace_id, query=query, error="retrieval_disabled")
        rate = self.rate_limiter.check(
            context.session_id, estimated_tokens=self.settings.max_injected_tokens
        )
        if not rate.allowed:
            return RouterResult(
                trace_id=trace_id,
                query=query,
                rate_limited=True,
                error=rate.reason,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        timeout_s = max(0.05, self.settings.prefetch_max_latency_ms / 1000.0)
        try:
            result = await asyncio.wait_for(
                self._run(
                    context,
                    query,
                    trace_id=trace_id,
                    recent_turns=recent_turns,
                    event=event,
                    budget=budget,
                    options=options,
                    caller=caller,
                    session_state=session_state,
                ),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            self.metrics.observe_error()
            latency = int((time.monotonic() - started) * 1000)
            logger.warning("记忆预取超过 %sms，已跳过本次注入", self.settings.prefetch_max_latency_ms)
            return RouterResult(
                trace_id=trace_id,
                query=query,
                latency_ms=latency,
                error="prefetch_timeout",
            )
        except Exception as exc:  # noqa: BLE001 - 召回失败不能阻塞对话
            self.metrics.observe_error()
            logger.exception("记忆预取失败")
            return RouterResult(
                trace_id=trace_id,
                query=query,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"prefetch_failed:{exc}",
            )

        result.latency_ms = int((time.monotonic() - started) * 1000)
        if result.retrieval is not None and result.gate is not None:
            self.metrics.observe_recall(
                latency_ms=result.latency_ms,
                candidates=len(result.retrieval.candidates),
                injected_items=result.injection.item_count,
                injected_tokens=result.injection.estimated_tokens,
                blocked_sensitive=result.gate.blocked_sensitive,
                blocked_injection=result.gate.blocked_injection,
            )
        if session_state is not None and not result.injection.empty:
            session_state.mark_injected(
                result.injection.memory_ids, query, trace_id
            )
        if result.injection.empty:
            self.rate_limiter.consume(context.session_id, tokens=0)
        else:
            self.rate_limiter.consume(
                context.session_id, tokens=result.injection.estimated_tokens
            )
        if record_trace:
            await self._record(result, context, trace_id=trace_id, caller=caller)
        return result

    async def _run(
        self,
        context: AccessContext,
        query: str,
        *,
        trace_id: str,
        recent_turns: Iterable[Any],
        event: Any,
        budget: Mapping[str, Any] | None,
        options: Mapping[str, Any] | None,
        caller: str,
        session_state: Any = None,
    ) -> RouterResult:
        filters = SearchFilters()
        retrieval = await self.retriever.retrieve(
            context,
            query,
            filters,
            recent_turns=recent_turns,
            event=event,
            limit=self.settings.candidate_limit,
        )
        candidates = await self.reranker.rerank(
            retrieval.plan, retrieval.candidates, context, filters, event=event
        )
        options_map = dict(options or {})
        options_map.setdefault("require_user_auth", False)
        decision = await self.gate.evaluate(
            candidates,
            context,
            filters,
            caller=caller,
            budget=budget,
            options=options_map,
        )
        # 同一会话短时间内已经注入过的记忆不重复占用上下文。
        if session_state is not None:
            kept = [
                item for item in decision.items
                if not session_state.was_recently_injected(item.memory_id)
            ]
            if len(kept) != len(decision.items):
                decision.items = kept
                decision.used_tokens = sum(item.estimated_tokens for item in kept)
                decision.truncated = True
        injection = (
            self.injector.build(decision, trace_id=trace_id, context=context)
            if self.settings.injection_enabled else InjectionPayload()
        )
        return RouterResult(
            trace_id=trace_id,
            query=query,
            plan=retrieval.plan,
            retrieval=retrieval,
            gate=decision,
            injection=injection,
        )

    async def _record(
        self,
        result: RouterResult,
        context: AccessContext,
        *,
        trace_id: str,
        caller: str,
    ) -> None:
        if result.retrieval is None or result.gate is None:
            return
        entries: list[dict[str, Any]] = []
        for candidate in result.retrieval.candidates:
            entries.append(
                {
                    "memory_id": candidate.memory_id,
                    "caller": caller,
                    "mode": "prefetch",
                    "event_type": "candidate_recalled",
                    "session_id": context.session_id,
                    "trace_id": trace_id,
                    "meta": {"relevance": round(candidate.relevance, 4)},
                }
            )
        for item in result.gate.selected:
            entries.append(
                {
                    "memory_id": item.memory_id,
                    "caller": caller,
                    "mode": "prefetch",
                    "event_type": "injected",
                    "session_id": context.session_id,
                    "trace_id": trace_id,
                    "meta": {
                        "gate": round(item.gate_score, 4),
                        "tokens": item.estimated_tokens,
                    },
                }
            )
        try:
            if entries:
                await self.store.record_access_batch(entries)
            await self.store.record_trace(
                {
                    "trace_id": trace_id,
                    "session_id": context.session_id,
                    "owner_scope": context.write_scope()[0],
                    "owner_id": context.write_scope()[1],
                    "query": mask_sensitive(result.query),
                    "rewritten_query": mask_sensitive(
                        result.plan.normalized if result.plan else ""
                    ),
                    "candidate_count": len(result.retrieval.candidates),
                    "candidates": [
                        candidate.to_trace()
                        for candidate in result.retrieval.candidates[:30]
                    ],
                    "gate_decision": result.gate.to_trace(),
                    "injected_text": mask_sensitive(result.injection.text)[:8000],
                    "injected_ids": result.injection.memory_ids,
                    "used": 0,
                    "created_at": now_iso(),
                    "latency_ms": result.latency_ms,
                }
            )
        except Exception:  # noqa: BLE001 - 观测失败不影响回复
            logger.warning("记忆 trace 写入失败", exc_info=True)
