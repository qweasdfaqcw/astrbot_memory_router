"""多路召回：FTS / LIKE / 实体 / 标签 / 可选向量。

召回阶段只负责“尽量不漏”，精排与门控交给 gate。所有候选最终必须再次经过
owner_scope 过滤，防止索引层漏掉租户条件。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable

from config.settings import MemorySettings
from retriever.query_rewrite import QueryPlan, QueryRewriter
from store.models import AccessContext, AccessStats, SearchFilters
from store.sqlite_store import SQLiteStore
from store.vector_backend import VectorBackend
from utils.text import jaccard, tokenize


@dataclass(slots=True)
class RetrievedMemory:
    memory: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    access_stats: AccessStats = field(default_factory=AccessStats)
    relevance: float = 0.0
    lexical_score: float = 0.0
    fts_score: float = 0.0
    like_score: float = 0.0
    vector_score: float = 0.0
    tag_score: float = 0.0
    entity_score: float = 0.0
    channel_scores: dict[str, float] = field(default_factory=dict)
    explain: list[str] = field(default_factory=list)

    @property
    def memory_id(self) -> str:
        return str(self.memory.get("id") or "")

    def to_trace(self) -> dict[str, Any]:
        """召回 trace 只保存 ID、分数和原因，不保存全文。"""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory.get("memory_type"),
            "relevance": round(self.relevance, 4),
            "channels": {key: round(value, 4) for key, value in self.channel_scores.items()},
            "lifecycle": self.memory.get("lifecycle"),
            "sensitivity": self.memory.get("sensitivity"),
        }


@dataclass(slots=True)
class RetrievalResult:
    plan: QueryPlan
    candidates: list[RetrievedMemory] = field(default_factory=list)
    channel_counts: dict[str, int] = field(default_factory=dict)
    total: int = 0

    @property
    def empty(self) -> bool:
        return not self.candidates


class MemoryRetriever:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        vector_backend: VectorBackend,
        rewriter: QueryRewriter,
    ):
        self.store = store
        self.settings = settings
        self.vector_backend = vector_backend
        self.rewriter = rewriter

    async def retrieve(
        self,
        context: AccessContext,
        query: str,
        filters: SearchFilters | None = None,
        *,
        limit: int | None = None,
        recent_turns: Iterable[Any] = (),
        event: Any = None,
        include_inactive: bool = False,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        max_candidates = max(5, min(100, int(limit or self.settings.candidate_limit)))
        fetch_limit = min(200, max_candidates + max(0, filters.offset))
        plan = await self.rewriter.plan(
            query, context, recent_turns=recent_turns, event=event
        )
        channel_scores: dict[str, dict[str, float]] = {
            "fts": {}, "like": {}, "entity": {}, "vector": {},
        }

        queries = plan.all_queries()
        tasks: list[tuple[str, Any]] = []
        for item in queries[:4]:
            if item:
                tasks.append(
                    ("fts", self._collect_fts(item, context, filters, fetch_limit))
                )
                tasks.append(
                    ("like", self._collect_like(item, context, filters, fetch_limit))
                )
        if plan.entities:
            tasks.append(
                ("entity", self._collect_entities(
                    plan.entities, context, filters, fetch_limit
                ))
            )
        if self.vector_backend.enabled and plan.original:
            tasks.append(
                ("vector", self._collect_vector(
                    plan.original, context, filters, fetch_limit
                ))
            )
        if not tasks:
            # 无查询时按过滤条件列出最近记忆，作为预取候选。
            # 这里重置 offset，稍后统一在候选排序后切片，避免双重偏移。
            fallback_filters = SearchFilters.from_mapping(filters.to_public_dict())
            fallback_filters.offset = 0
            fallback = await self.store.list_memories(
                context,
                fallback_filters,
                limit=fetch_limit,
                include_inactive=include_inactive,
            )
            for item in fallback:
                channel_scores["like"].setdefault(str(item["id"]), 0.3)
        else:
            results = await asyncio.gather(
                *(coro for _, coro in tasks), return_exceptions=True
            )
            for (name, _), result in zip(tasks, results):
                if isinstance(result, Exception) or not result:
                    continue
                for memory_id, score in result:
                    current = channel_scores[name].get(memory_id, 0.0)
                    if score > current:
                        channel_scores[name][memory_id] = score

        ids = set()
        for values in channel_scores.values():
            ids.update(values)
        if not ids:
            return RetrievalResult(plan=plan, total=0, channel_counts={})

        memories = await self.store.get_memories_by_ids(
            list(ids),
            context,
            filters,
            include_inactive=include_inactive,
        )
        memory_ids = [str(item["id"]) for item in memories]
        tags_map, conflicts_map, stats_map = await asyncio.gather(
            self.store.get_tags_map(memory_ids),
            self.store.get_conflicts_for(memory_ids),
            self.store.get_access_stats(memory_ids),
        )

        candidates: list[RetrievedMemory] = []
        query_tokens = set(tokenize(plan.normalized + " " + " ".join(plan.keywords)))
        for memory in memories:
            memory_id = str(memory["id"])
            tags = tags_map.get(memory_id, [])
            text = f"{memory.get('summary', '')} {memory.get('content', '')}"
            memory_tokens = set(tokenize(text))
            lexical = jaccard(query_tokens, memory_tokens) if query_tokens else 0.0
            tag_hits = sum(
                1 for keyword in plan.keywords
                if any(keyword and keyword.lower() in tag.lower() for tag in tags)
            )
            tag_score = min(1.0, tag_hits / max(1, min(len(plan.keywords), 5)))
            entity_hits = sum(
                1 for entity in plan.entities
                if any(entity.lower() in tag.lower() for tag in tags)
            )
            entity_score = channel_scores["entity"].get(memory_id, 0.0)
            entity_score = max(entity_score, min(1.0, entity_hits / max(1, min(len(plan.entities), 4))))
            fts_score = channel_scores["fts"].get(memory_id, 0.0)
            like_score = channel_scores["like"].get(memory_id, 0.0)
            vector_score = channel_scores["vector"].get(memory_id, 0.0)

            parts = [
                (lexical, 0.30),
                (tag_score, 0.18),
                (entity_score, 0.12),
                (fts_score, 0.28),
                (like_score, 0.08),
                (vector_score, self.settings.vector_weight),
            ]
            present = [(value, weight) for value, weight in parts if value > 0]
            relevance = (
                sum(value * weight for value, weight in present)
                / sum(weight for _, weight in present)
                if present else 0.25
            )
            # 召回渠道越多，略增可信度，但不超过 0.08。
            channels_present = sum(1 for name in channel_scores if channel_scores[name].get(memory_id))
            relevance = min(1.0, relevance + min(0.08, channels_present * 0.02))
            explain: list[str] = []
            if fts_score:
                explain.append(f"FTS={fts_score:.2f}")
            if like_score:
                explain.append(f"LIKE={like_score:.2f}")
            if vector_score:
                explain.append(f"向量={vector_score:.2f}")
            if tag_score:
                explain.append(f"标签={tag_score:.2f}")
            if entity_score:
                explain.append(f"实体={entity_score:.2f}")
            if lexical:
                explain.append(f"词面={lexical:.2f}")
            candidates.append(
                RetrievedMemory(
                    memory=memory,
                    tags=tags,
                    conflicts=conflicts_map.get(memory_id, []),
                    access_stats=stats_map.get(memory_id, AccessStats()),
                    relevance=max(0.0, min(1.0, relevance)),
                    lexical_score=lexical,
                    fts_score=fts_score,
                    like_score=like_score,
                    vector_score=vector_score,
                    tag_score=tag_score,
                    entity_score=entity_score,
                    channel_scores={
                        "fts": fts_score,
                        "like": like_score,
                        "vector": vector_score,
                        "tag": tag_score,
                        "entity": entity_score,
                        "lexical": lexical,
                    },
                    explain=explain,
                )
            )
        candidates.sort(key=lambda item: item.relevance, reverse=True)
        total = len(candidates)
        offset = max(0, filters.offset)
        return RetrievalResult(
            plan=plan,
            candidates=candidates[offset:offset + max_candidates],
            channel_counts={name: len(values) for name, values in channel_scores.items()},
            total=total,
        )

    async def _collect_fts(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        raw = await self.store.search_fts(query, context, filters, limit=limit)
        # bm25 越小越相关，转成 0~1；避免负数导致分数异常。
        return [(memory_id, 1.0 / (1.0 + abs(score))) for memory_id, score in raw]

    async def _collect_like(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        return await self.store.search_like(query, context, filters, limit=limit)

    async def _collect_entities(
        self,
        entities: Iterable[str],
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        return await self.store.search_by_entities(
            entities, context, filters, limit=limit
        )

    async def _collect_vector(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        return await self.vector_backend.search(query, context, filters, limit)
