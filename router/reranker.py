"""候选精排接口：小模型重排 + 可选 cross-encoder 适配点。

小模型重排只给“相关性/场景/任务”打分，不能裁决事实真伪。任何失败都保留
启发式排序，不影响主对话。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Sequence

from adapter.llm_adapter import LLMAdapter, LLMUnavailable
from config.settings import MemorySettings
from prompts import render_prompt
from retriever.query_rewrite import QueryPlan
from retriever.retriever import RetrievedMemory
from store.models import AccessContext, SearchFilters
from utils.json_utils import as_list, coerce_float


class Reranker(ABC):
    enabled = False

    @abstractmethod
    async def rerank(
        self,
        query: QueryPlan,
        candidates: list[RetrievedMemory],
        context: AccessContext,
        filters: SearchFilters,
        *,
        event: Any = None,
    ) -> list[RetrievedMemory]:
        raise NotImplementedError


class NoopReranker(Reranker):
    enabled = False

    async def rerank(
        self,
        query: QueryPlan,
        candidates: list[RetrievedMemory],
        context: AccessContext,
        filters: SearchFilters,
        *,
        event: Any = None,
    ) -> list[RetrievedMemory]:
        return candidates


class LLMReranker(Reranker):
    enabled = True

    def __init__(self, llm: LLMAdapter, settings: MemorySettings):
        self.llm = llm
        self.settings = settings

    async def rerank(
        self,
        query: QueryPlan,
        candidates: list[RetrievedMemory],
        context: AccessContext,
        filters: SearchFilters,
        *,
        event: Any = None,
    ) -> list[RetrievedMemory]:
        if not candidates:
            return candidates
        subset = candidates[: self.settings.rerank_candidate_limit]
        payload = [
            {
                "id": item.memory_id,
                "type": item.memory.get("memory_type"),
                "summary": str(item.memory.get("summary") or "")[:180],
                "tags": item.tags[:6],
            }
            for item in subset
        ]
        prompt = render_prompt(
            "routing.txt",
            query=query.normalized or query.original,
            scene=filters.scene,
            task=filters.task,
            candidates=json.dumps(payload, ensure_ascii=False),
        )
        try:
            raw = await self.llm.generate_json(prompt, event=event, timeout=12.0)
        except LLMUnavailable:
            return candidates
        scores = {
            str(entry.get("id")): entry
            for entry in as_list(raw.get("scores"))
            if isinstance(entry, dict) and entry.get("id")
        }
        for item in subset:
            entry = scores.get(item.memory_id)
            if not entry:
                continue
            llm_relevance = coerce_float(entry.get("relevance"), item.relevance, 0.0, 1.0)
            scene_match = coerce_float(entry.get("scene_match"), 0.5, 0.0, 1.0)
            task_match = coerce_float(entry.get("task_match"), 0.5, 0.0, 1.0)
            item.relevance = max(
                0.0, min(1.0, 0.68 * item.relevance + 0.32 * llm_relevance)
            )
            item.channel_scores["llm"] = llm_relevance
            item.channel_scores["llm_scene"] = scene_match
            item.channel_scores["llm_task"] = task_match
            item.explain.append(f"小模型重排={llm_relevance:.2f}")
        subset.sort(key=lambda item: item.relevance, reverse=True)
        return subset + candidates[self.settings.rerank_candidate_limit:]


class CrossEncoderReranker(ABC):
    """真实 cross-encoder 的接入点。

    实现方可以加载本地 bge-reranker 或调用异步 HTTP 服务；默认不启用。
    """

    enabled = False

    @abstractmethod
    async def score_pairs(
        self, query: str, documents: Sequence[str]
    ) -> list[float]:
        raise NotImplementedError


class NoopCrossEncoder(CrossEncoderReranker):
    enabled = False

    async def score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        return [0.0 for _ in documents]


class CrossEncoderRerankerAdapter(Reranker):
    """把 cross-encoder 分数融合进召回相关性的适配器。"""

    enabled = True

    def __init__(self, encoder: CrossEncoderReranker):
        self.encoder = encoder

    async def rerank(
        self,
        query: QueryPlan,
        candidates: list[RetrievedMemory],
        context: AccessContext,
        filters: SearchFilters,
        *,
        event: Any = None,
    ) -> list[RetrievedMemory]:
        if not candidates or not self.encoder.enabled:
            return candidates
        subset = candidates[:20]
        documents = [
            f"{item.memory.get('summary', '')} {item.memory.get('content', '')}"[:1200]
            for item in subset
        ]
        scores = await self.encoder.score_pairs(query.normalized or query.original, documents)
        for item, score in zip(subset, scores):
            item.relevance = max(0.0, min(1.0, 0.5 * item.relevance + 0.5 * float(score)))
            item.explain.append(f"cross-encoder={float(score):.2f}")
        subset.sort(key=lambda item: item.relevance, reverse=True)
        return subset + candidates[20:]
