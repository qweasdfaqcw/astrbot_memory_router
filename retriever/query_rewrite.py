"""查询改写与查询计划。

默认走确定性规则，避免每次召回都额外请求小模型；开启
``query_rewrite_with_llm`` 后才会调用小模型，并保留规则兜底。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from adapter.llm_adapter import LLMAdapter, LLMUnavailable
from prompts import render_prompt
from store.models import AccessContext, unique_strings
from utils.json_utils import as_list
from utils.text import normalize_text, top_terms, truncate_text

_PRONOUN_RE = re.compile(
    r"你(还)?记得|你还记得吗|记得吗|上次|之前|以前|先前|那个|那件事|那件事儿|"
    r"我说过|我们说好|我们说过|老样子|照旧|他|她|它"
)
_QUOTE_RE = re.compile(r"[“\"「『](.*?)[”\"」』]")
_SYNONYMS = {
    "住处": ["住址", "住在", "地址"],
    "家": ["住址", "住在"],
    "忌口": ["过敏", "不能吃", "饮食禁忌"],
    "生日": ["出生", "生日"],
    "工作": ["职业", "上班", "公司"],
    "对象": ["男朋友", "女朋友", "伴侣"],
    "宠物": ["猫", "狗", "养的"],
}


@dataclass(slots=True)
class QueryPlan:
    original: str
    normalized: str
    queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    rewritten_by: str = "rules"

    def all_queries(self) -> list[str]:
        return unique_strings([self.normalized, *self.queries], limit=5)


def _recent_text(turns: Iterable[Any], limit: int = 4) -> str:
    lines: list[str] = []
    for turn in list(turns)[-limit:]:
        content = turn.get("content") if isinstance(turn, dict) else getattr(turn, "content", "")
        if content:
            lines.append(str(content))
    return "\n".join(lines)[-2000:]


def heuristic_plan(query: str, recent_turns: Iterable[Any]) -> QueryPlan:
    original = normalize_text(query)
    if not original:
        recent = _recent_text(recent_turns, 3)
        original = top_terms(recent, 4)[0] if top_terms(recent, 4) else ""
    normalized = _PRONOUN_RE.sub(" ", original)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ，。,.!?！？")
    queries: list[str] = []
    quotes = [item.strip() for item in _QUOTE_RE.findall(original) if item.strip()]
    queries.extend(quotes[:2])
    keywords = top_terms(normalized or original, limit=8)
    if keywords:
        queries.append(" ".join(keywords[:6]))
    for term, replacements in _SYNONYMS.items():
        if term in original:
            queries.append(" ".join(replacements))
    if not normalized:
        normalized = original
    entities = [item for item in keywords if len(item) >= 2][:8]
    return QueryPlan(
        original=original,
        normalized=normalized or original,
        queries=unique_strings(queries, limit=4),
        entities=entities,
        keywords=keywords,
        rewritten_by="rules",
    )


class QueryRewriter:
    def __init__(self, llm: LLMAdapter | None = None, *, use_llm: bool = False):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    async def plan(
        self,
        query: str,
        context: AccessContext,
        *,
        recent_turns: Iterable[Any] = (),
        event: Any = None,
    ) -> QueryPlan:
        fallback = heuristic_plan(query, recent_turns)
        if not self.use_llm or not fallback.original:
            return fallback
        assert self.llm is not None
        prompt = render_prompt(
            "query_rewrite.txt",
            query=fallback.original,
            context=_recent_text(recent_turns, 6),
        )
        try:
            raw = await self.llm.generate_json(prompt, event=event, timeout=12.0)
        except LLMUnavailable:
            return fallback
        queries = [
            truncate_text(str(item), 200)
            for item in as_list(raw.get("queries"))
            if str(item).strip()
        ]
        entities = [
            truncate_text(str(item), 60)
            for item in as_list(raw.get("entities"))
            if str(item).strip()
        ]
        keywords = [
            truncate_text(str(item), 60)
            for item in as_list(raw.get("keywords"))
            if str(item).strip()
        ]
        if not queries:
            return fallback
        return QueryPlan(
            original=fallback.original,
            normalized=queries[0],
            queries=unique_strings(queries[1:], limit=4),
            entities=unique_strings([*entities, *fallback.entities], limit=10),
            keywords=unique_strings([*keywords, *fallback.keywords], limit=10),
            rewritten_by="llm",
        )
