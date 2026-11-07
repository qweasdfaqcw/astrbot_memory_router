"""门控打分：相关性、场景/任务、时效、重要性、来源、权限、成本、冲突。

公式可解释，所有分量都写入 trace。访问次数不进入门控分，避免“被召回越多次
越容易被召回”的自我强化；访问只影响留存/压缩/归档。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from config.settings import MemorySettings
from governance.policy import can_read_sensitive
from retriever.retriever import RetrievedMemory
from store.models import HALF_LIFE_DAYS, AccessContext, SearchFilters
from utils.text import estimate_tokens
from utils.time_utils import age_days, half_life_decay, now_iso, parse_iso

_SOURCE_CONFIDENCE = {
    "user": 1.0,
    "system": 0.9,
    "assistant": 0.75,
    "third_party": 0.55,
    "unknown": 0.6,
}
_OPEN_CONFLICT_STATUSES = {"open", "disputed"}


@dataclass(slots=True)
class GateItem:
    candidate: RetrievedMemory
    gate_score: float
    components: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    estimated_tokens: int = 0
    summary: str = ""

    @property
    def memory_id(self) -> str:
        return self.candidate.memory_id

    @property
    def blocked(self) -> bool:
        return bool(self.blocked_reason)

    def to_trace(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "gate": round(self.gate_score, 4),
            "relevance": round(self.candidate.relevance, 4),
            "components": {key: round(value, 4) for key, value in self.components.items()},
            "blocked_reason": self.blocked_reason,
            "estimated_tokens": self.estimated_tokens,
            "reasons": self.reasons[:6],
        }


@dataclass(slots=True)
class GateDecision:
    items: list[GateItem] = field(default_factory=list)
    considered: int = 0
    truncated: bool = False
    threshold: float = 0.42
    max_items: int = 5
    max_tokens: int = 700
    used_tokens: int = 0
    blocked_sensitive: int = 0
    blocked_injection: int = 0
    blocked_third_party: int = 0
    excluded: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> list[GateItem]:
        return [item for item in self.items if not item.blocked]

    @property
    def empty(self) -> bool:
        return not self.selected

    def injected_ids(self) -> list[str]:
        return [item.memory_id for item in self.selected]

    def to_trace(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "max_items": self.max_items,
            "max_tokens": self.max_tokens,
            "used_tokens": self.used_tokens,
            "considered": self.considered,
            "selected": [item.to_trace() for item in self.selected],
            "excluded": self.excluded[:20],
            "blocked_sensitive": self.blocked_sensitive,
            "blocked_injection": self.blocked_injection,
            "blocked_third_party": self.blocked_third_party,
            "truncated": self.truncated,
        }


class Gate:
    def __init__(self, settings: MemorySettings):
        self.settings = settings

    async def evaluate(
        self,
        candidates: Iterable[RetrievedMemory],
        context: AccessContext,
        filters: SearchFilters,
        *,
        caller: str,
        budget: Mapping[str, Any] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> GateDecision:
        options = options or {}
        budget = budget or {}
        max_tokens = self._clamp_int(
            budget.get("max_tokens", self.settings.max_injected_tokens), 80, 4000
        )
        max_items = self._clamp_int(
            budget.get("max_items", self.settings.top_k), 1, 20
        )
        threshold = float(options.get("threshold", self.settings.gate_threshold))
        threshold = max(0.0, min(1.0, threshold))
        require_user_auth = bool(options.get("require_user_auth", False))
        explicit_search = bool(options.get("explicit_search", False))
        diversify = bool(options.get("diversify", True))
        if explicit_search:
            threshold = min(threshold, 0.25)
        max_latency = self._clamp_int(
            budget.get("max_latency_ms", self.settings.prefetch_max_latency_ms), 20, 10000
        )

        decision = GateDecision(
            threshold=threshold,
            max_items=max_items,
            max_tokens=max_tokens,
            budget={
                "max_tokens": max_tokens,
                "max_items": max_items,
                "max_latency_ms": max_latency,
            },
        )
        current = parse_iso(now_iso())
        scored: list[GateItem] = []
        for candidate in candidates:
            decision.considered += 1
            item = self._score_one(
                candidate,
                context,
                filters,
                caller=caller,
                require_user_auth=require_user_auth,
                current=current,
            )
            scored.append(item)
            if item.blocked:
                if "敏感" in item.blocked_reason:
                    decision.blocked_sensitive += 1
                elif "指令" in item.blocked_reason or "不可注入" in item.blocked_reason:
                    decision.blocked_injection += 1
                elif "第三方" in item.blocked_reason:
                    decision.blocked_third_party += 1
                decision.excluded.append(item.to_trace())
        scored.sort(key=lambda item: item.gate_score, reverse=True)

        selected: list[GateItem] = []
        per_type: dict[str, int] = {}
        # 预留不可执行分隔块、使用说明和每条元数据的固定开销。
        header_tokens = 90
        used_tokens = header_tokens
        for item in scored:
            if item.blocked:
                continue
            if len(selected) >= max_items:
                decision.truncated = True
                break
            memory_type = str(item.candidate.memory.get("memory_type") or "fact")
            # 多样性：同一类型最多 3 条，避免整块上下文被一类记忆占满。
            if diversify and per_type.get(memory_type, 0) >= 3 and len(selected) < max_items:
                decision.excluded.append(
                    {
                        "memory_id": item.memory_id,
                        "gate": round(item.gate_score, 4),
                        "blocked_reason": "diversity_type_limit",
                    }
                )
                decision.truncated = True
                continue
            if used_tokens + item.estimated_tokens > max_tokens:
                decision.truncated = True
                decision.excluded.append(
                    {
                        "memory_id": item.memory_id,
                        "gate": round(item.gate_score, 4),
                        "blocked_reason": "token_budget",
                    }
                )
                continue
            if item.gate_score < threshold:
                decision.excluded.append(
                    {
                        "memory_id": item.memory_id,
                        "gate": round(item.gate_score, 4),
                        "blocked_reason": "below_threshold",
                    }
                )
                decision.truncated = True
                continue
            selected.append(item)
            per_type[memory_type] = per_type.get(memory_type, 0) + 1
            used_tokens += item.estimated_tokens

        decision.items = selected
        decision.used_tokens = used_tokens
        return decision

    def _score_one(
        self,
        candidate: RetrievedMemory,
        context: AccessContext,
        filters: SearchFilters,
        *,
        caller: str,
        require_user_auth: bool,
        current: Any,
    ) -> GateItem:
        memory = candidate.memory
        summary = str(memory.get("summary") or memory.get("content") or "")[:400]
        estimated = estimate_tokens(summary) + 28
        components: dict[str, float] = {}
        reasons: list[str] = []

        sensitivity = str(memory.get("sensitivity") or "normal")
        permission = 1.0
        if sensitivity == "secret":
            permission = 0.0
            reasons.append("secret 级记忆禁止注入")
        elif sensitivity == "sensitive":
            if can_read_sensitive(context, caller, require_user_auth):
                permission = 1.0
                reasons.append("用户显式授权读取敏感记忆")
            else:
                permission = 0.0
                reasons.append("敏感记忆被拦截，需用户显式授权")
        components["permission"] = permission

        injectable = bool(memory.get("injectable", 1))
        if not injectable or caller in {"main_model", "router_small"}:
            if not injectable:
                reasons.append("记忆含指令性内容，标记为不可注入")
                components["injectable"] = 0.0
            else:
                components["injectable"] = 1.0
        else:
            components["injectable"] = 1.0

        source = str(memory.get("source") or "unknown")
        if source == "third_party" and not self.settings.inject_third_party:
            reasons.append("第三方陈述默认不注入，需用户确认")
            components["third_party"] = 0.0
        else:
            components["third_party"] = 1.0

        memory_type = str(memory.get("memory_type") or "fact")
        reference = (
            str(memory.get("last_verified_at") or "")
            or str(memory.get("updated_at") or "")
            or str(memory.get("created_at") or "")
        )
        age = age_days(reference, now=current)
        decay = half_life_decay(age, HALF_LIFE_DAYS.get(memory_type, 180.0))
        recency = 0.65 + 0.35 * decay
        components["recency"] = recency
        components["decay"] = decay

        importance = max(0.0, min(1.0, float(memory.get("importance") or 0.5)))
        arousal = max(0.0, min(1.0, float(memory.get("emotion_arousal") or 0.0)))
        valence = abs(max(-1.0, min(1.0, float(memory.get("emotion_valence") or 0.0))))
        salience = min(1.0, 0.78 * importance + 0.17 * arousal + 0.05 * valence)
        components["salience"] = salience
        components["importance"] = importance

        source_factor = _SOURCE_CONFIDENCE.get(source, 0.6)
        confidence = max(0.0, min(1.0, float(memory.get("confidence") or 0.5)))
        source_score = source_factor * (0.55 + 0.45 * confidence)
        components["source_confidence"] = source_score

        scene_score = self._context_match(
            filters.scene, str(memory.get("scene") or ""), candidate.tags, "scene:"
        )
        task_score = self._context_match(
            filters.task, str(memory.get("task") or ""), candidate.tags, "task:"
        )
        components["scene"] = scene_score
        components["task"] = task_score

        relevance = max(0.0, min(1.0, candidate.relevance))
        components["relevance"] = relevance
        base = (
            0.50 * relevance
            + 0.10 * scene_score
            + 0.10 * task_score
            + 0.18 * salience
            + 0.12 * source_score
        )
        cost_penalty = min(0.08, estimated * 0.0015)
        conflict_penalty = 0.0
        open_conflicts = [
            conflict for conflict in candidate.conflicts
            if str(conflict.get("status")) in _OPEN_CONFLICT_STATUSES
        ]
        if open_conflicts:
            conflict_penalty = min(0.12, 0.04 + 0.02 * len(open_conflicts))
            reasons.append(f"存在 {len(open_conflicts)} 条未解决冲突")
        archived_penalty = 0.0
        if str(memory.get("lifecycle")) in {"archived", "compressed"}:
            archived_penalty = 0.10
            reasons.append("归档/压缩记忆，可能过时")
        third_party_penalty = 0.05 if source == "third_party" else 0.0
        components["cost_penalty"] = cost_penalty
        components["conflict_penalty"] = conflict_penalty
        components["archived_penalty"] = archived_penalty

        permission_factor = permission * components.get("injectable", 1.0) * components.get("third_party", 1.0)
        gate_score = (
            base * recency * permission_factor
            - cost_penalty
            - conflict_penalty
            - archived_penalty
            - third_party_penalty
        )
        gate_score = max(0.0, min(1.0, gate_score))
        if relevance >= 0.5:
            reasons.append("查询相关性较高")
        if decay >= 0.7:
            reasons.append("时效性较好")
        if importance >= 0.7:
            reasons.append("重要/约束性内容")
        blocked_reason = ""
        if sensitivity in {"sensitive", "secret"} and permission == 0.0:
            blocked_reason = "敏感记忆被拦截"
        elif not injectable:
            blocked_reason = "含指令性内容，不可注入"
        elif source == "third_party" and not self.settings.inject_third_party:
            blocked_reason = "第三方陈述需确认"
        return GateItem(
            candidate=candidate,
            gate_score=gate_score,
            components=components,
            reasons=reasons,
            blocked_reason=blocked_reason,
            estimated_tokens=estimated,
            summary=summary,
        )

    @staticmethod
    def _context_match(
        filter_value: str, memory_value: str, tags: list[str], tag_prefix: str
    ) -> float:
        if filter_value:
            if memory_value and memory_value == filter_value:
                return 1.0
            if any(tag.lower() == (tag_prefix + filter_value).lower() for tag in tags):
                return 0.9
            return 0.45
        return 0.75

    @staticmethod
    def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
        try:
            return max(minimum, min(maximum, int(value)))
        except (TypeError, ValueError):
            return minimum
