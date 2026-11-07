"""记忆抽取：小模型优先，规则兜底。

小模型必须返回严格 JSON；任何失败都不会阻塞主对话，而是返回规则抽取结果，
并通过 ``degraded`` 标记让调用方记录。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from adapter.llm_adapter import LLMAdapter, LLMUnavailable
from config.settings import MemorySettings
from prompts import render_prompt
from store.models import (
    DEFAULT_IMPORTANCE,
    MEMORY_TYPES,
    AccessContext,
    unique_strings,
)
from utils.json_utils import (
    as_dict,
    as_list,
    coerce_bool,
    coerce_float,
    loads_lenient,
)
from utils.text import jaccard, normalize_text, top_terms, truncate_text
from utils.time_utils import now_iso
from writer.sanitizer import (
    clamp_sensitivity,
    detect_sensitivity,
    detect_third_party,
    parse_memory_directive,
)

_TYPE_ALIASES = {
    "事实": "fact",
    "偏好": "preference",
    "喜好": "preference",
    "事件": "event",
    "任务": "task_state",
    "任务状态": "task_state",
    "关系": "relationship",
    "约束": "constraint",
    "禁忌": "constraint",
}

_PREFERENCE_RE = re.compile(
    r"我(?:很|特别|比较|更)?(?:喜欢|偏爱|讨厌|不喜欢|习惯|爱)|"
    r"请(?:以后|下次|始终|一直)?(?:用|按|以).{0,16}(?:回复|回答|格式|风格)|"
    r"(?:不要|别)(?:再)?用.{0,16}(?:回复|回答|格式)|"
    r"我(?:更)?希望(?:你)?"
)
_FACT_RE = re.compile(
    r"我叫|我是|我的(?:名字|生日|职业|工作|住址|家乡|学校|年龄|血型|星座)|"
    r"我住在|我在.{0,12}(?:工作|上学|读书)|我养了|我有(?:一个|一只|两|三)|"
    r"我今年|我来自|我的.{0,8}是"
)
_CONSTRAINT_RE = re.compile(
    r"过敏|忌口|不能吃|不能碰|禁止|绝对不要|务必|必须|"
    r"(?:不要|别)(?:再)?(?:提|说|问)|对.{1,16}过敏|禁忌"
)
_RELATIONSHIP_RE = re.compile(
    r"我(?:的)?(?:爸爸|妈妈|父亲|母亲|老婆|老公|妻子|丈夫|孩子|儿子|女儿|"
    r"男朋友|女朋友|朋友|同事|同学|室友|宠物|猫|狗|哥哥|姐姐|弟弟|妹妹)"
)
_TASK_RE = re.compile(
    r"正在|还没|待办|计划|明天|后天|下周|下个月|任务|项目|提醒我|截止|deadline|"
    r"需要完成|准备做"
)
_EVENT_RE = re.compile(
    r"今天|昨天|前天|上次|刚刚|刚才|约好|答应|去了|见了|完成|发生|经历|"
    r"上周|去年|生日会|旅行|面试"
)
_ENTITY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_\-]{1,24}\b|[\u4e00-\u9fff]{2,10}(?:公司|大学|学校|医院|城市|app|App|软件|项目)"
)
_QUESTION_RE = re.compile(r"[?？]\s*$|^(?:吗|呢|什么|为什么|怎么|如何|是否)")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


@dataclass(slots=True)
class CandidateMemory:
    content: str
    summary: str
    memory_type: str
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    event_type: str = ""
    scene: str = ""
    task: str = ""
    confidence: float = 0.65
    importance: float = 0.5
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    source_speaker: str = "user"
    sensitivity_hint: str = "normal"
    contains_instruction: bool = False
    reason: str = ""
    should_remember: bool = True
    # 写通路回填：便于报告精确记录创建/合并结果。
    last_id: str = ""
    conflict_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExtractionBatch:
    items: list[CandidateMemory] = field(default_factory=list)
    degraded: bool = False
    error: str = ""
    provider: str = ""

    def __bool__(self) -> bool:
        return bool(self.items)


def _turn_text(turns: Iterable[Any]) -> str:
    lines: list[str] = []
    for turn in turns:
        if isinstance(turn, Mapping):
            role = str(turn.get("role") or "user")
            content = str(turn.get("content") or "")
        else:
            role = str(getattr(turn, "role", "user") or "user")
            content = str(getattr(turn, "content", "") or "")
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _normalize_type(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = _TYPE_ALIASES.get(raw, raw)
    return raw if raw in MEMORY_TYPES else "fact"


def _classify(text: str) -> str:
    if _CONSTRAINT_RE.search(text):
        return "constraint"
    if _PREFERENCE_RE.search(text):
        return "preference"
    if _RELATIONSHIP_RE.search(text):
        return "relationship"
    if _TASK_RE.search(text):
        return "task_state"
    if _FACT_RE.search(text):
        return "fact"
    if _EVENT_RE.search(text):
        return "event"
    return ""


def _extract_entities(text: str) -> list[str]:
    values = [item.strip() for item in _ENTITY_RE.findall(text) if item.strip()]
    return unique_strings(values, limit=8)


def _summary_for(text: str, memory_type: str) -> str:
    if memory_type in {"preference", "constraint"}:
        return truncate_text(text, 90)
    if memory_type == "fact":
        return truncate_text(text, 80)
    return truncate_text(text, 100)


def _validate_candidate(raw: Mapping[str, Any], settings: MemorySettings) -> CandidateMemory | None:
    if not coerce_bool(raw.get("should_remember", True), True):
        return None
    content = normalize_text(str(raw.get("content") or ""))
    if len(content) < 2:
        return None
    content = truncate_text(content, 4000)
    memory_type = _normalize_type(raw.get("memory_type"))
    summary = normalize_text(str(raw.get("summary") or "")) or _summary_for(content, memory_type)
    summary = truncate_text(summary, 180)
    sensitivity = detect_sensitivity(content)
    hint = str(raw.get("sensitivity_hint") or "normal")
    level = clamp_sensitivity(sensitivity.level, hint)
    if coerce_bool(raw.get("contains_instruction"), False):
        level = "secret" if level == "secret" else "sensitive"
    source_speaker = str(raw.get("source_speaker") or "user").lower()
    if source_speaker not in {"user", "third_party", "assistant", "unknown"}:
        source_speaker = "unknown"
    confidence = coerce_float(raw.get("confidence"), 0.65, 0.0, 1.0)
    if source_speaker in {"third_party", "unknown"}:
        confidence = min(confidence, 0.5)
    importance = coerce_float(
        raw.get("importance"), DEFAULT_IMPORTANCE.get(memory_type, 0.5), 0.0, 1.0
    )
    return CandidateMemory(
        content=content,
        summary=summary,
        memory_type=memory_type,
        tags=unique_strings(str(item) for item in as_list(raw.get("tags")))[:20],
        entities=unique_strings(str(item) for item in as_list(raw.get("entities")))[:12],
        event_type=truncate_text(str(raw.get("event_type") or ""), 40),
        scene=truncate_text(str(raw.get("scene") or ""), 60),
        task=truncate_text(str(raw.get("task") or ""), 60),
        confidence=confidence,
        importance=importance,
        emotion_valence=coerce_float(raw.get("emotion_valence"), 0.0, -1.0, 1.0),
        emotion_arousal=coerce_float(raw.get("emotion_arousal"), 0.0, 0.0, 1.0),
        source_speaker=source_speaker,
        sensitivity_hint=level,
        contains_instruction=coerce_bool(raw.get("contains_instruction"), False),
        reason=truncate_text(str(raw.get("reason") or ""), 120),
    )


class BaseExtractor:
    async def extract(
        self,
        turns: Iterable[Any],
        context: AccessContext,
        *,
        event: Any = None,
        explicit: bool = False,
    ) -> ExtractionBatch:
        raise NotImplementedError


class RuleExtractor(BaseExtractor):
    """无模型时的保守规则抽取。"""

    def __init__(self, settings: MemorySettings):
        self.settings = settings

    async def extract(
        self,
        turns: Iterable[Any],
        context: AccessContext,
        *,
        event: Any = None,
        explicit: bool = False,
    ) -> ExtractionBatch:
        candidates: list[CandidateMemory] = []
        for turn in turns:
            if isinstance(turn, Mapping):
                role = str(turn.get("role") or "user")
                text = str(turn.get("content") or "")
            else:
                role = str(getattr(turn, "role", "user") or "user")
                text = str(getattr(turn, "content", "") or "")
            if role not in {"user", "assistant"} or not text:
                continue
            directive = parse_memory_directive(text)
            if directive.action == "suppress":
                continue
            if directive.action == "remember":
                source_text = directive.content
                forced = True
            else:
                source_text = text
                forced = explicit and role == "user"
            if _QUESTION_RE.search(source_text) and not forced:
                continue
            pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(source_text) if piece.strip()]
            if not pieces:
                continue
            for piece in pieces[:4]:
                memory_type = _classify(piece)
                if not memory_type:
                    if not forced:
                        continue
                    memory_type = "fact"
                third_party = detect_third_party(piece)
                sensitivity = detect_sensitivity(piece)
                candidate = CandidateMemory(
                    content=truncate_text(piece, 2000),
                    summary=_summary_for(piece, memory_type),
                    memory_type=memory_type,
                    tags=top_terms(piece, limit=6),
                    entities=_extract_entities(piece),
                    event_type="任务" if memory_type == "task_state" else "",
                    confidence=0.86 if forced else (0.45 if third_party else 0.66),
                    importance=DEFAULT_IMPORTANCE.get(memory_type, 0.5),
                    source_speaker="third_party" if third_party else role,
                    sensitivity_hint=sensitivity.level,
                    reason="规则抽取" if not forced else "用户显式要求记住",
                )
                candidates.append(candidate)
        # 同批去重
        deduped: list[CandidateMemory] = []
        for item in candidates:
            if any(jaccard(item.content, old.content) >= 0.9 for old in deduped):
                continue
            deduped.append(item)
        return ExtractionBatch(
            items=deduped[: self.settings.extraction_max_items],
            degraded=True,
            error="rule_fallback",
            provider="rules",
        )


class HybridExtractor(BaseExtractor):
    """小模型主路径 + 规则降级。"""

    def __init__(
        self,
        settings: MemorySettings,
        llm: LLMAdapter | None,
        *,
        enabled: bool = True,
    ):
        self.settings = settings
        self.llm = llm
        self.enabled = enabled and llm is not None
        self.rules = RuleExtractor(settings)

    async def extract(
        self,
        turns: Iterable[Any],
        context: AccessContext,
        *,
        event: Any = None,
        explicit: bool = False,
    ) -> ExtractionBatch:
        turn_list = list(turns)
        if not turn_list:
            return ExtractionBatch()
        if not self.enabled:
            return await self.rules.extract(
                turn_list, context, event=event, explicit=explicit
            )
        assert self.llm is not None
        context_text = _turn_text(turn_list)[-12000:]
        prompt = render_prompt(
            "extraction.txt",
            current_time=now_iso(),
            owner_scope=context.write_scope()[0],
            owner_id=context.write_scope()[1],
            scene="",
            task="",
            max_items=self.settings.extraction_max_items,
            context=context_text,
        )
        if explicit:
            prompt += "\n\n注意：用户本次明确要求记住，不要因为信息看起来琐碎而漏掉。"
        try:
            raw = await self.llm.extract_json(
                prompt,
                event=event,
                timeout=self.settings.extraction_timeout_seconds,
            )
        except LLMUnavailable as exc:
            fallback = await self.rules.extract(
                turn_list, context, event=event, explicit=explicit
            )
            fallback.degraded = True
            fallback.error = str(exc)
            return fallback

        items: list[CandidateMemory] = []
        raw_items = as_list(raw.get("memories"))
        for entry in raw_items[: self.settings.extraction_max_items]:
            if not isinstance(entry, Mapping):
                continue
            candidate = _validate_candidate(entry, self.settings)
            if candidate is None:
                continue
            if any(jaccard(candidate.content, old.content) >= 0.92 for old in items):
                continue
            items.append(candidate)
        return ExtractionBatch(
            items=items,
            degraded=False,
            provider=getattr(self.llm, "model_provider_id", "") or "small_model",
        )


def parse_extraction_json(text: str, settings: MemorySettings) -> ExtractionBatch:
    """测试/离线工具：从模型原始输出解析候选。"""
    try:
        data = as_dict(loads_lenient(text))
    except Exception as exc:  # noqa: BLE001
        return ExtractionBatch(degraded=True, error=str(exc), provider="rules")
    items = []
    for entry in as_list(data.get("memories"))[: settings.extraction_max_items]:
        if isinstance(entry, Mapping):
            validated = _validate_candidate(entry, settings)
            if validated:
                items.append(validated)
    return ExtractionBatch(items=items, provider="small_model")


def dump_extraction(batch: ExtractionBatch) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "content": item.content,
                    "summary": item.summary,
                    "memory_type": item.memory_type,
                    "sensitivity_hint": item.sensitivity_hint,
                }
                for item in batch.items
            ],
            "degraded": batch.degraded,
            "error": batch.error,
        },
        ensure_ascii=False,
    )
