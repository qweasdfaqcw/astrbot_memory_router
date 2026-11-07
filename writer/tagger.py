"""标签生成与受控词表。

标签采用 ``namespace:tag`` 的形式。命名空间包括：
- type / source / sensitivity / event / scene / task / time
- entity / topic / free / emotion / security

受控词表用于稳定的结构化过滤；自由标签用于后续聚类演化。每个标签保留
version，合并/淘汰时不物理抹掉历史版本。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

from store.models import AccessContext, MemoryRecord, TagInput, unique_strings
from utils.text import compact_text, top_terms

_CONTROLLED_NAMESPACES = {
    "type", "source", "sensitivity", "event", "scene", "task",
    "time", "entity", "topic", "free", "emotion", "security",
}
_TAG_CLEAN_RE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff:.\- ]+", re.UNICODE)


def normalize_tag(value: str, max_len: int = 48) -> str:
    """清洗标签值，保留可读的中英文与少量符号。"""
    text = (value or "").strip().lower()
    text = _TAG_CLEAN_RE.sub("", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("_.:- ")[:max_len]


def make_tag(namespace: str, value: str, weight: float = 1.0) -> TagInput | None:
    ns = normalize_tag(namespace, 32)
    if ns not in _CONTROLLED_NAMESPACES:
        ns = "free"
    tag = normalize_tag(value)
    if not tag:
        return None
    return TagInput(namespace=ns, tag=tag, weight=max(0.0, min(1.0, weight)))


def derive_tags(
    record: MemoryRecord,
    context: AccessContext,
    *,
    extra_tags: Iterable[str] = (),
    injection_risky: bool = False,
    third_party: bool = False,
) -> list[TagInput]:
    """从结构化字段、内容关键词与模型标签生成最终标签集。"""
    tags: list[TagInput] = []

    def add(namespace: str, value: str, weight: float = 1.0) -> None:
        item = make_tag(namespace, value, weight)
        if item:
            tags.append(item)

    add("type", record.memory_type, 1.0)
    add("source", record.source, 0.9)
    add("sensitivity", record.sensitivity, 1.0)
    if record.event_type:
        add("event", record.event_type, 0.9)
    if record.scene:
        add("scene", record.scene, 0.9)
    if record.task:
        add("task", record.task, 0.9)
    if record.owner_scope:
        add("time", datetime.now(timezone.utc).strftime("month_%Y_%m"), 0.4)

    for entity in record.entities[:12]:
        add("entity", entity, 0.9)
    for term in top_terms(record.content, limit=8):
        add("topic", term, 0.65)

    if record.emotion_valence >= 0.25:
        add("emotion", "positive", 0.8)
    elif record.emotion_valence <= -0.25:
        add("emotion", "negative", 0.8)
    if record.emotion_arousal >= 0.55:
        add("emotion", "high_arousal", 0.8)

    if injection_risky:
        add("security", "prompt_injection", 1.0)
        add("security", "not_executable", 1.0)
    if third_party:
        add("source", "third_party", 0.8)
        add("security", "needs_confirmation", 0.8)
    if record.sensitivity != "normal":
        add("security", "restricted_injection", 1.0)

    for raw in extra_tags:
        text = str(raw)
        if ":" in text:
            namespace, value = text.split(":", 1)
            add(namespace, value, 0.7)
        else:
            # 模型自由标签放入 free，避免污染受控命名空间。
            add("free", compact_text(text) or text, 0.55)

    # 同一 (namespace, tag) 只保留最高权重。
    by_key: dict[tuple[str, str], TagInput] = {}
    for item in tags:
        key = (item.namespace, item.tag)
        current = by_key.get(key)
        if current is None or item.weight > current.weight:
            by_key[key] = item
    result = sorted(by_key.values(), key=lambda item: (item.namespace, item.tag))
    return result[:40]


def tag_strings(tags: Iterable[TagInput]) -> list[str]:
    """TagInput -> ``namespace:tag`` 列表。"""
    return unique_strings(f"{tag.namespace}:{tag.tag}" for tag in tags)
