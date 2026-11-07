"""写通路：抽取 → 安全检查 → 去重/合并/冲突 → 打标签 → 落库 → 异步索引。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from config.settings import MemorySettings
from store.models import (
    AccessContext,
    MemoryRecord,
    TagInput,
    unique_strings,
)
from store.sqlite_store import SQLiteStore
from store.vector_backend import VectorBackend
from utils.ids import new_memory_id
from utils.text import normalize_text, truncate_text
from utils.time_utils import now_iso
from writer.conflict import detect_conflicts, is_near_duplicate
from writer.extractor import CandidateMemory, HybridExtractor
from writer.sanitizer import (
    clamp_sensitivity,
    detect_prompt_injection,
    detect_sensitivity,
    detect_third_party,
    mask_sensitive,
)
from writer.tagger import derive_tags, tag_strings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WriteReport:
    created_ids: list[str] = field(default_factory=list)
    merged_ids: list[str] = field(default_factory=list)
    conflict_ids: list[str] = field(default_factory=list)
    skipped: int = 0
    degraded: bool = False
    error: str = ""
    write_epoch: int = 0
    details: list[str] = field(default_factory=list)

    @property
    def touched(self) -> int:
        return len(self.created_ids) + len(self.merged_ids)


def _content_hash(owner_scope: str, owner_id: str, content: str) -> str:
    payload = f"{owner_scope}\x1f{owner_id}\x1f{normalize_text(content).lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tags_from_strings(tags: Iterable[str]) -> list[TagInput]:
    result: list[TagInput] = []
    for raw in tags:
        text = str(raw)
        if ":" in text:
            namespace, value = text.split(":", 1)
        else:
            namespace, value = "free", text
        value = normalize_text(value)[:80]
        if value:
            result.append(TagInput(namespace=namespace[:40], tag=value, weight=0.7))
    return result


def _merge_tag_lists(*groups: Iterable[TagInput]) -> list[TagInput]:
    merged: dict[tuple[str, str], TagInput] = {}
    for group in groups:
        for tag in group:
            key = (tag.namespace, tag.tag)
            current = merged.get(key)
            if current is None or tag.weight > current.weight:
                merged[key] = tag
    return list(merged.values())


class MemoryWriter:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        extractor: HybridExtractor,
        vector_backend: VectorBackend,
    ):
        self.store = store
        self.settings = settings
        self.extractor = extractor
        self.vector_backend = vector_backend

    async def process_turns(
        self,
        turns: Iterable[Any],
        context: AccessContext,
        *,
        event: Any = None,
        explicit: bool = False,
    ) -> WriteReport:
        report = WriteReport()
        turn_list = list(turns)
        if not turn_list:
            return report
        try:
            batch = await self.extractor.extract(
                turn_list, context, event=event, explicit=explicit
            )
        except Exception as exc:  # noqa: BLE001 - 写通路不能拖垮主对话
            logger.exception("记忆抽取失败")
            report.error = f"extract_failed:{exc}"
            report.degraded = True
            return report

        report.degraded = batch.degraded
        report.error = batch.error
        for candidate in batch.items:
            try:
                outcome = await self._write_candidate(
                    candidate, context, explicit=explicit
                )

            except Exception as exc:  # noqa: BLE001
                logger.exception("单条记忆写入失败: %s", mask_sensitive(candidate.content)[:80])
                report.details.append(f"error:{exc}")
                report.skipped += 1
                continue
            if outcome == "created":
                report.created_ids.append(candidate.last_id)
            elif outcome == "merged":
                report.merged_ids.append(candidate.last_id)
            else:
                report.skipped += 1
            report.conflict_ids.extend(candidate.conflict_ids)
        report.write_epoch = await self.store.get_write_epoch()
        return report

    async def _write_candidate(
        self,
        candidate: CandidateMemory,
        context: AccessContext,
        *,
        explicit: bool,
    ) -> str:
        content = normalize_text(candidate.content)
        if len(content) < 2:
            return "skipped"
        summary = normalize_text(candidate.summary) or truncate_text(content, 100)
        injection = detect_prompt_injection(content + "\n" + summary)
        contains_instruction = candidate.contains_instruction or injection.risky
        rule_sensitivity = detect_sensitivity(content)
        sensitivity = clamp_sensitivity(
            rule_sensitivity.level,
            candidate.sensitivity_hint,
        )
        if contains_instruction:
            sensitivity = "secret" if sensitivity == "secret" else "sensitive"
        third_party = (
            candidate.source_speaker == "third_party"
            or detect_third_party(content)
        )
        source = "user"
        if third_party:
            source = "third_party"
        elif candidate.source_speaker == "assistant":
            source = "assistant"
        confidence = candidate.confidence
        if third_party:
            confidence = min(confidence, 0.45)
        if source == "assistant":
            confidence = min(confidence, 0.5)
        if contains_instruction:
            confidence = min(confidence, 0.55)
        owner_scope, owner_id = context.write_scope()
        content_hash = _content_hash(owner_scope, owner_id, content)

        exact = await self.store.find_by_hash(owner_scope, owner_id, content_hash)
        if exact:
            merged_id = await self._merge(
                exact,
                candidate,
                content,
                summary,
                confidence,
                sensitivity,
                source,
                contains_instruction,
                context,
                explicit=explicit,
            )
            candidate.last_id = merged_id
            return "merged"

        similar = await self.store.find_similar_memories(
            owner_scope,
            owner_id,
            candidate.memory_type,
            content,
            limit=20,
        )
        near = next(
            (
                old for old in similar
                if is_near_duplicate(content, str(old.get("content") or ""), threshold=0.86)
            ),
            None,
        )
        if near:
            merged_id = await self._merge(
                near,
                candidate,
                content,
                summary,
                confidence,
                sensitivity,
                source,
                contains_instruction,
                context,
                explicit=explicit,
            )
            candidate.last_id = merged_id
            return "merged"

        conflicts = detect_conflicts(
            content, candidate.memory_type, source, similar
        )
        record = MemoryRecord(
            id=new_memory_id(),
            content=content,
            summary=summary,
            memory_type=candidate.memory_type,
            owner_scope=owner_scope,
            owner_id=owner_id,
            source=source,
            source_ref=(context.umo or context.session_id)[:200],
            confidence=confidence,
            sensitivity=sensitivity,
            importance=candidate.importance,
            emotion_valence=candidate.emotion_valence,
            emotion_arousal=candidate.emotion_arousal,
            scene=candidate.scene,
            task=candidate.task,
            event_type=candidate.event_type,
            entities=unique_strings(candidate.entities, 12),
            injectable=not contains_instruction,
            content_hash=content_hash,
            status="disputed" if conflicts else "active",
            # 用户明确要求记住的内容默认长期保留，自动归档/压缩不会删除。
            pinned=explicit,
            created_at=now_iso(),
            updated_at=now_iso(),
            last_verified_at=now_iso(),
        )
        tags = derive_tags(
            record,
            context,
            extra_tags=candidate.tags,
            injection_risky=contains_instruction,
            third_party=third_party,
        )
        await self.store.create_memory(record, tags)
        candidate.last_id = record.id

        if conflicts:
            conflict_ids: list[str] = []
            for finding in conflicts[:5]:
                old_id = str(finding.existing.get("id") or "")
                if not old_id:
                    continue
                conflict_id = await self.store.add_conflict(
                    record.id,
                    old_id,
                    finding.conflict_type,
                    resolution=f"pending_user_confirmation:{finding.reason}",
                )
                conflict_ids.append(conflict_id)
                await self.store.set_status(old_id, "disputed")
            candidate.conflict_ids = conflict_ids
            await self.store.add_audit(
                actor="memory_writer",
                action="memory.conflict_detected",
                target=record.id,
                detail={
                    "conflicts": conflict_ids,
                    "memory_type": candidate.memory_type,
                    "summary": mask_sensitive(summary),
                },
            )

        if explicit or contains_instruction:
            await self.store.add_audit(
                actor="memory_writer",
                action="memory.write",
                target=record.id,
                detail={
                    "explicit": explicit,
                    "injectable": record.injectable,
                    "sensitivity": sensitivity,
                    "summary": mask_sensitive(summary),
                    "injection_patterns": injection.patterns,
                },
            )

        await self._index_vector(record.id, content, summary, tags)
        return "created"

    async def _merge(
        self,
        old: Mapping[str, Any],
        candidate: CandidateMemory,
        content: str,
        summary: str,
        confidence: float,
        sensitivity: str,
        source: str,
        contains_instruction: bool,
        context: AccessContext,
        *,
        explicit: bool = False,
    ) -> str:
        old_id = str(old.get("id") or "")
        old_content = str(old.get("content") or "")
        merged_content = content if len(content) > len(old_content) else old_content
        merged_summary = summary if len(summary) > len(str(old.get("summary") or "")) else str(old.get("summary") or summary)
        old_confidence = float(old.get("confidence") or 0.0)
        old_importance = float(old.get("importance") or 0.0)
        old_tags_map = await self.store.get_tags_map([old_id])
        old_tags = _tags_from_strings(old_tags_map.get(old_id, []))
        record_stub = MemoryRecord(
            content=merged_content,
            summary=merged_summary,
            memory_type=str(old.get("memory_type") or candidate.memory_type),
            owner_scope=str(old.get("owner_scope") or context.write_scope()[0]),
            owner_id=str(old.get("owner_id") or context.write_scope()[1]),
        )
        new_tags = derive_tags(
            record_stub,
            context,
            extra_tags=candidate.tags,
            injection_risky=contains_instruction,
            third_party=source == "third_party",
        )
        tags = _merge_tag_lists(old_tags, new_tags)
        # 用户本人陈述可以升级来源与置信度；第三方/助手陈述不能覆盖用户来源。
        effective_source = str(old.get("source") or source)
        if source == "user":
            effective_source = "user"
            confidence = max(confidence, old_confidence)
        else:
            confidence = max(confidence, old_confidence)
        effective_sensitivity = sensitivity
        if str(old.get("sensitivity") or "normal") == "secret":
            effective_sensitivity = "secret"
        injectable = bool(old.get("injectable", 1)) and not contains_instruction
        await self.store.update_memory(
            old_id,
            {
                "content": merged_content,
                "summary": merged_summary,
                "confidence": min(1.0, confidence),
                "importance": max(old_importance, candidate.importance),
                "sensitivity": effective_sensitivity,
                "source": effective_source,
                "injectable": injectable,
                "content_hash": _content_hash(
                    str(old.get("owner_scope") or ""),
                    str(old.get("owner_id") or ""),
                    merged_content,
                ),
                "last_verified_at": now_iso(),
                "pinned": True if explicit else bool(old.get("pinned", 0)),
            },
            tags=tags,
        )
        if self.vector_backend.enabled:
            await self._index_vector(
                old_id, merged_content, merged_summary, tags
            )
        return old_id

    async def _index_vector(
        self,
        memory_id: str,
        content: str,
        summary: str,
        tags: Iterable[TagInput],
    ) -> None:
        if not self.vector_backend.enabled:
            return
        text = content + " " + summary + " " + " ".join(tag_strings(tags))
        try:
            await self.vector_backend.index(memory_id, text)
        except Exception:  # noqa: BLE001 - 向量索引失败不影响主存
            logger.warning("向量索引更新失败: %s", memory_id, exc_info=True)
