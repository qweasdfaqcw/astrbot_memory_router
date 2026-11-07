"""生命周期：留存更新、压缩、归档、软删除、硬删除、复活、纠正。

自动流程只做降权/压缩/归档，**绝不自动硬删除**。所有删除/纠正/复活都有审计。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from config.settings import MemorySettings
from governance.audit import AuditLogger
from governance.policy import (
    PermissionDenied,
    ensure_memory_access,
    ensure_memory_mutation,
)
from stats.retention import (
    RetentionBreakdown,
    compute_retention,
    retention_reason,
    should_archive,
    should_compress,
)
from store.models import (
    HALF_LIFE_DAYS,
    SENSITIVITY_RANK,
    AccessContext,
    MemoryRecord,
    TagInput,
    unique_strings,
)
from store.sqlite_store import SQLiteStore
from store.vector_backend import VectorBackend
from utils.ids import new_memory_id
from utils.text import normalize_text, truncate_text
from utils.time_utils import age_days, now_iso
from writer.sanitizer import detect_sensitivity, mask_sensitive


@dataclass(slots=True)
class MaintenanceReport:
    retention_updated: int = 0
    archived: list[str] = field(default_factory=list)
    compressed: list[str] = field(default_factory=list)
    tags_pruned: int = 0
    errors: list[str] = field(default_factory=list)


class LifecycleManager:
    def __init__(
        self,
        store: SQLiteStore,
        settings: MemorySettings,
        audit: AuditLogger,
        vector_backend: VectorBackend,
    ):
        self.store = store
        self.settings = settings
        self.audit = audit
        self.vector_backend = vector_backend

    async def breakdown(self, memory: Mapping[str, Any]) -> RetentionBreakdown:
        stats_map = await self.store.get_access_stats([str(memory.get("id") or "")])
        stats = stats_map.get(str(memory.get("id") or ""))
        return compute_retention(memory, stats)

    async def update_retention(self, limit: int = 1000) -> int:
        memories = await self.store.active_memories_for_maintenance(limit)
        if not memories:
            return 0
        stats_map = await self.store.get_access_stats(
            [str(item["id"]) for item in memories]
        )
        updated = 0
        for memory in memories:
            try:
                breakdown = compute_retention(
                    memory, stats_map.get(str(memory["id"]))
                )
                current = float(memory.get("retention_score") or 0.0)
                if abs(current - breakdown.score) < 0.005:
                    continue
                await self.store.update_memory(
                    str(memory["id"]),
                    {"retention_score": breakdown.score},
                )
                updated += 1
            except Exception:  # noqa: BLE001 - 单条失败继续
                continue
        return updated

    async def compress(
        self,
        memory_ids: Iterable[str],
        *,
        context: AccessContext,
        actor: str = "system",
        reason: str = "manual_compression",
        summary: str = "",
    ) -> str:
        ids = unique_strings(memory_ids, limit=50)
        if len(ids) < 2:
            raise ValueError("压缩至少需要 2 条记忆")
        memories = await self.store.get_memories_by_ids(ids, context)
        if len(memories) < 2:
            raise PermissionDenied("可访问的待压缩记忆不足")
        owner_scope = str(memories[0].get("owner_scope"))
        owner_id = str(memories[0].get("owner_id"))
        if any(
            str(item.get("owner_scope")) != owner_scope
            or str(item.get("owner_id")) != owner_id
            for item in memories
        ):
            raise PermissionDenied("不能跨 owner 压缩记忆")
        if any(str(item.get("lifecycle")) != "active" for item in memories):
            raise ValueError("只能压缩 active 记忆")
        tags_map = await self.store.get_tags_map(ids)
        tag_inputs: list[TagInput] = []
        for values in tags_map.values():
            for value in values:
                if ":" in value:
                    namespace, tag = value.split(":", 1)
                    tag_inputs.append(TagInput(namespace=namespace, tag=tag, weight=0.8))
        sensitivity = "normal"
        for memory in memories:
            level = str(memory.get("sensitivity") or "normal")
            if SENSITIVITY_RANK.get(level, 0) > SENSITIVITY_RANK.get(sensitivity, 0):
                sensitivity = level
        combined_content = summary or "；".join(
            str(item.get("summary") or item.get("content") or "")[:300]
            for item in memories[:20]
        )
        combined_content = truncate_text(combined_content, 3000)
        combined_summary = truncate_text(
            summary or f"压缩事件（{len(memories)} 条）：{combined_content}", 500
        )
        importance = max(float(item.get("importance") or 0.0) for item in memories)
        confidence = max(0.35, min(float(item.get("confidence") or 0.5) for item in memories) * 0.85)
        arousal = sum(float(item.get("emotion_arousal") or 0.0) for item in memories) / len(memories)
        valence = sum(float(item.get("emotion_valence") or 0.0) for item in memories) / len(memories)
        now = now_iso()
        record = MemoryRecord(
            id=new_memory_id(),
            content=combined_content,
            summary=combined_summary,
            memory_type="event",
            owner_scope=owner_scope,
            owner_id=owner_id,
            source="system",
            source_ref="compression",
            confidence=confidence,
            sensitivity=sensitivity,
            importance=importance,
            emotion_valence=valence,
            emotion_arousal=arousal,
            injectable=all(bool(item.get("injectable", 1)) for item in memories),
            content_hash=hashlib.sha256(combined_content.encode("utf-8")).hexdigest(),
            compressed_from=ids,
            status="active",
            created_at=now,
            updated_at=now,
            last_verified_at=now,
        )
        await self.store.create_memory(record, tag_inputs)
        for memory in memories:
            await self.store.set_lifecycle(
                str(memory["id"]),
                "compressed",
                compressed_into=record.id,
                write_cold_reason="compressed",
            )
            if self.vector_backend.enabled:
                await self.vector_backend.remove(str(memory["id"]))
        if self.vector_backend.enabled:
            try:
                await self.vector_backend.index(
                    record.id, combined_content + " " + combined_summary
                )
            except Exception:  # noqa: BLE001
                pass
        await self.audit.log(
            actor=actor,
            action="memory.compress",
            target=record.id,
            detail={
                "compressed_from": ids,
                "reason": reason,
                "summary": mask_sensitive(combined_summary),
            },
        )
        return record.id

    async def archive(
        self,
        memory_ids: Iterable[str],
        *,
        actor: str = "system",
        reason: str = "manual_archive",
        allow_pinned: bool = False,
    ) -> list[str]:
        archived: list[str] = []
        for memory_id in unique_strings(memory_ids, limit=100):
            memory = await self.store.get_memory(memory_id)
            if not memory:
                continue
            if bool(memory.get("pinned")) and not allow_pinned:
                continue
            if str(memory.get("lifecycle")) != "active":
                continue
            if await self.store.archive_memory(memory_id, reason):
                if self.vector_backend.enabled:
                    await self.vector_backend.remove(memory_id)
                await self.audit.log(
                    actor=actor,
                    action="memory.archive",
                    target=memory_id,
                    detail={"reason": reason, "summary": mask_sensitive(str(memory.get("summary") or ""))},
                )
                archived.append(memory_id)
        return archived

    async def soft_delete(
        self,
        memory_id: str,
        *,
        context: AccessContext,
        actor: str,
        reason: str = "user_forget",
    ) -> bool:
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return False
        ensure_memory_mutation(context, memory)
        if await self.store.soft_delete_memory(memory_id, reason):
            if self.vector_backend.enabled:
                await self.vector_backend.remove(memory_id)
            await self.audit.log(
                actor=actor,
                action="memory.soft_delete",
                target=memory_id,
                detail={"reason": reason, "summary": mask_sensitive(str(memory.get("summary") or ""))},
            )
            return True
        return False

    async def hard_delete(
        self,
        memory_id: str,
        *,
        context: AccessContext,
        actor: str,
        reason: str,
        authorized: bool,
    ) -> bool:
        if not authorized or not self.settings.hard_delete_enabled:
            raise PermissionDenied("硬删除未授权或未启用")
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return False
        ensure_memory_mutation(context, memory)
        # 先写审计，再删除；审计失败则中止。
        await self.audit.log(
            actor=actor,
            action="memory.hard_delete",
            target=memory_id,
            detail={
                "reason": reason,
                "memory_type": memory.get("memory_type"),
                "owner_scope": memory.get("owner_scope"),
                "summary": mask_sensitive(str(memory.get("summary") or "")),
            },
        )
        deleted = await self.store.hard_delete_memory(memory_id)
        if not deleted:
            return False
        if self.vector_backend.enabled:
            await self.vector_backend.remove(memory_id)
        return True

    async def revive(
        self,
        memory_id: str,
        *,
        context: AccessContext,
        actor: str,
        reason: str = "user_revive",
    ) -> dict[str, Any]:
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return {}
        ensure_memory_mutation(context, memory)
        if str(memory.get("lifecycle")) not in {"archived", "compressed", "soft_deleted"}:
            return {"memory_id": memory_id, "warning": "记忆当前不是归档/压缩/删除状态"}
        old_confidence = float(memory.get("confidence") or 0.5)
        warning_parts: list[str] = ["归档记忆，可能过时"]
        confidence = old_confidence * 0.9
        reference = str(memory.get("updated_at") or memory.get("created_at") or "")
        half_life = HALF_LIFE_DAYS.get(str(memory.get("memory_type")), 180.0)
        if age_days(reference) > half_life:
            confidence = min(confidence, old_confidence * 0.75)
            warning_parts.append("超过该类型半衰期，已进一步降低置信度")
        conflicts = await self.store.get_conflicts_for([memory_id])
        if conflicts.get(memory_id):
            warning_parts.append("存在历史冲突，请向用户确认")
        ok = await self.store.revive_memory(memory_id, max(0.05, min(0.9, confidence)))
        if not ok:
            return {}
        await self.store.record_access(
            memory_id,
            caller=actor,
            mode="system",
            event_type="revived",
            session_id=context.session_id,
            meta={"reason": reason},
        )
        if self.vector_backend.enabled:
            try:
                tags_map = await self.store.get_tags_map([memory_id])
                await self.vector_backend.index(
                    memory_id,
                    str(memory.get("content") or "")
                    + " "
                    + " ".join(tags_map.get(memory_id, [])),
                )
            except Exception:  # noqa: BLE001
                pass
        await self.audit.log(
            actor=actor,
            action="memory.revive",
            target=memory_id,
            detail={"reason": reason, "warning": "；".join(warning_parts)},
        )
        return {
            "memory_id": memory_id,
            "confidence": round(max(0.05, min(0.9, confidence)), 4),
            "warning": "；".join(warning_parts),
            "conflicts": len(conflicts.get(memory_id, [])),
        }

    async def confirm(
        self, memory_id: str, *, context: AccessContext, actor: str
    ) -> dict[str, Any]:
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return {}
        ensure_memory_mutation(context, memory)
        new_confidence = min(1.0, max(float(memory.get("confidence") or 0.5), 0.9))
        fields: dict[str, Any] = {
            "confidence": new_confidence,
            "status": "active",
            "last_verified_at": now_iso(),
        }
        if str(memory.get("source")) == "third_party":
            # 用户确认后升级为“用户认可”的来源，同时保留 third_party 标签以追溯原出处。
            fields["source"] = "user"
        tags_map = await self.store.get_tags_map([memory_id])
        tag_inputs = []
        for value in tags_map.get(memory_id, []):
            if ":" in value:
                namespace, tag = value.split(":", 1)
                tag_inputs.append(TagInput(namespace=namespace, tag=tag, weight=0.8))
        tag_inputs.append(TagInput(namespace="security", tag="user_confirmed", weight=1.0))
        await self.store.update_memory(memory_id, fields, tags=tag_inputs)
        await self.store.resolve_conflicts_for(
            memory_id, "user_confirmed", status="resolved"
        )
        await self.store.record_access(
            memory_id,
            caller=actor,
            mode="system",
            event_type="user_confirmed",
            session_id=context.session_id,
        )
        await self.audit.log(
            actor=actor,
            action="memory.confirm",
            target=memory_id,
            detail={"confidence": new_confidence},
        )
        return {"memory_id": memory_id, "confidence": new_confidence}

    async def correct(
        self,
        memory_id: str,
        corrected_text: str,
        *,
        context: AccessContext,
        actor: str,
    ) -> dict[str, Any]:
        old = await self.store.get_memory(memory_id)
        if not old:
            return {}
        ensure_memory_mutation(context, old)
        text = normalize_text(corrected_text)
        if len(text) < 2:
            raise ValueError("纠正内容太短")
        sensitivity = detect_sensitivity(text)
        now = now_iso()
        new = MemoryRecord(
            id=new_memory_id(),
            content=text,
            summary=truncate_text(text, 180),
            memory_type=str(old.get("memory_type") or "fact"),
            owner_scope=str(old.get("owner_scope")),
            owner_id=str(old.get("owner_id")),
            source="user",
            source_ref=context.umo or context.session_id,
            confidence=0.95,
            sensitivity=sensitivity.level,
            importance=max(0.5, float(old.get("importance") or 0.5)),
            injectable=not sensitivity.is_sensitive,
            content_hash=hashlib.sha256(text.lower().encode("utf-8")).hexdigest(),
            pinned=bool(old.get("pinned")),
            status="active",
            created_at=now,
            updated_at=now,
            last_verified_at=now,
        )
        tags_map = await self.store.get_tags_map([memory_id])
        tag_inputs = []
        for value in tags_map.get(memory_id, []):
            if ":" in value:
                namespace, tag = value.split(":", 1)
                tag_inputs.append(TagInput(namespace=namespace, tag=tag, weight=0.8))
        await self.store.create_memory(new, tag_inputs)
        await self.store.set_status(memory_id, "superseded")
        await self.store.record_access(
            memory_id,
            caller=actor,
            mode="system",
            event_type="user_corrected",
            session_id=context.session_id,
            meta={"corrected_into": new.id},
        )
        await self.store.resolve_conflicts_for(
            memory_id, f"corrected_into:{new.id}", status="resolved"
        )
        await self.audit.log(
            actor=actor,
            action="memory.correct",
            target=memory_id,
            detail={
                "corrected_into": new.id,
                "old_summary": mask_sensitive(str(old.get("summary") or "")),
                "new_summary": mask_sensitive(new.summary),
            },
        )
        return {"old_memory_id": memory_id, "new_memory_id": new.id}

    async def promote_global(
        self,
        memory_id: str,
        *,
        context: AccessContext,
        actor: str,
    ) -> dict[str, Any]:
        """把一条记忆提升为全局共享记忆（仅管理员，明确审计）。

        个人/群记忆默认隔离；只有显式提升后才进入 global 层。
        """
        if not context.is_admin:
            raise PermissionDenied("全局记忆提升需要管理员权限")
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return {}
        ensure_memory_access(context, memory)
        old_scope = str(memory.get("owner_scope") or "personal")
        old_owner = str(memory.get("owner_id") or "")
        tags_map = await self.store.get_tags_map([memory_id])
        tag_inputs = []
        for value in tags_map.get(memory_id, []):
            if ":" in value:
                namespace, tag = value.split(":", 1)
                tag_inputs.append(TagInput(namespace=namespace, tag=tag, weight=0.8))
        tag_inputs.append(TagInput(namespace="security", tag="shared", weight=1.0))
        ok = await self.store.update_memory(
            memory_id,
            {"owner_scope": "global", "owner_id": "global"},
            tags=tag_inputs,
        )
        if not ok:
            return {}
        await self.audit.log(
            actor=actor,
            action="memory.promote_global",
            target=memory_id,
            detail={"from_scope": old_scope, "from_owner": old_owner},
        )
        return {"memory_id": memory_id, "owner_scope": "global", "owner_id": "global"}

    async def pin(self, memory_id: str, *, context: AccessContext, pinned: bool, actor: str) -> bool:
        memory = await self.store.get_memory(memory_id)
        if not memory:
            return False
        ensure_memory_mutation(context, memory)
        ok = await self.store.set_pinned(memory_id, pinned)
        if ok:
            await self.audit.log(
                actor=actor,
                action="memory.keep" if pinned else "memory.unkeep",
                target=memory_id,
                detail={},
            )
        return ok

    async def run_maintenance(self) -> MaintenanceReport:
        report = MaintenanceReport()
        try:
            report.retention_updated = await self.update_retention()
            report.tags_pruned = await self.store.prune_orphan_tags()
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"retention:{exc}")
            return report

        memories = await self.store.active_memories_for_maintenance(1000)
        stats_map = await self.store.get_access_stats(
            [str(item["id"]) for item in memories]
        )
        archive_candidates: list[str] = []
        compress_groups: dict[str, list[str]] = {}
        for memory in memories:
            breakdown = compute_retention(
                memory, stats_map.get(str(memory["id"]))
            )
            if should_archive(
                memory,
                breakdown,
                auto_archive=self.settings.auto_archive,
                archive_after_days=self.settings.archive_after_days,
                threshold=self.settings.archive_retention_threshold,
            ):
                archive_candidates.append(str(memory["id"]))
            if should_compress(
                memory,
                breakdown,
                auto_compress=self.settings.auto_compress,
                compress_after_days=self.settings.compress_after_days,
            ):
                tags_map = await self.store.get_tags_map([str(memory["id"])])
                key_tags = [
                    tag for tag in tags_map.get(str(memory["id"]), [])
                    if tag.startswith("topic:") or tag.startswith("scene:")
                ]
                if key_tags:
                    key = f"{memory.get('owner_scope')}:{memory.get('owner_id')}:{key_tags[0]}"
                    compress_groups.setdefault(key, []).append(str(memory["id"]))
        for memory_id in archive_candidates[:50]:
            try:
                archived = await self.archive(
                    [memory_id], actor="maintenance", reason="auto_archive"
                )
                report.archived.extend(archived)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"archive:{exc}")
        for group in list(compress_groups.values()):
            if len(group) < self.settings.compress_min_group:
                continue
            # 自动压缩以系统身份执行；上下文必须匹配目标 owner，否则会被隔离层拒绝。
            try:
                first = await self.store.get_memory(group[0])
                if not first:
                    continue
                scope = str(first.get("owner_scope") or "personal")
                owner_id = str(first.get("owner_id") or "system")
                if scope == "group":
                    context = AccessContext(
                        user_id="system",
                        group_id=owner_id,
                        session_id="maintenance",
                        is_group=True,
                        is_admin=True,
                    )
                elif scope == "personal":
                    context = AccessContext(
                        user_id=owner_id,
                        session_id="maintenance",
                        is_admin=True,
                    )
                else:
                    context = AccessContext(
                        user_id="system",
                        session_id="maintenance",
                        is_admin=True,
                    )
                new_id = await self.compress(
                    group[:20],
                    context=context,
                    actor="maintenance",
                    reason="auto_compress",
                )
                report.compressed.append(new_id)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"compress:{exc}")
            if len(report.compressed) >= 3:
                break
        return report

    async def explain(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        breakdown = await self.breakdown(memory)
        conflicts = await self.store.get_conflicts_for([str(memory.get("id") or "")])
        return {
            "memory_id": memory.get("id"),
            "retention": breakdown.to_dict(),
            "retention_explain": breakdown.explain(),
            "reason": retention_reason(memory, breakdown),
            "conflicts": conflicts.get(str(memory.get("id") or ""), []),
            "cold_copies": len(await self.store.get_cold(str(memory.get("id") or ""))),
        }
