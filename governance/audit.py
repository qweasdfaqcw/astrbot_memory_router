"""审计封装。所有删除/纠正/复活/维护动作都应经过这里。"""

from __future__ import annotations

from typing import Any, Mapping

from governance.policy import sanitize_audit_detail
from store.sqlite_store import SQLiteStore
from writer.sanitizer import mask_sensitive


class AuditLogger:
    def __init__(self, store: SQLiteStore):
        self.store = store

    async def log(
        self,
        *,
        actor: str,
        action: str,
        target: str = "",
        detail: Mapping[str, Any] | None = None,
        trace_id: str = "",
    ) -> str:
        safe_detail = sanitize_audit_detail(detail or {})
        return await self.store.add_audit(
            actor=mask_sensitive(str(actor))[:120],
            action=str(action)[:120],
            target=str(target)[:200],
            detail=safe_detail,
            trace_id=str(trace_id)[:100],
        )
