"""SQLite + FTS5 存储实现。

读连接每次从线程池新建，写操作统一走 ``WriteQueue``。这样：
- 读不阻塞写，写不阻塞读（WAL）；
- 所有写操作串行化，避免 “database is locked”；
- 写入函数返回时事务已提交，关键事实立刻可检索。
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from store.migrations import SCHEMA_VERSION, migrate, rebuild_fts
from store.models import (
    SENSITIVITY_RANK,
    AccessContext,
    AccessStats,
    MemoryRecord,
    SearchFilters,
    TagInput,
    decode_memory_row,
    unique_strings,
)
from store.vector_backend import cosine_similarity
from store.write_queue import WriteQueue
from utils.ids import new_audit_id, new_conflict_id, new_memory_id
from utils.json_utils import dump
from utils.text import normalize_text, tokenize
from utils.time_utils import now_iso

T = TypeVar("T")
_MAX_VECTOR_SCAN = 5000


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def _safe_json(value: str | None, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, ValueError):
        return default


def _tag_query_values(tag: str) -> tuple[str, str]:
    """把 ``namespace:tag`` 或裸 ``tag`` 拆成两种匹配形式。"""
    raw = tag.strip()
    if ":" in raw:
        namespace, value = raw.split(":", 1)
        return f"{namespace}:{value}", value
    return raw, raw


class SQLiteStore:
    """记忆库。生命周期内长驻，插件 ``terminate`` 时关闭。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.schema_version = SCHEMA_VERSION
        self.fts_mode = "trigram"
        self._queue: WriteQueue | None = None
        self._opened = False
        self._open_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def open(self) -> None:
        """同步建库/迁移。只应在插件初始化时调用一次。"""
        if self._opened:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            version = migrate(conn)
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"数据库 schema_version={version} 高于插件支持的 {SCHEMA_VERSION}"
                )
            self.schema_version = version
            row = conn.execute(
                "SELECT value FROM settings WHERE key='fts_mode'"
            ).fetchone()
            self.fts_mode = str(row[0]) if row else "trigram"
            # 上次异常退出可能留下不一致的 FTS；启动时自检重建。
            if self.fts_mode in {"trigram", "unicode61"}:
                memory_count = conn.execute(
                    "SELECT COUNT(*) FROM memories "
                    "WHERE lifecycle='active' AND status IN ('active','disputed')"
                ).fetchone()[0]
                fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                if memory_count != fts_count:
                    rebuild_fts(conn, self.fts_mode)
                    conn.commit()
        finally:
            conn.close()
        self._queue = WriteQueue(self.db_path)
        self._opened = True

    async def start(self) -> None:
        """启动写队列（惰性，首个写操作也会启动）。"""
        if not self._opened:
            raise RuntimeError("store is not open")
        if self._queue is None:
            self._queue = WriteQueue(self.db_path)

    async def close(self) -> None:
        if self._queue:
            await self._queue.close()
        self._queue = None
        self._opened = False

    async def flush(self) -> None:
        if self._queue:
            await self._queue.flush()

    # ------------------------------------------------------------------
    # 基础连接
    # ------------------------------------------------------------------
    def _connect_read(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path), timeout=10.0, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    async def _read(self, func: Callable[[sqlite3.Connection], T]) -> T:
        def reader() -> T:
            conn = self._connect_read()
            try:
                return func(conn)
            finally:
                conn.close()

        return await asyncio.to_thread(reader)

    async def _write(self, func: Callable[[sqlite3.Connection], T]) -> T:
        if not self._opened:
            raise RuntimeError("store is not open")
        if self._queue is None:
            await self.start()
        assert self._queue is not None
        return await self._queue.submit(func)

    # ------------------------------------------------------------------
    # 设置 / epoch
    # ------------------------------------------------------------------
    async def get_setting(self, key: str, default: str = "") -> str:
        def op(conn: sqlite3.Connection) -> str:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return str(row[0]) if row else default

        return await self._read(op)

    async def set_setting(self, key: str, value: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

        await self._write(op)

    async def get_write_epoch(self) -> int:
        value = await self.get_setting("write_epoch", "0")
        try:
            return int(value)
        except ValueError:
            return 0

    def _next_write_epoch(self, conn: sqlite3.Connection) -> int:
        conn.execute(
            "INSERT INTO settings(key, value) VALUES('write_epoch', '1') "
            "ON CONFLICT(key) DO UPDATE SET value=CAST(value AS INTEGER)+1"
        )
        row = conn.execute("SELECT value FROM settings WHERE key='write_epoch'").fetchone()
        return int(row[0]) if row else 1

    # ------------------------------------------------------------------
    # owner / filter SQL
    # ------------------------------------------------------------------
    def _owner_clause(
        self,
        context: AccessContext,
        filters: SearchFilters | None = None,
        alias: str = "m",
    ) -> tuple[str, list[Any]]:
        """构造严格 owner 过滤，并处理 filters 中的窄化条件。"""
        scopes = context.accessible_scopes()
        requested_scope = filters.owner_scope if filters else ""
        requested_id = filters.owner_id if filters else ""

        if requested_scope:
            allowed = [item for item in scopes if item[0] == requested_scope]
            if requested_scope == "global":
                allowed = [("global", "*")]
            elif requested_id:
                allowed = [
                    item for item in allowed
                    if item[1] == requested_id and context.can_access(item[0], item[1])
                ]
            if not allowed:
                return "0=1", []
            scopes = allowed
        elif requested_id:
            scopes = [item for item in scopes if item[1] == requested_id]
            if not scopes and requested_id in {"global", "*"}:
                scopes = [("global", "*")]
            if not scopes:
                return "0=1", []

        clauses: list[str] = []
        params: list[Any] = []
        for scope, owner_id in scopes:
            if scope == "global":
                clauses.append(f"{alias}.owner_scope='global'")
            elif owner_id:
                clauses.append(f"({alias}.owner_scope=? AND {alias}.owner_id=?)")
                params.extend([scope, owner_id])
        if not clauses:
            return "0=1", []
        return "(" + " OR ".join(clauses) + ")", params

    def _filter_clauses(
        self,
        context: AccessContext,
        filters: SearchFilters,
        *,
        include_inactive: bool = False,
        alias: str = "m",
    ) -> tuple[list[str], list[Any]]:
        owner_sql, owner_params = self._owner_clause(context, filters, alias)
        clauses = [owner_sql]
        params: list[Any] = list(owner_params)

        lifecycle = filters.lifecycle
        if include_inactive:
            lifecycle = []
        if lifecycle:
            placeholders = ",".join("?" for _ in lifecycle)
            clauses.append(f"{alias}.lifecycle IN ({placeholders})")
            params.extend(lifecycle)
        if filters.status and not include_inactive:
            placeholders = ",".join("?" for _ in filters.status)
            clauses.append(f"{alias}.status IN ({placeholders})")
            params.extend(filters.status)
        if filters.memory_ids:
            placeholders = ",".join("?" for _ in filters.memory_ids)
            clauses.append(f"{alias}.id IN ({placeholders})")
            params.extend(filters.memory_ids)
        if filters.time_range.get("from"):
            clauses.append(f"{alias}.created_at >= ?")
            params.append(filters.time_range["from"])
        if filters.time_range.get("to"):
            clauses.append(f"{alias}.created_at <= ?")
            params.append(filters.time_range["to"])
        if filters.scene:
            clauses.append(f"{alias}.scene = ?")
            params.append(filters.scene)
        if filters.task:
            clauses.append(f"{alias}.task = ?")
            params.append(filters.task)
        if filters.event_type:
            clauses.append(f"{alias}.event_type = ?")
            params.append(filters.event_type)
        if filters.source:
            clauses.append(f"{alias}.source = ?")
            params.append(filters.source)
        if filters.confidence_min > 0:
            clauses.append(f"{alias}.confidence >= ?")
            params.append(filters.confidence_min)
        if filters.sensitivity_max:
            allowed_sensitivity = [
                name for name, rank in SENSITIVITY_RANK.items()
                if rank <= SENSITIVITY_RANK.get(filters.sensitivity_max, 0)
            ]
            placeholders = ",".join("?" for _ in allowed_sensitivity)
            clauses.append(f"{alias}.sensitivity IN ({placeholders})")
            params.extend(allowed_sensitivity)
        if filters.topic:
            clauses.append(
                f"({alias}.summary LIKE ? OR {alias}.content LIKE ? OR EXISTS ("
                f"SELECT 1 FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id "
                f"WHERE mt.memory_id={alias}.id AND t.status='active' "
                f"AND (t.tag LIKE ? OR (t.namespace || ':' || t.tag) LIKE ?)))"
            )
            like = f"%{filters.topic}%"
            params.extend([like, like, like, like])
        for entity in filters.entities[:12]:
            clauses.append(
                f"EXISTS (SELECT 1 FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id "
                f"WHERE mt.memory_id={alias}.id AND t.status='active' "
                f"AND t.namespace IN ('entity','topic','free') "
                f"AND (t.tag=? OR t.tag LIKE ?))"
            )
            params.extend([entity, f"%{entity}%"])
        for tag in filters.tags_any[:16]:
            combined, bare = _tag_query_values(tag)
            clauses.append(
                f"EXISTS (SELECT 1 FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id "
                f"WHERE mt.memory_id={alias}.id AND t.status='active' "
                f"AND ((t.namespace || ':' || t.tag)=? OR t.tag=?))"
            )
            params.extend([combined, bare])
        for tag in filters.tags_all[:16]:
            combined, bare = _tag_query_values(tag)
            clauses.append(
                f"EXISTS (SELECT 1 FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id "
                f"WHERE mt.memory_id={alias}.id AND t.status='active' "
                f"AND ((t.namespace || ':' || t.tag)=? OR t.tag=?))"
            )
            params.extend([combined, bare])
        if filters.emotion:
            emotion = filters.emotion.lower()
            if emotion in {"positive", "正向", "开心"}:
                clauses.append(f"{alias}.emotion_valence >= 0.2")
            elif emotion in {"negative", "负向", "难过"}:
                clauses.append(f"{alias}.emotion_valence <= -0.2")
            elif emotion in {"high", "high_arousal", "强烈"}:
                clauses.append(f"{alias}.emotion_arousal >= 0.55")
            elif emotion in {"low", "low_arousal", "平静"}:
                clauses.append(f"{alias}.emotion_arousal < 0.3")
        return clauses, params

    @staticmethod
    def _where(clauses: list[str]) -> str:
        return " AND ".join(f"({clause})" for clause in clauses if clause)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    async def create_memory(self, record: MemoryRecord, tags: Iterable[TagInput] = ()) -> str:
        tag_list = list(tags)
        if not record.id:
            record.id = new_memory_id()
        if not record.content_hash:
            import hashlib

            record.content_hash = hashlib.sha256(
                normalize_text(record.content).lower().encode("utf-8")
            ).hexdigest()

        def op(conn: sqlite3.Connection) -> str:
            epoch = self._next_write_epoch(conn)
            record.write_epoch = epoch
            record.schema_version = self.schema_version
            record.updated_at = now_iso()
            conn.execute(
                """
                INSERT INTO memories(
                    id, content, summary, memory_type, owner_scope, owner_id,
                    source, source_ref, confidence, sensitivity, importance,
                    emotion_valence, emotion_arousal, scene, task, event_type,
                    entities, injectable, content_hash, created_at, updated_at,
                    last_verified_at, last_access_at, lifecycle, compressed_from,
                    compressed_into, status, write_epoch, schema_version,
                    retention_score, pinned, deleted_at, archived_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.id,
                    normalize_text(record.content),
                    normalize_text(record.summary),
                    record.memory_type,
                    record.owner_scope,
                    record.owner_id,
                    record.source,
                    record.source_ref,
                    record.confidence,
                    record.sensitivity,
                    record.importance,
                    record.emotion_valence,
                    record.emotion_arousal,
                    record.scene,
                    record.task,
                    record.event_type,
                    _json_list(record.entities),
                    1 if record.injectable else 0,
                    record.content_hash,
                    record.created_at,
                    record.updated_at,
                    record.last_verified_at,
                    record.last_access_at,
                    record.lifecycle,
                    _json_list(record.compressed_from),
                    record.compressed_into,
                    record.status,
                    record.write_epoch,
                    record.schema_version,
                    record.retention_score,
                    1 if record.pinned else 0,
                    record.deleted_at,
                    record.archived_at,
                ),
            )
            self._replace_tags(conn, record.id, tag_list)
            self._sync_fts(conn, record.id)
            return record.id

        return await self._write(op)

    async def update_memory(
        self,
        memory_id: str,
        fields: Mapping[str, Any],
        *,
        tags: Iterable[TagInput] | None = None,
    ) -> bool:
        allowed = {
            "content", "summary", "memory_type", "owner_scope", "owner_id",
            "source", "source_ref", "confidence", "sensitivity", "importance",
            "emotion_valence", "emotion_arousal", "scene", "task", "event_type",
            "entities", "injectable", "content_hash", "last_verified_at",
            "last_access_at", "lifecycle", "compressed_from", "compressed_into",
            "status", "retention_score", "pinned", "deleted_at", "archived_at",
        }
        clean: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in {"entities", "compressed_from"}:
                value = _json_list(value or [])
            elif key in {"injectable", "pinned"}:
                value = 1 if value else 0
            elif isinstance(value, str) and key in {"content", "summary"}:
                value = normalize_text(value)
            clean[key] = value
        tag_list = list(tags) if tags is not None else None
        fts_fields = {"content", "summary", "lifecycle", "status"}
        sync_fts = tags is not None or bool(set(clean) & fts_fields)

        def op(conn: sqlite3.Connection) -> bool:
            row = conn.execute("SELECT id FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return False
            epoch = self._next_write_epoch(conn)
            clean["updated_at"] = now_iso()
            clean["write_epoch"] = epoch
            assignments = ", ".join(f"{key}=?" for key in clean)
            conn.execute(
                f"UPDATE memories SET {assignments} WHERE id=?",
                [*clean.values(), memory_id],
            )
            if tag_list is not None:
                self._replace_tags(conn, memory_id, tag_list)
            if sync_fts:
                self._sync_fts(conn, memory_id)
            return True

        return await self._write(op)

    async def merge_memory(
        self,
        memory_id: str,
        *,
        content: str,
        summary: str,
        confidence: float,
        importance: float,
        tags: Iterable[TagInput] | None = None,
    ) -> bool:
        return await self.update_memory(
            memory_id,
            {
                "content": content,
                "summary": summary,
                "confidence": confidence,
                "importance": importance,
                "last_verified_at": now_iso(),
            },
            tags=tags,
        )

    def _replace_tags(
        self, conn: sqlite3.Connection, memory_id: str, tags: Iterable[TagInput]
    ) -> None:
        conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (memory_id,))
        for tag in tags:
            namespace = normalize_text(str(tag.namespace)).lower()[:40] or "free"
            value = normalize_text(str(tag.tag)).lower()[:80]
            if not value:
                continue
            weight = max(0.0, min(1.0, float(tag.weight)))
            conn.execute(
                """
                INSERT INTO tags(namespace, tag, weight, status, version, created_at, updated_at)
                VALUES (?,?,?, 'active', 1, ?, ?)
                ON CONFLICT(namespace, tag) DO UPDATE SET
                    weight=excluded.weight,
                    status='active',
                    version=tags.version+1,
                    updated_at=excluded.updated_at
                """,
                (namespace, value, weight, now_iso(), now_iso()),
            )
            tag_row = conn.execute(
                "SELECT id FROM tags WHERE namespace=? AND tag=?", (namespace, value)
            ).fetchone()
            if tag_row:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_tags(memory_id, tag_id, weight) VALUES(?,?,?)",
                    (memory_id, int(tag_row[0]), weight),
                )

    def _sync_fts(self, conn: sqlite3.Connection, memory_id: str) -> None:
        row = conn.execute(
            "SELECT id, content, summary, lifecycle, status FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
        if not row or row["lifecycle"] != "active" or row["status"] not in {"active", "disputed"}:
            return
        tags_text = conn.execute(
            "SELECT COALESCE(GROUP_CONCAT(t.namespace || ':' || t.tag, ' '), '') "
            "FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id WHERE mt.memory_id=?",
            (memory_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO memories_fts(memory_id, content, summary, tags_text) VALUES(?,?,?,?)",
            (memory_id, row["content"], row["summary"], tags_text),
        )

    async def set_pinned(self, memory_id: str, pinned: bool) -> bool:
        return await self.update_memory(memory_id, {"pinned": pinned})

    async def set_status(self, memory_id: str, status: str) -> bool:
        return await self.update_memory(memory_id, {"status": status})

    async def set_lifecycle(
        self,
        memory_id: str,
        lifecycle: str,
        *,
        compressed_into: str = "",
        write_cold_reason: str = "",
    ) -> bool:
        now = now_iso()
        fields: dict[str, Any] = {"lifecycle": lifecycle}
        if lifecycle == "archived":
            fields["archived_at"] = now
        if lifecycle == "soft_deleted":
            fields["deleted_at"] = now
        if lifecycle == "active":
            fields["deleted_at"] = None
            fields["archived_at"] = None
        if compressed_into:
            fields["compressed_into"] = compressed_into

        def op(conn: sqlite3.Connection) -> bool:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return False
            if write_cold_reason:
                conn.execute(
                    "INSERT INTO cold_memories(memory_id, content, reason, created_at) "
                    "VALUES(?,?,?,?)",
                    (memory_id, row["content"], write_cold_reason, now),
                )
            return True

        existed = await self._write(op)
        if not existed:
            return False
        return await self.update_memory(memory_id, fields)

    async def archive_memory(self, memory_id: str, reason: str = "archive") -> bool:
        return await self.set_lifecycle(
            memory_id, "archived", write_cold_reason=reason
        )

    async def soft_delete_memory(self, memory_id: str, reason: str = "soft_delete") -> bool:
        return await self.set_lifecycle(
            memory_id, "soft_deleted", write_cold_reason=reason
        )

    async def revive_memory(self, memory_id: str, confidence: float) -> bool:
        ok = await self.update_memory(
            memory_id,
            {
                "lifecycle": "active",
                "status": "active",
                "deleted_at": None,
                "archived_at": None,
                "confidence": confidence,
                "last_verified_at": now_iso(),
            },
        )
        return ok

    async def hard_delete_memory(self, memory_id: str) -> dict[str, Any]:
        """硬删除并返回删除前的行；审计由调用方写入。"""

        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return {}
            data = dict(row)
            tag_rows = conn.execute(
                "SELECT tag_id FROM memory_tags WHERE memory_id=?", (memory_id,)
            ).fetchall()
            tag_ids = [int(tag_row[0]) for tag_row in tag_rows]
            conn.execute("DELETE FROM memory_tags WHERE memory_id=?", (memory_id,))
            if tag_ids:
                placeholders = ",".join("?" for _ in tag_ids)
                conn.execute(
                    f"DELETE FROM tags WHERE id IN ({placeholders}) AND NOT EXISTS ("
                    "SELECT 1 FROM memory_tags mt WHERE mt.tag_id=tags.id)",
                    tag_ids,
                )
            conn.execute("DELETE FROM conflicts WHERE memory_a=? OR memory_b=?", (memory_id, memory_id))
            conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))
            conn.execute("DELETE FROM cold_memories WHERE memory_id=?", (memory_id,))
            conn.execute("DELETE FROM memories_fts WHERE memory_id=?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            # 保留 trace/access 的审计骨架，但清除可能含正文的注入文本。
            conn.execute(
                "UPDATE recall_traces SET injected_text='' "
                "WHERE injected_ids LIKE ?",
                (f'%"{memory_id}"%',),
            )
            # access_logs 保留用于审计；memory_id 不再是外键，避免删除审计。
            return data

        return await self._write(op)

    async def add_cold(self, memory_id: str, content: str, reason: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO cold_memories(memory_id, content, reason, created_at) VALUES(?,?,?,?)",
                (memory_id, content, reason, now_iso()),
            )

        await self._write(op)

    async def get_cold(self, memory_id: str, limit: int = 5) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM cold_memories WHERE memory_id=? ORDER BY id DESC LIMIT ?",
                (memory_id, limit),
            ).fetchall()
            return [dict(row) for row in rows]

        return await self._read(op)

    # ------------------------------------------------------------------
    # 单条 / 列表读取
    # ------------------------------------------------------------------
    async def get_memory(self, memory_id: str, *, include_deleted: bool = True) -> dict[str, Any]:
        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if not row:
                return {}
            data = decode_memory_row(row)
            if not include_deleted and data.get("lifecycle") in {"soft_deleted", "archived"}:
                return {}
            return data

        return await self._read(op)

    async def get_memories_by_ids(
        self,
        memory_ids: Iterable[str],
        context: AccessContext | None = None,
        filters: SearchFilters | None = None,
        *,
        include_inactive: bool = True,
    ) -> list[dict[str, Any]]:
        ids = unique_strings(memory_ids, limit=200)
        if not ids:
            return []
        filters = filters or SearchFilters(lifecycle=[], status=[])
        placeholders = ",".join("?" for _ in ids)

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            clauses = [f"m.id IN ({placeholders})"]
            params: list[Any] = list(ids)
            if context is not None:
                owner_sql, owner_params = self._owner_clause(context, filters, "m")
                clauses.append(owner_sql)
                params.extend(owner_params)
            if not include_inactive:
                clauses.append("m.lifecycle='active'")
                clauses.append("m.status IN ('active','disputed')")
            sql = f"SELECT m.* FROM memories m WHERE {self._where(clauses)}"
            rows = conn.execute(sql, params).fetchall()
            by_id = {row["id"]: decode_memory_row(row) for row in rows}
            return [by_id[item] for item in ids if item in by_id]

        return await self._read(op)

    async def list_memories(
        self,
        context: AccessContext,
        filters: SearchFilters | None = None,
        *,
        limit: int = 100,
        include_inactive: bool = False,
        order_by: str = "pinned DESC, updated_at DESC",
    ) -> list[dict[str, Any]]:
        filters = filters or SearchFilters()
        limit = max(1, min(500, int(limit)))
        safe_order = order_by if order_by in {
            "pinned DESC, updated_at DESC",
            "updated_at DESC",
            "created_at DESC",
            "last_access_at DESC",
            "retention_score DESC",
        } else "pinned DESC, updated_at DESC"

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            clauses, params = self._filter_clauses(
                context, filters, include_inactive=include_inactive, alias="m"
            )
            sql = (
                f"SELECT m.* FROM memories m WHERE {self._where(clauses)} "
                f"ORDER BY {safe_order} LIMIT ? OFFSET ?"
            )
            rows = conn.execute(
                sql, [*params, limit, max(0, filters.offset)]
            ).fetchall()
            return [decode_memory_row(row) for row in rows]

        return await self._read(op)

    async def find_by_hash(
        self, owner_scope: str, owner_id: str, content_hash: str
    ) -> dict[str, Any]:
        if not content_hash:
            return {}

        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM memories WHERE owner_scope=? AND owner_id=? "
                "AND content_hash=? AND lifecycle='active' "
                "ORDER BY updated_at DESC LIMIT 1",
                (owner_scope, owner_id, content_hash),
            ).fetchone()
            return decode_memory_row(row) if row else {}

        return await self._read(op)

    async def find_similar_memories(
        self,
        owner_scope: str,
        owner_id: str,
        memory_type: str,
        content: str,
        *,
        exclude_id: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        terms = [token for token in tokenize(content) if len(token) >= 2][:6]
        like_clauses = " OR ".join("content LIKE ? OR summary LIKE ?" for _ in terms)
        params: list[Any] = []
        for term in terms:
            params.extend([f"%{term}%", f"%{term}%"])

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            clauses = [
                "owner_scope=?", "owner_id=?", "memory_type=?",
                "lifecycle='active'", "status IN ('active','disputed')",
            ]
            sql_params: list[Any] = [owner_scope, owner_id, memory_type]
            if exclude_id:
                clauses.append("id<>?")
                sql_params.append(exclude_id)
            if like_clauses:
                clauses.append(f"({like_clauses})")
                sql_params.extend(params)
            sql = (
                "SELECT * FROM memories WHERE " + " AND ".join(clauses)
                + " ORDER BY pinned DESC, updated_at DESC LIMIT ?"
            )
            sql_params.append(max(1, min(100, limit)))
            rows = conn.execute(sql, sql_params).fetchall()
            return [decode_memory_row(row) for row in rows]

        return await self._read(op)

    async def count_memories(self, context: AccessContext, filters: SearchFilters | None = None) -> int:
        filters = filters or SearchFilters(lifecycle=[])

        def op(conn: sqlite3.Connection) -> int:
            clauses, params = self._filter_clauses(context, filters, include_inactive=True)
            row = conn.execute(
                f"SELECT COUNT(*) FROM memories m WHERE {self._where(clauses)}", params
            ).fetchone()
            return int(row[0]) if row else 0

        return await self._read(op)

    # ------------------------------------------------------------------
    # FTS / LIKE / 标签 / 实体
    # ------------------------------------------------------------------
    def _fts_match_query(self, query: str) -> str:
        normalized = normalize_text(query)
        if self.fts_mode == "trigram":
            # trigram 索引必须用至少 3 个字符的查询项；中文按 3-gram 补充。
            import re

            terms: list[str] = []
            for word in re.findall(r"[A-Za-z0-9_]{3,}", normalized):
                terms.append(word.lower())
            for run in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]{3,}", normalized):
                if len(run) <= 12:
                    terms.append(run)
                terms.extend(run[i:i + 3] for i in range(max(0, len(run) - 2)))
        else:
            terms = [token for token in tokenize(query) if len(token) >= 2]
        terms = unique_strings(terms, limit=16)
        return " OR ".join(f'"{term}"' for term in terms)

    async def search_fts(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        *,
        limit: int = 40,
    ) -> list[tuple[str, float]]:
        match = self._fts_match_query(query)
        if not match:
            return []

        def op(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            if self.fts_mode not in {"trigram", "unicode61"}:
                return []
            clauses, params = self._filter_clauses(context, filters, alias="m")
            clauses.append("memories_fts MATCH ?")
            params.append(match)
            sql = (
                "SELECT f.memory_id AS memory_id, bm25(memories_fts) AS rank "
                "FROM memories_fts f JOIN memories m ON m.id=f.memory_id "
                f"WHERE {self._where(clauses)} ORDER BY rank LIMIT ?"
            )
            params.append(max(1, min(200, limit)))
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
            return [(str(row["memory_id"]), float(row["rank"])) for row in rows]

        return await self._read(op)

    async def search_like(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        *,
        limit: int = 30,
    ) -> list[tuple[str, float]]:
        raw = normalize_text(query)
        if not raw:
            return []
        like = f"%{raw}%"

        def op(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            clauses, params = self._filter_clauses(context, filters, alias="m")
            clauses.append(
                "(m.content LIKE ? OR m.summary LIKE ? OR m.task LIKE ? OR m.scene LIKE ?)"
            )
            params.extend([like, like, like, like])
            sql = (
                f"SELECT m.id FROM memories m WHERE {self._where(clauses)} "
                "ORDER BY m.pinned DESC, m.updated_at DESC LIMIT ?"
            )
            params.append(max(1, min(100, limit)))
            rows = conn.execute(sql, params).fetchall()
            return [(str(row["id"]), 0.8) for row in rows]

        return await self._read(op)

    async def search_by_entities(
        self,
        entities: Iterable[str],
        context: AccessContext,
        filters: SearchFilters,
        *,
        limit: int = 30,
    ) -> list[tuple[str, float]]:
        values = unique_strings(entities, limit=12)
        if not values:
            return []

        def op(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            clauses, params = self._filter_clauses(context, filters, alias="m")
            placeholders = ",".join("?" for _ in values)
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_tags mt JOIN tags t ON t.id=mt.tag_id "
                "WHERE mt.memory_id=m.id AND t.status='active' "
                "AND t.namespace IN ('entity','topic','free') "
                f"AND t.tag IN ({placeholders}))"
            )
            params.extend(values)
            sql = (
                f"SELECT DISTINCT m.id FROM memories m WHERE {self._where(clauses)} "
                "ORDER BY m.updated_at DESC LIMIT ?"
            )
            params.append(max(1, min(100, limit)))
            rows = conn.execute(sql, params).fetchall()
            return [(str(row["id"]), 0.7) for row in rows]

        return await self._read(op)

    async def merge_tags(self, source: str, target: str) -> int:
        """把 source 标签合并到 target，保留 source 的 version/status 作为历史。"""
        if ":" not in source or ":" not in target:
            raise ValueError("标签必须使用 namespace:tag 格式")

        def op(conn: sqlite3.Connection) -> int:
            source_ns, source_tag = source.split(":", 1)
            target_ns, target_tag = target.split(":", 1)
            source_row = conn.execute(
                "SELECT id FROM tags WHERE namespace=? AND tag=?",
                (source_ns, source_tag),
            ).fetchone()
            target_row = conn.execute(
                "SELECT id FROM tags WHERE namespace=? AND tag=?",
                (target_ns, target_tag),
            ).fetchone()
            if not source_row or not target_row:
                return 0
            source_id, target_id = int(source_row[0]), int(target_row[0])
            if source_id == target_id:
                return 0
            memory_rows = conn.execute(
                "SELECT memory_id, weight FROM memory_tags WHERE tag_id=?",
                (source_id,),
            ).fetchall()
            memory_ids = [str(row["memory_id"]) for row in memory_rows]
            conn.execute(
                "INSERT OR IGNORE INTO memory_tags(memory_id, tag_id, weight) "
                "SELECT memory_id, ?, weight FROM memory_tags WHERE tag_id=?",
                (target_id, source_id),
            )
            conn.execute("DELETE FROM memory_tags WHERE tag_id=?", (source_id,))
            conn.execute(
                "UPDATE tags SET status='merged', version=version+1, updated_at=? "
                "WHERE id=?",
                (now_iso(), source_id),
            )
            for memory_id in memory_ids:
                self._sync_fts(conn, memory_id)
            return len(memory_ids)

        return await self._write(op)

    async def prune_orphan_tags(self) -> int:
        def op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "UPDATE tags SET status='inactive', version=version+1, updated_at=? "
                "WHERE status='active' AND NOT EXISTS ("
                "SELECT 1 FROM memory_tags mt WHERE mt.tag_id=tags.id)",
                (now_iso(),),
            )
            return int(cursor.rowcount or 0)

        return await self._write(op)

    async def tag_stats(self, *, limit: int = 100) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT t.namespace, t.tag, t.status, t.version, COUNT(mt.memory_id) AS usage "
                "FROM tags t LEFT JOIN memory_tags mt ON mt.tag_id=t.id "
                "GROUP BY t.id ORDER BY usage DESC, t.updated_at DESC LIMIT ?",
                (max(1, min(500, limit)),),
            ).fetchall()
            return [dict(row) for row in rows]

        return await self._read(op)

    async def get_tags_map(self, memory_ids: Iterable[str]) -> dict[str, list[str]]:
        ids = unique_strings(memory_ids, limit=200)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)

        def op(conn: sqlite3.Connection) -> dict[str, list[str]]:
            rows = conn.execute(
                "SELECT mt.memory_id, t.namespace, t.tag FROM memory_tags mt "
                "JOIN tags t ON t.id=mt.tag_id WHERE mt.memory_id IN "
                f"({placeholders}) AND t.status='active' "
                "ORDER BY t.namespace, t.tag",
                ids,
            ).fetchall()
            result: dict[str, list[str]] = {memory_id: [] for memory_id in ids}
            for row in rows:
                result.setdefault(row["memory_id"], []).append(
                    f"{row['namespace']}:{row['tag']}"
                )
            return result

        return await self._read(op)

    # ------------------------------------------------------------------
    # 冲突
    # ------------------------------------------------------------------
    async def add_conflict(
        self,
        memory_a: str,
        memory_b: str,
        conflict_type: str,
        *,
        resolution: str = "",
    ) -> str:
        conflict_id = new_conflict_id()

        def op(conn: sqlite3.Connection) -> str:
            conn.execute(
                "INSERT INTO conflicts(id, memory_a, memory_b, conflict_type, status, "
                "resolution, created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    conflict_id,
                    memory_a,
                    memory_b,
                    conflict_type,
                    "open",
                    resolution,
                    now_iso(),
                ),
            )
            return conflict_id

        return await self._write(op)

    async def get_conflicts_for(
        self, memory_ids: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]:
        ids = unique_strings(memory_ids, limit=200)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)

        def op(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
            rows = conn.execute(
                "SELECT * FROM conflicts WHERE (memory_a IN "
                f"({placeholders}) OR memory_b IN ({placeholders})) "
                "ORDER BY created_at DESC",
                [*ids, *ids],
            ).fetchall()
            result: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                data = dict(row)
                for key in ("memory_a", "memory_b"):
                    result.setdefault(data[key], []).append(data)
            return result

        return await self._read(op)

    async def resolve_conflicts_for(
        self, memory_id: str, resolution: str, *, status: str = "resolved"
    ) -> int:
        def op(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                "UPDATE conflicts SET status=?, resolution=?, resolved_at=? "
                "WHERE (memory_a=? OR memory_b=?) AND status='open'",
                (status, resolution, now_iso(), memory_id, memory_id),
            )
            return int(cursor.rowcount or 0)

        return await self._write(op)

    # ------------------------------------------------------------------
    # 访问统计 / trace / 审计
    # ------------------------------------------------------------------
    async def record_access(
        self,
        memory_id: str,
        caller: str,
        mode: str,
        event_type: str,
        *,
        session_id: str = "",
        trace_id: str = "",
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        await self.record_access_batch(
            [
                {
                    "memory_id": memory_id,
                    "caller": caller,
                    "mode": mode,
                    "event_type": event_type,
                    "session_id": session_id,
                    "trace_id": trace_id,
                    "meta": dict(meta or {}),
                }
            ]
        )

    async def record_access_batch(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        timestamp = now_iso()

        def op(conn: sqlite3.Connection) -> None:
            for entry in entries:
                memory_id = str(entry.get("memory_id") or "")
                if not memory_id:
                    continue
                conn.execute(
                    "INSERT INTO access_logs(memory_id, caller, mode, event_type, "
                    "session_id, trace_id, created_at, meta) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        memory_id,
                        str(entry.get("caller") or "system"),
                        str(entry.get("mode") or "search"),
                        str(entry.get("event_type") or "candidate_recalled"),
                        str(entry.get("session_id") or ""),
                        str(entry.get("trace_id") or ""),
                        timestamp,
                        dump(entry.get("meta") or {}),
                    ),
                )
                conn.execute(
                    "UPDATE memories SET last_access_at=? WHERE id=?",
                    (timestamp, memory_id),
                )

        await self._write(op)

    async def get_access_stats(
        self, memory_ids: Iterable[str]
    ) -> dict[str, AccessStats]:
        ids = unique_strings(memory_ids, limit=200)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)

        def op(conn: sqlite3.Connection) -> dict[str, AccessStats]:
            from utils.time_utils import days_ago

            cutoff = days_ago(30)
            rows = conn.execute(
                "SELECT memory_id, event_type, COUNT(*) AS total, "
                "COUNT(DISTINCT session_id) AS sessions, MAX(created_at) AS last_at, "
                "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS recent "
                "FROM access_logs WHERE memory_id IN "
                f"({placeholders}) GROUP BY memory_id, event_type",
                [cutoff, *ids],
            ).fetchall()
            result: dict[str, AccessStats] = {}
            for row in rows:
                memory_id = str(row["memory_id"])
                stats = result.setdefault(memory_id, AccessStats())
                stats.counts[str(row["event_type"])] = int(row["total"])
                stats.recent_counts[str(row["event_type"])] = int(row["recent"] or 0)
                stats.session_counts[str(row["event_type"])] = int(row["sessions"])
                last_at = row["last_at"]
                if last_at and (stats.last_access_at is None or last_at > stats.last_access_at):
                    stats.last_access_at = str(last_at)
            return result

        return await self._read(op)

    async def aggregate_access(self, memory_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        stats = await self.get_access_stats(memory_ids)
        return {key: value.to_dict() for key, value in stats.items()}

    async def record_trace(self, trace: Mapping[str, Any]) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO recall_traces(
                    trace_id, session_id, owner_scope, owner_id, query,
                    rewritten_query, candidate_count, candidates, gate_decision,
                    injected_text, injected_ids, used, used_ids, feedback,
                    latency_ms, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    candidates=excluded.candidates,
                    gate_decision=excluded.gate_decision,
                    injected_text=excluded.injected_text,
                    injected_ids=excluded.injected_ids,
                    used=excluded.used,
                    used_ids=excluded.used_ids,
                    feedback=excluded.feedback,
                    latency_ms=excluded.latency_ms
                """,
                (
                    str(trace.get("trace_id") or ""),
                    str(trace.get("session_id") or ""),
                    str(trace.get("owner_scope") or ""),
                    str(trace.get("owner_id") or ""),
                    str(trace.get("query") or "")[:2000],
                    str(trace.get("rewritten_query") or "")[:2000],
                    int(trace.get("candidate_count") or 0),
                    dump(trace.get("candidates") or []),
                    dump(trace.get("gate_decision") or {}),
                    str(trace.get("injected_text") or "")[:8000],
                    dump(trace.get("injected_ids") or []),
                    1 if trace.get("used") else 0,
                    dump(trace.get("used_ids") or []),
                    str(trace.get("feedback") or "")[:500],
                    int(trace.get("latency_ms") or 0),
                    str(trace.get("created_at") or now_iso()),
                ),
            )

        await self._write(op)

    async def mark_trace_used(
        self, trace_id: str, memory_ids: Iterable[str], *, feedback: str = ""
    ) -> dict[str, Any]:
        ids = unique_strings(memory_ids, limit=50)

        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            row = conn.execute(
                "SELECT * FROM recall_traces WHERE trace_id=?", (trace_id,)
            ).fetchone()
            if not row:
                return {}
            data = dict(row)
            injected = set(_safe_json(data.get("injected_ids"), []))
            used = set(_safe_json(data.get("used_ids"), []))
            used.update(ids & injected if injected else ids)
            conn.execute(
                "UPDATE recall_traces SET used=1, used_ids=?, feedback=? WHERE trace_id=?",
                (dump(sorted(used)), feedback[:500], trace_id),
            )
            data["used"] = 1
            data["used_ids"] = sorted(used)
            data["feedback"] = feedback
            return data

        return await self._write(op)

    async def update_trace_feedback(self, trace_id: str, feedback: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE recall_traces SET feedback=? WHERE trace_id=?",
                (str(feedback)[:500], trace_id),
            )

        await self._write(op)

    async def get_recent_traces(
        self, session_id: str = "", *, limit: int = 20
    ) -> list[dict[str, Any]]:
        limit = max(1, min(200, limit))

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM recall_traces WHERE session_id=? "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM recall_traces ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                for key in ("candidates", "gate_decision", "injected_ids", "used_ids"):
                    data[key] = _safe_json(data.get(key), [] if key.endswith("ids") or key == "candidates" else {})
                result.append(data)
            return result

        return await self._read(op)

    async def find_trace_for_memory(
        self, memory_id: str, session_id: str = "", *, limit: int = 100
    ) -> dict[str, Any]:
        traces = await self.get_recent_traces(session_id, limit=limit)
        for trace in traces:
            candidates = trace.get("candidates") or []
            for candidate in candidates:
                if isinstance(candidate, Mapping) and candidate.get("memory_id") == memory_id:
                    return trace
            if memory_id in (trace.get("injected_ids") or []):
                return trace
        return {}

    async def add_audit(
        self,
        *,
        actor: str,
        action: str,
        target: str = "",
        detail: Mapping[str, Any] | None = None,
        trace_id: str = "",
    ) -> str:
        audit_id = new_audit_id()

        def op(conn: sqlite3.Connection) -> str:
            conn.execute(
                "INSERT INTO audit_logs(id, actor, action, target, detail, created_at, trace_id) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    audit_id,
                    actor,
                    action,
                    target,
                    dump(detail or {}),
                    now_iso(),
                    trace_id,
                ),
            )
            return audit_id

        return await self._write(op)

    async def list_audit(
        self, *, actor: str = "", limit: int = 30
    ) -> list[dict[str, Any]]:
        limit = max(1, min(200, limit))

        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            if actor:
                rows = conn.execute(
                    "SELECT * FROM audit_logs WHERE actor=? ORDER BY created_at DESC LIMIT ?",
                    (actor, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
            result = []
            for row in rows:
                data = dict(row)
                data["detail"] = _safe_json(data.get("detail"), {})
                result.append(data)
            return result

        return await self._read(op)

    # ------------------------------------------------------------------
    # 向量
    # ------------------------------------------------------------------
    async def upsert_vector(self, memory_id: str, vector: list[float]) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO memory_vectors(memory_id, vector, dim, updated_at) "
                "VALUES(?,?,?,?) ON CONFLICT(memory_id) DO UPDATE SET "
                "vector=excluded.vector, dim=excluded.dim, updated_at=excluded.updated_at",
                (
                    memory_id,
                    dump([round(float(item), 6) for item in vector]),
                    len(vector),
                    now_iso(),
                ),
            )

        await self._write(op)

    async def delete_vector(self, memory_id: str) -> None:
        def op(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM memory_vectors WHERE memory_id=?", (memory_id,))

        await self._write(op)

    async def vector_search(
        self,
        embedding: list[float],
        context: AccessContext,
        filters: SearchFilters,
        *,
        limit: int = 30,
    ) -> list[tuple[str, float]]:
        def op(conn: sqlite3.Connection) -> list[tuple[str, float]]:
            clauses, params = self._filter_clauses(context, filters, alias="m")
            sql = (
                "SELECT v.memory_id AS memory_id, v.vector AS vector FROM memory_vectors v "
                "JOIN memories m ON m.id=v.memory_id "
                f"WHERE {self._where(clauses)} ORDER BY m.updated_at DESC LIMIT ?"
            )
            params.append(_MAX_VECTOR_SCAN)
            rows = conn.execute(sql, params).fetchall()
            scored: list[tuple[str, float]] = []
            for row in rows:
                vector = _safe_json(row["vector"], [])
                if not isinstance(vector, list):
                    continue
                score = cosine_similarity(embedding, vector)
                if score > 0:
                    scored.append((str(row["memory_id"]), (score + 1.0) / 2.0))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[: max(1, min(100, limit))]

        return await self._read(op)

    # ------------------------------------------------------------------
    # 维护 / 概览
    # ------------------------------------------------------------------
    async def active_memories_for_maintenance(self, limit: int = 1000) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM memories WHERE lifecycle='active' "
                "ORDER BY updated_at ASC LIMIT ?",
                (max(1, min(5000, limit)),),
            ).fetchall()
            return [decode_memory_row(row) for row in rows]

        return await self._read(op)

    async def old_event_memories(
        self, older_than_iso: str, *, limit: int = 1000
    ) -> list[dict[str, Any]]:
        def op(conn: sqlite3.Connection) -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type='event' AND lifecycle='active' "
                "AND pinned=0 AND updated_at < ? ORDER BY updated_at ASC LIMIT ?",
                (older_than_iso, max(1, min(5000, limit))),
            ).fetchall()
            return [decode_memory_row(row) for row in rows]

        return await self._read(op)

    async def access_event_totals(self, context: AccessContext) -> dict[str, int]:
        """统计当前会话可见记忆的访问事件总量（用于纠错率等指标）。"""

        def op(conn: sqlite3.Connection) -> dict[str, int]:
            clauses, params = self._filter_clauses(
                context, SearchFilters(lifecycle=[], status=[]), include_inactive=True
            )
            rows = conn.execute(
                "SELECT al.event_type, COUNT(*) AS n FROM access_logs al "
                "JOIN memories m ON m.id=al.memory_id "
                f"WHERE {self._where(clauses)} GROUP BY al.event_type",
                params,
            ).fetchall()
            return {str(row["event_type"]): int(row["n"]) for row in rows}

        return await self._read(op)

    async def audit_action_counts(self, actions: Iterable[str]) -> dict[str, int]:
        values = unique_strings(actions, limit=30)
        if not values:
            return {}
        placeholders = ",".join("?" for _ in values)

        def op(conn: sqlite3.Connection) -> dict[str, int]:
            rows = conn.execute(
                "SELECT action, COUNT(*) AS n FROM audit_logs WHERE action IN "
                f"({placeholders}) GROUP BY action",
                values,
            ).fetchall()
            return {str(row["action"]): int(row["n"]) for row in rows}

        return await self._read(op)

    async def stats_overview(self, context: AccessContext) -> dict[str, Any]:
        def op(conn: sqlite3.Connection) -> dict[str, Any]:
            clauses, params = self._filter_clauses(
                context, SearchFilters(lifecycle=[], status=[]), include_inactive=True
            )
            where = self._where(clauses)
            total = conn.execute(
                f"SELECT COUNT(*) FROM memories m WHERE {where}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT memory_type, lifecycle, status, COUNT(*) AS n FROM memories m "
                f"WHERE {where} GROUP BY memory_type, lifecycle, status",
                params,
            ).fetchall()
            by_type: dict[str, int] = {}
            by_lifecycle: dict[str, int] = {}
            by_status: dict[str, int] = {}
            for row in rows:
                n = int(row["n"])
                by_type[str(row["memory_type"])] = by_type.get(str(row["memory_type"]), 0) + n
                by_lifecycle[str(row["lifecycle"])] = by_lifecycle.get(str(row["lifecycle"]), 0) + n
                by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + n
            return {
                "total": int(total),
                "by_type": by_type,
                "by_lifecycle": by_lifecycle,
                "by_status": by_status,
            }

        return await self._read(op)
