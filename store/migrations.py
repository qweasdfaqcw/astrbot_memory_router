"""SQLite schema 与迁移。

约定：
- 任何结构变化都新增一个 ``(version, function)``，绝不能修改已发布的迁移；
- 通过 ``PRAGMA user_version`` 记录当前版本；
- ``memories.schema_version`` 记录单条记忆写入时的版本，便于旧记录升级；
- FTS5 使用 trigram tokenizer（对中文友好），不支持时回退 unicode61，
  再失败则退化为普通表和 LIKE 查询，插件仍能工作。
"""

from __future__ import annotations

import sqlite3
from typing import Callable

SCHEMA_VERSION = 1


def _exec(conn: sqlite3.Connection, sql: str) -> None:
    conn.execute(sql)


def create_fts_table(conn: sqlite3.Connection) -> str:
    """创建 FTS5 表并返回实际模式。

    返回值：``trigram``、``unicode61`` 或 ``like``。
    """
    for tokenizer in ("trigram", "unicode61"):
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5("
                "memory_id UNINDEXED, content, summary, tags_text, "
                f"tokenize='{tokenizer}')"
            )
            return tokenizer
        except sqlite3.OperationalError:
            # 已存在或当前 SQLite 不支持该 tokenizer；已存在时继续判断。
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='memories_fts'"
            ).fetchone()
            if exists:
                return tokenizer
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories_fts("
        "memory_id TEXT PRIMARY KEY, content TEXT, summary TEXT, tags_text TEXT)"
    )
    return "like"


def migrate_v1(conn: sqlite3.Connection) -> None:
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            summary TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            owner_scope TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'user',
            source_ref TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.7,
            sensitivity TEXT NOT NULL DEFAULT 'normal',
            importance REAL NOT NULL DEFAULT 0.5,
            emotion_valence REAL NOT NULL DEFAULT 0.0,
            emotion_arousal REAL NOT NULL DEFAULT 0.0,
            scene TEXT NOT NULL DEFAULT '',
            task TEXT NOT NULL DEFAULT '',
            event_type TEXT NOT NULL DEFAULT '',
            entities TEXT NOT NULL DEFAULT '[]',
            injectable INTEGER NOT NULL DEFAULT 1,
            content_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_verified_at TEXT NOT NULL,
            last_access_at TEXT,
            lifecycle TEXT NOT NULL DEFAULT 'active',
            compressed_from TEXT NOT NULL DEFAULT '[]',
            compressed_into TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            write_epoch INTEGER NOT NULL DEFAULT 0,
            schema_version INTEGER NOT NULL DEFAULT 1,
            retention_score REAL NOT NULL DEFAULT 0.0,
            pinned INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT,
            archived_at TEXT
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            tag TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            status TEXT NOT NULL DEFAULT 'active',
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(namespace, tag)
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS memory_tags (
            memory_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (memory_id, tag_id),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            caller TEXT NOT NULL,
            mode TEXT NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT '',
            trace_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            meta TEXT NOT NULL DEFAULT '{}'
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS conflicts (
            id TEXT PRIMARY KEY,
            memory_a TEXT NOT NULL,
            memory_b TEXT NOT NULL,
            conflict_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            resolution TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            resolved_at TEXT
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS recall_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL DEFAULT '',
            owner_scope TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL DEFAULT '',
            query TEXT NOT NULL DEFAULT '',
            rewritten_query TEXT NOT NULL DEFAULT '',
            candidate_count INTEGER NOT NULL DEFAULT 0,
            candidates TEXT NOT NULL DEFAULT '[]',
            gate_decision TEXT NOT NULL DEFAULT '{}',
            injected_text TEXT NOT NULL DEFAULT '',
            injected_ids TEXT NOT NULL DEFAULT '[]',
            used INTEGER NOT NULL DEFAULT 0,
            used_ids TEXT NOT NULL DEFAULT '[]',
            feedback TEXT NOT NULL DEFAULT '',
            latency_ms INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            trace_id TEXT NOT NULL DEFAULT ''
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS cold_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            content TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
    )
    _exec(
        conn,
        """
        CREATE TABLE IF NOT EXISTS memory_vectors (
            memory_id TEXT PRIMARY KEY,
            vector TEXT NOT NULL,
            dim INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
    )
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_scope, owner_id, lifecycle, status)",
        "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type, lifecycle, status)",
        "CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle, status)",
        "CREATE INDEX IF NOT EXISTS idx_tags_lookup ON tags(namespace, tag, status)",
        "CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id, memory_id)",
        "CREATE INDEX IF NOT EXISTS idx_access_memory ON access_logs(memory_id, event_type, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_access_session ON access_logs(session_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_conflicts_pair ON conflicts(memory_a, memory_b)",
        "CREATE INDEX IF NOT EXISTS idx_recall_session ON recall_traces(session_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_cold_memory ON cold_memories(memory_id)",
    ]
    for sql in indexes:
        _exec(conn, sql)

    fts_mode = create_fts_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO settings(key, value) VALUES('fts_mode', ?)", (fts_mode,)
    )
    conn.execute(
        "INSERT OR REPLACE INTO settings(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.execute(
        "INSERT OR IGNORE INTO settings(key, value) VALUES('write_epoch', '0')"
    )


MIGRATIONS: list[tuple[int, Callable[[sqlite3.Connection], None]]] = [
    (1, migrate_v1),
]


def migrate(conn: sqlite3.Connection) -> int:
    """执行所有未应用的迁移，返回目标版本。"""
    current = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    for version, func in MIGRATIONS:
        if version <= current:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            func(conn)
            conn.execute(f"PRAGMA user_version={version}")
            conn.execute(
                "INSERT OR REPLACE INTO settings(key, value) VALUES('schema_version', ?)",
                (str(version),),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return int(conn.execute("PRAGMA user_version").fetchone()[0] or SCHEMA_VERSION)


def rebuild_fts(conn: sqlite3.Connection, mode: str) -> None:
    """从 memories 重建 FTS 索引。"""
    conn.execute("DELETE FROM memories_fts")
    if mode in {"trigram", "unicode61"}:
        conn.execute(
            """
            INSERT INTO memories_fts(memory_id, content, summary, tags_text)
            SELECT m.id, m.content, m.summary, COALESCE(
                (SELECT GROUP_CONCAT(t.namespace || ':' || t.tag, ' ')
                 FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id
                 WHERE mt.memory_id = m.id), ''
            )
            FROM memories m
            WHERE m.lifecycle = 'active' AND m.status IN ('active', 'disputed')
            """
        )
    else:
        conn.execute(
            """
            INSERT INTO memories_fts(memory_id, content, summary, tags_text)
            SELECT m.id, m.content, m.summary, COALESCE(
                (SELECT GROUP_CONCAT(t.namespace || ':' || t.tag, ' ')
                 FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id
                 WHERE mt.memory_id = m.id), ''
            )
            FROM memories m
            WHERE m.lifecycle = 'active' AND m.status IN ('active', 'disputed')
            """
        )
