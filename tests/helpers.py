"""测试公共工具。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MemorySettings  # noqa: E402
from governance.audit import AuditLogger  # noqa: E402
from governance.rate_limit import RateLimiter  # noqa: E402
from memory_tool.service import MemoryToolService  # noqa: E402
from retriever.query_rewrite import QueryRewriter  # noqa: E402
from retriever.retriever import MemoryRetriever  # noqa: E402
from router.reranker import NoopReranker  # noqa: E402
from gate.gate import Gate  # noqa: E402
from lifecycle.manager import LifecycleManager  # noqa: E402
from stats.metrics import MetricsCollector  # noqa: E402
from store.models import AccessContext  # noqa: E402
from store.sqlite_store import SQLiteStore  # noqa: E402
from store.vector_backend import build_vector_backend  # noqa: E402
from writer.extractor import HybridExtractor  # noqa: E402
from writer.pipeline import MemoryWriter  # noqa: E402
from commands.memory_commands import MemoryCommandService  # noqa: E402


class FakeLLM:
    """可替换的小模型适配器。"""

    model_provider_id = "fake"

    def __init__(self, payload: dict | None = None):
        self.payload = payload or {"memories": []}
        self.calls: list[str] = []

    async def extract_json(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        return self.payload

    async def generate_json(self, prompt: str, **kwargs):
        self.calls.append(prompt)
        return self.payload


async def build_stack(tmp_path: Path, settings: MemorySettings | None = None, llm=None):
    settings = settings or MemorySettings(
        auto_maintenance=False,
        auto_archive=False,
        auto_compress=False,
        prefetch_max_latency_ms=1500,
        candidate_limit=30,
        top_k=5,
        extraction_debounce_seconds=0,
    )
    store = SQLiteStore(tmp_path / "memory.db")
    store.open()
    await store.start()
    vector = build_vector_backend(settings.vector_backend, store, settings.embedding_dim)
    extractor = HybridExtractor(
        settings, llm, enabled=llm is not None and settings.extract_enabled
    )
    if llm is None:
        extractor.enabled = False
    writer = MemoryWriter(store, settings, extractor, vector)
    metrics = MetricsCollector()
    rate_limiter = RateLimiter(
        per_minute=settings.rate_limit_per_minute,
        tokens_per_hour=settings.rate_limit_tokens_per_hour,
    )
    retriever = MemoryRetriever(
        store,
        settings,
        vector,
        QueryRewriter(llm, use_llm=False),
    )
    gate = Gate(settings)
    audit = AuditLogger(store)
    lifecycle = LifecycleManager(store, settings, audit, vector)
    tools = MemoryToolService(
        store, settings, retriever, NoopReranker(), gate, metrics, rate_limiter
    )
    commands = MemoryCommandService(
        store, settings, lifecycle, tools, metrics, audit
    )
    return SimpleNamespace(
        settings=settings,
        store=store,
        vector=vector,
        writer=writer,
        retriever=retriever,
        gate=gate,
        audit=audit,
        lifecycle=lifecycle,
        tools=tools,
        commands=commands,
        metrics=metrics,
    )


def user_context(user_id: str = "u1", session_id: str = "s1", **kwargs) -> AccessContext:
    return AccessContext(user_id=user_id, session_id=session_id, **kwargs)
