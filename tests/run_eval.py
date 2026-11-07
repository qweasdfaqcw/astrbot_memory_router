"""离线召回回归评测。

用法：
    python tests/run_eval.py [--dataset tests/eval_dataset.jsonl] [--threshold 0.6]

构造临时 SQLite，写入“场景-应召回记忆”数据，跑 retrieval + gate，输出：
- recall@10：候选召回是否包含期望记忆；
- injection_recall：门控后是否仍包含期望记忆；
- injection_precision：注入中期望记忆占比；
- 平均延迟。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import MemorySettings  # noqa: E402
from gate.gate import Gate  # noqa: E402
from retriever.query_rewrite import QueryRewriter  # noqa: E402
from retriever.retriever import MemoryRetriever  # noqa: E402
from store.models import (  # noqa: E402
    AccessContext,
    MemoryRecord,
    SearchFilters,
    TagInput,
)
from store.sqlite_store import SQLiteStore  # noqa: E402
from store.vector_backend import NullVectorBackend  # noqa: E402


def _tag_inputs(tags: list[str]) -> list[TagInput]:
    result = []
    for raw in tags:
        if ":" in raw:
            namespace, tag = raw.split(":", 1)
        else:
            namespace, tag = "free", raw
        result.append(TagInput(namespace=namespace, tag=tag, weight=0.8))
    return result


async def evaluate(dataset_path: Path) -> dict:
    settings = MemorySettings(
        auto_maintenance=False,
        candidate_limit=30,
        gate_threshold=0.42,
        top_k=5,
        prefetch_max_latency_ms=3000,
    )
    eval_dir = Path(os.environ.get("MEMORY_ROUTER_EVAL_DIR", ROOT / ".eval_tmp"))
    shutil.rmtree(eval_dir, ignore_errors=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(eval_dir / "eval.db")
    store.open()
    await store.start()
    retriever = MemoryRetriever(
        store, settings, NullVectorBackend(), QueryRewriter(None, use_llm=False)
    )
    gate = Gate(settings)
    cases = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # 每个 case 的 owner 可能不同，按 case 重建数据，避免用户/群之间互相干扰。
    total_recall = 0
    total_injection_recall = 0
    injected_total = 0
    expected_injected_total = 0
    latencies: list[int] = []
    details: list[dict] = []
    try:
        for case in cases:
            scope = case.get("owner_scope", "personal")
            owner_id = case.get("owner_id", "eval_u1")
            for memory in case["memories"]:
                record = MemoryRecord(
                    id=memory["id"],
                    content=memory["content"],
                    summary=memory["summary"],
                    memory_type=memory["memory_type"],
                    owner_scope=scope,
                    owner_id=owner_id,
                    source="user",
                    confidence=0.9,
                    importance=float(memory.get("importance", 0.5)),
                )
                await store.create_memory(record, _tag_inputs(memory.get("tags", [])))
            if scope == "group":
                context = AccessContext(
                    user_id="eval_u1",
                    group_id=owner_id,
                    session_id=f"eval_{case['case_id']}",
                    is_group=True,
                )
            else:
                context = AccessContext(
                    user_id=owner_id, session_id=f"eval_{case['case_id']}"
                )
            started = time.monotonic()
            result = await retriever.retrieve(
                context, case["query"], SearchFilters(), limit=10
            )
            candidate_ids = [item.memory_id for item in result.candidates]
            expected = set(case["expected"])
            recall_hit = bool(expected & set(candidate_ids))
            decision = await gate.evaluate(
                result.candidates,
                context,
                SearchFilters(),
                caller="main_model",
                budget={"max_items": 5, "max_tokens": 500},
                options={"explicit_search": True},
            )
            injected = [item.memory_id for item in decision.selected]
            expected_injected = expected & set(injected)
            latency = int((time.monotonic() - started) * 1000)
            latencies.append(latency)
            total_recall += int(recall_hit)
            total_injection_recall += int(bool(expected_injected))
            injected_total += len(injected)
            expected_injected_total += len(expected_injected)
            details.append(
                {
                    "case_id": case["case_id"],
                    "query": case["query"],
                    "expected": sorted(expected),
                    "recalled": candidate_ids,
                    "injected": injected,
                    "recall_hit": recall_hit,
                    "injection_hit": bool(expected_injected),
                    "latency_ms": latency,
                }
            )
            # 清理本 case 记忆，防止后续 case 被历史数据干扰。
            for memory in case["memories"]:
                await store.hard_delete_memory(memory["id"])
        count = max(1, len(cases))
        return {
            "cases": len(cases),
            "recall_at_10": round(total_recall / count, 4),
            "injection_recall": round(total_injection_recall / count, 4),
            "injection_precision": round(
                expected_injected_total / max(1, injected_total), 4
            ),
            "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
            "details": details,
        }
    finally:
        await store.close()
        shutil.rmtree(eval_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "tests" / "eval_dataset.jsonl",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.dataset))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["recall_at_10"] < args.threshold:
        print(
            f"FAIL: recall@10 {report['recall_at_10']} < {args.threshold}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
