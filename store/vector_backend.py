"""可选向量后端接口。

默认 ``none``，只使用 SQLite FTS5 + 标签 + 实体。启用 ``hash`` 时使用本地
确定性哈希向量，仅作为可运行示例与离线测试，不代表语义 embedding 质量。
生产环境可实现本接口接入 bge / OpenAI embeddings / pgvector 等后端。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

from store.models import AccessContext, SearchFilters
from utils.text import tokenize


class VectorBackend(ABC):
    """向量索引后端协议。"""

    name = "base"
    enabled = False

    @abstractmethod
    async def index(self, memory_id: str, text: str) -> None:
        """写入/更新一条记忆向量。"""

    @abstractmethod
    async def remove(self, memory_id: str) -> None:
        """删除一条记忆向量。"""

    @abstractmethod
    async def search(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        """返回 [(memory_id, cosine_score)]。"""


class NullVectorBackend(VectorBackend):
    """关闭向量召回。"""

    name = "none"
    enabled = False

    async def index(self, memory_id: str, text: str) -> None:
        return None

    async def remove(self, memory_id: str) -> None:
        return None

    async def search(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        return []


class HashVectorBackend(VectorBackend):
    """本地确定性哈希向量后端（可运行示例，非语义模型）。"""

    name = "hash"
    enabled = True

    def __init__(self, store: object, dim: int = 384):
        self.store = store
        self.dim = max(64, min(4096, int(dim)))

    def embed(self, text: str) -> list[float]:
        """把 token 哈希到固定维度，L2 归一化。"""
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % self.dim
            sign = 1.0 if (number >> 8) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return vector
        return [value / norm for value in vector]

    async def index(self, memory_id: str, text: str) -> None:
        vector = self.embed(text)
        await self.store.upsert_vector(memory_id, vector)  # type: ignore[attr-defined]

    async def remove(self, memory_id: str) -> None:
        await self.store.delete_vector(memory_id)  # type: ignore[attr-defined]

    async def search(
        self,
        query: str,
        context: AccessContext,
        filters: SearchFilters,
        limit: int,
    ) -> list[tuple[str, float]]:
        embedding = self.embed(query)
        return await self.store.vector_search(  # type: ignore[attr-defined]
            embedding, context, filters, limit
        )


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    """余弦相似度，映射到 [-1, 1]。"""
    va, vb = list(a), list(b)
    if not va or len(va) != len(vb):
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def build_vector_backend(name: str, store: object, dim: int) -> VectorBackend:
    """工厂方法。未知名称安全回退到 none。"""
    if name == "hash":
        return HashVectorBackend(store, dim=dim)
    return NullVectorBackend()
