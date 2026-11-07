"""文本处理工具：分词、相似度、token 估算、转义。

设计目标是零第三方依赖，中英文混合场景可用。
中文按「单字 + 双字 bigram」切分，既能拉回短词，也能承受 FTS/LIKE 的召回压力。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_STOPWORDS = {
    "的", "了", "着", "吗", "呢", "啊", "呀", "吧", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "这", "那", "这个", "那个", "就是", "然后", "但是",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or", "in",
    "on", "for", "with", "it", "this", "that", "i", "you", "we", "they",
}
_PUNCT_RE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_text(text: str) -> str:
    """基础清洗：去控制字符、压缩空白。"""
    if not text:
        return ""
    cleaned = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return _SPACE_RE.sub(" ", cleaned).strip()


def compact_text(text: str) -> str:
    """去掉标点与空白的紧凑文本，用于去重哈希与相似度。"""
    return _PUNCT_RE.sub("", normalize_text(text).lower())


def tokenize(text: str) -> list[str]:
    """轻量分词。

    英文/数字按词；中文保留单字并补充 bigram。单字和 bigram 同时进入集合，
    对「花生过敏」这类短实体更友好。
    """
    raw = _WORD_RE.findall(normalize_text(text).lower())
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        for i in range(len(cjk_run) - 1):
            tokens.append(cjk_run[i] + cjk_run[i + 1])
        cjk_run.clear()

    for item in raw:
        if _CJK_RE.fullmatch(item):
            cjk_run.append(item)
        else:
            flush_cjk()
            if item not in _STOPWORDS and len(item) > 0:
                tokens.append(item)
    flush_cjk()
    return [t for t in tokens if t not in _STOPWORDS]


def token_set(text: str) -> set[str]:
    """返回去重 token 集合。"""
    return set(tokenize(text))


def jaccard(a: str | Iterable[str], b: str | Iterable[str]) -> float:
    """Jaccard 相似度，适合快速去重与冲突预筛。"""
    set_a = token_set(a) if isinstance(a, str) else set(a)
    set_b = token_set(b) if isinstance(b, str) else set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def overlap_ratio(a: Iterable[str], b: Iterable[str]) -> float:
    """两个集合的重合比例（以较小集合为分母）。"""
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    中文约 1 字 1 token，英文约 4 字符 1 token。
    这是预算保护值，不追求与具体 tokenizer 完全一致。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return max(1, cjk + math.ceil(other / 4))


def truncate_text(text: str, max_chars: int, suffix: str = "…") -> str:
    """按字符截断，保留末尾省略号。"""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def top_terms(text: str, limit: int = 8) -> list[str]:
    """提取频次较高的词项，过滤单字和停用词。"""
    counts = Counter(tokenize(text))
    ranked = sorted(
        ((token, count) for token, count in counts.items() if len(token) >= 2),
        key=lambda item: (-item[1], -len(item[0]), item[0]),
    )
    return [token for token, _ in ranked[:limit]]


def escape_markup(text: str) -> str:
    """转义会被主模型误读为结构标签的字符。"""
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("```", "'''")
    )


def json_safe_text(text: str, max_chars: int = 1200) -> str:
    """输出到命令/日志前的基础清洗与截断。"""
    return truncate_text(normalize_text(text), max_chars)
