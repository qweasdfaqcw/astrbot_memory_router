"""写入前的安全检测：提示注入、敏感信息、第三方陈述。

原则：规则只做“标记”和“降权”，不做事实真伪裁决；敏感/指令性内容由门控
和注入器决定是否可见，绝不因为检测到文本就自动删除用户记忆。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from store.models import SENSITIVITY_RANK, SENSITIVITY_LEVELS

# 只匹配明确的“覆盖指令/角色/工具调用”表达，避免把普通祈使句全部误杀。
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_previous", re.compile(
        r"(?i)\b(ignore|disregard|forget)\b.{0,30}\b(previous|above|prior|system)\b.{0,30}\b(instruction|prompt|rule)s?\b"
    )),
    ("override_system", re.compile(
        r"(?i)\b(system\s*prompt|developer\s*message|you\s+are\s+now|act\s+as\s+system)\b"
    )),
    ("role_marker", re.compile(
        r"(?im)^\s*(system|assistant|developer|tool)\s*[:：]|<\s*/?\s*(system|assistant|developer|tool)\s*>"
    )),
    ("execute_command", re.compile(
        r"(?i)\b(execute|run)\b.{0,20}\b(command|shell|tool)\b|调用.{0,8}工具|执行以下.{0,8}(命令|操作)"
    )),
    ("cn_ignore", re.compile(
        r"忽略(掉)?(之前|以上|先前|所有|全部)?的?(指令|提示|设定|要求)|"
        r"无视(之前|以上|所有)?的?(指令|提示|设定)|"
        r"覆盖(之前|以上)?的?(设定|提示|指令)|"
        r"不要(遵守|理会)(之前|以上)的?(指令|提示)"
    )),
    ("cn_role", re.compile(
        r"你现在是.{0,20}(系统|管理员|开发者)|"
        r"从现在开始你(只|必须|要)听|"
        r"接下来你(只|必须|要)执行"
    )),
    ("exfiltration", re.compile(
        r"把.{0,20}(记忆|隐私|密钥|密码|token).{0,20}(发送|告诉|输出|泄漏)|"
        r"(输出|显示|打印).{0,10}(系统提示|system prompt|隐藏指令)"
    )),
    ("memory_tamper", re.compile(
        r"删除(掉)?(所有|全部|你的)?记忆|清空(所有)?记忆|"
        r"修改(你的)?记忆库|写入(一条)?假记忆|忽略记忆中的安全"
    )),
)

_SENSITIVE_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("credential", "secret", re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret|password|passwd|密码|密钥|口令)"
        r"\s*[:：=是为]?\s*[\w\-./+=]{4,}"
    )),
    ("id_card", "secret", re.compile(
        r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    )),
    ("phone", "sensitive", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("bank_card", "secret", re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)")),
    ("email", "sensitive", re.compile(
        r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
    )),
    ("address", "sensitive", re.compile(
        r"[\u4e00-\u9fff]{1,12}(?:省|市|区|县).{0,20}(?:路|街|道|巷|小区|楼|号|室)"
    )),
    ("medical", "sensitive", re.compile(
        r"病历|诊断|处方|用药|服药|手术|抑郁症|焦虑症|传染病|化验单|检查报告|"
        r"医院|医生|住院|医保|病史|癌|糖尿病|高血压"
    )),
    # 过敏/禁忌属于安全约束，不按“私密医疗史”拦截，否则关键安全记忆无法注入。
    ("medical_constraint", "normal", re.compile(r"过敏|忌口|禁忌|不能吃|不能碰")),
    ("financial", "sensitive", re.compile(
        r"银行卡|余额|工资|收入|欠款|贷款|信用卡|花呗|借呗|存款|理财|投资收益|账单"
    )),
)

_THIRD_PARTY_RE = re.compile(
    r"我(的)?(朋友|同事|同学|家人|妈妈|爸爸|老婆|老公|孩子|对象)|"
    r"他(说|讲|提到)|她(说|讲|提到)|听说|据说|别人说|有人说|朋友说|同事说"
)

_REMEMBER_RE = re.compile(
    r"^\s*(?:请|麻烦)?(?:帮我|给我)?(?:记住|记一下|记下|记着|保存一下)"
    r"(?:吧|哦|啊)?\s*[:：,，。!！\s]*(?P<content>.*)$",
    re.IGNORECASE,
)
_SUPPRESS_RE = re.compile(
    r"(?i)\b(do\s+not\s+remember|don't\s+remember|forget\s+this)\b|"
    r"(别记|不要记|不用记|别记住|不要保存|不用保存|忘掉这个|忘了这个|别存)"
)
_DELETE_REQUEST_RE = re.compile(
    r"把.{0,20}(记忆|那条|这条).{0,10}(删掉|删除|忘掉|抹掉)|"
    r"(删掉|删除|忘掉).{0,20}(记忆|那条|这条)"
)


@dataclass(slots=True)
class InjectionRisk:
    risky: bool
    patterns: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return ",".join(self.patterns)


@dataclass(slots=True)
class SensitivityResult:
    level: str = "normal"
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)

    @property
    def is_sensitive(self) -> bool:
        return self.level != "normal"


@dataclass(slots=True)
class MemoryDirective:
    """用户显式指令。action 为 remember/suppress/delete/None。"""

    action: str | None = None
    content: str = ""

    @property
    def explicit(self) -> bool:
        return self.action in {"remember", "suppress", "delete"}


def detect_prompt_injection(text: str) -> InjectionRisk:
    """检测可能诱导主模型执行指令的文本。"""
    raw = text or ""
    hits: list[str] = []
    for name, pattern in _INJECTION_PATTERNS:
        if pattern.search(raw):
            hits.append(name)
    return InjectionRisk(risky=bool(hits), patterns=hits)


def detect_sensitivity(text: str) -> SensitivityResult:
    """规则兜底敏感识别；返回最高敏感级别与类别。"""
    raw = text or ""
    categories: list[str] = []
    matches: list[str] = []
    level = "normal"
    for name, candidate_level, pattern in _SENSITIVE_PATTERNS:
        found = pattern.search(raw)
        if not found:
            continue
        categories.append(name)
        snippet = found.group(0)
        matches.append(snippet[:6] + "***" if len(snippet) > 6 else "***")
        if SENSITIVITY_RANK[candidate_level] > SENSITIVITY_RANK[level]:
            level = candidate_level
    if level not in SENSITIVITY_LEVELS:
        level = "normal"
    return SensitivityResult(level=level, categories=sorted(set(categories)), matches=matches)


def mask_sensitive(text: str) -> str:
    """脱敏日志/审计文本。"""
    if not text:
        return ""
    masked = text
    for name, _, pattern in _SENSITIVE_PATTERNS:
        if name in {"phone", "id_card", "bank_card"}:
            masked = pattern.sub("***", masked)
    # 邮箱只保留首字符
    masked = re.sub(
        r"(?i)\b([A-Z0-9])[A-Z0-9._%+-]*@([A-Z0-9.-]+\.[A-Z]{2,})\b",
        r"\1***@\2",
        masked,
    )
    return masked


def detect_third_party(text: str) -> bool:
    """是否在转述第三方信息。转述默认低置信，需用户确认后升级。"""
    return bool(_THIRD_PARTY_RE.search(text or ""))


def parse_memory_directive(text: str) -> MemoryDirective:
    """解析“记住/别记/删掉”等显式指令。"""
    raw = (text or "").strip()
    if not raw:
        return MemoryDirective()
    if _SUPPRESS_RE.search(raw):
        return MemoryDirective(action="suppress", content=raw)
    if _DELETE_REQUEST_RE.search(raw):
        return MemoryDirective(action="delete", content=raw)
    match = _REMEMBER_RE.match(raw)
    if match:
        content = match.group("content").strip()
        # “你记住”单独出现时没有可写内容，不视为显式写入。
        if len(content) >= 2:
            return MemoryDirective(action="remember", content=content)
    return MemoryDirective()


def clamp_sensitivity(level: str, hint: str = "") -> str:
    """合并规则结果与模型 hint，取更严格级别。"""
    candidate = hint if hint in SENSITIVITY_RANK else "normal"
    base = level if level in SENSITIVITY_RANK else "normal"
    return candidate if SENSITIVITY_RANK[candidate] > SENSITIVITY_RANK[base] else base
