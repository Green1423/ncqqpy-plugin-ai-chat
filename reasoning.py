"""思考强度（reasoning effort / thinking）控制。

为什么单独一个模块
------------------
「思考强度」不是一个参数，而是**一族方言**。同一个模型能力，不同端点的开关名字
完全不同，而且给错名字的代价很不对称：

* 多数端点对**未知字段**直接 400（请求整条失败）；
* 少数端点**静默忽略**（你以为关掉了思考，其实还在烧 token 与延迟）。

所以这里把「用户想表达什么强度」和「往 payload 里塞哪个字段」拆开：
`resolve_spec()` 负责前者，`build_params()` 负责后者。想换端点只改 `dialect`，
不用碰会话里存的值。

已知方言（实测/文档来源见 `docs/plugins/ai-chat/07-reasoning.md`）
-------------------------------------------
====================  ==========================================================
dialect               生成的 payload 字段
====================  ==========================================================
`effort`（默认）       `reasoning_effort: "<level>"`（OpenAI / DeepSeek / 通义）
`thinking`             `thinking: {"type": "enabled"}`（DeepSeek V3.2 风格）
                       ⚠️ `off` 时**什么都不发**，不是发 `disabled`：见 `build_params()`
                       的首行提前返回 —— 「不干预」是所有方言共同的语义
`enable_thinking`      `enable_thinking: true|false`（Qwen / vLLM 部署常见）
`none`                 什么都不发（让端点用自己的默认值）
`raw`                  由 `params` 原样合并（自建端点万能兜底）
====================  ==========================================================

强度别名
--------
`off` / `inherit` / `default` / `auto` / `none`  → 不发思考字段（表达式里的「不干预」）
`minimal` / `low` / `medium` / `high` / `max`    → 归一化后交给方言渲染

> `low` 在一些 DeepSeek 部署上会被**服务端忽略**（实测：reasoning 454 vs none 的 0）。
> 想真正省 token 请用 `off`（= 不干预）或 `none`，不要指望 `low`。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# ---- 强度级别 ----
LEVEL_MINIMAL = "minimal"
LEVEL_LOW = "low"
LEVEL_MEDIUM = "medium"
LEVEL_HIGH = "high"
LEVEL_MAX = "max"

LEVELS = (LEVEL_MINIMAL, LEVEL_LOW, LEVEL_MEDIUM, LEVEL_HIGH, LEVEL_MAX)

# 「不干预」的各种写法（语义相同，保留原样只为让用户少查文档）
NULL_WORDS = frozenset({"", "off", "inherit", "default", "auto", "none", "-"})

# 用户可能敲出来的别名 → 规范级别
LEVEL_ALIASES = {
    "min": LEVEL_MINIMAL, "minimum": LEVEL_MINIMAL, "least": LEVEL_MINIMAL,
    "最少": LEVEL_MINIMAL, "最低": LEVEL_MINIMAL,
    "low": LEVEL_LOW, "l": LEVEL_LOW, "light": LEVEL_LOW,
    "低": LEVEL_LOW, "少": LEVEL_LOW,
    "medium": LEVEL_MEDIUM, "mid": LEVEL_MEDIUM, "m": LEVEL_MEDIUM,
    "normal": LEVEL_MEDIUM, "default+": LEVEL_MEDIUM,
    "中": LEVEL_MEDIUM,
    "high": LEVEL_HIGH, "h": LEVEL_HIGH, "deep": LEVEL_HIGH,
    "高": LEVEL_HIGH, "多": LEVEL_HIGH,
    "max": LEVEL_MAX, "maximum": LEVEL_MAX, "highest": LEVEL_MAX,
    "最高": LEVEL_MAX, "最大": LEVEL_MAX,
}

# 端点可能对强度级别有自己的口味：这里只做「能表达」的保证，不做白名单校验 ——
# 未知级别原样送出（自建端点常常自定义），被 400 拒绝时用户可以立刻退回。
VALIDATION_HINT = (
    "若端点报 400/参数不识别，用 /aichat think off 退回「不干预」，"
    "或把 llm.reasoning.dialect 改成 thinking / enable_thinking / none"
)

# ---- 方言 ----
DIALECT_EFFORT = "effort"
DIALECT_THINKING = "thinking"
DIALECT_ENABLE_THINKING = "enable_thinking"
DIALECT_NONE = "none"
DIALECT_RAW = "raw"

DIALECTS = (DIALECT_EFFORT, DIALECT_THINKING, DIALECT_ENABLE_THINKING,
            DIALECT_NONE, DIALECT_RAW)

# 「开/关」的布尔词（用在 enable_thinking 方言与 `on`/`off` 选择里）
TRUE_WORDS = frozenset({"on", "true", "1", "yes", "y", "enable", "enabled",
                        "开", "开启", "是"})
FALSE_WORDS = frozenset({"off", "false", "0", "no", "n", "disable", "disabled",
                         "关", "关闭", "否"})


def normalize_level(value: Any) -> Optional[str]:
    """把用户输入归一成规范级别；「不干预」与无法识别都返回 None。

    无法识别的返回值是 None（= 不干预）而不是抛异常：配置项写错了应该是
    「按默认跑」而不是把整个插件启动流程炸掉。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return LEVEL_MEDIUM if value else None
    if isinstance(value, (int, float)):
        # 数字当预算（见 parse_budget），级别位置上不接受 → 不干预
        return None
    text = str(value).strip().lower()
    if text in NULL_WORDS:
        return None
    if text in LEVEL_ALIASES:
        return LEVEL_ALIASES[text]
    if text in LEVELS:
        return text
    return None


# 预算写法：`4096` / `4k` / `8K` / `1.5k` / `2048tokens`。
# 必须严格匹配，否则 `high` 这类级别词会被当成「不是数字」而误判成预算失败。
_BUDGET_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([km])?(?:\s*tokens?|t)?$", re.I)


def parse_budget(value: Any) -> Optional[int]:
    """解析思考预算（token）：`4096` / `"4k"` / `"8K"` / `"2048 tokens"`。

    不是预算写法就返回 None（调用方据此区分「级别」与「预算」）。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return n if n > 0 else None
    text = str(value).strip().replace("_", "").replace(",", "")
    m = _BUDGET_RE.match(text)
    if not m:
        return None
    mult = {"k": 1024, "m": 1024 * 1024}.get((m.group(2) or "").lower(), 1)
    n = int(float(m.group(1)) * mult)
    return n if n > 0 else None


def looks_like_budget(value: Any) -> bool:
    """这个输入是不是在表达预算（而不是强度级别）。"""
    return parse_budget(value) is not None


def format_budget(tokens: int) -> str:
    """`4096` → `4096`；`16384` → `16k`（给人看）。"""
    if tokens and tokens % 1024 == 0 and tokens >= 2048:
        return f"{tokens // 1024}k"
    return str(tokens)


@dataclass
class ReasoningSpec:
    """**用户意图**：想用多强的思考。与具体端点无关。"""

    mode: str = "off"             # off | effort | thinking | budget
    level: Optional[str] = None   # mode=effort 时的规范级别
    enabled: Optional[bool] = None  # mode=thinking 时
    budget: Optional[int] = None  # mode=budget 时的 token 数
    note: str = ""                # 解析说明（给 /aichat think 回显）

    @property
    def is_off(self) -> bool:
        """是否「不干预」—— 不含任何思考参数。"""
        if self.mode == "off":
            return True
        if self.mode == "thinking":
            return self.enabled is False
        return False

    @property
    def wants_thinking(self) -> bool:
        """是否明确要求开思考（用于 max_tokens 下界与状态提示）。"""
        return not self.is_off

    def label(self) -> str:
        """一行的可读描述。"""
        if self.mode == "off":
            return "关（不发思考参数，由端点决定）"
        if self.mode == "effort":
            return f"reasoning_effort={self.level}"
        if self.mode == "thinking":
            return f"thinking={'enabled' if self.enabled else 'disabled'}"
        if self.mode == "budget":
            return f"思考预算 {format_budget(self.budget or 0)} token"
        # 其余模式由 dialect 决定渲染，label 只描述意图
        if self.mode == "raw":
            return "自定义参数（见 raw_params）"
        return self.mode

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"mode": self.mode}
        if self.level:
            out["level"] = self.level
        if self.enabled is not None:
            out["enabled"] = self.enabled
        if self.budget:
            out["budget"] = self.budget
        return out

    @staticmethod
    def from_dict(d: dict) -> "ReasoningSpec":
        if not isinstance(d, dict):
            return ReasoningSpec()
        mode = str(d.get("mode") or "off")
        return ReasoningSpec(
            mode=mode if mode in ("off", "effort", "thinking", "budget", "raw")
            else "off",
            level=normalize_level(d.get("level")),
            enabled=bool(d["enabled"]) if "enabled" in d else None,
            budget=parse_budget(d.get("budget")),
        )


def spec_from_text(text: str, *, default_level: str = LEVEL_MEDIUM) -> ReasoningSpec:
    """把 `/aichat think` 的一个参数解析成意图。

    接受的写法（大小写不敏感）：

    ==========================  ====================================
    写法                        含义
    ==========================  ====================================
    `off` / `inherit` / `auto`  不干预（回到由端点/全局决定）
    `low` / `medium` / `high`   指定强度级别
    `4096` / `4k`               思考预算（budget 模式）
    `thinking:on` / `thinking:off`  DeepSeek 风格 thinking 开关
    `on` / `enable`             等同于 `medium`（开思考的便捷写法）
    ==========================  ====================================
    """
    raw = (text or "").strip()
    if not raw:
        return ReasoningSpec(note="空值 → 不干预")
    low = raw.lower()

    # 带模式前缀：`mode:value`
    if ":" in low:
        mode, _, value = low.partition(":")
        mode = mode.strip()
        value = value.strip()
        if mode in ("thinking", "think", "enable_thinking"):
            if value in TRUE_WORDS or value in LEVELS or value in LEVEL_ALIASES:
                level = normalize_level(value)
                return ReasoningSpec(
                    mode="thinking", enabled=True,
                    note=f"开启了思考（{level or default_level}）")
            if value in FALSE_WORDS:
                return ReasoningSpec(mode="thinking", enabled=False, note="关闭思考")
            return ReasoningSpec(note=f"不认识的取值 {value!r}")
        if mode in ("budget", "tokens", "token"):
            budget = parse_budget(value)
            if budget:
                return ReasoningSpec(mode="budget", budget=budget,
                                     note=f"思考预算 {format_budget(budget)} token")
            return ReasoningSpec(note=f"不认识的预算 {value!r}")
        if mode in ("effort", "reasoning", "reasoning_effort"):
            level = normalize_level(value)
            if level:
                return ReasoningSpec(mode="effort", level=level)
            return ReasoningSpec(mode="off", note="不干预（effort 无具体级别）")
        return ReasoningSpec(note=f"不认识的模式 {mode!r}")

    if low in TRUE_WORDS:
        return ReasoningSpec(mode="effort", level=default_level,
                             note=f"开启思考（{default_level}）")
    if low in FALSE_WORDS or low in NULL_WORDS:
        return ReasoningSpec(mode="off", note="不干预")

    if looks_like_budget(low):
        budget = parse_budget(low)
        return ReasoningSpec(mode="budget", budget=budget,
                             note=f"思考预算 {format_budget(budget)} token")

    level = normalize_level(low)
    if level:
        return ReasoningSpec(mode="effort", level=level)
    # 认不出来 → 不干预（而不是猜一个级别往端点上送）；note 留原文便于排查
    return ReasoningSpec(note=f"不认识的值 {raw!r}（按不干预处理）")


# ==================== 策略（多层合并） ====================

def override_text(policy: Optional["ReasoningPolicy"], model: str) -> Optional[str]:
    """把策略在某模型上的生效值转成 `chat(reasoning=...)` 要的文本。

    返回 `None` 表示**不覆盖**（client 那边用自己配置的默认值）。返回**字符串**
    而不是 `ReasoningSpec`：client 统一走 `spec_from_text()` 一条解析路径，
    避免两处各解析一遍、两处各自漂移。

    ⚠️ 这段分支以前只写在 `Compactor._reasoning_override()` 里。压缩摘要与
    历史清单摘要要的是同一档强度（短、密，思考链纯属浪费），复制一份的话
    以后加一种 mode 就会漏改一处 —— 所以提出来共用。
    """
    if policy is None:
        return None
    spec = policy.spec_for(model or "", None)
    if spec.mode == "off":
        return "off"
    if spec.mode == "budget" and spec.budget:
        return str(spec.budget)
    if spec.mode == "thinking":
        return "thinking:on" if spec.enabled is not False else "thinking:off"
    return spec.level or "off"


@dataclass
class ReasoningPolicy:
    """端点侧的思考参数渲染方式 + 多层默认值。

    合并优先级（后者覆盖前者）：`default` → `per_model[当前模型]` → 会话覆盖。
    """

    dialect: str = DIALECT_EFFORT
    default: str = "off"                 # 字符串（级别/“off”）或 dict（完整 spec）
    per_model: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)      # dialect=raw 时原样合并
    max_tokens_floor: int = 0            # >0：开了思考时把 max_tokens 抬到这个下界

    def spec_for(self, model: str, session_override: Any = None,
                 *, text: bool = False) -> ReasoningSpec:
        """算出当前该用哪个 spec。

        `session_override` 可以是 `/aichat think` 存下的字符串（如 `"high"`），
        也可以是 `ReasoningSpec`；为空则往上一层找。
        """
        candidate = session_override
        if candidate in (None, "", {}):
            candidate = (self.per_model or {}).get(model or "", None)
        if candidate in (None, "", {}):
            candidate = self.default
        if isinstance(candidate, ReasoningSpec):
            return candidate
        if isinstance(candidate, dict):
            # 允许配置里直接写 {"mode": "effort", "level": "high"}
            if "mode" in candidate:
                return ReasoningSpec.from_dict(candidate)
            text_value = candidate.get("level") or candidate.get("value") \
                or candidate.get("effort")
            if text_value is None:
                return ReasoningSpec()
            candidate = text_value
        if isinstance(candidate, bool):
            return ReasoningSpec(mode="effort",
                                 level=LEVEL_MEDIUM if candidate else None)
        return spec_from_text(candidate if isinstance(candidate, str)
                              else str(candidate))

    # ---- 渲染 ----

    def build_params(self, spec: ReasoningSpec) -> dict:
        """把意图渲染成 payload 字段。`off` 方言下永远返回空 dict。"""
        if spec is None or spec.is_off:
            return {}
        if self.dialect == DIALECT_NONE:
            return {}
        if self.dialect == DIALECT_RAW:
            return dict(self.params or {})
        if self.dialect == DIALECT_ENABLE_THINKING:
            return {"enable_thinking": True}
        if self.dialect == DIALECT_THINKING:
            # 这一族只认开关；「强度」无从表达 → 退化成 enabled
            return {"thinking": {"type": "enabled"}}
        # 默认：effort 方言。`thinking:*` 与「预算」都要翻译成级别，
        # 否则用户写的强度在这个端点上会被悄悄吞掉。
        return {"reasoning_effort": self.effective_level(spec)}

    def effective_level(self, spec: ReasoningSpec) -> str:
        """在只有「级别」可用的方言里，这个意图等价于哪个级别。"""
        if spec.mode == "thinking":
            return LEVEL_HIGH if spec.enabled is not False else LEVEL_LOW
        if spec.mode == "budget":
            budget = spec.budget or 0
            if budget >= 16384:
                return LEVEL_MAX
            if budget >= 8192:
                return LEVEL_HIGH
            if budget >= 4096:
                return LEVEL_MEDIUM
            return LEVEL_LOW
        return spec.level or LEVEL_MEDIUM

    def payload_extras(self, spec: ReasoningSpec) -> dict:
        """与方言无关的额外字段（如预算方言的 `thinking_budget`）。"""
        out: dict[str, Any] = {}
        if spec is None or spec.is_off:
            return out
        if spec.mode == "budget" and spec.budget:
            if self.dialect in (DIALECT_EFFORT, DIALECT_RAW):
                # effort 方言的端点普遍不认 budget；只在 raw 里由用户自己写。
                # 这里仍给出一个通用字段名，便于自建端点直接吃。
                out["thinking_budget"] = int(spec.budget)
        return out

    def min_max_tokens(self, spec: ReasoningSpec, base: int) -> int:
        """开了思考时 max_tokens 的下界。

        推理 token **计入 completion**（deepseek-flash 实测约占 93%），
        `max_tokens` 给小了会拿到 `finish_reason=length` 且 **content 为空串** ——
        这是静默失败，用户只看到「没反应」。所以给出下界保护。
        """
        if spec is None or spec.is_off:
            return int(base)
        floor = int(self.max_tokens_floor or 0)
        if spec.mode == "budget" and spec.budget:
            # 预算 + 余量：推理 token 与正文**共用** completion 配额，
            # 只留预算本身会得到「思考完但没字数回答」的空响应。
            floor = max(floor, spec.budget + 1024)
        elif spec.wants_thinking:
            # 没有明确预算时给个经验下界，避免用户开了 high 却只给 256 token
            floor = max(floor, 4096)
        return max(int(base), floor)


    def describe(self, spec: ReasoningSpec, *, model: str = "") -> str:
        """一行诊断：会往 payload 里发什么。"""
        params = self.build_params(spec)
        extras = self.payload_extras(spec)
        merged = {**params, **extras}
        if not merged:
            return f"{spec.label()} → 不发思考参数"
        shown = ", ".join(f"{k}={v!r}" for k, v in merged.items())
        out = f"{spec.label()} → payload 增加 {shown}"
        if spec.mode == "budget" and self.dialect == DIALECT_THINKING:
            out += "（该方言只认开关，预算退化为「开」）"
        return out
