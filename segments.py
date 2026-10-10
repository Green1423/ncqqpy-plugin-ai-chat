"""对话数据模型 —— 仿 `@deepseek-ai/dsh-llm` 的 block/消息结构。

DSH 里一条消息不是一个字符串，而是
`{role, content: [block, ...], metadata}`：正文块（text / tool-call / tool-result）
被拆成有序 block 列表，消息本身带元信息（谁说的、什么时候说的、来自哪个会话）。
本模块把这套结构搬进 ncqqpy：

* `Block`      —— 正文块。`text` / `tool_call` / `tool_result` 三类。
* `Message`    —— 一条消息。`role` ∈ {system, user, assistant, tool} + metadata。
* 工厂函数      —— `system_message()` / `user_message()` / `assistant_message()`
                  / `tool_message()`，与 DSH 的 `createUserMessage` 等一一对应。

**为什么不用纯字符串**：压缩（compaction）必须能把「工具调用 ↔ 工具结果」成对
地留下或丢掉（OpenAI 协议要求 `tool_calls` 与紧随其后的 `tool` 消息严格配对，
拆散会被上游 400）。block 列表让配对判定变成结构问题，而不是字符串正则。

序列化到 OpenAI 兼容协议由 `to_api_message()` 负责，是**唯一**的协议出入口。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---- 消息角色 ----

ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"

# ---- block 类型 ----

BLOCK_TEXT = "text"
BLOCK_TOOL_CALL = "tool_call"
BLOCK_TOOL_RESULT = "tool_result"

# ---- 正文块的来源（metadata.origin）----
#
# 只有 USER_* 三种来源算「人类真实输入」，用于给模型加身份前缀；
# 其余都是系统注入或回填，绝不能被打上「某某说」的标签，否则模型会把
# 「历史摘要」当成是某个群成员刚说的话。
ORIGIN_USER_PRIVATE = "user.private"
ORIGIN_USER_GROUP = "user.group"
ORIGIN_COMPACT = "system.compact"
ORIGIN_TOOL = "tool"
ORIGIN_ASSISTANT = "assistant"

USER_ORIGINS = frozenset({ORIGIN_USER_PRIVATE, ORIGIN_USER_GROUP})


def _now_iso() -> str:
    """本地时区 ISO8601（秒精度）—— 人读友好，且用于给模型报时。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Block:
    """一个正文块。`type` 决定读哪个字段。"""

    type: str
    text: str = ""
    # tool_call
    call_id: str = ""
    name: str = ""
    arguments: dict = field(default_factory=dict)
    # tool_result
    is_error: bool = False
    # 展示用（发给用户时若模型没写正文，可用它兜底）
    summary: str = ""

    # ---- 构造 ----

    @staticmethod
    def text_block(text: str) -> "Block":
        return Block(type=BLOCK_TEXT, text=text)

    @staticmethod
    def tool_call(call_id: str, name: str, arguments: dict) -> "Block":
        return Block(type=BLOCK_TOOL_CALL, call_id=call_id, name=name,
                     arguments=arguments or {})

    @staticmethod
    def tool_result(call_id: str, text: str, *, name: str = "",
                    is_error: bool = False) -> "Block":
        return Block(type=BLOCK_TOOL_RESULT, call_id=call_id, text=text,
                     name=name, is_error=is_error)

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        """完整结构（落盘用，与 DSH 的 block 字段名保持一致）。"""
        d: dict[str, Any] = {"type": self.type}
        if self.type == BLOCK_TEXT:
            d["text"] = self.text
        elif self.type == BLOCK_TOOL_CALL:
            d["id"] = self.call_id
            d["name"] = self.name
            d["arguments"] = self.arguments
        elif self.type == BLOCK_TOOL_RESULT:
            d["id"] = self.call_id
            d["text"] = self.text
            if self.name:
                d["name"] = self.name
            if self.is_error:
                d["isError"] = True
        if self.summary:
            d["summary"] = self.summary
        return d

    @staticmethod
    def from_dict(d: dict) -> "Block":
        t = str(d.get("type", BLOCK_TEXT))
        if t == BLOCK_TOOL_CALL:
            args = d.get("arguments")
            if isinstance(args, str):        # 容错：模型有时给 JSON 字符串
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {"_raw": args}
            return Block(type=BLOCK_TOOL_CALL, call_id=str(d.get("id", "")),
                         name=str(d.get("name", "")),
                         arguments=args if isinstance(args, dict) else {},
                         summary=str(d.get("summary", "")))
        if t == BLOCK_TOOL_RESULT:
            return Block(type=BLOCK_TOOL_RESULT, call_id=str(d.get("id", "")),
                         text=str(d.get("text", "")), name=str(d.get("name", "")),
                         is_error=bool(d.get("isError")),
                         summary=str(d.get("summary", "")))
        return Block(type=BLOCK_TEXT, text=str(d.get("text", "")),
                     summary=str(d.get("summary", "")))

    # ---- 取值 ----

    def approx_tokens(self, chars_per_token: float = 1.6) -> float:
        """粗略 token 估算。见 `Message.approx_tokens` 的说明。"""
        if self.type == BLOCK_TEXT:
            return len(self.text) / chars_per_token
        if self.type == BLOCK_TOOL_CALL:
            payload = json.dumps(self.arguments, ensure_ascii=False)
            return (len(self.name) + len(payload)) / chars_per_token
        return len(self.text) / chars_per_token


@dataclass
class Message:
    """一条对话消息。`metadata` 是**唯一**允许挂载自由信息的地方。"""

    role: str
    blocks: list[Block] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    seq: int = 0            # 会话内单调递增（落盘后回填）
    ts: str = ""            # ISO8601

    # ---- 构造 ----

    def __post_init__(self):
        if not self.ts:
            self.ts = _now_iso()

    # ---- 取值 ----

    @property
    def text(self) -> str:
        """所有 text block 拼接（不含工具块）。"""
        return "".join(b.text for b in self.blocks if b.type == BLOCK_TEXT)

    @property
    def origin(self) -> str:
        return str(self.metadata.get("origin", ""))

    @property
    def tool_calls(self) -> list[Block]:
        return [b for b in self.blocks if b.type == BLOCK_TOOL_CALL]

    @property
    def is_blank(self) -> bool:
        return not any(b.type in (BLOCK_TEXT, BLOCK_TOOL_CALL) for b in self.blocks)

    def approx_tokens(self, chars_per_token: float = 1.6) -> float:
        """粗略 token 估算 —— **不引第三方 tokenizer**。

        中文为主的聊天里 1 token ≈ 1.6 个字符（中文约 1 字/token、英文约 4 字符/token
        的折中）。它只用来决定「什么时候该压缩」，不参与计费，偏保守即可。
        值可通过配置 `compaction.chars_per_token` 调整。
        """
        if not self.blocks:
            return len(json.dumps(self.metadata, ensure_ascii=False)) / chars_per_token
        return sum(b.approx_tokens(chars_per_token) for b in self.blocks)

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "blocks": [b.to_dict() for b in self.blocks],
            "metadata": self.metadata,
            # ⚠️ `seq` 必须落盘。它是 `SentReplyIndex` 回指原文的**唯一**线索
            # （机器人自己发的消息 id → 会话里第 seq 条），丢了就只剩「引用的是
            # 一张图片，读不到」（见 store.SessionStore.find / _own_original）。
            # 早先没写这个字段：落盘后重新加载全部退化成 0，热重载一次就永久失效。
            "seq": int(self.seq or 0),
            "ts": self.ts,
        }

    @staticmethod
    def from_dict(d: dict) -> "Message":
        blocks = [Block.from_dict(b) for b in d.get("blocks", [])
                  if isinstance(b, dict)]
        meta = d.get("metadata")
        try:
            seq = int(d.get("seq") or 0)
        except (TypeError, ValueError):
            seq = 0
        return Message(role=str(d.get("role", ROLE_USER)), blocks=blocks,
                       metadata=meta if isinstance(meta, dict) else {},
                       seq=seq, ts=str(d.get("ts", "")))

    # ---- 协议出口 ----

    def to_api_message(self) -> dict:
        """转成 OpenAI `chat/completions` 的 message。

        三条硬约束（都对不上会被上游 400 或被模型误解）：

        1. `tool` 角色必须带 `tool_call_id`，且**紧跟在**发起调用的 assistant 之后；
        2. assistant 带 `tool_calls` 时 `content` 允许为空串，但不能缺字段；
        3. 推理模型看到多余的自定义字段容易跑偏，所以这里只输出协议白名单字段。
        """
        if self.role == ROLE_TOOL:
            res = next((b for b in self.blocks if b.type == BLOCK_TOOL_RESULT), None)
            api: dict[str, Any] = {
                "role": ROLE_TOOL,
                "content": res.text if res else self.text,
            }
            if res is not None:
                api["tool_call_id"] = res.call_id
            return api

        calls = self.tool_calls
        if not calls:
            return {"role": self.role, "content": self.text}

        return {
            "role": self.role,
            # 带 tool_calls 时 content 必须是字符串（可为空串），但**不能缺字段**
            "content": self.text or "",
            "tool_calls": [
                {
                    "id": b.call_id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.arguments, ensure_ascii=False),
                    },
                }
                for b in calls
            ],
        }

    def clone(self) -> "Message":
        """深拷贝（压缩时要切片重组历史，不能共享 block 引用）。"""
        return Message.from_dict(self.to_dict())


# ---- 工厂（与 DSH 的 createUserMessage 等对应）----

def system_message(text: str, **metadata) -> Message:
    return Message(role=ROLE_SYSTEM, blocks=[Block.text_block(text)],
                   metadata={"origin": "system", **metadata})


def user_message(text: str, *, origin: str = ORIGIN_USER_PRIVATE,
                 **metadata) -> Message:
    return Message(role=ROLE_USER, blocks=[Block.text_block(text)],
                   metadata={"origin": origin, **metadata})


def assistant_message(text: str, *, tool_calls: Optional[list[Block]] = None,
                      **metadata) -> Message:
    blocks: list[Block] = []
    if text:
        blocks.append(Block.text_block(text))
    blocks.extend(tool_calls or [])
    return Message(role=ROLE_ASSISTANT, blocks=blocks,
                   metadata={"origin": ORIGIN_ASSISTANT, **metadata})


def tool_message(call_id: str, text: str, *, name: str = "",
                 is_error: bool = False, **metadata) -> Message:
    return Message(role=ROLE_TOOL,
                   blocks=[Block.tool_result(call_id, text, name=name,
                                             is_error=is_error)],
                   metadata={"origin": ORIGIN_TOOL, **metadata})


def compact_message(summary: str, **metadata) -> Message:
    """压缩产物：一段「接手摘要」。

    刻意用 `role=user` + `origin=system.compact`：OpenAI 协议里 system 只能出现
    在开头（部分实现会拒绝中途插入 system），而 user 角色的注入消息在 DSH 的
    历史投影里也是同一手法 —— 并且它**不带** `user.*` 来源，所以不会被加上
    「某某说」的前缀（见 `core.wrap_user_text`）。
    """
    return Message(role=ROLE_USER, blocks=[Block.text_block(summary)],
                   metadata={"origin": ORIGIN_COMPACT, **metadata})


def now_ts() -> str:
    return _now_iso()


def unix_now() -> float:
    return time.time()
