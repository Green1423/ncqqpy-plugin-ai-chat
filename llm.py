"""OpenAI 兼容 `/chat/completions` 客户端（带 function calling）。

沿用本仓库在别的摘要链路上实测出来的几条硬约束：

* 推理模型（`deepseek-flash` 一类）的 reasoning token **计入 completion**，
  `max_tokens` 给小了会拿到 `finish_reason=length` **且 content 为空串** ——
  这是静默失败，必须显式报错而不是回一句空话给群友；
* 思考强度（reasoning_effort / thinking / enable_thinking）由 `reasoning.py`
  统一渲染 —— 本模块只负责「把算好的字段并进 payload」，不猜端点口味。
* 有些异常的 `str(e)` 是空串，甚至 `__str__` 自己会抛（aiohttp 的
  `ClientConnectorDNSError` 在缺 conn_key 时）——故 `_err_text()` 全程 try 住。

**默认走 SSE 流式**（`llm.stream`）。理由见 `_chat_stream()`：非流式请求在
「模型要想一两分钟」时和「上游彻底卡死」长得一模一样，只能靠一个总超时一刀切
（实测：同一句定积分问题 78s～240s 不等，90s 的总超时随机砍掉一半）；
流式能看见 token 在流动，于是可以把判据换成**「多久没有新数据」**。

本模块只负责「发一次请求、解一次响应」，重试/压缩/工具循环都在上层。
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from .reasoning import ReasoningPolicy, ReasoningSpec
from .segments import Block


def err_text(e: BaseException) -> str:
    """异常的可读描述 —— **绝不抛异常**（原因见模块 docstring）。"""
    name = type(e).__name__
    try:
        msg = str(e).strip()
    except Exception:  # noqa: BLE001
        try:
            msg = repr(e)
        except Exception:  # noqa: BLE001
            msg = ""
    return f"{name}: {msg}" if msg else name


class LlmError(RuntimeError):
    """一次调用彻底失败（含 HTTP 错误、空响应、解析失败）。

    `partial`：**已经被掐断但已经生成出来的正文**（流式才可能有）。
    上层在重试用尽后可以把它发出去 —— 「答了一半」比「什么都没有」有用得多。
    """

    def __init__(self, message: str, *, status: int = 0, retryable: bool = False,
                 partial: str = ""):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.partial = partial or ""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    raw_arguments: str = ""

    def to_block(self) -> Block:
        return Block.tool_call(self.id, self.name, self.arguments)


@dataclass
class ChatResult:
    """一次 assistant 回复。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_blocks(self) -> list[Block]:
        blocks: list[Block] = []
        if self.text:
            blocks.append(Block.text_block(self.text))
        blocks.extend(tc.to_block() for tc in self.tool_calls)
        return blocks


def _parse_arguments(raw: Any) -> dict:
    """把模型给的 arguments 变成 dict。

    模型有时给 JSON 字符串、有时直接给对象、偶尔给半截 JSON（被截断）。
    半截 JSON 也要**保留原文**，否则工具调用会变成一个没有参数的静默失败。
    """
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, (list, int, float, bool)):
        return {"_value": raw}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        val = json.loads(text)
    except (ValueError, TypeError):
        return {"_raw": text}
    if isinstance(val, dict):
        return val
    return {"_value": val}


class SseBuffer:
    """SSE 行缓冲：喂字节，吐出解析好的 JSON 载荷。

    单独成一个类是为了**能单测**（见 `tests/test_ai_chat.py::test_sse_parse`）：
    真实网络里 `data:` 行的切分位置由 TCP 决定，半行、多行、`[DONE]`、注释心跳
    都可能出现在任意位置 —— 这些分支不该只在联调时才被走到。

    ⚠️ 不用 `async for line in resp.content`：aiohttp 的行迭代有 64KB 的行长上限
    （`LineTooLong`），而工具调用的 `arguments` 分片、长公式都可能超过它。
    自己攒缓冲区就没有这个上限。
    """

    def __init__(self) -> None:
        self._buf = b""
        self.done = False

    def feed(self, chunk: bytes) -> list[dict]:
        self._buf += chunk
        out: list[dict] = []
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            obj = self._line(line)
            if obj is not None:
                out.append(obj)
        return out

    def flush(self) -> list[dict]:
        """收尾：最后一行可能没有换行符。"""
        if not self._buf:
            return []
        obj = self._line(self._buf)
        self._buf = b""
        return [obj] if obj is not None else []

    def _line(self, line: bytes) -> Optional[dict]:
        text = line.strip().decode("utf-8", "ignore")
        if not text or text.startswith(":"):      # 空行 / 心跳注释
            return None
        if not text.startswith("data:"):
            return None                            # event: / id: / retry: 之类
        payload = text[5:].strip()
        if payload == "[DONE]":
            self.done = True
            return None
        try:
            obj = json.loads(payload)
        except (ValueError, TypeError):
            return None                            # 半行/非 JSON：丢掉这一行就好
        return obj if isinstance(obj, dict) else None


class _StreamAcc:
    """把流式分片累加成一次完整回复（正文 / 思考链 / 工具调用 / usage）。"""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning: list[str] = []
        self.calls: dict[int, dict] = {}
        self.finish = ""
        self.usage: dict = {}
        self.model = ""
        self.chunks = 0

    # ---- 合并 ----

    def merge(self, obj: dict) -> None:
        self.chunks += 1
        if isinstance(obj.get("model"), str) and obj["model"]:
            self.model = obj["model"]
        # `stream_options.include_usage` 的用量在最后一个 chunk 上，且它往往
        # 带一个空的 choices —— 所以 usage 要在遍历 choices 之前先收
        if isinstance(obj.get("usage"), dict) and obj["usage"]:
            self.usage = obj["usage"]
        choices = obj.get("choices")
        if not isinstance(choices, list):
            return
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            if ch.get("finish_reason"):
                self.finish = str(ch["finish_reason"])
            delta = ch.get("delta")
            if not isinstance(delta, dict):
                # 有的端点每片都回完整 `message`（而不是 `delta`）
                delta = ch.get("message") if isinstance(ch.get("message"), dict) else {}
            self._merge_delta(delta)

    def _merge_delta(self, d: dict) -> None:
        content = d.get("content")
        if isinstance(content, list):
            content = "".join(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text", "output_text"))
        if content:
            self.text.append(str(content))
        reason = d.get("reasoning_content")
        if reason is None:
            reason = d.get("reasoning")
        if isinstance(reason, str) and reason:
            self.reasoning.append(reason)
        for item in d.get("tool_calls") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index") or 0)
            except (TypeError, ValueError):
                idx = 0
            slot = self.calls.setdefault(idx, {"id": "", "name": "", "args": ""})
            if item.get("id"):
                slot["id"] = str(item["id"])
            fn = item.get("function") if isinstance(item.get("function"), dict) else {}
            name = fn.get("name") or item.get("name")
            if name:
                slot["name"] = str(name)
            args = fn.get("arguments", item.get("arguments"))
            if isinstance(args, str) and args:
                slot["args"] += args
            elif isinstance(args, dict) and args:
                # 少数端点直接给对象：转回字符串，和分片路径统一收口
                slot["args"] += json.dumps(args, ensure_ascii=False)

    # ---- 出结果 ----

    @property
    def partial(self) -> str:
        return "".join(self.text)

    def result(self) -> ChatResult:
        calls: list[ToolCall] = []
        for idx in sorted(self.calls):
            slot = self.calls[idx]
            if not slot["name"]:
                continue
            calls.append(ToolCall(
                id=slot["id"] or f"call_{uuid.uuid4().hex[:12]}",
                name=slot["name"],
                arguments=_parse_arguments(slot["args"]),
                raw_arguments=slot["args"]))
        text = self.partial
        if not text.strip() and not calls:
            # 推理模型在 max_tokens 不足时的典型症状 —— 必须报出来，
            # 否则用户只看到「没有回复」，运维完全无从下手。
            raise LlmError(
                f"空响应（finish_reason={self.finish or '无'}，"
                f"思考链 {sum(len(x) for x in self.reasoning)} 字）；"
                "若为推理模型请调大 ai_chat.llm.max_tokens（建议 ≥ 16384："
                "思考 token 与正文共用配额，给少了就会「想完但没字数回答」）")
        return ChatResult(
            text=text, tool_calls=calls, model=self.model,
            finish_reason=self.finish, usage=dict(self.usage),
            raw={"stream": True, "chunks": self.chunks,
                 "reasoning_chars": sum(len(x) for x in self.reasoning)})


class LlmClient:
    """OpenAI 兼容客户端。"""

    def __init__(self, *, base_url: str = "", api_key: str = "", model: str = "",
                 max_tokens: int = 4096, temperature: Optional[float] = None,
                 reasoning_effort: str = "off", timeout: float = 120.0,
                 extra_headers: Optional[dict] = None, logger=None,
                 verify_ssl: bool = True, stream: bool = True,
                 stall_timeout: float = 60.0,
                 reasoning_policy: Optional[ReasoningPolicy] = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.max_tokens = max(64, int(max_tokens))
        self.temperature = temperature
        # 兼容旧配置：单个 `reasoning_effort` 串仍可用；新的分层策略走
        # `reasoning_policy`（支持 per-model / 会话覆盖 / 多方言）。
        self.reasoning_policy = reasoning_policy or ReasoningPolicy(
            default=reasoning_effort or "off")
        # `timeout` = 单次调用的**总时长**上限（流式下是兜底，不是判据）
        self.timeout = float(timeout)
        # `stall_timeout` = 流式下「多久没有新数据」就判定卡死（真正的掐断判据）
        self.stall_timeout = max(1.0, float(stall_timeout))
        self.stream = bool(stream)
        # 最近一次实际发送的思考参数（诊断用：让用户看到「到底发了什么」）
        self.last_reasoning: dict = {}
        self.extra_headers = dict(extra_headers or {})
        self.verify_ssl = bool(verify_ssl)
        self.logger = logger
        self.last_error: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.calls = 0

    # ---- 生命周期 ----

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def reconfigure(self, *, reasoning_policy: Optional[ReasoningPolicy] = None,
                    **kwargs) -> None:
        """热更新配置（`/aichat model` / `/aichat think` 用）。

        `reasoning_policy` 走显式关键字：它在构造器里是从 `reasoning_effort`
        派生出来的，不在 `__dict__` 的初始字段里 —— 靠 `hasattr` 兜底的通用
        循环会把它静默丢掉，表现为「改了 config 但思考强度没变」。
        """
        if reasoning_policy is not None:
            self.reasoning_policy = reasoning_policy
        for key, value in kwargs.items():
            if value is None or not hasattr(self, key):
                continue
            setattr(self, key, value)
        self.base_url = (self.base_url or "").rstrip("/")

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # 超时**按请求给**（见 `chat()`）：流式要的是「多久没新数据」，
            # 非流式要的是「整件事多久」，两种判据没法共用一个会话级默认值。
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, connect=15))
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # ---- 请求 ----

    def reasoning_spec(self, session_override: Any = None) -> ReasoningSpec:
        """当前该用哪个思考强度（会话覆盖 → per-model → 默认）。"""
        return self.reasoning_policy.spec_for(self.model, session_override)

    def build_payload(self, messages: list[dict], *, tools: Optional[list[dict]] = None,
                      max_tokens: Optional[int] = None,
                      stream: bool = False,
                      reasoning: Any = None) -> dict:
        """组装请求体。

        `reasoning` 传会话级覆盖值（如 `"high"`）；为空则走策略默认。
        思考字段由 `ReasoningPolicy` 渲染 —— 换端点只改 dialect。
        """
        spec = self.reasoning_spec(reasoning)
        budget = int(max_tokens or self.max_tokens)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # 开了思考时抬高下界：推理 token 与正文共用 completion 配额，
            # 给小了会「思考完但没字数回答」（content 为空串的静默失败）。
            "max_tokens": self.reasoning_policy.min_max_tokens(spec, budget),
            "stream": bool(stream),
        }
        if stream:
            # 让上游把 usage 放在最后一个 chunk 里 —— 否则流式拿不到 token 用量，
            # `/aichat current` 与日志里的统计会突然变成 0（vLLM / OpenAI 都认这个字段）。
            payload["stream_options"] = {"include_usage": True}
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        reasoning_params = dict(self.reasoning_policy.build_params(spec))
        reasoning_params.update(self.reasoning_policy.payload_extras(spec))
        if reasoning_params:
            payload.update(reasoning_params)
        self.last_reasoning = reasoning_params
        return payload

    async def chat(self, messages: list[dict], *, tools: Optional[list[dict]] = None,
                   max_tokens: Optional[int] = None,
                   reasoning: Any = None,
                   timeout: Optional[float] = None) -> ChatResult:
        """发一次请求并解析（默认流式，见 `_chat_stream`）。

        Raises:
            LlmError: 未配置 / HTTP 非 200 / 空响应 / 响应结构不认识 / 掐断。
        """
        if not self.enabled:
            raise LlmError("未配置 base_url / model / api_key")
        # 单次调用的总时长上限：会话可以用 `/aichat timeout` 覆盖（见 main.effective_timeout）。
        limit = float(timeout or self.timeout)
        use_stream = bool(self.stream)
        payload = self.build_payload(messages, tools=tools, max_tokens=max_tokens,
                                     stream=use_stream, reasoning=reasoning)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        session = await self._ensure_session()
        self.calls += 1
        # 总时长上限：流式下由 `_chat_stream` 自己按「已经跑了多久」判，
        # socket 层给一个略宽一点的硬上限兜底（防的是「判据逻辑写漏了 → 永久挂住」）；
        # 非流式没有别的手段，就直接用它当超时。
        total = (limit + 15) if use_stream else limit
        try:
            async with session.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers,
                    ssl=self.verify_ssl,
                    timeout=aiohttp.ClientTimeout(total=total, connect=15)) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    retryable = (resp.status in (408, 409, 425, 429)
                                 or resp.status >= 500)
                    raise LlmError(f"HTTP {resp.status}: {body[:300]}",
                                   status=resp.status, retryable=retryable)
                ctype = str(resp.headers.get("Content-Type") or "")
                if use_stream and "event-stream" in ctype:
                    return await self._chat_stream(resp, timeout=limit)
                # 没按 SSE 回（有的网关直接忽略 `stream`）→ 当普通 JSON 读完，
                # 这样「流式」只是请求方式，不影响解析口径。
                body = await resp.text()
        except LlmError:
            raise
        except Exception as e:  # noqa: BLE001 - 网络层任何异常都归一到 LlmError
            msg = err_text(e)
            self.last_error = msg
            raise LlmError(msg, retryable=True) from None
        return self._parse_body(body)

    # ---- 流式 ----

    async def _chat_stream(self, resp, *, timeout: Optional[float] = None) -> ChatResult:
        """读 SSE 流，边收边判「是不是卡住了」。

        为什么值得为它写一整套：非流式请求在「模型正在想 100 秒」和「上游彻底
        卡死」这两种情况下的表现**完全一样** —— 都是没有任何字节。于是只能定一个
        总超时一刀切：定小了（90s）把正常但慢的回答砍掉，定大了（300s）真要卡死
        时又让用户白等。实测同一句定积分问题在 78s～240s 之间随机分布，90s 正好
        切在中间，成功率全看运气。

        流式把这两个情况分开了：只要 token 还在流，就说明它活着，可以一直等
        （上限 `timeout`）；**多久没有新数据**（`stall_timeout`）才是「卡死」的判据。
        """
        acc = _StreamAcc()
        buf = SseBuffer()
        loop = asyncio.get_running_loop()
        limit = float(timeout or self.timeout)
        deadline = loop.time() + limit
        content = resp.content
        while True:
            left = deadline - loop.time()
            if left <= 0:
                raise LlmError(
                    f"超过单次调用总时长上限 {limit:.0f}s（已生成 "
                    f"{len(acc.partial)} 字正文）", retryable=False,
                    partial=acc.partial)
            try:
                chunk = await asyncio.wait_for(
                    content.readany(), timeout=min(self.stall_timeout, left))
            except asyncio.TimeoutError:
                if loop.time() >= deadline - 0.05:
                    raise LlmError(
                        f"超过单次调用总时长上限 {limit:.0f}s（已生成 "
                        f"{len(acc.partial)} 字正文）", retryable=False,
                        partial=acc.partial) from None
                raise LlmError(
                    f"流式读取停滞 {self.stall_timeout:.0f}s（上游不再吐数据，"
                    f"已生成 {len(acc.partial)} 字正文）", retryable=True,
                    partial=acc.partial) from None
            except Exception as e:  # noqa: BLE001 - 连接被断等
                msg = err_text(e)
                self.last_error = msg
                raise LlmError(msg, retryable=True, partial=acc.partial) from None
            if not chunk:                     # EOF：流正常结束
                break
            for obj in buf.feed(chunk):
                acc.merge(obj)
            if buf.done:
                break
        for obj in buf.flush():
            acc.merge(obj)
        return acc.result()

    # ---- 非流式 ----

    def _parse_body(self, body: str) -> ChatResult:
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            self.last_error = "响应不是 JSON"
            # 「上游给了个不能用的东西」多半是代理/网关抽了一下 —— 与网络错误同类，
            # 标成可重试（重试由上层做，见 main._chat_with_retry）。
            raise LlmError(f"响应不是 JSON（前 200 字：{body[:200]!r}）",
                           retryable=True) from None
        if not isinstance(data, dict):
            raise LlmError("响应结构不是对象", retryable=True)
        if isinstance(data.get("error"), dict):
            err = data["error"]
            raise LlmError(f"上游错误: {err.get('message') or err}")

        choices = data.get("choices") or []
        if not choices:
            raise LlmError(f"响应没有 choices（前 200 字：{body[:200]!r}）",
                           retryable=True)
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, list):
            # 少数端点回 content 数组；拼出 text part
            content = "".join(
                str(p.get("text", "")) for p in content
                if isinstance(p, dict) and p.get("type") in (None, "text", "output_text"))
        text = str(content or "")
        finish = str(choice.get("finish_reason") or "")

        calls: list[ToolCall] = []
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            for item in raw_calls:
                if not isinstance(item, dict):
                    continue
                fn = item.get("function") if isinstance(item.get("function"), dict) else {}
                name = str(fn.get("name") or item.get("name") or "").strip()
                if not name:
                    continue
                raw_args = fn.get("arguments", item.get("arguments"))
                calls.append(ToolCall(
                    id=str(item.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                    name=name, arguments=_parse_arguments(raw_args),
                    raw_arguments=raw_args if isinstance(raw_args, str) else ""))

        if not text.strip() and not calls:
            # 推理模型在 max_tokens 不足时的典型症状 —— 必须报出来，
            # 否则用户只看到「没有回复」，运维完全无从下手。
            self.last_error = f"空响应（finish_reason={finish}）"
            raise LlmError(
                f"空响应（finish_reason={finish}）；"
                "若为推理模型请调大 ai_chat.llm.max_tokens（建议 ≥ 16384）")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatResult(text=text, tool_calls=calls,
                          model=str(data.get("model") or self.model),
                          finish_reason=finish, usage=usage, raw=data)
