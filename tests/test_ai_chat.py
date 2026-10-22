"""ai_chat 插件测试 —— 真起 OneBot WebSocket 服务端 + 真跑 NapCatClient。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_ai_chat.py

与 `tests/test_watchdog.py` 同套路：**不打桩框架**。事件以原始 OneBot JSON
从 WS 推进去，走完 `parse_event → EventBus → 插件 → ApiClient → WS` 全链路；
只把「外部 LLM」换成本文件里的 `StubLlm`（否则测试要联网、要花钱、还不稳定）。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP_DIR = ROOT / "tests" / ".test-tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)   # 新 clone 里没有它（.gitignore 排除）

# ---- 插件包位置：两种仓库布局都要能跑 --------------------------------------
# 框架仓库布局 = <repo>/plugins/ai_chat/（直接可用）；
# 独立插件仓库布局 = 仓库根就是包目录，而且 clone 出来的目录名未必等于插件名。
# 后者搭一个「包农场」<tmp>/pkgfarm/plugins/ai_chat -> 插件包目录，让
# `import plugins.ai_chat.*` 与 `plugin_dir=<农场>/plugins` 两种口径都成立。
PLUGIN_NAME = "ai_chat"
PLUGIN_DIR = ROOT / "plugins" / PLUGIN_NAME
PLUGIN_ROOT_DIR = ROOT / "plugins"
if not (PLUGIN_DIR / "__init__.py").exists() and (ROOT / "__init__.py").exists():
    import tempfile
    PLUGIN_ROOT_DIR = (Path(tempfile.mkdtemp(prefix=f"pkgfarm-{PLUGIN_NAME}-"))
                       / "plugins")
    PLUGIN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if not (PLUGIN_ROOT_DIR / PLUGIN_NAME).exists():
        (PLUGIN_ROOT_DIR / PLUGIN_NAME).symlink_to(ROOT, target_is_directory=True)
    sys.path.insert(0, str(PLUGIN_ROOT_DIR.parent))
    PLUGIN_DIR = PLUGIN_ROOT_DIR / PLUGIN_NAME
PLUGIN_PKG = f"plugins.{PLUGIN_NAME}"

from websockets.asyncio.server import serve            # noqa: E402
from napcat.client import NapCatClient                 # noqa: E402

PASSED = 0
FAILED = 0
FAILURES: list[str] = []

BOT_ID = 10001
SUPERUSER = 1
ALICE = 20001
BOB = 20002
GROUP = 30001
OTHER_GROUP = 30002


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


async def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ==================== LLM 打桩 ====================

#: 假 Chromium（与 `test_history_command.py` 同一替身）。
#: 画图用例必须用它：真浏览器要联网取 KaTeX、还要 fork，慢且不稳。
FAKE_CHROME_SRC = '''#!/usr/bin/env python3
"""假 Chromium：只实现本插件用到的两个开关（--dump-dom / --screenshot）。"""
import re, sys, zlib, struct

def png(w, h):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    row = b"\\x00" + b"\\x33\\x66\\x99" * w
    return (b"\\x89PNG\\r\\n\\x1a\\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(row * h)) + chunk(b"IEND", b""))

args = sys.argv[1:]
for a in args:
    if a.startswith("--screenshot="):
        with open(a.split("=", 1)[1], "wb") as f:
            f.write(png(720, 300))
if "--dump-dom" in args:
    sys.stdout.write("<html><body><div id=\\"__size__\\">H=420</div></body></html>")
'''


def make_fake_chrome(path: Path) -> Path:
    path.write_text(FAKE_CHROME_SRC, encoding="utf-8")
    path.chmod(0o755)
    return path


class StubResult:
    def __init__(self, text="", tool_calls=None, usage=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.model = "stub-model"
        self.finish_reason = "stop"
        self.usage = usage or {}
        self.raw = {}

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)

    def to_blocks(self):
        Block = plugin_module("segments").Block
        blocks = []
        if self.text:
            blocks.append(Block.text_block(self.text))
        for tc in self.tool_calls:
            blocks.append(tc.to_block())
        return blocks


class StubCall:
    def __init__(self, name, arguments, call_id="c1"):
        self.id = call_id
        self.name = name
        self.arguments = arguments
        self.raw_arguments = json.dumps(arguments, ensure_ascii=False)

    def to_block(self):
        Block = plugin_module("segments").Block
        return Block.tool_call(self.id, self.name, self.arguments)


class StubLlm:
    """接住 `LlmClient.chat()`，按脚本回话并记录每一次请求。"""

    def __init__(self, script_text="好的，我收到了。"):
        self.enabled = True
        self.model = "stub-model"
        self.base_url = "http://stub/v1"
        self.api_key = "stub"
        self.max_tokens = 4096
        self.reasoning_effort = "off"
        self.temperature = None
        self.timeout = 5
        self.stream = True          # 与真客户端同形（见 llm.LlmClient.__init__）
        self.stall_timeout = 60.0
        self.last_error = None
        self.calls = 0
        self.requests: list[dict] = []
        self.script_text = script_text
        self.responder = None      # 可选：callable(messages) -> StubResult
        self._real_client = None   # Harness 注入：用于读取 reasoning_policy
        self.last_reasoning_params: dict = {}
        self.last_max_tokens = 0
        self.last_timeout = 0.0
        # 供 compaction 压缩调用使用（摘要必须够长才不被下界拦掉）
        self.summary = (
            "## 用户诉求\n- 用户在测试里反复确认会话与上下文行为\n\n"
            "## 关键信息\n- 会话与压缩相关的具体事实\n\n"
            "## 已做的事\n- 已执行多轮对话与工具调用\n\n"
            "## 未决问题\n- （无）\n\n## 当前状态\n- 正在测试压缩\n\n"
            "## 下一步\n- （无）\n\n## 关键上下文\n- 压缩必须让上下文变短"
        )

    async def close(self):
        return None

    def reconfigure(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None and hasattr(self, k):
                setattr(self, k, v)

    def build_payload(self, messages, **kwargs):     # 兼容接口
        return {"messages": messages}

    async def chat(self, messages, *, tools=None, max_tokens=None, reasoning=None,
                   timeout=None):
        """复刻 `LlmClient.chat()` 的思考强度行为（打桩也要能验证 payload）。

        刻意**不**在桩里存 policy 快照：从真 client 现取，这样 `/aichat think`
        改了策略后，断言看到的就是真正会发出去的东西。
        """
        self.calls += 1
        if self._real_client is not None:
            # 插件改过模型名（`/aichat model`）时，策略要按新名字查 per_model
            self.model = self._real_client.model
        policy = getattr(self._real_client, "reasoning_policy", None) if self._real_client else None
        spec = None
        params = {}
        if policy is not None:
            spec_from_text = plugin_module("reasoning").spec_from_text
            spec = (spec_from_text(str(reasoning)) if isinstance(reasoning, str)
                    else policy.spec_for(self.model, reasoning))
            params = dict(policy.build_params(spec))
            params.update(policy.payload_extras(spec))
            floor = policy.min_max_tokens(spec, max_tokens or self.max_tokens)
        else:
            floor = max_tokens or self.max_tokens
        self.last_reasoning_params = params
        self.last_max_tokens = floor
        # 会话级 `/aichat timeout` 走的是 chat(timeout=…)：桩必须记下来，
        # 否则「真的用上了吗」这条断言永远看到 0。
        self.last_timeout = float(timeout or 0.0)
        self.requests.append({"messages": messages, "tools": tools,
                              "reasoning": reasoning, "reasoning_params": params,
                              "max_tokens": floor})
        if self.responder is not None:
            return self.responder(messages, tools)
        sys_prompt = messages[0].get("content", "") if messages else ""
        if "压缩引擎" in sys_prompt:
            return StubResult(text=self.summary)
        # 模型看到工具结果后给最终答复
        if any(m.get("role") == "tool" for m in messages):
            return StubResult(text=f"工具结果我已看到。（第 {self.calls} 次调用）")
        return StubResult(text=self.script_text)

    # ---- 断言辅助 ----

    def last_user_content(self) -> str:
        for m in reversed(self.requests[-1]["messages"]):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def all_system_prompts(self) -> list[str]:
        return [r["messages"][0].get("content", "") for r in self.requests if r["messages"]]


# ==================== OneBot 服务端 ====================

class Harness:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.calls: list[dict] = []
        self.replies: list[dict] = []
        self.clients: list = []
        self._reply_event = asyncio.Event()
        self.client: NapCatClient | None = None
        self.plugin = None
        self.llm = StubLlm()

    async def start(self, *, superusers=(SUPERUSER,)):
        plugin_dir = self.tmp / "plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PLUGIN_DIR, plugin_dir / "ai_chat", ignore=shutil.ignore_patterns("__pycache__", "data", "data.*", "tests",
                                      "docs", ".git", "README.md", "LICENSE",
                                      ".gitignore"))
        data_dir = self.tmp / "ai_chat_data"
        fake_chrome = make_fake_chrome(self.tmp / "fake-chrome")
        cfg = {
            "ws_url": f"ws://127.0.0.1:{await free_port()}", "token": "",
            "plugin_dir": str(plugin_dir), "superusers": list(superusers),
            "command_prefixes": ["/"], "bound_groups": [GROUP],
            "all_groups": False, "api_delay": 0.05,
            "reconnect_interval": 0.2, "watchdog_enabled": False,
            "restart_enabled": False,
            "plugins": {
                "ai_chat": {
                    "data_dir": str(data_dir),
                    # 显式写全 llm.reasoning：不依赖「缺键时用默认值」，
                    # 于是用例顺序不会影响这里的思考强度断言。
                    "llm": {"base_url": "http://stub/v1", "api_key": "sk-stub",
                            "model": "stub-model",
                            "reasoning": {"dialect": "effort", "default": "off",
                                          "per_model": {}, "params": {},
                                          "max_tokens_floor": 0,
                                          "compaction": "off"}},
                    "reply": {"quote": False},
                    # 假浏览器：画图/history 出图都走它，测试不依赖真 Chromium
                    "history": {"chromium": str(fake_chrome)},
                    "log_verbose": False,
                    # 服务范围白名单：不配的话（默认 deny）谁都收不到服务，
                    # 所有用例都会先撞到那句回绝说明。
                    "binding": {"groups": [GROUP, OTHER_GROUP],
                                "users": [ALICE, BOB, SUPERUSER],
                                "notify_cooldown": 0},
                }
            },
        }
        cfg_path = self.tmp / "config.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        self.cfg_path = cfg_path
        self.cfg = cfg
        self.data_dir = data_dir
        # `get_msg` 桩：{message_id: 返回体}，用于测「引用某条消息」
        self.msg_lookup: dict = {}
        #: `get_msg` 的人为延迟（秒）。线上框架给**每次** API 调用都加了
        #: `api_delay`（1~5 秒）并串行化，所以「读引用」要等好几秒 ——
        #: 测「发内容后紧接着按结束符」这类竞态时必须能复现这个窗口。
        self.msg_delay: float = 0.0
        # 桩发出的每条消息（含分配到的 message_id）
        self.sent: list[dict] = []
        self._next_mid = 900000

        async def handler(ws):
            self.clients.append(ws)
            try:
                async for raw in ws:
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    echo = payload.get("echo")
                    action = payload.get("action")
                    params = payload.get("params") or {}
                    self.calls.append({"action": action, "params": params})
                    if action in ("send_private_msg", "send_group_msg", "send_msg"):
                        self.replies.append({"action": action, "params": params})
                        self._reply_event.set()
                    if echo is None:
                        continue
                    if action == "get_msg" and self.msg_delay:
                        await asyncio.sleep(self.msg_delay)
                    result = self._result_for(action, params)
                    await ws.send(json.dumps({"status": "ok", "retcode": 0,
                                              "echo": echo, "data": result}))
            except Exception:
                pass

        self.port = int(cfg["ws_url"].rsplit(":", 1)[1])
        self.server = await serve(handler, "127.0.0.1", self.port)
        self.client = NapCatClient(config_path=str(cfg_path))
        started = asyncio.Event()

        async def _on_connected():
            started.set()

        self.client.on_ws_connected(_on_connected)
        self.runner = asyncio.create_task(self.client.start())
        await asyncio.wait_for(started.wait(), timeout=10)
        # 插件加载完成后把 LLM 换成打桩版
        self.plugin = self.client.plugins.plugins.get("ai_chat")
        if self.plugin is None:
            raise RuntimeError(f"ai_chat 插件没加载：{list(self.client.plugins.plugins)}")
        self._install_stub()      # 内部会把真 client 记到桩上，再替换
        # `reload_config()` 会 `reconfigure` 回真 client（生产里是同一个对象，
        # 测试里是替换动作），所以包一层，重载后把桩装回去。
        original_reload = self.plugin.reload_config

        def reload_and_restub(*args, **kwargs):
            result = original_reload(*args, **kwargs)
            self._install_stub()
            return result

        self.plugin.reload_config = reload_and_restub
        return self

    def _install_stub(self):
        """把打桩 LLM 装到插件与压缩器上（两侧都要，压缩器抓的是旧引用）。

        ⚠️ 两处顺序陷阱：

        1. `_real_client` 必须在**替换之前**记录，否则记到的是桩自己；
        2. `plugin.reload_config()` 内部会 `self.llm.reconfigure(...)` —— 装桩之后
           那句就落在桩上（桩当然不认 `reasoning_policy`），真 client 的策略永远
           不更新。所以这里把 config 应用动作**改派**给真 client，而桩只负责回话。
        """
        if not isinstance(self.plugin.llm, StubLlm):
            self.llm._real_client = self.plugin.llm
        self.plugin.llm = self.llm
        self.plugin.compactor.bind_llm(self.llm)
        self.plugin.compactor.bind_reasoning(self.plugin.compaction_reasoning)

        # 把「应用配置」转发给真 client（见上方说明）
        real = self.llm._real_client
        if real is not None:
            # 把「应用配置」转发给真 client：桩没有 reasoning_policy，
            # `reload_config()` 里那句 reconfigure 落在桩上就等于没生效。
            self.plugin.llm.reconfigure = real.reconfigure

    def _result_for(self, action, params):
        if action in ("send_private_msg", "send_group_msg", "send_msg"):
            # 真实现会返回 message_id，插件靠它建立「自己发的消息 → 原文」的索引
            # （用户引用机器人发的**图片**时要用，见 test_quote_ai_image）。
            self._next_mid += 1
            self.sent.append({"message_id": self._next_mid, "params": params})
            return {"message_id": self._next_mid}
        if action == "get_msg":
            mid = int(params.get("message_id") or 0)
            return (self.msg_lookup or {}).get(mid)
        if action == "get_login_info":
            return {"user_id": BOT_ID, "nickname": "测试机器人"}
        if action == "get_group_info":
            return {"group_id": int(params.get("group_id") or 0),
                    "group_name": "测试群", "member_count": 42}
        if action == "get_group_member_list":
            return [{"user_id": ALICE, "nickname": "爱丽丝", "card": "",
                     "role": "admin"},
                    {"user_id": BOB, "nickname": "鲍勃", "card": "", "role": "member"}]
        return None

    async def stop(self):
        try:
            await self.client.stop()
        except Exception:
            pass
        self.runner.cancel()
        self.server.close()
        await self.server.wait_closed()

    # ---- 事件注入 ----

    async def _send_raw(self, payload: dict, settle: float = 0.35):
        ws = self.clients[-1]
        await ws.send(json.dumps(payload, ensure_ascii=False))
        await asyncio.sleep(settle)

    async def private(self, text: str, *, user_id=ALICE, settle=0.35,
                      with_reply=0, **extra):
        await self._send_raw(self._msg_payload("private", text, user_id=user_id,
                                               reply_to=with_reply, **extra), settle)

    async def group(self, text: str, *, user_id=ALICE, group_id=GROUP,
                    at_bot=True, role="member", settle=0.35, with_reply=0, **extra):
        await self._send_raw(self._msg_payload("group", text, user_id=user_id,
                                               group_id=group_id, at_bot=at_bot,
                                               role=role, reply_to=with_reply,
                                               **extra), settle)

    def _msg_payload(self, kind, text, *, user_id, group_id=None, at_bot=False,
                     message_id=0, role="member", reply_to=0):
        segs = []
        if reply_to:
            segs.append({"type": "reply", "data": {"id": str(reply_to)}})
        if at_bot:
            segs.append({"type": "at", "data": {"qq": str(BOT_ID)}})
            segs.append({"type": "text", "data": {"text": " "}})
        segs.append({"type": "text", "data": {"text": text}})
        payload = {
            "post_type": "message", "message_type": kind, "sub_type": "normal",
            "message_id": message_id or (id(text) % 100000 + 1),
            "user_id": user_id, "message": segs, "raw_message": text,
            "self_id": BOT_ID, "time": 1700000000,
            "sender": {"user_id": user_id, "nickname": f"用户{user_id}",
                       "card": "", "role": role},
        }
        if group_id is not None:
            payload["group_id"] = group_id
        return payload

    async def wait_for_replies(self, count: int, timeout=6.0):
        """等回复条数达到 count（轮询计数，不用 Event 标志）。

        ⚠️ 别用 `Event.clear()` + `wait()` 的写法：回复在 `settle` 期间就到达时，
        `set()` 已经发生过 —— 随后 `clear()` 把它抹掉，`wait()` 便一直等到超时。
        表现为「回复明明发出去了，测试却报超时」。
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.replies) < count:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"等第 {count} 条回复超时（已有 {len(self.replies)} 条："
                    f"{[reply_text(r)[:40] for r in self.replies]}）")
            await asyncio.sleep(0.05)

    async def wait_reply(self, timeout=6.0, *, index=0) -> str:
        """等第 index+1 条回复，返回其文本。"""
        await self.wait_for_replies(index + 1, timeout)
        return reply_text(self.replies[index])

    async def wait_new_reply(self, timeout=6.0) -> str:
        """等一条**新**回复（相对当前已有条数）。"""
        base = len(self.replies)
        await self.wait_for_replies(base + 1, timeout)
        return reply_text(self.replies[base])

    async def wait_replies(self, count: int, timeout=6.0) -> list[str]:
        try:
            await self.wait_for_replies(count, timeout)
        except TimeoutError:
            pass
        return [reply_text(r) for r in self.replies[:count]]

    def clear(self):
        """清空观测窗口（回复/API 调用/LLM 请求与计数）。

        计数一起清，用例才能断言「本轮恰好调用 N 次」而不是跨用例累加。
        """
        self.replies.clear()
        self.calls.clear()
        self._reply_event.clear()
        self.llm.requests.clear()
        self.llm.calls = 0
        self.llm.last_reasoning_params = {}
        self.llm.last_max_tokens = 0
        self.llm.last_timeout = 0.0


def reply_text(rec: dict) -> str:
    """从 send_* 的参数里抽纯文本。"""
    msg = rec["params"].get("message")
    if isinstance(msg, str):
        return msg
    parts = []
    for seg in msg or []:
        if isinstance(seg, dict) and seg.get("type") == "text":
            parts.append(seg.get("data", {}).get("text", ""))
    return "".join(parts)


# ==================== 用例 ====================

async def test_private_flow(h: Harness):
    print("\n[1] 私聊：直接说话 + 结束符提交 + 分段拼接")
    await establish(h)
    h.clear()
    await h.private("你好")
    check("没有结束符时模型不被调用", h.llm.calls == 0, f"calls={h.llm.calls}")
    check("没有结束符时不回复", len(h.replies) == 0, f"{h.replies}")

    await h.private("第一段")
    await h.private("第二段")
    await h.private("//", settle=0.8)
    check("结束符触发一次模型调用", h.llm.calls == 1, f"calls={h.llm.calls}")
    content = h.llm.last_user_content()
    # 分段的语义：用户是因为单条消息有长度上限才拆开发，对模型来说就是**一整段**。
    # 所以拼接结果里**不该**出现 `(1/3)` 这类序号 —— 那既不是用户写的，
    # 又会诱导模型在回复里模仿同样的编号（提示词专门叮嘱过不要写）。
    check("分段直接拼接、不加序号",
          "(1/3)" not in content and "(3/3)" not in content, repr(content))
    check("没有多余的标记，只有用户写的内容",
          content.strip() == "你好\n第一段\n第二段", repr(content))
    check("分段内容按顺序保留",
          content.index("第一段") < content.index("第二段"), repr(content))
    reply = await h.wait_reply()
    check("回复发给了正确的私聊", h.replies[0]["params"].get("user_id") == str(ALICE),
          f"{h.replies[0]}")
    check("回复是模型文本", "我收到了" in reply, reply)

    # 结束符与正文同一条消息
    h.clear()
    await h.private("粘在末尾//", settle=0.8)
    check("行尾结束符同样提交", h.llm.calls == 1, f"calls={h.llm.calls}")
    check("结束符不进入正文", "//" not in h.llm.last_user_content(),
          h.llm.last_user_content())

    # 退役回归：`/end` 命令已删除 —— 收尾只认行尾结束符。
    # 发 `/end` 现在是普通文本（进暂存区，不提交），绝不能提交给模型。
    h.clear()
    await h.private("用行尾结束符收尾")
    await h.private("/end", settle=0.8)
    check("已删的命令 /end 不再提交", h.llm.calls == 0, f"calls={h.llm.calls}")
    await h.private("补上行尾//", settle=0.8)
    check("行尾 // 仍能提交（含 /end 那段）", h.llm.calls == 1, f"calls={h.llm.calls}")
    check("/end 不会被当成结束符剥掉（留在正文里）",
          "/end" in h.llm.last_user_content(), h.llm.last_user_content())

    # 空暂存区 + 只有结束符
    h.clear()
    await h.private("//", settle=0.5)
    check("空暂存区只发结束符时有提示且不调模型", h.llm.calls == 0
          and len(h.replies) == 1, f"calls={h.llm.calls} replies={h.replies}")


async def test_group_trigger(h: Harness):
    print("\n[2] 群聊：必须 @机器人")
    await establish(h, group_id=GROUP)
    h.clear()
    await h.group("不@机器人", at_bot=False)
    await h.group("//", at_bot=False, settle=0.5)
    check("群里不 @ 时模型不被调用", h.llm.calls == 0, f"calls={h.llm.calls}")

    h.clear()
    await h.group("你看看这个", at_bot=True)
    await h.group("//", at_bot=True, settle=0.8)
    check("群里 @ 后能触发", h.llm.calls == 1, f"calls={h.llm.calls}")
    reply = await h.wait_reply()
    check("回复发到群里", h.replies[0]["params"].get("group_id") == str(GROUP),
          f"{h.replies[0]}")


async def test_command_messages_not_dialogue(h: Harness):
    print("\n[3] 命令消息不进对话")
    await establish(h)
    h.clear()
    await h.group("/aichat list", at_bot=True, settle=0.6)
    check("命令走命令系统（有回复）", len(h.replies) >= 1, f"{h.replies}")
    check("命令不调用模型", h.llm.calls == 0, f"calls={h.llm.calls}")
    check("命令不会被当成待提交内容",
          not (h.plugin.buffer.peek(f"group:{GROUP}").get("items")),
          f"{h.plugin.buffer.peek(f'group:{GROUP}')}")


async def test_session_lifecycle(h: Harness):
    print("\n[4] 会话：新建 / 列表 / 切换 / 解附 / 附着")
    h.clear()
    # 会话要显式创建（不再自动建）
    await h.private("第一次说话//", settle=0.8)
    check("说话不会自动建会话，而是给出提示",
          "还没有选中的会话" in await h.wait_reply(), "应给出提示")
    check("确实没有建会话", not h.plugin.store.list(), "不该自动建")

    h.clear()
    await h.private("/aichat new 第一个会话", settle=1.0)
    await h.wait_reply()
    sessions = h.plugin.store.list()
    check("new 建出会话", len(sessions) == 1, f"{[s.describe() for s in sessions]}")
    auto_id = sessions[0].id

    # /aichat new
    h.clear()
    await h.private("/aichat new 学习计划", settle=0.7)
    text = await h.wait_reply()
    check("new 命令成功", "已新建会话" in text, text)
    new_sessions = h.plugin.store.list()
    check("会话数变成 2", len(new_sessions) == 2, f"{[s.id for s in new_sessions]}")
    current = _current_session(h)
    check("new 后当前会话就是新会话", current is not None and current.name == "学习计划",
          f"{current}")
    study = [s for s in new_sessions if s.name == "学习计划"][0]

    # 切换回自动建的会话：**直接切**，不需要 --force，也不需要先 detach
    h.clear()
    # 先在暂存区留一段没提交的（切换不该丢它，也不该因此拒绝切换）
    await h.private("切会话前打了一半")
    check("切换前暂存区有一段",
          len(h.plugin.buffer.peek(f"private:{ALICE}").get("items") or []) == 1)
    await h.private(f"/aichat attach {auto_id}", settle=0.7)
    text = await h.wait_reply()
    check("attach 直接切换（不再要求 --force）",
          "已切到" in text and "--force" not in text, text)
    check("回执标出原来的会话", f"原 {study.id}" in text, text)
    check("回执说明暂存区那几段会并入", "暂存区里还有 1 段" in text, text)
    check("切换没丢暂存区",
          len(h.plugin.buffer.peek(f"private:{ALICE}").get("items") or []) == 1,
          f"{h.plugin.buffer.peek(f'private:{ALICE}')}")
    check("当前会话已换成目标", _current_session(h).id == auto_id,
          f"{_current_session(h)}")

    # 暂存的那段仍会并入**新**会话（回执说过会并入，这里验证真的并入了）
    h.clear()
    await h.private("补个结束符//", settle=1.2)
    await h.wait_reply()
    hist_new = [m.text for m in h.plugin.store.history(auto_id)]
    check("暂存的分段并入了新会话",
          any("切会话前打了一半" in t for t in hist_new), f"{hist_new}")
    hist_old = [m.text for m in h.plugin.store.history(study.id)]
    check("没有并进原来的会话", not any("切会话前打了一半" in t for t in hist_old),
          f"{hist_old}")

    # 历史隔离：两个会话的历史互不串
    h.clear()
    await h.private("这是A会话的内容//", settle=0.9)
    await h.wait_reply()
    hist_a = [m.text for m in h.plugin.store.history(auto_id)]
    check("A 会话历史含本条", any("这是A会话的内容" in t for t in hist_a), f"{hist_a}")
    study = [s for s in h.plugin.store.list() if s.name == "学习计划"][0]
    hist_b = [m.text for m in h.plugin.store.history(study.id)]
    check("B 会话历史不受影响", not any("这是A会话的内容" in t for t in hist_b),
          f"{hist_b}")

    # detach 后不再入会话
    h.clear()
    await h.private("/aichat detach", settle=0.6)
    text = await h.wait_reply()
    check("detach 成功", "已取消你在" in text, text)

    # list
    h.clear()
    await h.private("/aichat list", settle=0.6)
    text = await h.wait_reply()
    check("list 列出两个会话",
          "学习计划" in text and auto_id in text, text)

    # attach
    h.clear()
    await h.private(f"/aichat attach {study.id}", settle=0.6)
    text = await h.wait_reply()
    check("attach 成功", "已切到" in text and study.id in text, text)


async def test_group_session_is_per_group(h: Harness):
    """群里用的是**群聊会话**（归本群）；且群聊/私聊互不相通。"""
    ChatTarget = plugin_module("store").ChatTarget

    print("\n[5] 群聊：群会话 + 类型隔离")

    # ---- 1) 不会自动建会话 ----
    h.clear()
    await h.group("群里第一次说话//", at_bot=True, user_id=ALICE, settle=1.2)
    text = await h.wait_reply()
    check("群里说话不自动建会话", not h.plugin.store.list(),
          f"{[x.describe() for x in h.plugin.store.list()]}")
    check("而是提示怎么建", "new" in text, text[:140])

    # ---- 2) 显式建群会话 ----
    session = await establish(h, user_id=ALICE, group_id=GROUP)
    check("建出来的是群聊会话",
          session is not None and session.owner_kind == "group", f"{session}")
    check("群会话归本群所有", session.owner_id == GROUP, f"{session.owner_id}")
    check("群会话名来自模板（不是第一个说话的人）",
          session.name == "预热群会话", session.name)
    check("焦点指向它",
          _current_session(h, user_id=ALICE, group_id=GROUP).id == session.id,
          "焦点不对")

    # ---- 3) 邀请前别人用不了；受邀后可用 ----
    h.clear()
    await h.group("B 说一句//", at_bot=True, user_id=BOB, settle=1.2)
    check("未受邀的群成员没有焦点（提示去要邀请）",
          _current_session(h, user_id=BOB, group_id=GROUP) is None, "不该有焦点")
    check("提示里给出邀请路径",
          "invite" in await h.wait_reply(), "应提示邀请")
    check("没有为 B 建出第二个会话",
          len(h.plugin.store.list()) == 1,
          f"{[x.describe() for x in h.plugin.store.list()]}")

    h.plugin.store.invite(session.id, BOB)
    await attach_to(h, session.id, user_id=BOB, group_id=GROUP)
    check("受邀并 attach 后两人共用一个群会话",
          _current_session(h, user_id=ALICE, group_id=GROUP).id == session.id
          and _current_session(h, user_id=BOB, group_id=GROUP).id == session.id,
          "焦点不一致")

    # ---- 4) 类型隔离：群会话进不了私聊，私聊会话进不了群 ----
    h.clear()
    await h.private("/aichat new 我的私聊会话", user_id=ALICE, settle=1.1)
    await h.wait_for_replies(1)
    ps = _current_session(h, user_id=ALICE)
    check("私聊建的是私聊会话",
          ps is not None and ps.owner_kind == "user", f"{ps}")

    ok, why = h.plugin._may_attach(ps, ChatTarget("group", GROUP), ALICE)
    check("私聊会话不能在群里用", not ok, f"ok={ok}")
    check("拒绝信息说明原因并给路径", "私聊会话" in why and "new" in why, why[:140])
    ok2, why2 = h.plugin._may_attach(session, ChatTarget("private", ALICE), ALICE)
    check("群聊会话不能在私聊里用", not ok2, f"ok={ok2}")
    check("拒绝信息给出正确路径", "群聊会话" in why2 and "new" in why2, why2[:120])

    h.clear()
    await h.group(f"/aichat attach {ps.id}", at_bot=True, user_id=ALICE, settle=1.0)
    check("群里 attach 私聊会话被拒", "私聊会话" in await h.wait_reply(), "应被拒")
    h.clear()
    await h.private(f"/aichat attach {session.id}", user_id=ALICE, settle=1.0)
    check("私聊 attach 群会话被拒", "群聊会话" in await h.wait_reply(), "应被拒")


async def test_invite_flow(h: Harness):
    """受邀才能参与别人的会话；且只能在**同类型**的聊天里用。"""
    print("\n[5b] 邀请流程：别人的会话要受邀才能参与")
    h.clear()
    sa = await establish(h, user_id=ALICE, group_id=GROUP)   # 群会话（归本群）
    h.clear()
    await h.group("Alice 开场//", at_bot=True, user_id=ALICE, settle=1.2)
    await h.wait_for_replies(1)

    # 未受邀 attach 失败
    h.clear()
    await h.group(f"/aichat attach {sa.id}", at_bot=True, user_id=BOB, settle=0.9)
    text = await h.wait_reply()
    check("未受邀者 attach 被拒", "需被邀请" in text or "不在受邀名单" in text,
          text[:140])
    check("拒绝信息给出邀请命令", f"/aichat invite {sa.id}" in text, text[:170])
    bob_focus = _current_session(h, user_id=BOB, group_id=GROUP)
    check("未受邀者焦点没变", bob_focus is None or bob_focus.id != sa.id,
          "焦点被改动了")

    # 非管理者不能邀请
    h.clear()
    await h.group(f"/aichat invite {sa.id} {BOB}", at_bot=True, user_id=BOB, settle=0.9)
    text = await h.wait_reply()
    check("非管理者不能邀请", "只有群主/管理员" in text or "只有所有者" in text,
          text[:140])

    # 创建者（=管理者）邀请
    h.clear()
    await h.group(f"/aichat invite {sa.id} {BOB}", at_bot=True, user_id=ALICE, settle=0.9)
    text = await h.wait_reply()
    check("创建者能邀请", "已邀请" in text, text[:140])
    check("受邀名单落库", h.plugin.store.guests_of(sa.id) == [BOB],
          f"{h.plugin.store.guests_of(sa.id)}")
    after_invite = _current_session(h, user_id=BOB, group_id=GROUP)
    check("邀请不代替对方 attach", after_invite is None or after_invite.id != sa.id,
          "不该自动切过去")

    # 受邀者自己 attach（群里）
    h.clear()
    await h.group(f"/aichat attach {sa.id} --force=true", at_bot=True, user_id=BOB,
                  settle=1.0)
    text = await h.wait_reply()
    check("受邀者能在群里 attach", "已切到" in text, text[:120])
    check("焦点已指向共享会话",
          _current_session(h, user_id=BOB, group_id=GROUP).id == sa.id, "没切过去")

    # ⚠️ 但**不能**在私聊里用这个群会话（群聊/私聊互不相通）
    h.clear()
    await h.private(f"/aichat attach {sa.id}", user_id=BOB, settle=0.9)
    text = await h.wait_reply()
    check("群会话即使受邀也不能在私聊里用", "群聊会话" in text and "不能在私聊" in text,
          text[:140])
    check("私聊焦点没被改动", _current_session(h, user_id=BOB) is None,
          "私聊里不该有焦点")

    # 私聊里要用得另建一个私聊会话
    h.clear()
    await h.private("/aichat new 我的私聊会话", user_id=BOB, settle=1.0)
    await h.wait_for_replies(1)
    ps = _current_session(h, user_id=BOB)
    check("可以另建私聊会话", ps is not None and ps.owner_kind == "user", f"{ps}")
    h.clear()
    await h.private("私聊里说的话//", user_id=BOB, settle=1.2)
    await h.wait_for_replies(1)
    check("私聊会话与群会话上下文互不干扰",
          any("私聊里说的话" in m.text
              for m in h.plugin.store.history(ps.id) if m.role == "user")
          and not any("私聊里说的话" in m.text
                      for m in h.plugin.store.history(sa.id) if m.role == "user"),
          "上下文串了")

    # 取消邀请会摘掉对方的焦点
    h.clear()
    await h.group(f"/aichat uninvite {sa.id} {BOB}", at_bot=True, user_id=ALICE,
                  settle=1.0)
    text = await h.wait_reply()
    check("可以取消邀请", "已取消" in text, text[:120])
    check("取消邀请后对方在本群的焦点被摘掉",
          _current_session(h, user_id=BOB, group_id=GROUP) is None, "焦点还在")


async def test_wrapping(h: Harness):
    print("\n[6] DSH 式消息包装")
    await establish(h)
    h.clear()
    await h.private("看看现在几点//", settle=0.9)
    await h.wait_reply()
    req = h.llm.requests[-1]["messages"]
    check("请求首条是 system", req[0]["role"] == "system", f"{req[0]['role']}")
    check("系统提示含运行上下文", "当前运行上下文" in req[0]["content"],
          req[0]["content"][:200])
    # 排版要求分两类：表格/公式**要主动用**（会渲染成图），其它 Markdown 不要用。
    # 旧断言查的是「纯文本排版」那句 —— 它已经改成这个更细的版本了。
    prompt0 = req[0]["content"]
    check("系统提示要求主动用表格与公式",
          "表格与公式要主动用" in prompt0 and "渲染成图片" in prompt0,
          prompt0[:400])
    check("系统提示仍禁止不渲染的 Markdown",
          "加粗" in prompt0 and ("别用" in prompt0 or "不要用" in prompt0),
          prompt0[:400])
    check("系统提示含会话名", "会话：" in req[0]["content"], req[0]["content"][:400])
    check("系统提示含能力清单", "可用能力" in req[0]["content"]
          and "get_current_time" in req[0]["content"], req[0]["content"][-400:])
    check("system 提示不落盘（历史里没有 system）",
          all(m.role != "system" for m in h.plugin.store.history(
              _current_session(h).id)),
          "历史里出现了 system 消息")
    # 私聊一对一 → 不加前缀
    check("一对一私聊不加说话人前缀",
          not h.llm.last_user_content().startswith("["),
          repr(h.llm.last_user_content()))


async def test_tool_calling(h: Harness):
    print("\n[7] 工具调用")
    await establish(h)
    state = {"n": 0}

    def responder(messages, tools):
        state["n"] += 1
        if state["n"] == 1:
            return StubResult(text="", tool_calls=[StubCall("get_current_time", {}, "c1")])
        return StubResult(text="现在的时间我查到了。")

    h.llm.responder = responder
    try:
        await h.private("现在几点//", settle=1.2)
        reply = await h.wait_reply()
        check("工具后给出最终回复", "查到了" in reply, reply)
        check("发起了 2 次模型调用", state["n"] == 2, f"{state['n']}")
        # 第二次请求里必须带上 tool 结果，且 tool_call 与 tool 结果配对
        second = h.llm.requests[-1]["messages"]
        roles = [m["role"] for m in second]
        check("第二次请求含 assistant 的 tool_calls 与 tool 结果",
              "tool" in roles and "assistant" in roles, f"{roles}")
        check("assistant 消息带 tool_calls",
              any(m.get("tool_calls") for m in second if m["role"] == "assistant"),
              f"{second}")
        tool_msg = [m for m in second if m["role"] == "tool"][0]
        check("tool 结果带 tool_call_id", tool_msg.get("tool_call_id") == "c1",
              f"{tool_msg}")
        check("工具结果里确实有时间",
              "20" in tool_msg["content"] and "-" in tool_msg["content"],
              tool_msg["content"])
        check("工具调用进历史（可追溯）",
              any(b.get("type") == "tool_call" for m in
                  h.plugin.store.history(_current_session(h).id) for b in m.to_dict()["blocks"]),
              "历史里没有 tool_call block")
        names = [t["function"]["name"] for t in (h.llm.requests[0]["tools"] or [])]
        # 私聊工具集 = 基础工具 + 联网搜索（与群无关，私聊也能用）；
        # 关键是**否定面**：群专属工具一个都不能出现（否则等于私聊里给了禁言权限）。
        check("私聊工具集=对话基础工具+联网（无群专属）",
              {"get_current_time", "get_session_info", "search_history"}
              <= set(names)
              and not (set(names) & {"query_group_members", "query_group_info",
                                     "mute_member", "list_group_bans"}),
              f"{names}")
    finally:
        h.llm.responder = None


async def test_plot_tool(h: Harness):
    """函数图像：工具真的被模型看到、调用后图真的发出去。

    这是画图功能的**端到端**验收（工具注册 → 模型请求里的 tools → 执行 → 渲染
    → 标记换成图片）。表达式解析与几何细节在 `tests/test_plot.py` 里单独测。
    """
    print("\n[7.1] 函数图像：plot_function 出图并发出")
    await establish(h)

    def responder(messages, tools):
        # 第一次：模型决定画图；第二次：用自己的话收尾。
        #
        # ⚠️ 第二次**故意不转述工具结果** —— 这是真模型的实际行为（实测会话记录）：
        # 工具返回的那段文字里带着 `[[plot:…]]` 标记，但模型会把它概括成
        # 「画好了，图在下面」，标记就此消失。所以图片必须从**工具结果**里取
        # （`_plots_from_tools`），不能指望模型转述。
        # 这个桩要是写成「原样转发工具结果」，就会掩盖这个 bug（早期版本正是如此）。
        if not any(m["role"] == "tool" for m in messages):
            return StubResult(text="", tool_calls=[StubCall(
                "plot_function", {"expression": "x^2-4", "title": "抛物线"}, "p1")])
        return StubResult(text="画好了，顶点在 (0,-4)，图在下面。")

    h.llm.responder = responder
    try:
        await h.private("画一下 y=x^2-4//", settle=1.5)
        # ⚠️ 正文与图片是**两条**消息（图单独发，避免把正文断成两截），
        # 所以这里等两条。只等一条会随机只看到正文，断言图片时假失败。
        try:
            await h.wait_for_replies(2, timeout=8.0)
        except TimeoutError:
            pass
        # 工具必须在模型看到的清单里（否则模型根本调不到）
        names = [t["function"]["name"] for t in (h.llm.requests[0]["tools"] or [])]
        check("plot_function 在工具清单里", "plot_function" in names, f"{names}")
        # 找到图片消息：QQ 的图片段是 {"type": "image", ...}
        images = []
        texts = []
        for r in h.replies:
            for seg in (r["params"].get("message") or []):
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") == "image":
                    images.append(seg)
                elif seg.get("type") == "text":
                    texts.append(seg.get("data", {}).get("text", ""))
        check("发出了图片", bool(images), f"replies={h.replies}")
        check("图片是本地文件",
              all("file://" in (i.get("data", {}).get("file") or "") for i in images),
              f"{images}")
        joined = "\n".join(texts)
        check("正文里没有留下标记", "[[plot:" not in joined, joined)
        check("正文保留了模型的说明", "顶点" in joined, joined)
        check("模型没转述标记也照样发图（真模型行为）", bool(images),
              "图片是从工具结果里取的，不该依赖模型转述")
        check("工具结果进了历史",
              any(b.get("type") == "tool_call" for m in
                  h.plugin.store.history(_current_session(h).id)
                  for b in m.to_dict()["blocks"]),
              "历史里没有 tool_call block")
    finally:
        h.llm.responder = None


async def test_fetch_image_tool(h: Harness):
    """网图配图：`fetch_image` 被模型看到、调用后图片真的作为单独一条消息发出。

    端到端验收（工具注册 → 出现在请求的 tools 里 → 执行 → 下载校验 → 标记换成
    图片）。魔数判定 / SSRF 闸 / 大小与张数上限这些**安全边界**在
    `tests/test_webimg.py` 里单独测。
    """
    print("\n[7.2] 网图配图：fetch_image 下载并配图")
    await establish(h)
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    calls: list = []

    async def fake_fetch(url, *, max_bytes):
        calls.append((url, max_bytes))
        return PNG

    h.plugin.web._injected_fetch = fake_fetch

    def responder(messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return StubResult(text="", tool_calls=[StubCall(
                "fetch_image", {"url": "https://img.example.com/cat.png",
                                "caption": "一只橘猫"}, "i1")])
        return StubResult(text="找到一张猫的图片，配在下面。")

    h.llm.responder = responder
    h.clear()
    try:
        await h.private("来张猫的图//", settle=1.6)
        try:
            await h.wait_for_replies(2, timeout=8.0)
        except TimeoutError:
            pass
        names = [t["function"]["name"] for t in (h.llm.requests[0]["tools"] or [])]
        check("fetch_image 在工具清单里", "fetch_image" in names, f"{names}")
        images = [seg for r in h.replies
                  for seg in (r["params"].get("message") or [])
                  if isinstance(seg, dict) and seg.get("type") == "image"]
        check("图片真的发出去了", bool(images),
              f"{[r['params'].get('message') for r in h.replies]}")
        check("正文照常发（图单独一条）",
              any("猫" in reply_text(r) for r in h.replies),
              f"{[reply_text(r) for r in h.replies]}")
        check("下载的是模型给的直链",
              bool(calls) and calls[0][0].endswith("cat.png"), f"{calls}")
        system = h.llm.requests[0]["messages"][0]["content"]
        check("系统提示里讲了配图怎么用", "fetch_image" in system, system[:160])
    finally:
        h.llm.responder = None
        h.plugin.web._injected_fetch = None

    # 关掉配图开关后：工具不再出现（讲不存在的工具会让模型空调用一轮）
    cfg = h.plugin.cfg["web_image"]
    old = cfg.get("enabled")
    cfg["enabled"] = False
    h.clear()
    try:
        await h.plugin.on_enable()          # 工具表按当前配置重建
        names2 = h.plugin.tools.names()
        check("关掉后工具不再注册", "fetch_image" not in names2, f"{names2}")
    finally:
        cfg["enabled"] = old
        await h.plugin.on_enable()


async def test_fetch_page_lists_images(h: Harness):
    """`fetch_page` 要列出页面里的**图片直链** —— 模型找到配图地址的唯一途径。

    纯文本里看不到 `<img>`，所以模型「为了找图反复读页面」是必然的：线上实测
    （用户问「给出这几种封装的图片」）它连读 6 轮，一张图没配上、一个字没说。
    """
    print("\n[7.3] fetch_page 列出页面图片直链")
    await establish(h)
    page = ('<html><body><p>产品页</p>'
            '<img src="/img/ind.jpg"><img src="https://cdn.example.com/b.png">'
            '<img src="https://cdn.example.com/logo.svg"></body></html>')

    async def fake_fetch(url, *, max_bytes):
        return page

    h.plugin.web._injected_fetch = fake_fetch

    def responder(messages, tools):
        if not any(m["role"] == "tool" for m in messages):
            return StubResult(text="", tool_calls=[StubCall(
                "fetch_page", {"url": "https://example.com/p.html"}, "f1")])
        return StubResult(text="页面里有两张图。")

    h.llm.responder = responder
    h.clear()
    try:
        await h.private("看看这个页面//", settle=1.6)
        await h.wait_reply()
        joined = " ".join(str(m.get("content") or "")
                          for m in h.llm.requests[-1]["messages"]
                          if m.get("role") == "tool")
        check("相对路径拼成绝对直链", "https://example.com/img/ind.jpg" in joined,
              joined[-260:])
        check("绝对直链原样给出", "https://cdn.example.com/b.png" in joined,
              joined[-260:])
        check("svg 不进清单", ".svg" not in joined, joined[-260:])
        check("写明可以直接 fetch_image", "fetch_image" in joined, joined[-200:])
    finally:
        h.llm.responder = None
        h.plugin.web._injected_fetch = None


async def test_last_round_answers_without_tools(h: Harness):
    """轮次用尽时**最后一轮不给工具**，逼模型把话说出来。

    线上实测（用户问「给出这几种封装的图片」）：模型把 6 轮全花在搜/读上，
    用户最后只收到「（已达单轮上限 6 次模型调用，先说到这里）」—— 什么都没说。
    """
    print("\n[7.4] 轮次用尽也要给个交代")
    await establish(h)
    h.clear()
    rounds: list[list[str]] = []

    def responder(messages, tools):
        names = [t["function"]["name"] for t in (tools or [])]
        rounds.append(names)
        if not names:
            # 没有工具可调了 —— 真模型这时才会开始说话
            return StubResult(text="查到的信息够回答你了：结论是这样。")
        return StubResult(text="", tool_calls=[StubCall(
            "get_current_time", {}, f"c{len(rounds)}")])

    h.llm.responder = responder
    try:
        await h.private("一直查一直查//", settle=2.0)
        reply = await h.wait_reply()
        limit = int(h.plugin.cfg["reply"]["max_rounds"])
        check("调用轮数就是上限", len(rounds) == limit, f"{len(rounds)}")
        check("最后一轮没有工具", rounds[-1] == [], f"{rounds}")
        check("前面几轮有工具", all(r for r in rounds[:-1]), f"{rounds}")
        check("用户拿到了真正的回答", "结论是这样" in reply, reply)
    finally:
        h.llm.responder = None


async def test_tool_permission(h: Harness):
    print("\n[8] 工具权限：群里普通成员拿不到禁言工具")
    gs = await establish(h, group_id=GROUP)
    h.plugin.store.invite(gs.id, BOB)            # 群里两个人都要能说话
    await attach_to(h, gs.id, user_id=BOB, group_id=GROUP)

    def responder(messages, tools):
        return StubResult(text="好的")

    h.llm.responder = responder
    try:
        await h.group("试试//", at_bot=True, user_id=BOB, settle=1.0)
        await h.wait_reply()
        names = [t["function"]["name"] for t in (h.llm.requests[-1]["tools"] or [])]
        check("普通成员看不到 mute_member", "mute_member" not in names, f"{names}")
        check("普通成员仍能用群信息工具", "query_group_info" in names, f"{names}")

        h.clear()
        await h.group("管理员试试//", at_bot=True, user_id=ALICE, role="admin",
                      settle=1.0)
        await h.wait_reply()
        names = [t["function"]["name"] for t in (h.llm.requests[-1]["tools"] or [])]
        check("群管理员能看到 mute_member", "mute_member" in names, f"{names}")
    finally:
        h.llm.responder = None

    # 私聊里不该出现「仅群聊」工具（私聊要有自己的私聊会话）
    await establish(h, user_id=ALICE)
    h.clear()
    await h.private("私聊看看//", settle=1.2)
    await h.wait_reply()
    names = [t["function"]["name"] for t in (h.llm.requests[-1]["tools"] or [])]
    check("私聊里没有群专属工具", "mute_member" not in names
          and "query_group_info" not in names, f"{names}")


async def test_role_command(h: Harness):
    """`/aichat role`：会话级角色设定。

    两条硬要求：
    ① 设定后**每轮**都要进系统提示（否则「设了没反应」）；
    ② 上下文压缩**原样保留**它 —— 它走系统提示而不是消息历史，所以压缩
       （只处理历史消息）根本不碰它。这条是用户明确要求的，单独钉死。
    """
    print("\n[9.2] 角色设定：/aichat role")
    session = await establish(h)
    h.clear()

    # 没设过 → 说明怎么设
    await h.private("/aichat role", settle=0.8)
    reply = await h.wait_reply()
    check("未设角色时给出用法", "还没设角色" in reply and "/aichat role" in reply, reply)

    # 设定角色（带空格的描述要能整体收下）
    h.clear()
    await h.private("/aichat role 你是一只猫娘，说话要带喵", settle=0.8)
    reply = await h.wait_reply()
    check("设定成功", "已设定" in reply, reply)
    check("存到了会话上", "猫娘" in h.plugin.store.get(session.id).role,
          h.plugin.store.get(session.id).role)

    # ① 下一轮的系统提示里必须带上角色
    h.clear()
    await h.private("在吗//", settle=1.0)
    await h.wait_reply()
    system = h.llm.requests[-1]["messages"][0]
    check("角色进了系统提示", "猫娘" in system["content"],
          system["content"][:200])
    check("系统提示里标明优先于默认人格",
          "优先于上面的默认人格" in system["content"], "")
    check("角色不在普通消息里（只走系统提示）",
          not any("猫娘" in str(m.get("content") or "")
                  for m in h.llm.requests[-1]["messages"][1:]),
          "角色不该出现在消息历史里")

    # 查看
    h.clear()
    await h.private("/aichat role", settle=0.8)
    reply = await h.wait_reply()
    check("能看回当前角色", "猫娘" in reply, reply)

    # ② 压缩之后角色原样保留
    policy = h.plugin.policy
    policy.max_tokens = 400
    try:
        h.clear()
        for i in range(8):
            await h.private(f"第{i}条比较长的消息" + "内容" * 30 + "//", settle=0.9)
            await h.wait_reply()
        check("确实压缩过", h.plugin.store.get(session.id).compact_count >= 1,
              f"compact_count={h.plugin.store.get(session.id).compact_count}")
        check("压缩后角色一字未改",
              h.plugin.store.get(session.id).role == "你是一只猫娘，说话要带喵",
              h.plugin.store.get(session.id).role)
        # 摘要请求里要告知角色（否则接手摘要可能与角色冲突），但不能改写它
        summary_reqs = [r for r in h.llm.requests
                        if "压缩引擎" in r["messages"][0]["content"]]
        check("摘要请求里告知了角色",
              any("猫娘" in r["messages"][1]["content"] for r in summary_reqs),
              "摘要请求没带角色设定")
        check("摘要请求要求不要改写角色",
              any("不要改写" in r["messages"][1]["content"] for r in summary_reqs),
              "没有明确要求别动角色")
        # 压缩之后下一轮的系统提示仍然带着角色
        h.clear()
        await h.private("压缩之后还在吗//", settle=1.0)
        await h.wait_reply()
        check("压缩后角色仍进系统提示",
              "猫娘" in h.llm.requests[-1]["messages"][0]["content"], "")
        # 压缩产生的检查点里不该出现角色正文（角色不参与压缩）
        history = h.plugin.store.history(session.id)
        checkpoints = [m for m in history
                       if m.metadata.get("origin") == "system.compact"]
        check("检查点里没有角色正文",
              not any("猫娘" in m.text for m in checkpoints),
              "角色被写进摘要了（应当只存在于系统提示）")
    finally:
        policy.max_tokens = 16384

    # 只给会话名/id 时应当是「查看」而不是把 id 当角色
    h.clear()
    await h.private(f"/aichat role {session.id}", settle=0.8)
    reply = await h.wait_reply()
    check("只给会话时是查看（不覆盖角色）",
          "猫娘" in reply
          and h.plugin.store.get(session.id).role == "你是一只猫娘，说话要带喵",
          reply)

    # 指定会话 + 角色
    h.clear()
    await h.private(f"/aichat role {session.id} 你是一位数学老师", settle=0.8)
    await h.wait_reply()
    check("指定会话也能设",
          h.plugin.store.get(session.id).role == "你是一位数学老师",
          h.plugin.store.get(session.id).role)

    # 清除
    h.clear()
    await h.private("/aichat role --clear=true", settle=0.8)
    reply = await h.wait_reply()
    check("能清除角色", h.plugin.store.get(session.id).role == "", reply)
    h.clear()
    await h.private("清掉之后//", settle=1.0)
    await h.wait_reply()
    check("清除后系统提示不再有角色段",
          "你在本会话中扮演的角色" not in h.llm.requests[-1]["messages"][0]["content"],
          "")

    # 引号：框架会把 CJK 引号归一成 ASCII 再按 shlex 分词，**成对的引号会被
    # 当成定界符吃掉**（`带「喵」字` → `带喵字`）。这是命令解析层的既有约定
    # （第一节 / 命令框架文档），不是角色功能的问题；想保留引号本身要转义。
    h.clear()
    await h.private('/aichat role 说话要带\\"喵\\"字', settle=0.8)
    await h.wait_reply()
    check("转义的引号能原样进角色",
          h.plugin.store.get(session.id).role == '说话要带"喵"字',
          repr(h.plugin.store.get(session.id).role))

    # 长度上限
    h.clear()
    await h.private("/aichat role " + "啊" * 2500, settle=0.8)
    reply = await h.wait_reply()
    check("超长角色被拒绝", "太长" in reply, reply[:150])


async def test_role_permission(h: Harness):
    """角色设定是会话级配置，只有所有者/超管能改。"""
    print("\n[9.3] 角色设定的权限")
    gs = await establish(h, group_id=GROUP)
    h.plugin.store.invite(gs.id, BOB)
    await attach_to(h, gs.id, user_id=BOB, group_id=GROUP)

    # ⚠️ 群会话的焦点落在**群**里（`focus_key` 带 chat 这一段），所以这些命令
    # 必须从群里发。用私聊发会显示「你在此处没有当前会话」—— 那是**正确行为**
    # （同一个人在私聊和群里是两套焦点），不是权限问题。
    h.clear()
    await h.group("/aichat role", at_bot=True, user_id=BOB, settle=0.9)
    reply = await h.wait_reply()
    check("受邀者能查看角色", "还没设角色" in reply or "当前角色" in reply, reply)

    h.clear()
    await h.group("/aichat role 我要当管理员", at_bot=True, user_id=BOB, settle=0.9)
    reply = await h.wait_reply()
    check("受邀者不能改角色", "⚠️" in reply or "只有" in reply, reply)
    check("受邀者的尝试没有生效",
          h.plugin.store.get(gs.id).role == "", h.plugin.store.get(gs.id).role)

    # 所有者可以
    h.clear()
    await h.group(f"/aichat role {gs.id} 你是一位数学老师", at_bot=True,
                  user_id=ALICE, settle=0.9)
    await h.wait_reply()
    check("所有者可以设", h.plugin.store.get(gs.id).role == "你是一位数学老师",
          h.plugin.store.get(gs.id).role)

    # 超管可以
    h.clear()
    await h.group(f"/aichat role {gs.id} 超管改的", at_bot=True,
                  user_id=SUPERUSER, settle=0.9)
    await h.wait_reply()
    check("超管可以设", h.plugin.store.get(gs.id).role == "超管改的",
          h.plugin.store.get(gs.id).role)

    # 状态里要看得到角色。
    # ⚠️ 必须问**有焦点的本人**（ALICE 是群会话所有者，建立时有焦点）。
    # 超管虽然能改，但超管在群里并没有 attach 过 —— `/aichat current` 报的是
    # 「**你**在此处的焦点」，没有就是没有，这不是权限问题（实测踩过）。
    h.clear()
    await h.group("/aichat current", at_bot=True, user_id=ALICE, settle=0.9)
    reply = await h.wait_reply()
    check("current 里显示角色", "角色" in reply and "超管改的" in reply, reply)


async def test_role_persistence(h: Harness):
    """角色设定必须落盘，重启后还在。

    它每轮都决定 AI 怎么说话，丢了就是「重启之后人设突然没了」——
    而用户唯一的线索只是「感觉它变了」。
    """
    print("\n[9.4] 角色设定的持久化")
    SessionStore = plugin_module("store").SessionStore
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="role-persist-",
                                   dir=str(ROOT / "tests" / ".test-tmp")))
    st = SessionStore(str(tmpdir))
    s1 = st.create("带角色的会话", owner_kind="user", owner_id=ALICE)
    st.set_role(s1.id, "你是一位数学老师\n多用生活例子")
    st.append(s1.id, plugin_module("segments").user_message("随便一句"))

    # 重新打开（等价于进程重启）
    st2 = SessionStore(str(tmpdir))
    got = st2.get(s1.id)
    check("角色落盘并在重开后还在", got is not None
          and got.role == "你是一位数学老师\n多用生活例子",
          f"{got.role if got else None!r}")
    check("重开后历史仍在", len(st2.history(s1.id)) == 1,
          f"{len(st2.history(s1.id))}")
    # 多行内容要原样保留（角色常常是分行写的）
    check("多行角色原样保留", "\n" in got.role, repr(got.role))

    # 清除后也要落盘
    st2.set_role(s1.id, "")
    st3 = SessionStore(str(tmpdir))
    check("清除后重开仍为空", (st3.get(s1.id).role or "") == "",
          repr(st3.get(s1.id).role))

    # 旧 state.json（没有 role 键）要能读，默认空
    import json as _json
    state = _json.loads((Path(tmpdir) / "sessions" / "state.json").read_text("utf-8"))
    check("state.json 里有 role 字段",
          "role" in list(state["sessions"].values())[0],
          f"{list(state['sessions'].values())[0]}")
    for v in state["sessions"].values():
        v.pop("role", None)
    (Path(tmpdir) / "sessions" / "state.json").write_text(
        _json.dumps(state, ensure_ascii=False), encoding="utf-8")
    st4 = SessionStore(str(tmpdir))
    check("旧记录缺 role 键也能读", (st4.get(s1.id).role or "") == "",
          repr(st4.get(s1.id).role))


async def test_compaction(h: Harness):
    print("\n[9] 上下文压缩")
    policy = h.plugin.policy
    session = await establish(h)
    policy.max_tokens = 400          # 压小预算，几条消息就能触发
    h.clear()
    try:
        # 塞满历史
        for i in range(8):
            await h.private(f"第{i}条比较长的消息" + "内容" * 30 + "//", settle=0.9)
            await h.wait_reply()
        history = h.plugin.store.history(session.id)
        check("历史被压缩过", session.compact_count >= 1,
              f"compact_count={session.compact_count}")
        check("压缩后历史变短（消息数少于累积轮数×2）", len(history) < 16,
              f"{len(history)}")
        # 摘要以 user 角色、带 compact 来源落在历史开头
        first = history[0]
        check("首条是压缩检查点", first.metadata.get("origin") == "system.compact",
              f"{first.metadata}")
        check("检查点用 user 角色（协议安全）", first.role == "user", first.role)
        check("检查点带包框标签", "<compacted-summary>" in first.text,
              first.text[:120])
        check("检查点带前言（别复述）", "不要复述" in first.text, first.text[:200])
        check("检查点记了吸收条数",
              int(first.metadata.get("absorbed") or 0) > 0, f"{first.metadata}")
        # 压缩后仍然配对平衡
        pairing_balanced = plugin_module("compaction").pairing_balanced
        check("压缩后 tool 配对仍然平衡", pairing_balanced(history),
              "配对被破坏")
        # 摘要调用本身也走了 LLM
        check("摘要是一次真实模型调用",
              any("压缩引擎" in r["messages"][0]["content"] for r in h.llm.requests),
              "没有找到摘要请求")
        # 压缩后仍能继续对话
        h.clear()
        await h.private("压缩之后接着说//", settle=1.0)
        reply = await h.wait_reply()
        check("压缩后对话仍正常", "我收到了" in reply or "看到" in reply, reply)
    finally:
        policy.max_tokens = 16384


async def test_compaction_refuses_bad_summary(h: Harness):
    print("\n[10] 压缩守门：摘要不比原文短就放弃")
    policy = h.plugin.policy
    session = await establish(h)
    policy.max_tokens = 400
    h.clear()
    saved = h.llm.summary
    h.llm.summary = "太短"          # 会被下界拦掉
    try:
        for i in range(6):
            await h.private(f"填充{i}" + "内容" * 30 + "//", settle=0.9)
            await h.wait_reply()
        check("劣质摘要被拒绝，未记压缩次数", session.compact_count == 0,
              f"compact_count={session.compact_count}")
        check("拒绝原因可查", h.plugin.compactor.last_error is not None,
              "last_error 为空")
    finally:
        h.llm.summary = saved
        policy.max_tokens = 16384


async def test_compaction_split_safety(h=None):
    print("\n[11] 压缩切点不破坏 tool 配对")
    CompactionPolicy = plugin_module("compaction").CompactionPolicy
    pairing_balanced = plugin_module("compaction").pairing_balanced
    _seg = plugin_module("segments")
    assistant_message = _seg.assistant_message
    tool_message = _seg.tool_message
    user_message = _seg.user_message
    Block = plugin_module("segments").Block

    history = []
    for i in range(6):
        history.append(user_message(f"问题{i}" + "字" * 40))
        history.append(assistant_message(
            "", tool_calls=[Block.tool_call(f"c{i}", "get_current_time", {})]))
        history.append(tool_message(f"c{i}", "结果" * 40, name="get_current_time"))
        history.append(assistant_message(f"回答{i}" + "字" * 40))
    policy = CompactionPolicy(max_tokens=200, threshold_ratio=0.8, retain_ratio=0.16,
                              chars_per_token=1.6)
    split = policy.find_split(history)
    check("找到了切点", split > 0, f"split={split}")
    check("切点前配对平衡", pairing_balanced(history[:split]), f"split={split}")
    check("切点后配对平衡", pairing_balanced(history[split:]), f"split={split}")
    check("切点不会落在 assistant(tool_call) 与其 tool 结果之间",
          history[split - 1].role != "assistant"
          or not any(b.type == "tool_call" for b in history[split - 1].blocks),
          f"前一条={history[split - 1].role}")

    # 尾部不带配对 → 整体放弃（宁可超预算也不送非法历史）
    broken = history[:1] + [assistant_message(
        "", tool_calls=[Block.tool_call("zz", "get_current_time", {})])]
    policy2 = CompactionPolicy(max_tokens=300, retain_ratio=0.9, retain_tokens=1)
    check("尾部不完整时放弃压缩", policy2.find_split(broken) == 0,
          f"split={policy2.find_split(broken)}")


async def test_target_commands(h: Harness):
    """服务范围管理：`add` / `del` / `targets`。"""
    ChatTarget = plugin_module("store").ChatTarget

    print("\n[12] 服务范围：add / del / targets（超级用户 · 仅私聊）")
    await establish(h)
    h.clear()
    await h.private("/aichat add group 999999", user_id=BOB, settle=0.7)
    check("普通用户不能用 add", "未知命令" in await h.wait_reply(), "不该放行")

    # 超级用户私聊：给一个新群开通
    h.clear()
    await h.private("/aichat add group 888888", user_id=SUPERUSER, settle=1.0)
    text = await h.wait_reply()
    check("超级用户可开通群", "已开通群 888888" in text, text[:120])
    check("写进了 binding.groups", 888888 in h.plugin.cfg["binding"]["groups"],
          f"{h.plugin.cfg['binding']['groups']}")
    saved = json.loads(h.cfg_path.read_text(encoding="utf-8"))
    check("并落盘到 config.json",
          888888 in saved["plugins"]["ai_chat"]["binding"]["groups"],
          f"{saved['plugins']['ai_chat']['binding']['groups']}")
    check("开通后该群可用",
          h.plugin.check_access(ChatTarget("group", 888888), ALICE)[0], "仍被拒")

    h.clear()
    await h.private("/aichat add group 888888", user_id=SUPERUSER, settle=0.9)
    check("重复开通给出提示", "已开通" in await h.wait_reply(), "应提示已开通")

    # 用户类型
    h.clear()
    await h.private("/aichat add user 777777", user_id=SUPERUSER, settle=1.0)
    check("可开通具体用户", 777777 in h.plugin.cfg["binding"]["users"],
          f"{h.plugin.cfg['binding']['users']}")
    check("开通后该用户私聊可用",
          h.plugin.check_access(ChatTarget("private", 777777), 777777)[0], "仍被拒")

    # targets：同时展示服务范围与各人焦点
    h.clear()
    await h.private("/aichat targets", user_id=SUPERUSER, settle=0.9)
    text = await h.wait_reply()
    check("targets 展示服务范围", "服务范围" in text and "群（" in text, text[:150])
    check("targets 展示会话焦点", "会话焦点" in text, text[:300])
    check("targets 给出用法提示", "/aichat add" in text, text[-200:])

    # 停止服务
    h.clear()
    await h.private("/aichat del group 888888", user_id=SUPERUSER, settle=1.0)
    text = await h.wait_reply()
    check("可以停止服务", "已停止" in text, text[:120])
    check("停止后该群不可用",
          not h.plugin.check_access(ChatTarget("group", 888888), ALICE)[0], "仍可用")
    h.clear()
    await h.private("/aichat del user 999999", user_id=SUPERUSER, settle=0.8)
    check("停止未开通的目标有提示",
          "不在服务范围内" in await h.wait_reply(), "应有提示")

    # 群里省略号码 = 当前群
    h.clear()
    await h.group("/aichat add group", at_bot=True, user_id=SUPERUSER, settle=1.0)
    texts = [reply_text(r) for r in h.replies]
    check("群里 add group 省略号码 → 当前群",
          any(str(GROUP) in t or "未知命令" in t for t in texts), f"{texts}")


async def test_session_access_guard(h: Harness):
    """会话可用性：只有所有者与受邀者能用；超级用户也不豁免。"""
    ChatTarget = plugin_module("store").ChatTarget

    print("\n[13] 会话可用性守卫（归属 + 受邀）")
    await establish(h)
    own = _current_session(h)                    # Alice 的私聊会话
    check("当前会话存在", own is not None, "没有当前会话")

    group = ChatTarget("group", GROUP)
    # 类型隔离优先：私聊会话在群里对**任何人**都不可用（超级用户也不豁免）
    check("私聊会话在群里一律不可用（类型隔离优先于归属）",
          not h.plugin._may_attach(own, group, SUPERUSER)[0], "不该放行")

    # 群会话：创建者能用，别人要受邀
    gs = await establish(h, user_id=ALICE, group_id=GROUP)
    check("创建者能用自己建的群会话",
          h.plugin._may_attach(gs, group, ALICE)[0], "创建者被拦")
    ok, why = h.plugin._may_attach(gs, group, BOB)
    check("群成员未受邀不能 attach", not ok, f"ok={ok}")
    check("拒绝信息给出邀请命令", "/aichat invite" in why, why[:110])

    h.plugin.store.invite(gs.id, BOB)
    ok3, why3 = h.plugin._may_attach(gs, group, BOB)
    check("受邀后能在群里用", ok3, f"ok={ok3} why={why3}")
    ok4, why4 = h.plugin._may_attach(gs, ChatTarget("private", BOB), BOB)
    check("受邀后也不能在私聊里用群会话（类型隔离）", not ok4, f"ok={ok4}")
    check("私聊拒绝信息说明类型不符", "群聊会话" in why4, why4[:110])

    # 私聊会话：本人可用
    okp, _ = h.plugin._may_attach(own, ChatTarget("private", ALICE), ALICE)
    check("私聊会话本人可用", okp, "本人被拦")

    # 超级用户不豁免「用别人的会话」
    ok5, why5 = h.plugin._may_attach(own, group, SUPERUSER)
    check("超级用户也不能直接翻别人的会话", not ok5, f"ok={ok5} why={why5}")

    # 群主/管理员可以代管群归属会话，但不能代管个人会话
    h.plugin.perm.set_group_role(GROUP, ALICE, "admin")   # 模拟群角色缓存
    okm, why_m = h.plugin._may_manage(gs, ALICE)
    check("群管理员可代管群归属会话", okm, f"管理员被拦：{why_m[:60]}")
    okm_bob, _ = h.plugin._may_manage(gs, BOB)
    check("普通群成员不能代管群归属会话", not okm_bob, "普通成员不该能管")
    okm2, why_m2 = h.plugin._may_manage(own, BOB)
    check("别人不能管理你的个人会话", not okm2, f"ok={okm2}")
    check("拒绝信息说明只有所有者能管", "只有所有者" in why_m2, why_m2[:100])


async def test_model_and_current(h: Harness):
    print("\n[14] model / current（stats、switch 已删除）")
    await establish(h, user_id=SUPERUSER)
    for gone in ("/aichat stats", "/aichat switch x"):
        h.clear()
        await h.private(gone, user_id=SUPERUSER, settle=0.6)
        text = await h.wait_reply()
        # 子命令名写错 = 用法不正确 → 框架重定向到命令组帮助页（列着现存的子命令）
        check(f"{gone} 已不存在（回命令组帮助页，页里没有它）",
              text.startswith("📖 /aichat")
              and f"/aichat {gone.split()[1]}" not in text, text[:120])

    h.clear()
    await h.private("/aichat model", user_id=SUPERUSER, settle=0.7)
    text = await h.wait_reply()
    check("model 显示当前值", "stub-model" in text, text)
    check("model 不泄露完整 key", "sk-stub" not in text, text)

    h.clear()
    await h.private("/aichat model --context_limit=2048", user_id=SUPERUSER, settle=0.7)
    text = await h.wait_reply()
    check("model 可改上下文预算", h.plugin.policy.max_tokens == 2048, text)
    h.plugin.policy.max_tokens = 16384

    h.clear()
    await h.private("/aichat current", user_id=SUPERUSER, settle=0.7)
    text = await h.wait_reply()
    check("current 显示会话与用量", "会话" in text and "token" in text, text)


async def test_detach_drops_staged(h: Harness):
    print("\n[15.5] detach 丢弃暂存区的未提交内容")
    await establish(h)
    h.clear()
    # 分段发两段（没有结束符 → 留在暂存区）
    await h.private("第一段没写完")
    await h.private("第二段也没写完")
    key = f"private:{ALICE}"          # ChatTarget.key：私聊是 `private:<uid>`
    state = h.plugin.buffer.peek(key)
    check("两段都已暂存", len(state.get("items") or []) == 2, f"{state}")
    check("没有结束符就没提交", h.llm.calls == 0, f"calls={h.llm.calls}")

    h.clear()
    await h.private("/aichat detach", settle=0.8)
    text = await h.wait_reply()
    check("detach 不再拒绝（解除成功）", "已取消你在" in text, text)
    check("回执说明丢了几段", "2 段" in text and "丢弃" in text, text)
    check("暂存区已清空", not (h.plugin.buffer.peek(key).get("items")),
          f"{h.plugin.buffer.peek(key)}")
    check("丢弃的内容没有被提交", h.llm.calls == 0, f"calls={h.llm.calls}")
    ChatTarget = plugin_module("store").ChatTarget
    check("当前会话已解除",
          h.plugin.store.focus_of(ChatTarget("private", ALICE), ALICE) is None,
          f"{h.plugin.store.focus_of(ChatTarget('private', ALICE), ALICE)}")

    # 暂存区本来就空时：照旧正常解除，不出现「丢弃」字样
    h.clear()
    await h.private("/aichat attach 预热会话", settle=0.8)
    await h.wait_reply()
    h.clear()
    await h.private("/aichat detach", settle=0.8)
    text2 = await h.wait_reply()
    check("空暂存区时回执不提丢弃", "已取消你在" in text2 and "丢弃" not in text2, text2)


async def test_staging_shared_across_windows(h: Harness):
    """同一个人在主私聊与**群小窗**里共用一个暂存区（也共用一个焦点会话）。

    ⚠️ 暂存区原来按 `target.key` 存，而群临时会话（小窗）的 key 是
    `private:<uid>@group:<gid>` —— 于是在主私聊里压栈、在小窗里发结束符，
    会被判成「暂存区是空的」，**压进去的内容再也提交不了**（用户报过）。
    人就是同一个人，键不该跟着「从哪个窗口打字」走。

    同一个 bug 也在**会话焦点**上（`focus_key` 原先也建在 `chat.key` 上）：
    换个窗口说话就「没有会话」了。两处一起改。
    """
    print("\n[21.8] 主私聊与群小窗共用一个暂存区 / 焦点")
    old_terms = list(h.plugin.cfg["trigger"]["terminators"])
    h.plugin.cfg["trigger"]["terminators"] = ["#"]
    try:
        await establish(h)
        conv = f"private:{ALICE}"
        h.clear()
        # ① 主私聊里压栈
        await h.private("画y=sinx+1000，x、y每格1个单位", settle=0.3)
        check("暂存键是「人」而不是「窗口」", list(h.plugin.buffer.keys()) == [conv],
              f"{list(h.plugin.buffer.keys())}")
        # ② 从群小窗发结束符
        await h.private("#", settle=1.2, group_id=GROUP)
        check("小窗发结束符能提交主私聊压的内容", h.llm.calls == 1,
              f"calls={h.llm.calls}")
        check("提交的正是那段内容",
              "画y=sinx+1000" in h.llm.last_user_content(),
              h.llm.last_user_content())
        check("暂存区已清空",
              not (h.plugin.buffer.peek(conv).get("items")),
              f"{h.plugin.buffer.peek(conv)}")

        # ③ 焦点也是同一个：换窗口说话仍认得当前会话
        h.clear()
        await h.private("换个窗口接着聊#", settle=1.2, group_id=GROUP)
        check("小窗里也有会话（焦点没分裂）", h.llm.calls == 1,
              f"calls={h.llm.calls}")
    finally:
        h.plugin.cfg["trigger"]["terminators"] = old_terms


async def test_window_key_migration(h: Harness):
    """老状态里的「群小窗键」要并回主键，不能因为改了键就丢。"""
    print("\n[21.9] 群小窗键的迁移")
    store_mod = plugin_module("store")
    check("焦点键：小窗并回主键",
          store_mod.SessionStore._migrate_bindings(
              {"private:7@group:9@user:7": "s1"}) == {"private:7@user:7": "s1"},
          f"{store_mod.SessionStore._migrate_bindings({'private:7@group:9@user:7': 's1'})}")
    check("主键已有值时保留主键",
          store_mod.SessionStore._migrate_bindings(
              {"private:7@user:7": "s1", "private:7@group:9@user:7": "s2"})
          == {"private:7@user:7": "s1"}, "")
    check("群键不受影响",
          store_mod.SessionStore._migrate_bindings({"group:9@user:7": "s1"})
          == {"group:9@user:7": "s1"}, "")

    # 暂存区：老的 `private:U@group:G` 要并进 `private:U`
    path = h.tmp / "staging-migrate.json"
    path.write_text(json.dumps({
        "private:7@group:9": {"items": [{"text": "小窗里压的一段"}]},
        "private:7": {"items": [{"text": "主私聊的一段"}]},
        "group:9": {"items": [{"text": "群里的"}]},
    }, ensure_ascii=False), encoding="utf-8")
    buf = plugin_module("main")._StagingBuffer(str(path))
    items = [it.get("text") for it in buf.peek("private:7").get("items") or []]
    check("小窗那段并进了主键", "小窗里压的一段" in items, f"{items}")
    check("并进去的顺序是「主键原本的 + 小窗的」",
          items == ["主私聊的一段", "小窗里压的一段"], f"{items}")
    check("原有的那段还在", "主私聊的一段" in items, f"{items}")
    check("群键原样保留",
          [it.get("text") for it in buf.peek("group:9").get("items") or []]
          == ["群里的"], f"{buf.peek('group:9')}")


async def test_recall_removes_staged(h: Harness):
    print("\n[15] 撤回的消息离开暂存区")
    await establish(h)
    h.clear()
    await h.group("这条要撤回", at_bot=True, message_id=555)
    state = h.plugin.buffer.peek(f"group:{GROUP}")
    check("消息已暂存", len(state.get("items") or []) == 1, f"{state}")
    await h._send_raw({
        "post_type": "notice", "notice_type": "group_recall",
        "group_id": GROUP, "user_id": ALICE, "operator_id": ALICE,
        "message_id": 555, "self_id": BOT_ID, "time": 1700000001,
    }, settle=0.5)
    state = h.plugin.buffer.peek(f"group:{GROUP}")
    check("撤回后暂存区清空", not (state.get("items")), f"{state}")


async def test_paragraph_labels(h: Harness):
    """请求里每条**文本段落**带 `[#N]` 编号（N = 会话内 seq），系统提示讲清含义。

    编号是「引用用标号代替原文」的前提：没有编号就没法指代，只能重抄。
    """
    print("\n[19.8] 段落编号 [#N]")
    await establish(h)
    h.clear()
    h.llm.script_text = "第一答"
    await h.private("第一问//", settle=1.2)
    h.llm.script_text = "第二答"
    await h.private("第二问//", settle=1.2)

    msgs = h.llm.requests[-1]["messages"]
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    assists = [m["content"] for m in msgs if m.get("role") == "assistant"]
    session = _current_session(h)
    seqs = [int(m.seq or 0) for m in h.plugin.store.history(session.id) if m.seq]
    check("用户消息带编号", all(u.startswith("[#") for u in users), f"{users}")
    check("回答也带编号", all(a.startswith("[#") for a in assists), f"{assists}")
    check("编号就是会话 seq",
          [int(u[2:u.index("]")]) for u in users] == [seqs[0], seqs[2]],
          f"{[u[:8] for u in users]} vs seq={seqs}")
    check("工具消息不加编号",
          all("[#" not in str(m.get("content") or "")
              for m in msgs if m.get("role") == "tool"),
          f"{[m for m in msgs if m.get('role') == 'tool']}")
    check("系统提示里说明了编号",
          "[#N]" in h.llm.requests[-1]["messages"][0]["content"],
          h.llm.requests[-1]["messages"][0]["content"][:200])


async def test_quote_uses_label(h: Harness):
    """引用上文时：文本还在上下文里 → 只给 `[#N]`，**不重抄原文**、也不调 get_msg。"""
    print("\n[19.9] 引用在上文里 → 用标号代替原文")
    session = await establish(h)
    h.clear()
    h.llm.script_text = "这是要被引用的那段回答。"
    await h.private("讲一下椭圆//", settle=1.2)
    answered = h.replies  # noqa: F841 - 只为确保上一轮真的答完
    hist = h.plugin.store.history(session.id)
    bot_msg = next(m for m in reversed(hist) if m.role == "assistant")
    label = int(bot_msg.seq or 0)
    check("回答有 seq（可被编号）", label > 0, f"{label}")
    # 让机器人自己那条也能被「按 QQ 消息 id 认出来」
    sent_id = 950001
    h.plugin.sent_reply.add([sent_id], session.id, label)
    h.msg_lookup[sent_id] = {
        "message_id": sent_id, "message_type": "private",
        "sender": {"user_id": BOT_ID, "nickname": "测试机器人"},
        "message": [{"type": "text", "data": {"text": "这是要被引用的那段回答。"}}],
    }
    h.clear()
    before = len(h.calls)
    h.llm.script_text = "好的，接着说。"
    await h.private("那它的焦点呢？//", with_reply=sent_id, settle=1.3)
    seen = h.llm.last_user_content()
    check(f"用标号指代 [#{label}]", f"[#{label}]" in seen, seen)
    check("没有重抄原文", "这是要被引用的那段回答" not in seen, seen)
    check("用户的问题还在", "那它的焦点呢" in seen, seen)
    gets = [c for c in h.calls[before:] if c.get("action") == "get_msg"]
    check("不再为这次引用调 get_msg（省掉 1~5 秒节流）", not gets, f"{gets}")


async def test_quote_chain_labels(h: Harness):
    """链式引用：被引用的那条自己也引用了上文 → 两个标号都要带出来。"""
    print("\n[19.95] 链式引用（引用的引用）")
    session = await establish(h)
    h.clear()
    # 造两条历史消息：m1 是普通回答；m2 是一条**引用着 m1** 的用户输入
    h.llm.script_text = "第一段回答。"
    await h.private("第一问//", settle=1.2)
    hist = h.plugin.store.history(session.id)
    first = next(m for m in reversed(hist) if m.role == "assistant")
    first_label = int(first.seq or 0)
    q1 = next(m for m in reversed(hist) if m.role == "user")
    q1_label = int(q1.seq or 0)
    # 让 q1 也「可被按 QQ id 认出」，并把它的 reply_to 指向更早的一条
    q1.metadata.setdefault("message_ids", []).append(961001)
    h.plugin.store.replace_history(session.id, h.plugin.store.history(session.id))
    # 第二层：另一条消息引用 q1（q1 自己又引用着 first）
    h.msg_lookup[961001] = {
        "message_id": 961001, "message_type": "private",
        "sender": {"user_id": ALICE, "nickname": "爱丽丝"},
        "message": [{"type": "text", "data": {"text": "第一问"}}],
    }
    q1.metadata["reply_to"] = 950001
    # 第二环指向的是一条**机器人回答**：要能被按消息 id 认出来，得先进索引
    # （线上由 `_remember_sent` 在发送后写入）
    h.plugin.sent_reply.add([950001], session.id, first_label)
    h.msg_lookup[950001] = {
        "message_id": 950001, "message_type": "private",
        "sender": {"user_id": BOT_ID, "nickname": "测试机器人"},
        "message": [{"type": "text", "data": {"text": "第一段回答。"}}],
    }
    h.clear()
    h.llm.script_text = "第二段回答。"
    await h.private("继续//", with_reply=961001, settle=1.3)
    seen = h.llm.last_user_content()
    check("第一层用标号", f"[#{q1_label}]" in seen, seen)
    check("第二层（引用的引用）也用标号", f"[#{first_label}]" in seen, seen)
    check("两段原文都没重抄",
          "第一问" not in seen and "第一段回答" not in seen, seen)


async def test_quote_falls_back_to_text(h: Harness):
    """引用的上文**已不在上下文里**时，退回贴原文（老行为）。"""
    print("\n[19.97] 引用不在上下文里 → 贴原文")
    await establish(h)
    h.clear()
    h.msg_lookup[970001] = {
        "message_id": 970001, "message_type": "private",
        "sender": {"user_id": BOB, "nickname": "鲍勃"},
        "message": [{"type": "text", "data": {"text": "很久以前的一条消息"}}],
    }
    h.llm.script_text = "收到。"
    await h.private("这个怎么理解？//", with_reply=970001, settle=1.3)
    seen = h.llm.last_user_content()
    check("原文被贴进来了", "很久以前的一条消息" in seen, seen)
    check("带上发送人", "鲍勃" in seen, seen)
    check("没有凭空编标号", "[#" not in seen.split("对方")[-1] if "对方" in seen
          else True, seen)


async def test_terminator_waits_for_ingest(h: Harness):
    """结束符要先**等**那条还在读引用的消息落进暂存区，不能直接说「暂存区是空的」。

    线上复现（2026-10-09 21:40，用户截图）：21:40:36 发「重试 + 引用」，
    21:40:38 发 `#`，机器人回「暂存区是空的」—— 因为读引用要走 `get_msg`，
    而框架给每次 API 调用加了 `api_delay`（线上 1~5 秒），内容 21:40:39 才进
    暂存区。用户那一按白费了，内容还卡在暂存区里等着下一次 `#`。
    """
    print("\n[19.7] 结束符要等「还在读引用」的内容")
    await establish(h)
    h.clear()
    qid = 880001
    h.msg_lookup[qid] = {
        "message_id": qid, "message_type": "private",
        "sender": {"user_id": ALICE, "nickname": "爱丽丝"},
        "message": [{"type": "text", "data": {"text": "被引用的历史问题"}}],
    }
    h.msg_delay = 0.8          # 读引用要 0.8 秒 → 结束符一定先到
    try:
        h.llm.script_text = "好的，这是重试后的回答。"
        await h.private("重试", with_reply=qid, settle=0.05)   # 故意不等它消化完
        await h.private("//", settle=1.6)                     # 紧接着按结束符
        await h.wait_reply()
        texts = [reply_text(r) for r in h.replies]
        check("没有回「暂存区是空的」",
              not any("暂存区是空的" in t for t in texts), f"{texts}")
        seen = h.llm.last_user_content()
        check("内容确实提交给了模型", "重试" in seen, seen)
        check("引用内容也在", "被引用的历史问题" in seen, seen)
    finally:
        h.msg_delay = 0.0

    # 反证：把这个「等一等」拿掉，同样的时序**确实**会误报 —— 说明这条用例
    # 不是摆设（改坏了会红）。顺手也解释了线上那句「暂存区是空的」是怎么来的。
    h.clear()
    h.msg_delay = 0.8
    real_wait = h.plugin._await_pending_content

    async def _no_wait(key):
        return []

    h.plugin._await_pending_content = _no_wait
    try:
        await h.private("再试一次", with_reply=qid, settle=0.05)
        await h.private("//", settle=1.2)
        texts = [reply_text(r) for r in h.replies]
        check("去掉等待后确实会误报「暂存区是空的」",
              any("暂存区是空的" in t for t in texts), f"{texts}")
    finally:
        h.plugin._await_pending_content = real_wait
        # 那半截内容这时**卡在暂存区里**（正是线上那段的处境）—— 等它消化完再清掉，
        # 否则清早了、它随后落进来，会把后面那条「真的没内容」的用例带偏。
        key = f"private:{ALICE}"
        for _ in range(40):
            if not h.plugin._ingesting.get(key):
                break
            await asyncio.sleep(0.1)
        h.plugin.buffer.take(key)
        h.msg_delay = 0.0

    # 反例：真的什么都没发时，还是要照实说「暂存区是空的」
    h.clear()
    await h.private("//", settle=0.8)
    text = await h.wait_reply()
    check("真的没内容时照实回绝", "暂存区是空的" in text, text)


async def test_submit_while_busy(h: Harness):
    """上一轮还在回答时用户按提交键 —— **不能静默吞掉**。

    用户截图报过：「发了内容、又发了结束符，却收到『暂存区是空的：先发内容』」。
    底下其实是两个 bug 叠在一起：

    ① `_schedule_submit` 撞上 inflight 时直接 `return`，**既不提交也不记下来** ——
       用户发的内容永远卡在暂存区，后台连一行日志都没有（看起来就是机器人死了）。
       现在用一个 `_pending_submit` 标记，本轮跑完立刻补提交。
    ② 只发结束符时一律回「暂存区是空的」，可内容其实**早就收到、正在回答** ——
       用户以为白发了一遍，于是又按一次，正好撞上这句误导的话。
    ③ 顺带：只按提交键时不再往暂存区塞一个**空段**（空段会让 `parts` 多算一段，
       模型会收到一个空段）。
    """
    print("\n[17] 回答期间提交：不能吞掉、也不能误导")
    await establish(h)
    key = f"private:{ALICE}"
    orig_chat = h.llm.chat

    async def slow_chat(messages, **kw):
        await asyncio.sleep(0.8)
        return await orig_chat(messages, **kw)

    h.llm.chat = slow_chat
    try:
        h.clear()
        await h.private("第一段//", settle=0.15)
        # 桩模型是慢的（0.8s），所以这里看的是「已经进入回答」而不是调用次数 ——
        # 调用次数要等桩模型 sleep 完才涨，判它会假失败
        check("第一段已提交、正在回答", key in h.plugin._inflight,
              f"inflight={sorted(h.plugin._inflight)}")

        # 第一轮还在跑：再发一段内容 + 结束符
        await h.private("第二段内容", settle=0.15)
        await h.private("//", settle=0.15)
        # 只在结束符上按一次：不能往暂存区塞空段
        staged = [it.get("text") for it in
                  (h.plugin.buffer.peek(key).get("items") or [])]
        check("只按结束符不会塞空段", all(str(t).strip() for t in staged),
              f"暂存区={staged!r}")

        await asyncio.sleep(2.2)              # 等第一轮跑完 + 补提交
        check("暂存区已清空（补提交生效）",
              not (h.plugin.buffer.peek(key).get("items")),
              f"残留={h.plugin.buffer.peek(key)}")
        check("第二段真的提交给了模型", h.llm.calls >= 2, f"calls={h.llm.calls}")
        seen = h.llm.last_user_content()
        check("模型看到的是第二段内容", "第二段内容" in seen, seen)
        check("第二段没有被标成多段、也没有序号",
              seen.strip() == "第二段内容", seen)

        # 回答期间再按一次结束符：回执要说明「正在回答」，不能说「暂存区是空的」
        h.clear()
        await h.private("第三段//", settle=0.15)
        await h.private("//", settle=0.35)
        texts = " ".join(str(r) for r in h.replies)
        check("回答期间按结束符：提示是「正在回答」",
              "正在回答" in texts, texts[:200])
        check("回答期间按结束符：不再说「暂存区是空的」",
              "暂存区是空的" not in texts, texts[:200])
        await asyncio.sleep(1.6)
    finally:
        h.llm.chat = orig_chat


async def test_staging_persistence(h: Harness):
    print("\n[16] 暂存区落盘（重启不丢）")
    await establish(h)
    h.clear()
    await h.group("重启前的一段话", at_bot=True, message_id=666, settle=0.5)
    path = h.plugin.buffer.path
    check("暂存区文件存在", Path(path).exists(), path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    check("暂存内容已落盘",
          any("重启前的一段话" in str(it.get("text"))
              for it in (data.get(f"group:{GROUP}") or {}).get("items", [])),
          f"{data}")


async def test_multi_part_reply(h: Harness):
    print("\n[17] 长回复分段发送")
    await establish(h)
    h.clear()
    long_text = "这是一段很长的回复。" * 200
    h.llm.script_text = long_text
    try:
        await h.private("来段长的//", settle=1.6)
        await asyncio.sleep(0.6)
        check("长回复被拆成多条", len(h.replies) >= 2, f"条数={len(h.replies)}")
        joined = "".join(reply_text(r) for r in h.replies)
        check("分段后内容不丢", joined.count("这是一段很长的回复。") == 200,
              f"实际 {joined.count('这是一段很长的回复。')} 段")
        check("每条都在配额内",
              all(len(reply_text(r)) <= 1200 for r in h.replies),
              f"{[len(reply_text(r)) for r in h.replies]}")
    finally:
        h.llm.script_text = "好的，我收到了。"


async def test_reply_text_as_context(h: Harness):
    print("\n[18] 引用文本消息作为上文")
    await establish(h, group_id=GROUP)
    h.clear()
    # `get_msg` 桩：引用的那条是别人发的文本
    quoted_id = 424242
    h.msg_lookup[quoted_id] = {
        "message_id": quoted_id, "message_type": "group",
        "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
        "message": [{"type": "text", "data": {"text": "原问题：图书馆国庆开吗？"}}],
    }
    # ⚠️ 断言「结束符被剥掉」：`//` 只可能出现在用户自己敲的那部分。
    # 曾经引文被拼在结束符之前，`//` 被挤到中间，用户怎么发都不提交。
    await h.group("那到底开不开//", at_bot=True, with_reply=quoted_id, settle=1.6)
    seen = h.llm.last_user_content()
    check("引用后仍能提交（结束符没被引文挤走）",
          h.llm.calls >= 1, f"调模型 {h.llm.calls} 次")
    check("引用的正文进了模型上下文", "图书馆国庆开吗" in seen, f"{seen!r}")
    check("引文带发送者名", "鲍勃" in seen, f"{seen!r}")
    check("结束符没漏进上下文", "//" not in seen, f"{seen!r}")
    check("用户自己那句也在", "那到底开不开" in seen, f"{seen!r}")


async def test_reply_media_submits_text(h: Harness):
    """引用多媒体时**把能提交的文字部分提交上去**，不再整轮回绝。

    原来只要引文是图片/语音，就回一句「暂不支持多模态」并丢掉整条输入 ——
    可用户写的那个问题明明是好的（「这个怎么说？」），却被一起扔了，
    连模型都没见到。现在那句话只是一句**提示**，落在引文的位置上。
    """
    print("\n[19] 引用多媒体：提交文字 + 一句提示（不回绝）")
    await establish(h, group_id=GROUP)
    for seg_type, label in (("image", "图片"), ("record", "语音")):
        h.clear()
        quoted_id = 500000 + len(seg_type)
        h.msg_lookup[quoted_id] = {
            "message_id": quoted_id, "message_type": "group",
            "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
            "message": [{"type": seg_type, "data": {"file": "x"}}],
        }
        await h.group("这个怎么说？//", at_bot=True, with_reply=quoted_id,
                      settle=1.2)
        check(f"引用{label}：仍然提交给模型", h.llm.calls == 1,
              f"calls={h.llm.calls}")
        seen = h.llm.last_user_content()
        check(f"引用{label}：用户的问题在里面", "这个怎么说" in seen, seen)
        check(f"引用{label}：引文位置有一句提示",
              "暂不支持多模态" in seen and label in seen, seen)

    # 只引用一张图、自己一个字都不写：也要提交（提示就是全部内容）
    h.clear()
    h.msg_lookup[500009] = {
        "message_id": 500009, "message_type": "group",
        "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
        "message": [{"type": "image", "data": {"file": "x"}}],
    }
    await h.group("//", at_bot=True, with_reply=500009, settle=1.2)
    check("只引用图、不写字也会提交", h.llm.calls == 1, f"calls={h.llm.calls}")
    check("提交内容就是那句提示",
          "暂不支持多模态" in h.llm.last_user_content(),
          h.llm.last_user_content())

    # 读不到引文（撤回/接口失败）也不能静默吞掉整轮
    h.clear()
    await h.group("读不到的那种？//", at_bot=True, with_reply=999999, settle=1.4)
    check("引文读不到时仍然提交", h.llm.calls >= 1, f"调模型 {h.llm.calls} 次")
    check("读不到引文有标注", "读不到" in h.llm.last_user_content(),
          f"{h.llm.last_user_content()!r}")


async def test_md_command(h: Harness):
    """顶层 `/md`：把**上一条 AI 回复**转成 Markdown 图片发过来。

    刻意**不带子命令**（`/md` 而不是 `/aichat md`）—— 这是日常用得最多的一句话
    动作，值得少打几个字。它仍然受服务范围限制（走同一个守卫）。
    """
    print("\n[19.1] /md：把上条 AI 回复转成图片")
    await establish(h)
    reply_text = "## 椭圆\n\n标准椭圆 $x^2/25+y^2/9=1$。\n\n| 量 | 值 |\n| --- | --- |\n| a | 5 |"
    h.llm.script_text = reply_text
    h.clear()
    await h.private("讲讲椭圆//", settle=1.4)
    await h.wait_reply()

    seen: list[str] = []
    orig = h.plugin.renderer.render_markdown

    async def spy(text, **kw):
        seen.append(text)
        return await orig(text, **kw)

    h.plugin.renderer.render_markdown = spy
    try:
        h.clear()
        await h.private("/md", settle=2.0)
    finally:
        h.plugin.renderer.render_markdown = orig

    check("拿去渲染的正是上一条 AI 回复", seen and "标准椭圆" in seen[0],
          f"{seen[:1]}")
    check("Markdown 原文一字不差（不是渲染后的产物）",
          bool(seen) and seen[0].strip() == reply_text.strip(),
          f"{seen[:1]}")
    pics = [m for m in h.sent
            if any(str(seg.get("type")) == "image"
                   for seg in (m["params"].get("message") or [])
                   if isinstance(seg, dict))]
    check("发出了一张图片", bool(pics), f"{[m['params'].get('message') for m in h.sent]}")
    check("没有再调模型", h.llm.calls == 0, f"calls={h.llm.calls}")

    # 再来一次：用的仍是**最后一条**（这次是新的那条）
    h.llm.script_text = "第二条回复。"
    h.clear()
    await h.private("再来一条//", settle=1.2)
    await h.wait_reply()
    seen.clear()
    h.plugin.renderer.render_markdown = spy
    try:
        await h.private("/md", settle=1.6)
    finally:
        h.plugin.renderer.render_markdown = orig
    check("取的是最后一条 AI 回复", bool(seen) and "第二条回复" in seen[0],
          f"{seen[:1]}")

    # 空会话：还没有 AI 回复
    h.clear()
    await h.private("/aichat new 空会话", settle=0.5)
    await h.wait_reply()
    h.clear()
    await h.private("/md", settle=1.0)
    text = await h.wait_reply()
    check("还没回复过时给一句说明", "还没有" in text, text)


async def test_quote_chain(h: Harness):
    """引用要能**沿菊花链往上读**：被引用的那条自己也是引用别人的。

    群聊里接龙式对话很常见（A 引用 B，B 引用 C），只读一层的话模型看到的是半截话。
    """
    print("\n[19.2] 引用支持菊花链")
    await establish(h, group_id=GROUP)
    h.clear()
    # C：最底层那条
    h.msg_lookup[700003] = {
        "message_id": 700003, "message_type": "group",
        "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
        "message": [{"type": "text", "data": {"text": "底层的原话：闭馆三天"}}],
    }
    # B：引用了 C
    h.msg_lookup[700002] = {
        "message_id": 700002, "message_type": "group",
        "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
        "message": [{"type": "reply", "data": {"id": "700003"}},
                    {"type": "text", "data": {"text": "中间那层说"}}],
    }
    await h.group("那到底怎么说？//", at_bot=True, with_reply=700002, settle=1.2)
    seen = h.llm.last_user_content()
    check("读到被引用的那一层", "中间那层说" in seen, seen)
    check("继续往上读到更底一层", "底层的原话" in seen, seen)
    check("两层都标了来源", seen.count("[引用上文]") == 2, seen)

    # 环：自己引用自己，不能转不出来
    h.clear()
    h.msg_lookup[700004] = {
        "message_id": 700004, "message_type": "group",
        "sender": {"user_id": BOB, "nickname": "鲍勃", "card": ""},
        "message": [{"type": "reply", "data": {"id": "700004"}},
                    {"type": "text", "data": {"text": "自引用"}}],
    }
    await h.group("环测试//", at_bot=True, with_reply=700004, settle=1.2)
    check("自引用环不会卡住", h.llm.calls >= 1, f"calls={h.llm.calls}")


async def test_quote_ai_image(h: Harness):
    """引用 AI 回答里**含图片**的那条 → 用 AI 当时生成的原文替代。

    Markdown 模式的整篇回答就是一张图、函数图与公式图也都是图 —— 它们在 QQ 那边
    没有文字，`get_msg` 读回来只有「一张图片」。可那段内容本来就是我们自己生成的，
    原文就在会话历史里（用户要求）。
    """
    print("\n[19.3] 引用 AI 的图片 → 用原文替代")
    await establish(h)
    h.clear()
    h.llm.script_text = "标准椭圆 x²/25+y²/9=1，长半轴 5，短半轴 3。"
    h.sent.clear()
    await h.private("讲讲椭圆//md", settle=1.6)      # md 模式：整篇出成一张图
    await h.wait_reply()

    pics = [m for m in h.sent
            if any(str(seg.get("type")) == "image"
                   for seg in (m["params"].get("message") or [])
                   if isinstance(seg, dict))]
    check("这一轮真的发了图片", bool(pics),
          f"{[m['params'].get('message') for m in h.sent]}")
    if not pics:
        return
    pic_id = pics[0]["message_id"]

    # QQ 那边读回来就是「一张图片」——没有文字
    h.msg_lookup[pic_id] = {
        "message_id": pic_id, "message_type": "private",
        "sender": {"user_id": BOT_ID, "nickname": "测试机器人"},
        "message": [{"type": "image", "data": {"file": "x.png"}}],
    }
    h.clear()
    h.llm.script_text = "好的。"
    await h.private("那张图里写的什么？//", with_reply=pic_id, settle=1.6)
    seen = h.llm.last_user_content()
    # 现在这段原文**还在上下文里**，所以按「引用上文用标号代替原文」的规矩
    # 只给编号（不再重抄一遍原文，也不再贴「机器人自己发的图」的原文段）。
    answers = [m for m in h.plugin.store.history(
        _current_session(h).id) if m.role == "assistant" and m.seq]
    label = int(answers[-1].seq) if answers else 0
    check("用标号指代那条回答", label and f"[#{label}]" in seen, seen)
    check("没有重抄原文", "长半轴 5" not in seen, seen)
    check("这句也进了正文", "那张图里写的什么" in seen, seen)

    # 别人发的图读不到内容，但用户的问题照常提交（不再整轮回绝）
    h.clear()
    other = 700009
    h.msg_lookup[other] = {
        "message_id": other, "message_type": "private",
        "sender": {"user_id": BOB, "nickname": "鲍勃"},
        "message": [{"type": "image", "data": {"file": "y.png"}}],
    }
    await h.private("这张呢？//", with_reply=other, settle=1.2)
    seen2 = h.llm.last_user_content()
    check("别人的图：内容读不到，但用户的问题照常提交",
          "这张呢" in seen2 and "暂不支持多模态" in seen2, seen2)


async def test_reply_index_survives_reload(h: Harness):
    """`SentReplyIndex` 记的 `(sid, seq)` 在**重新加载历史之后**仍要能取到原文。

    踩过的坑：`Message.to_dict()` 漏写 `seq`，落盘读回来的消息 seq 全是 0，
    `SessionStore.find()` 于是**一个都对不上** —— 表现是「引用机器人自己发的
    图片，永远只有一句『读不到』」，而用户要的是「替换成 AI 当时生成的原文」。
    热重载一次就复现，所以它是「偶尔灵、重启后必坏」的那类 bug。
    """
    print("\n[19.4] 引用索引跨重载")
    store_mod = plugin_module("store")
    seg_mod = plugin_module("segments")
    SessionStore, SentReplyIndex = store_mod.SessionStore, store_mod.SentReplyIndex
    import json as _json
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="reply-index-",
                                   dir=str(ROOT / "tests" / ".test-tmp")))
    st = SessionStore(str(tmpdir))
    s = st.create("索引会话", owner_kind="user", owner_id=ALICE)
    st.append(s.id, seg_mod.user_message("讲讲椭圆"))
    ans = seg_mod.assistant_message("标准椭圆 x²/25+y²/9=1，长半轴 5。")
    st.append(s.id, ans)
    seq = int(ans.seq or 0)
    check("assistant 消息拿到会话内序号", seq > 0, f"seq={seq}")

    idx = SentReplyIndex(str(Path(tmpdir) / "sent_replies.json"))
    idx.add([9001], s.id, seq)

    # 等价于一次热重载 / 进程重启：全部从 jsonl 重新读
    st2 = SessionStore(str(tmpdir))
    got = st2.find(s.id, seq)
    check("重载后仍能按 seq 取到原文",
          got is not None and "标准椭圆" in (got.text or ""), f"{got}")
    pair = SentReplyIndex(str(Path(tmpdir) / "sent_replies.json")).get(9001)
    check("索引落盘后能读回 (sid, seq)", pair == (s.id, seq), f"{pair}")
    got2 = st2.find(*pair) if pair else None
    check("「消息 id → 索引 → 原文」这条链完整",
          got2 is not None and "标准椭圆" in (got2.text or ""), f"{got2}")

    # 老文件（写入时还没有 seq 字段）也要能按会话计数倒推出来
    path = Path(tmpdir) / "sessions" / f"{s.id}.jsonl"
    rows = [_json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]
    for r in rows:
        r.pop("seq", None)
    path.write_text("\n".join(_json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    st3 = SessionStore(str(tmpdir))
    got3 = st3.find(s.id, seq)
    check("老 jsonl（无 seq 字段）倒推出的序号仍然正确",
          got3 is not None and "标准椭圆" in (got3.text or ""), f"{got3}")
    check("倒推的序号就是最后一条", int(got3.seq) == int(st3.get(s.id).message_count)
          if got3 else False, f"{getattr(got3, 'seq', None)}")


async def test_paging_long_reply(h: Harness):
    print("\n[20] 超长回复自动分页")
    await establish(h)
    h.clear()
    # 恰好 3000 字：按 1200 切应为 1200/1200/600 三条
    h.llm.script_text = "分页测试。" * 600
    try:
        await h.private("来段超长的//", settle=2.0)
        await asyncio.sleep(0.6)
        lens = [len(reply_text(r)) for r in h.replies]
        check("拆成 3 条", len(h.replies) == 3, f"条数={len(h.replies)} 长度={lens}")
        check("每条都在 1200 以内", all(n <= 1200 for n in lens), f"{lens}")
        check("分页后内容完整",
              "".join(reply_text(r) for r in h.replies).count("分页测试。") == 600,
              f"{lens}")
        check("分页不带省略号（不是截断）",
              not any("…" in reply_text(r) or "..." in reply_text(r)
                      for r in h.replies), f"{lens}")
    finally:
        h.llm.script_text = "好的，我收到了。"


async def test_llm_error_surfaces(h: Harness):
    print("\n[21] LLM 失败要说人话")
    await establish(h)
    h.clear()
    LlmError = plugin_module("llm").LlmError

    def responder(messages, tools):
        raise LlmError("HTTP 401: invalid api key", status=401)

    h.llm.responder = responder
    try:
        await h.private("会失败的一轮//", settle=1.2)
        reply = await h.wait_reply()
        check("用户看到错误提示", "出错" in reply or "401" in reply, reply)
        check("错误里带原因", "401" in reply, reply)
    finally:
        h.llm.responder = None


async def test_llm_timeout_hint(h: Harness):
    """上游「想太久 / 卡死」时要给一句能照做的提示，而不是光甩一个 TimeoutError。

    实测：`deepseek-v4-flash` 遇到「这段 C++ 会报什么运行时错误」这类输入会连想
    200 秒以上（4096 token 全花在思考链上、正文 0 字）。用户看到的只有
    `LlmError: TimeoutError`，既不知道能不能重试，也不知道该改什么。

    流式之后「卡死」与「只是慢」是两种失败（见 `llm._chat_stream`），提示必须分开给：
    对真正的网络卡顿说「把问题拆小」毫无帮助。
    """
    print("\n[21.6] 上游超时要说人话")
    await establish(h)
    h.clear()
    LlmError = plugin_module("llm").LlmError
    cfg = h.plugin.cfg["llm"]
    old = {k: cfg.get(k) for k in ("max_retries", "retry_backoff", "retry_max_wait",
                                   "retry_notice")}
    cfg.update({"max_retries": 0, "retry_backoff": 0, "retry_max_wait": 0})

    def responder(messages, tools):
        raise LlmError("超过单次调用总时长上限 270s（已生成 0 字正文）",
                       retryable=False)

    h.llm.responder = responder
    try:
        await h.private("一段会把它想死的问题//", settle=1.2)
        reply = await h.wait_reply()
        check("仍然报出错", "出错" in reply, reply)
        check("说明是「超过上限」", "上限" in reply, reply)
        check("给出可照做的动作（拆小 / 调 timeout）",
              "拆小" in reply and "timeout" in reply, reply)
    finally:
        h.llm.responder = None
        h.clear()

    # 卡死（stall）是另一种失败：该说的是「上游没吐数据、再发一次」
    def stalled(messages, tools):
        raise LlmError("流式读取停滞 60s（上游不再吐数据，已生成 0 字正文）",
                       retryable=True)

    h.llm.responder = stalled
    try:
        await h.private("卡住的一轮//", settle=1.2)
        reply = await h.wait_reply()
        check("卡死提示指向 stall_timeout 而不是「拆小问题」",
              "卡住" in reply and "stall_timeout" in reply and "拆小" not in reply,
              reply)
    finally:
        h.llm.responder = None

    # 别的失败（参数错）不该乱加这句 —— 提示要跟原因对得上
    def bad_key(messages, tools):
        raise LlmError("HTTP 401: invalid api key", status=401)

    h.clear()                       # 前面几轮的回复还堆在 h.replies 里
    h.llm.responder = bad_key
    try:
        await h.private("再来一次//", settle=1.2)
        reply = await h.wait_reply()
        check("401 不加超时提示", "上限" not in reply and "停滞" not in reply, reply)
    finally:
        h.llm.responder = None
        cfg.update(old)


async def test_escaped_prefix_paste(h: Harness):
    """用户自己加的「转义命令前缀」反斜杠要脱掉，别原样喂给模型。

    实测踩到：用户想粘一段 C++，为了不让 `#include…` 被当成命令（本部署里 `#`
    既是命令前缀又是结束符），自己补了个反斜杠写成 `\\#include<bits/stdc++.h>`。
    框架其实不需要这个转义（`#include<…>` 不是合法命令名，命令分发本来就静默
    丢掉），可模型收到的是带反斜杠的代码，于是答「第一行编译不过」。
    """
    print("\n[21.7] 转义前缀的粘贴")
    await establish(h)
    h.clear()
    h.llm.script_text = "第一行是 memset 的第三个参数用错了。"
    # 线上结束符配的就是 `#`（`//` 会被 C++ 代码里的注释行撞上），这里照搬
    trigger = h.plugin.cfg["trigger"]
    old_terms = list(trigger.get("terminators") or [])
    trigger["terminators"] = ["#"]
    code = ("\\#include<bits/stdc++.h>\nusing namespace std;\n"
            "int main(){return 0;}这段会报什么运行时错误#")
    try:
        await h.private(code, settle=1.6)
        seen = h.llm.last_user_content()
        check("正文里没有多余的转义反斜杠", "\\#include" not in seen, seen)
        check("代码本身原样保留", "#include<bits/stdc++.h>" in seen, seen)
        check("结束符已被剥掉（提交了而不是卡在暂存区）",
              "运行时错误" in seen and not seen.rstrip().endswith("#"), seen)

        # LaTeX 里的 `\#`（不在行首）不能被误伤
        h.clear()
        await h.private("$\\#$ 在 LaTeX 里表示什么？#", settle=1.2)
        seen2 = h.llm.last_user_content()
        check("行内 LaTeX 的 \\# 保持原样", "$\\#$" in seen2, seen2)
    finally:
        trigger["terminators"] = old_terms


async def test_sse_parse(h: Harness):
    """SSE 分片解析：任意位置切断、心跳注释、`[DONE]`、工具调用分片累积。"""
    print("\n[21.8] 流式分片解析")
    llm = plugin_module("llm")

    def sse(obj) -> bytes:
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n"

    blob = b"".join([
        b": keep-alive\n\n",                                   # 心跳注释
        sse({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        sse({"choices": [{"delta": {"reasoning_content": "想"}}]}),
        sse({"choices": [{"delta": {"content": "你"}}]}),
        sse({"choices": [{"delta": {"content": "好"}}]}),
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "type": "function",
             "function": {"name": "now", "arguments": ""}}]}}]}),
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"a"'}}]}}]}),
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ":1}"}}]}}]}),
        sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        sse({"choices": [{"delta": {}}], "usage": {"total_tokens": 7}}),
        b"data: not-json\n",                                   # 半行/垃圾
        b"data: [DONE]\n",
    ])
    buf = llm.SseBuffer()
    acc = llm._StreamAcc()
    for i in range(0, len(blob), 7):        # 每 7 字节喂一次 → 必然切断 JSON
        for obj in buf.feed(blob[i:i + 7]):
            acc.merge(obj)
    for obj in buf.flush():
        acc.merge(obj)
    res = acc.result()
    check("正文跨分片拼回", res.text == "你好", repr(res.text))
    check("[DONE] 被识别", buf.done, f"{buf.done}")
    check("垃圾行被丢掉而不是炸掉", res.finish_reason == "tool_calls",
          res.finish_reason)
    check("工具调用参数按 index 累积",
          [(c.name, c.arguments) for c in res.tool_calls] == [("now", {"a": 1})],
          f"{[(c.name, c.raw_arguments) for c in res.tool_calls]}")
    check("usage 收到", res.usage.get("total_tokens") == 7, f"{res.usage}")
    check("思考链单独计数（不进正文）", res.raw.get("reasoning_chars") == 1,
          f"{res.raw}")

    # 思考链吃光配额（content 为空）要说清楚，而不是回一句空话
    acc2 = llm._StreamAcc()
    acc2.merge({"choices": [{"delta": {"reasoning_content": "想了很久"}}]})
    acc2.merge({"choices": [{"delta": {}, "finish_reason": "length"}]})
    try:
        acc2.result()
        check("空正文要报错", False, "没有报错")
    except llm.LlmError as e:
        check("空正文报错并提示调大 max_tokens", "max_tokens" in str(e), str(e))


async def test_llm_stream_cutoff(h: Harness):
    """流式请求：正常流 / 端点不回 SSE / 上游卡死 / 超过总时长。

    这组断言对应「掐断逻辑」本身 —— 它不是按总时长一刀切，而是：
    **只要 token 还在流就一直等**（上限 `timeout`），**多久没新数据**（`stall_timeout`）
    才是卡死判据；真被掐断时把**已经生成的部分**带在异常里（`LlmError.partial`）。
    """
    print("\n[21.9] 流式掐断判据")
    from aiohttp import web
    llm = plugin_module("llm")
    seen_payloads: list[dict] = []

    def sse(obj) -> bytes:
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n"

    async def endpoint(request):
        kind = request.match_info["kind"]
        body = await request.json()
        seen_payloads.append(body)
        if kind == "plain":            # 端点忽略 stream，直接回 JSON
            return web.json_response({
                "choices": [{"message": {"content": "普通响应"},
                             "finish_reason": "stop"}],
                "usage": {"total_tokens": 3}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        try:
            await resp.write(sse({"choices": [{"delta": {"content": "前半"}}]}))
            if kind == "stall":
                await asyncio.sleep(3)          # 之后一个字节都不吐
                return resp
            if kind == "slow":
                for _ in range(50):             # 一直在吐，但总时长的上限会先到
                    await resp.write(sse({"choices": [{"delta": {"content": "x"}}]}))
                    await asyncio.sleep(0.2)
                return resp
            await resp.write(sse({"choices": [{"delta": {"content": "后半"}}]}))
            await resp.write(sse({"choices": [{"delta": {},
                                               "finish_reason": "stop"}]}))
            await resp.write(b"data: [DONE]\n")
        except (ConnectionResetError, RuntimeError):
            pass                                # 客户端掐断后写失败是预期的
        return resp

    app = web.Application()
    # 四种行为各占一个前缀：客户端固定 POST `{base_url}/chat/completions`，
    # 所以把「哪一种」放进 base_url
    app.router.add_post("/{kind}/chat/completions", endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]      # noqa: SLF001

    def client(kind: str, **kw):
        return llm.LlmClient(base_url=f"http://127.0.0.1:{port}/{kind}",
                             api_key="k", model="m", max_tokens=64, **kw)

    try:
        # ① 正常流
        c = client("ok", stream=True, timeout=10, stall_timeout=5)
        r = await c.chat([{"role": "user", "content": "hi"}])
        check("流式正文拼成完整回答", r.text == "前半后半", repr(r.text))
        check("流式请求带 stream_options",
              seen_payloads[-1].get("stream_options", {}).get("include_usage") is True,
              f"{seen_payloads[-1].get('stream_options')}")
        check("流式请求 stream=true", seen_payloads[-1].get("stream") is True,
              f"{seen_payloads[-1].get('stream')}")
        await c.close()

        # ② 端点没按 SSE 回 → 当普通 JSON 解析（不因为「开了流式」就废掉）
        c = client("plain", stream=True, timeout=10, stall_timeout=5)
        r = await c.chat([{"role": "user", "content": "hi"}])
        check("端点忽略 stream 时退回 JSON 解析", r.text == "普通响应", repr(r.text))
        await c.close()

        # ③ 卡死：stall_timeout 内没有新数据 → 可重试，且带上半截
        c = client("stall", stream=True, timeout=30, stall_timeout=0.6)
        try:
            await c.chat([{"role": "user", "content": "hi"}])
            check("卡死要被掐断", False, "居然成功了")
        except llm.LlmError as e:
            check("报的是「停滞」而不是总超时", "停滞" in str(e), str(e))
            check("卡死算可重试", e.retryable is True, f"{e.retryable}")
            check("带出已生成的部分", e.partial == "前半", repr(e.partial))
        await c.close()

        # ④ 一直在吐但超过总时长 → 掐断但**不可重试**（再试也是同样的时间）
        c = client("slow", stream=True, timeout=1.0, stall_timeout=30)
        try:
            await c.chat([{"role": "user", "content": "hi"}])
            check("超过总时长要被掐断", False, "居然成功了")
        except llm.LlmError as e:
            check("报的是总时长上限", "上限" in str(e), str(e))
            check("总超时不再重试", e.retryable is False, f"{e.retryable}")
            check("总超时也带出半截", e.partial.startswith("前半"),
                  repr(e.partial))
        await c.close()
    finally:
        await runner.cleanup()


async def test_partial_reply_on_cutoff(h: Harness):
    """上游被掐断时把**已经生成的部分**发给用户，而不是只回一句报错。"""
    print("\n[21.95] 掐断时先发半截")
    await establish(h)
    h.clear()
    LlmError = plugin_module("llm").LlmError

    def responder(messages, tools):
        raise LlmError("超过单次调用总时长上限 270s（已生成 42 字正文）",
                       retryable=False, partial="先讲到这里：定积分是……")

    h.llm.responder = responder
    try:
        await h.private("一个很长的问题//", settle=1.4)
        await h.wait_for_replies(2)          # ① 说明「被掐断了」 ② 半截正文
        texts = [reply_text(r) for r in h.replies]
        check("先说明上游中断了", any("已经生成的部分" in t for t in texts), f"{texts}")
        check("半截正文也发出来了", any("定积分是" in t for t in texts), f"{texts}")
        check("半截排在说明之后", "定积分是" in texts[-1], f"{texts}")
    finally:
        h.llm.responder = None


async def test_llm_stream_wiring(h: Harness):
    """`llm.stream` / `llm.stall_timeout` 要真的接到客户端上（不只是写进默认值）。

    这类「配置项加了、代码没读」的坑在本仓库出现过好几次（`plot.unit_px`、
    `trigger.auto_submit_seconds`），所以这里断言的是**客户端对象的属性**，
    以及 `/aichat current` 里能看到当前用的是哪一套掐断判据。
    """
    print("\n[21.96] 流式配置的接线")
    await establish(h)
    llm_cfg = h.plugin.cfg["llm"]
    check("默认开流式", llm_cfg.get("stream") is True, f"{llm_cfg.get('stream')}")
    check("默认有 stall_timeout", float(llm_cfg.get("stall_timeout", 0)) == 60.0,
          f"{llm_cfg.get('stall_timeout')}")
    # ⚠️ `h.plugin.llm` 是测试桩，所以要问插件自己的工厂要一个**真客户端**
    real = h.plugin._build_llm()
    check("真客户端拿到 stream", real.stream is True, f"{real.stream}")
    check("真客户端拿到 stall_timeout", float(real.stall_timeout) == 60.0,
          f"{real.stall_timeout}")
    check("真客户端拿到总时长上限", float(real.timeout) == float(llm_cfg["timeout"]),
          f"{real.timeout}")
    await real.close()

    # 关掉流式（退回老路）时，工厂造出来的客户端与 `current` 的话术都要跟着变
    old = llm_cfg.get("stream")
    llm_cfg["stream"] = False
    try:
        real2 = h.plugin._build_llm()
        check("关掉之后真客户端也不再流式", real2.stream is False, f"{real2.stream}")
        await real2.close()
        # `current` 读的是 `self.llm`（测试里是桩），所以这里同步桩上的开关
        stub_stream = h.plugin.llm.stream
        h.plugin.llm.stream = False
        h.clear()
        await h.private("/aichat current", settle=0.8)
        text = await h.wait_reply()
        check("current 里能看到掐断判据", "掐断" in text and "非流式" in text, text)
        h.plugin.llm.stream = stub_stream
    finally:
        llm_cfg["stream"] = old

    h.clear()
    await h.private("/aichat current", settle=0.8)
    text2 = await h.wait_reply()
    check("流式时 current 说明停滞判据", "流式" in text2 and "停滞" in text2, text2)


async def test_llm_retry(h: Harness):
    """模型调用失败要**自动重试**，次数可配。

    用户报过：上游 DNS 抽了一下（`Name or service not known`），那一轮就直接
    报错结束了 —— 而这类失败下一毫秒重发就好。判据是 `LlmError.retryable`
    （网络层异常 / 408·409·425·429 / 5xx / 上游回了看不懂的东西），
    **4xx 不重试**：参数错、key 失效，再试一百次也是同一个结果。
    """
    print("\n[21.5] 模型调用失败自动重试")
    await establish(h)
    LlmError = plugin_module("llm").LlmError
    cfg = h.plugin.cfg["llm"]
    old_cfg = {k: cfg.get(k) for k in
               ("max_retries", "retry_backoff", "retry_max_wait", "retry_notice")}
    cfg["retry_backoff"] = 0            # 测试里别真等
    cfg["retry_max_wait"] = 0

    def flaky(n_fail, *, retryable=True, status=503):
        """前 n 次失败，之后成功。"""
        state = {"n": 0}

        def responder(messages, tools):
            state["n"] += 1
            if state["n"] <= n_fail:
                raise LlmError(f"HTTP {status}: boom", status=status,
                               retryable=retryable)
            return StubResult(text="重试之后成功了")
        return state, responder

    try:
        # ① 失败两次后成功：模型被调 3 次，用户拿到正常回答
        cfg["max_retries"] = 2
        state, resp = flaky(2)
        h.llm.responder = resp
        h.clear()
        await h.private("上游抖两下//", settle=0.4)
        # 两条：①「正在重试」提示 ② 重试成功后的正式回答
        await h.wait_for_replies(2, timeout=8.0)
        reply = await h.wait_reply(index=1)
        check("重试后成功", "重试之后成功了" in reply, reply)
        check("一共调了 3 次（1 + 2 次重试）", state["n"] == 3, f"{state['n']}")
        check("重试期间给过用户一句提示",
              any("重试" in str(r) for r in h.replies),
              f"{[str(r)[:60] for r in h.replies]}")

        # ② 重试用尽：一共 1 + max_retries 次，然后把错误报给用户
        cfg["max_retries"] = 1
        state, resp = flaky(99)
        h.llm.responder = resp
        h.clear()
        await h.private("一直失败//", settle=0.4)
        await h.wait_for_replies(2, timeout=8.0)
        reply = await h.wait_reply(index=1)
        check("重试用尽后报错", "出错" in reply or "503" in reply, reply)
        check("次数正好 1 + max_retries", state["n"] == 2, f"{state['n']}")

        # ③ max_retries=0：一次都不重试
        cfg["max_retries"] = 0
        state, resp = flaky(99)
        h.llm.responder = resp
        h.clear()
        await h.private("不许重试//", settle=1.2)
        await h.wait_reply(index=0)
        check("max_retries=0 就不重试", state["n"] == 1, f"{state['n']}")

        # ④ 不可重试的失败（4xx）立刻放弃，别让用户白等
        cfg["max_retries"] = 3
        state, resp = flaky(99, retryable=False, status=401)
        h.llm.responder = resp
        h.clear()
        await h.private("key 错了//", settle=1.2)
        # 401 不可重试 → **没有**「正在重试」提示，只有错误那一条
        reply = await h.wait_reply(index=0)
        check("401 不重试", state["n"] == 1, f"{state['n']}")
        check("401 的错误照常报出来", "401" in reply, reply)

        # ⑤ 判据本身（用插件自己那份模块，别引第二份副本）
        retryable = plugin_module("main")._is_retryable
        check("网络层失败可重试",
              retryable(LlmError("dns", retryable=True)), "")
        check("5xx 可重试",
              retryable(LlmError("500", status=500, retryable=True)), "")
        check("401 不可重试", not retryable(LlmError("401", status=401)), "")
        check("超时可重试", retryable(asyncio.TimeoutError()), "")
    finally:
        h.llm.responder = None
        for k, v in old_cfg.items():
            cfg[k] = v


async def test_delete_and_rename(h: Harness):
    print("\n[19] 重命名 / 删除会话")
    await establish(h)
    await h.private("/aichat new 临时会话", settle=0.7)
    await h.wait_reply()
    sid = _current_session(h).id

    h.clear()
    await h.private(f"/aichat rename {sid} 改过的名字", settle=0.7)
    text = await h.wait_reply()
    check("重命名成功", "已更名为" in text, text)
    check("名字真的改了", h.plugin.store.get(sid).name == "改过的名字",
          h.plugin.store.get(sid).name)

    h.clear()
    await h.private(f"/aichat delete {sid}", settle=0.7)
    text = await h.wait_reply()
    check("删除需要确认", "确认" in text and h.plugin.store.get(sid) is not None, text)

    h.clear()
    await h.private(f"/aichat delete {sid} --yes=true", settle=0.7)
    text = await h.wait_reply()
    check("确认后删除", h.plugin.store.get(sid) is None, text)


async def test_schema_error_feedback(h: Harness):
    print("\n[21] 参数写错要有提示（不能静默）")
    await establish(h)
    h.clear()
    # `attach` 的会话名有长度上限（NAME_MAX）；超长会被 Schema 拒绝。
    # （原先这里用 `--force=maybe` 试布尔解析 —— `attach` 已不再有 `--force`，
    #  未知键值会被框架当位置参数吞掉，测不出 Schema 反馈了。）
    await h.private("/aichat attach " + "会" * 200, settle=0.9)
    text = await h.wait_reply()
    check("用法不正确有明确回复（框架统一回执）", "用法不正确" in text, text[:200])
    check("回复重定向到该子命令的帮助页",
          "/aichat attach" in text and "位置传参：" in text, text[:400])


async def test_reasoning_unit(h=None):
    """思考强度的纯单元测试：解析、方言渲染、max_tokens 下界。"""
    _r = plugin_module("reasoning")
    ReasoningPolicy = _r.ReasoningPolicy
    ReasoningSpec = _r.ReasoningSpec
    parse_budget = _r.parse_budget
    spec_from_text = _r.spec_from_text

    print("\n[22] 思考强度：解析")
    check("off/auto/inherit 都表示不干预",
          all(spec_from_text(v).is_off for v in ("off", "auto", "inherit", "default", "-")),
          "有不被当作不干预的")
    check("级别归一化", spec_from_text("HIGH").level == "high"
          and spec_from_text("max").level == "max"
          and spec_from_text("最少").level == "minimal",
          f"{spec_from_text('HIGH')} {spec_from_text('最少')}")
    check("`on` 等价于开思考（medium）",
          spec_from_text("on").level == "medium", f"{spec_from_text('on')}")
    check("预算是数字不是级别", spec_from_text("4k").mode == "budget"
          and spec_from_text("4k").budget == 4096, f"{spec_from_text('4k')}")
    check("预算换算正确",
          parse_budget("1.5k") == 1536 and parse_budget("2048 tokens") == 2048
          and parse_budget("high") is None, "预算解析有误")
    check("thinking:off 是显式关闭",
          spec_from_text("thinking:off").is_off, f"{spec_from_text('thinking:off')}")
    check("不认识的值退回不干预并留痕",
          spec_from_text("maybe").is_off and spec_from_text("maybe").note.startswith("不认识"),
          f"{spec_from_text('maybe')}")

    print("\n[23] 思考强度：各方言渲染")
    pol_effort = ReasoningPolicy(dialect="effort")
    check("effort 方言发 reasoning_effort",
          pol_effort.build_params(spec_from_text("high")) == {"reasoning_effort": "high"},
          f"{pol_effort.build_params(spec_from_text('high'))}")
    check("off 时什么都不发",
          pol_effort.build_params(spec_from_text("off")) == {},
          f"{pol_effort.build_params(spec_from_text('off'))}")
    pol_thinking = ReasoningPolicy(dialect="thinking")
    check("thinking 方言发 thinking.type",
          pol_thinking.build_params(spec_from_text("high")) == {"thinking": {"type": "enabled"}},
          f"{pol_thinking.build_params(spec_from_text('high'))}")
    check("thinking 方言下 off 不发",
          pol_thinking.build_params(spec_from_text("off")) == {},
          f"{pol_thinking.build_params(spec_from_text('off'))}")
    pol_et = ReasoningPolicy(dialect="enable_thinking")
    check("enable_thinking 方言发布尔",
          pol_et.build_params(spec_from_text("low")) == {"enable_thinking": True},
          f"{pol_et.build_params(spec_from_text('low'))}")
    check("dialect=none 永不发思考参数",
          ReasoningPolicy(dialect="none").build_params(spec_from_text("max")) == {},
          "none 方言仍发了字段")
    pol_raw = ReasoningPolicy(dialect="raw", params={"chat_template_kwargs": {"thinking": True}})
    check("raw 方言原样合并",
          pol_raw.build_params(spec_from_text("high")) == {"chat_template_kwargs": {"thinking": True}},
          f"{pol_raw.build_params(spec_from_text('high'))}")
    check("thinking:* 在 effort 方言里翻译成级别（不被吞掉）",
          pol_effort.build_params(spec_from_text("thinking:on")).get("reasoning_effort") == "high",
          f"{pol_effort.build_params(spec_from_text('thinking:on'))}")

    print("\n[24] 思考强度：max_tokens 下界与 per-model")
    check("开思考时抬高 max_tokens（防「想完没字数回答」）",
          pol_effort.min_max_tokens(spec_from_text("high"), 256) >= 4096,
          f"{pol_effort.min_max_tokens(spec_from_text('high'), 256)}")
    check("不开思考时不干预 max_tokens",
          pol_effort.min_max_tokens(spec_from_text("off"), 256) == 256,
          f"{pol_effort.min_max_tokens(spec_from_text('off'), 256)}")
    check("预算模式给预算 + 余量",
          pol_effort.min_max_tokens(spec_from_text("8k"), 512) >= 8192,
          f"{pol_effort.min_max_tokens(spec_from_text('8k'), 512)}")
    pol_model = ReasoningPolicy(default="off",
                                per_model={"deepseek-v4-flash": "off", "o4-mini": "high"})
    check("per_model 覆盖默认",
          pol_model.spec_for("o4-mini").level == "high"
          and pol_model.spec_for("deepseek-v4-flash").is_off
          and pol_model.spec_for("unknown-model").is_off,
          "per_model 未生效")
    check("会话覆盖优先于 per_model",
          pol_model.spec_for("o4-mini", "low").level == "low",
          f"{pol_model.spec_for('o4-mini', 'low')}")


async def test_think_command(h: Harness):
    print("\n[25] /aichat think：查看与切换")
    await establish(h)
    h.clear()
    await h.private("/aichat think", settle=0.9)
    text = await h.wait_reply()
    check("think 显示状态", "思考强度" in text and "方言" in text, text[:200])
    check("think 显示压缩摘要那一档", "压缩摘要" in text, text[:300])

    # 默认不发思考参数
    h.clear()
    await h.private("默认一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("默认不往 payload 加思考字段",
          h.llm.last_reasoning_params == {}, f"{h.llm.last_reasoning_params}")

    # 切到 high
    h.clear()
    await h.private("/aichat think high", settle=0.9)
    text = await h.wait_reply()
    check("think 切换成功", "已" in text and "reasoning_effort=high" in text, text[:200])

    h.clear()
    await h.private("现在带思考//", settle=1.0)
    await h.wait_for_replies(1)
    check("切换后 payload 真的带上 reasoning_effort",
          h.llm.last_reasoning_params == {"reasoning_effort": "high"},
          f"{h.llm.last_reasoning_params}")
    check("会话级切换落库（能持久化）",
          h.plugin.store.get(_current_session(h).id).meta.get("reasoning") == "high",
          f"{h.plugin.store.get(_current_session(h).id).meta}")

    # 切回 off
    h.clear()
    await h.private("/aichat think off", settle=0.9)
    await h.wait_for_replies(1)
    h.clear()
    await h.private("再一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("切回 off 后不再发思考字段",
          h.llm.last_reasoning_params == {}, f"{h.llm.last_reasoning_params}")

    # 预算模式 + max_tokens 下界
    h.clear()
    await h.private("/aichat think 8k", settle=0.9)
    await h.wait_for_replies(1)
    h.clear()
    await h.private("预算一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("预算模式抬高 max_tokens",
          h.llm.last_max_tokens >= 8192, f"{h.llm.last_max_tokens}")

    # 不认识的值要报错而不是静默
    h.clear()
    await h.private("/aichat think maybe", settle=0.9)
    text = await h.wait_reply()
    check("不认识的值有明确提示", "不认识" in text, text[:150])

    # 全局默认（写到 config.json）
    h.clear()
    await h.private("/aichat think low --global_=true", settle=0.9)
    text = await h.wait_reply()
    check("--global_=true 改全局默认", "全局默认" in text, text[:200])
    check("全局默认真的生效",
          h.plugin.reasoning.spec_for(h.plugin.llm.model).level == "low",
          f"{h.plugin.reasoning.spec_for(h.plugin.llm.model)}")
    cfg_file = Path(h.tmp) / "config.json"
    saved = json.loads(cfg_file.read_text(encoding="utf-8"))
    check("全局默认写进 config.json",
          saved["plugins"]["ai_chat"]["llm"]["reasoning"]["default"] == "low",
          f"{saved['plugins']['ai_chat']['llm'].get('reasoning')}")


async def test_reasoning_dialect_via_config(h: Harness):
    print("\n[26] 换方言端点：config 驱动")
    # 必须改**文件**再 reload：reload_config() 会重读 config.json，
    # 只改内存里的 cfg 会被下一次重载覆盖掉（这正是它的正确行为）。
    saved = json.loads(h.cfg_path.read_text(encoding="utf-8"))
    saved["plugins"]["ai_chat"]["llm"]["reasoning"] = {
        "dialect": "thinking", "default": "high"}
    h.cfg_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    # 默认的 reload_config() 不读文件（ConfigManager 只在构造时读一次）；
    # 手工改文件要走 from_disk=True。
    h.plugin.reload_config(from_disk=True)
    check("from_disk=True 真的重读了文件（方言已切换）",
          h.plugin.reasoning.dialect == "thinking", h.plugin.reasoning.dialect)
    await establish(h)
    h.clear()
    await h.private("方言一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("thinking 方言下发的字段正确",
          h.llm.last_reasoning_params == {"thinking": {"type": "enabled"}},
          f"{h.llm.last_reasoning_params}")
    check("旧配置 reasoning_effort 仍被接受",
          h.plugin._build_reasoning_policy() is not None, "构造失败")
    saved["plugins"]["ai_chat"]["llm"]["reasoning"] = {
        "dialect": "effort", "default": "off"}
    h.cfg_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    h.plugin.reload_config()


async def test_config_template_seeding(h: Harness):
    """config.json 配置模板：补齐、不覆盖用户值、幂等、与内置默认等价。"""
    DEFAULT_CONFIG = plugin_module("main").DEFAULT_CONFIG

    print("\n[27] config.json 配置模板")
    saved = json.loads(h.cfg_path.read_text(encoding="utf-8"))
    section = saved["plugins"]["ai_chat"]

    check("启动即把 plugins.ai_chat 段补进 config.json",
          isinstance(section, dict) and "llm" in section, f"{sorted(section or [])}")
    check("模板含全部顶层段",
          set(DEFAULT_CONFIG) <= set(section), f"缺 {set(DEFAULT_CONFIG) - set(section)}")
    leaf_default = _flatten_paths(DEFAULT_CONFIG)
    leaf_file = _flatten_paths(section)
    check("模板覆盖全部叶子配置项（文件即清单）",
          set(leaf_default) <= set(leaf_file),
          f"缺 {sorted(set(leaf_default) - set(leaf_file))[:6]}")

    # 用户改过的值不能被覆盖；缺键要补上
    saved["plugins"]["ai_chat"]["session"] = {"scope": "user"}      # 只剩一个键
    saved["plugins"]["ai_chat"]["trigger"]["terminators"] = ["??"]   # 用户自定义
    h.cfg_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    h.plugin._cfg_root.reload()
    wrote = h.plugin.seed_config_file()
    after = json.loads(h.cfg_path.read_text(encoding="utf-8"))["plugins"]["ai_chat"]
    check("补齐动作确实写了盘", wrote, "应返回 True")
    check("用户改过的值保持不变",
          after["trigger"]["terminators"] == ["??"]
          and after["session"]["scope"] == "user",
          f"{after['trigger']['terminators']} / {after['session']}")
    check("缺键被补回默认",
          after["session"].get("history_max_messages") == 400
          and "name_template" in after["session"],
          f"{after['session']}")
    check("幂等：再补一次不写盘", h.plugin.seed_config_file() is False, "重复写了盘")

    # 命令路径
    h.clear()
    # 先造一个缺键状态
    reloaded = json.loads(h.cfg_path.read_text(encoding="utf-8"))
    del reloaded["plugins"]["ai_chat"]["wrapping"]
    h.cfg_path.write_text(json.dumps(reloaded, ensure_ascii=False), encoding="utf-8")
    h.plugin._cfg_root.reload()
    await h.private("/aichat config", user_id=SUPERUSER, settle=0.9)
    text = await h.wait_reply()
    check("config 命令补齐并回报", "已把" in text and "wrapping" in text, text[:200])
    h.clear()
    await h.private("/aichat config", user_id=SUPERUSER, settle=0.9)
    text = await h.wait_reply()
    check("已完整时回报「无需补齐」", "无需补齐" in text, text[:200])

    # 非超级用户拿不到这个命令
    h.clear()
    await h.private("/aichat config", settle=0.8)
    text = await h.wait_reply()
    check("普通用户不能用 config", "未知命令" in text, text[:120])


async def test_data_dir_resolution(h=None):
    """data_dir 的相对基准：裸名字随插件，显式相对随 config.json 所在目录。"""
    print("\n[29] data_dir 路径解析")
    AiChatPlugin = plugin_module("main").AiChatPlugin

    # 插件实际加载自哪个目录（测试里是临时副本，不是仓库）
    plugin_dir = Path(plugin_module("main").__file__).resolve().parent

    class _Fake:
        # 三个方法一起挂上：_resolve_data_dir 现在复用 _resolve_relative
        _resolve_data_dir = AiChatPlugin._resolve_data_dir
        _resolve_relative = AiChatPlugin._resolve_relative
        _config_dir = AiChatPlugin._config_dir

        class _cfg_root:
            _path = str(ROOT / "config.json")

        def __init__(self, value):
            self.cfg = {"data_dir": value}

    check("留空 → 插件包内 data/",
          _Fake("")._resolve_data_dir() == str(plugin_dir / "data"),
          _Fake("")._resolve_data_dir())
    check("裸名字 → 插件包内",
          _Fake("data")._resolve_data_dir() == str(plugin_dir / "data"),
          _Fake("data")._resolve_data_dir())
    check("显式相对 → config.json 所在目录",
          _Fake("./data/ai_chat")._resolve_data_dir() == str(ROOT / "data" / "ai_chat"),
          _Fake("./data/ai_chat")._resolve_data_dir())
    check("绝对路径原样使用",
          _Fake("/tmp/abs")._resolve_data_dir() == "/tmp/abs",
          _Fake("/tmp/abs")._resolve_data_dir())

    # 同一套口径下，指到工作区任意位置的显式相对路径都必须落在 config 目录旁
    expected = ROOT / "plugins" / "some_plugin" / "data"
    got = _Fake("./plugins/some_plugin/data")._resolve_relative(
        "./plugins/some_plugin/data")
    check("显式相对路径解析到工作区（不落插件包里）",
          got == str(expected), f"{got} != {expected}")



def plugin_module(name: str):
    """取**插件自己加载的**子模块（`napcat_plugin_ai_chat.<name>`）。

    ⚠️ 别用 `from plugins.ai_chat.segments import ...`：插件是以合成包名
    （`napcat_plugin_ai_chat`）加载的，而仓库根又在 `sys.path` 上，于是同一个
    文件会存在**两个模块对象** —— 类不是同一个类、`isinstance` 失效，
    更阴的是方法解析可能落到另一份副本上（实测：修好的行为在测试里看起来没生效）。
    要断言插件的行为，就必须用它自己那份。
    """
    for key in (f"napcat_plugin_ai_chat.{name}", f"{PLUGIN_PKG}.{name}"):
        if key in sys.modules:
            return sys.modules[key]
    # 两边都不在（回归用例会故意摘掉模块）→ 退回按仓库路径加载
    import importlib
    return importlib.import_module(f"{PLUGIN_PKG}.{name}")

def _flatten_paths(data, prefix=""):
    out = []
    for k, v in (data or {}).items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.extend(_flatten_paths(v, prefix=f"{path}."))
        else:
            out.append(path)
    return out


async def test_prompt_tables_and_formulas(h=None):
    """提示词契约：AI 要**主动**用表格与公式（会被渲染成图片）。

    这条用例盯的是「提示词自相矛盾」这类回归：人格里曾经写着
    「`|表格|` 会原样显示成符号，很难看」—— 模型照做就不发表格，
    出图能力等于白做。所以这里把措辞钉住。
    """
    c = plugin_module("compaction")
    persona = c.DEFAULT_PERSONA

    print("\n[42] 文本模式提示词：主动用表格与公式")
    check("明确要求主动用表格", "表格" in persona and "主动用" in persona, persona[:120])
    check("说明会被渲染成图片", "渲染成图片" in persona, "没告诉模型表格/公式会变成图")
    check("给出表格分隔行写法（缺了就不识别）",
          "| --- |" in persona and "分隔行" in persona, "没说分隔行")
    check("给出行间公式定界符", "$$...$$" in persona, "没说 $$ 写法")
    check("给出行内公式定界符", "$...$" in persona, "没说 $ 写法")
    check("给常用 LaTeX 命令示例",
          "\\frac" in persona and "\\sqrt" in persona, "没有 LaTeX 示例")
    check("提醒别塞太多", "1~2 个" in persona or "别塞太多" in persona, "没提数量上限")

    print("\n[43] 仍然禁止不会被渲染的 Markdown（否则用户看到一堆符号）")
    # 断言「意图」而不是某一句原话：措辞会调整，但「不渲染的东西别用」这个意图
    # 必须在（否则用户看到一堆星号和井号）。
    check("加粗仍是禁用项",
          "加粗" in persona and ("别用" in persona or "不要用" in persona),
          persona[:200])
    check("代码围栏仍是禁用项", "围栏" in persona, "没提代码围栏")
    check("标题仍是禁用项", "#" in persona, "没提标题")
    check("不再说「表格原样显示成符号」这种自相矛盾的话",
          "很难看" not in persona, persona[:200])

    print("\n[44] 工具指引与 Markdown 模式都要跟着一致")
    g = c.tool_guidance(web=True, markdown=False)
    check("工具指引提醒把结构化结果整理成表格",
          "表格" in g, g[-120:])
    check("markdown=False 时的指引也包含表格公式", "表格" in g)

    md = c.build_system_prompt(chat_kind="private", markdown=True)
    check("Markdown 模式含人格里的表格公式指引",
          "表格与公式要主动用" in md, "人格没带上")
    check("Markdown 模式说明整篇可用 Markdown",
          "整篇" in md and "Markdown 模式" in md, md[:200])
    check("Markdown 模式不再谎称「纯文本要求作废」",
          "纯文本排版」的要求**作废**" not in md, "说法过时了")

    plain = c.build_system_prompt(chat_kind="private", markdown=False)
    check("两种模式都带表格公式指引（能力是插件固有的）",
          "表格与公式要主动用" in plain and "表格与公式要主动用" in md,
          "有一边缺了")


async def test_config_seed_unit(h=None):
    """干净配置下的补齐不变式（不经过 Harness，避免它自己的覆写干扰断言）。

    核心不变式：**补齐出来的段与内置默认深合并后逐键相等**。
    不成立就说明「文件里写的」与「实际生效的」不是一回事 —— 那这份模板就是在骗人。
    """
    import tempfile
    from napcat.config import ConfigManager
    DEFAULT_CONFIG = plugin_module("main").DEFAULT_CONFIG
    deep_merge = plugin_module("main").deep_merge
    AiChatPlugin = plugin_module("main").AiChatPlugin

    print("\n[28] 配置补齐不变式（干净配置）")
    tmpdir = Path(tempfile.mkdtemp(prefix="seed-unit-", dir=str(ROOT / "tests" / ".test-tmp")))
    path = tmpdir / "config.json"
    path.write_text(json.dumps({"ws_url": "ws://127.0.0.1:1", "plugins": {}}),
                    encoding="utf-8")
    cfg = ConfigManager(str(path))

    # 用插件真实的补齐实现（打桩掉它需要的 logger/cfg）
    class _Fake:
        config = cfg.plugin("ai_chat")
        logger = None
    written = AiChatPlugin.seed_config_file(_Fake())
    check("空配置被补齐", written, "应写入")

    section = json.loads(path.read_text(encoding="utf-8"))["plugins"]["ai_chat"]
    effective = deep_merge(DEFAULT_CONFIG, section)
    check("补齐段 ≡ 内置默认（逐键）",
          json.dumps(effective, sort_keys=True, ensure_ascii=False)
          == json.dumps(DEFAULT_CONFIG, sort_keys=True, ensure_ascii=False),
          "模板与内置默认不一致")
    check("再补一次是幂等的", AiChatPlugin.seed_config_file(_Fake()) is False,
          "重复写盘")

    # 用户值优先
    cfg.set("plugins.ai_chat.session", {"scope": "user"})
    cfg.set("plugins.ai_chat.llm.reasoning.default", "high")
    AiChatPlugin.seed_config_file(_Fake())
    after = cfg.plugin("ai_chat").all()
    check("补齐不覆盖已有的用户值",
          after["session"]["scope"] == "user"
          and after["llm"]["reasoning"]["default"] == "high",
          f"{after['session']} / {after['llm']['reasoning']}")
    check("同段的其它缺键仍被补上",
          after["session"].get("history_max_messages") == 400,
          f"{after['session']}")

    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_no_deferred_relative_imports(h: Harness):
    """回归：对话进行中不能有「函数内相对 import」。

    真实故障：热重载会先把 `napcat_plugin_ai_chat*` 整个子树从 `sys.modules` 里
    摘掉再重新导入；如果此时有函数**在调用时**才去 `from .llm import ...`，
    Python 会重新解析父包 → `ModuleNotFoundError: No module named
    'napcat_plugin_ai_chat'`，而它发生在一轮对话中间，用户只看到
    「⚠️ AI 对话出错」。

    这里直接复刻那一刻：把包从 `sys.modules` 里摘掉，然后跑完一整轮
    （含工具调用 + 异常路径 + 压缩），全程不许抛异常。
    """
    print("\n[30] 回归：摘掉 sys.modules 后仍能跑完一轮")
    await establish(h)
    h.clear()

    # 先让一个工具**失败**（覆盖工具异常里的 err_text 路径）
    state = {"n": 0}

    def responder(messages, tools):
        state["n"] += 1
        if state["n"] == 1:
            return StubResult(text="", tool_calls=[StubCall("no_such_tool", {}, "c1")])
        return StubResult(text="工具报错了但我还活着。")

    h.llm.responder = responder
    pkg = "napcat_plugin_ai_chat"
    stashed = {k: v for k, v in sys.modules.items()
               if k == pkg or k.startswith(pkg + ".")}
    try:
        for k in stashed:
            sys.modules.pop(k, None)
        await h.private("工具会失败//", settle=1.4)
        reply = await h.wait_reply()
        check("模块被摘掉后对话仍走通", "还活着" in reply, reply[:120])
        check("工具失败被回喂给模型（没有炸穿）", state["n"] == 2, f"n={state['n']}")

        # 压缩路径（会调用 compaction / segments，同样不能有延迟 import）
        h.clear()
        h.llm.responder = None
        h.plugin.policy.max_tokens = 400
        for i in range(4):
            await h.private(f"填充{i}" + "内容" * 30 + "//", settle=1.0)
            await h.wait_for_replies(i + 1)
        check("压缩路径也没炸", h.plugin.store.get(_current_session(h).id).compact_count >= 1,
              f"compact_count={h.plugin.store.get(_current_session(h).id).compact_count}")
        h.plugin.policy.max_tokens = 16384
    finally:
        h.llm.responder = None
        sys.modules.update(stashed)

    # 静态检查：源码里不许再有「函数内相对 import」
    offenders = []
    for f in sorted(PLUGIN_DIR.glob("*.py")):
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if line.startswith((" ", "\t")) and (
                    stripped.startswith("from .") or stripped.startswith("import .")):
                offenders.append(f"{f.name}:{lineno}: {stripped}")
    check("源码里没有函数内相对 import", not offenders, f"{offenders}")


async def test_no_coupling_with_other_plugins(h: Harness):
    """aichat 与 bupt_news **彻底解耦**：不读它的数据、不加载它的模块、不注册它的工具。

    这条用例是「防复发」的：耦合曾经真实存在（读 `plugins/bupt_news/data/notices.jsonl`
    + 按文件路径加载 `bupt_news.weather`），特点是**不 import 也能生效** ——
    所以只查 import 表是查不出来的，必须连源码文本一起卡住。
    """
    print("\n[31] 与其它插件彻底解耦")
    # ⚠️ 一律从**本轮 harness 的插件目录**取源码，别走 `plugin_module("main")`：
    # `[30]` 用例会故意把包从 `sys.modules` 里摘掉，那个合成模块名随后可能被
    # 重新填成上一轮那份**已删除临时目录**里的旧对象 —— 读源码就 FileNotFoundError。
    pkg_dir = Path(h.cfg["plugin_dir"]) / "ai_chat"

    check("不再有 _load_sibling_module（跨插件加载器已删）",
          "_load_sibling_module" not in
          (pkg_dir / "main.py").read_text(encoding="utf-8"),
          "源码里还有残留")
    check("不再有 _weather_line / _notice_records 等方法",
          not any(hasattr(h.plugin, n) for n in
                  ("_weather_line", "_read_notices", "_search_notices",
                   "_get_notice", "_notice_records")),
          f"{[n for n in dir(h.plugin) if 'notice' in n or 'weather' in n]}")

    # 全包源码文本扫描：任何子模块都不许出现兄弟插件的名字
    offenders = []
    for src in sorted(pkg_dir.glob("*.py")):
        text = src.read_text(encoding="utf-8")
        for word in ("bupt_news", "buptnews"):
            if word in text:
                for i, line in enumerate(text.splitlines(), 1):
                    if word in line:
                        offenders.append(f"{src.name}:{i}: {line.strip()[:70]}")
    check("插件源码里不再出现兄弟插件名字", not offenders, "; ".join(offenders))

    # 工具表里不该再有外部知识源工具
    names = set(h.plugin.tools.names())
    check("不再注册通知/天气工具",
          not (names & {"list_recent_notices", "search_notices", "get_notice",
                        "get_weather", "list_schedules"}),
          f"{sorted(names)}")
    # 但内置的「自给自足」工具必须还在（解耦不能把工具削没了）
    check("自给自足的内置工具仍在",
          {"get_current_time", "search_history", "get_session_info"} <= names,
          f"{sorted(names)}")
    template = h.plugin.cfg
    check("配置模板里不再有 notices / weather",
          "notices" not in template and "weather" not in template,
          f"{sorted(template)}")

    # 老部署的 config.json 里那两段失效配置必须被清理掉，而不是留在文件里误导人
    # （当年真配过：`notices.path` 指着 bupt_news 的私有数据目录）
    cfg_path = h.cfg_path
    on_disk = json.loads(cfg_path.read_text(encoding="utf-8"))
    on_disk["plugins"]["ai_chat"]["notices"] = {
        "enabled": True, "path": "./plugins/bupt_news/data", "limit": 10}
    on_disk["plugins"]["ai_chat"]["weather"] = {"enabled": True}
    cfg_path.write_text(json.dumps(on_disk, ensure_ascii=False), encoding="utf-8")

    # 模拟一次真实启动：重读文件 →（on_load 会调）清退休段
    h.plugin.reload_config(from_disk=True)
    check("退休段被读进来了（模拟老部署的前置条件）",
          h.plugin.config.get("notices") is not None,
          f"{h.plugin.config.get('notices')!r}")
    h.plugin.retire_config_sections()
    h.plugin.reload_config(from_disk=True)
    check("退休配置段已被清掉（内存）",
          h.plugin.config.get("notices") is None
          and h.plugin.config.get("weather") is None,
          f"notices={h.plugin.config.get('notices')!r}")

    saved_raw = json.loads(cfg_path.read_text(encoding="utf-8"))["plugins"]["ai_chat"]
    check("退休配置段已落盘清掉（config.json 不再误导人）",
          "notices" not in saved_raw and "weather" not in saved_raw,
          f"{sorted(saved_raw)}")
    # 深层退休键（hget 改单条后不再需要）也要能被清掉
    check("深层退休键也被清掉（history.max_images_per_call）",
          "max_images_per_call" not in (saved_raw.get("history") or {}),
          f"{sorted((saved_raw.get('history') or {}))}")
    check("清理后其它配置段一个没少",
          {"llm", "session", "trigger", "reply", "tools", "binding"} <= set(saved_raw),
          f"{sorted(saved_raw)}")


async def test_binding_allowlist(h: Harness):
    """服务范围：只对配置里列出的群/私聊提供服务，且与权限等级无关。"""
    ChatTarget = plugin_module("store").ChatTarget
    await establish(h, group_id=GROUP)

    print("\n[32] 服务范围白名单")
    plugin = h.plugin

    # ---- 默认：白名单为空 + empty_means=deny → 谁都不用 ----
    # ⚠️ 群白名单为空时会回落到框架的 `bound_groups`，所以要连它一起清掉，
    # 才是真正的「什么都没配」。
    saved_bound = set(h.client.groups.list_bound())
    h.client.groups.clear()

    def restore_bound():
        h.client.groups.clear()
        h.client.groups.bind_many(list(saved_bound))

    plugin.cfg["binding"] = {"groups": [], "users": [], "empty_means": "deny",
                             "notify_denied": True, "notify_cooldown": 0}
    for target, uid, label in [
        (ChatTarget("group", GROUP), ALICE, "群"),
        (ChatTarget("private", ALICE), ALICE, "私聊"),
        (ChatTarget("group", GROUP), SUPERUSER, "超级用户在群"),
        (ChatTarget("private", SUPERUSER), SUPERUSER, "超级用户私聊"),
    ]:
        ok, why = plugin.check_access(target, uid)
        check(f"空名单拒绝：{label}", not ok, f"allow={ok}")
        check(f"空名单的说明提到 binding：{label}", "binding" in why, why[:80])

    # ---- 配好白名单 ----
    restore_bound()
    plugin.cfg["binding"] = {"groups": [GROUP, OTHER_GROUP],
                             "users": [ALICE, SUPERUSER],
                             "empty_means": "deny",
                             "notify_denied": True, "notify_cooldown": 0}
    cases = [
        (ChatTarget("group", GROUP), ALICE, True, "已绑群 + 已授权的人"),
        (ChatTarget("group", OTHER_GROUP), BOB, True, "已绑群（group 作用域下全群可用）"),
        (ChatTarget("group", 99999), ALICE, False, "未绑定的群"),
        (ChatTarget("private", ALICE), ALICE, True, "已授权的私聊"),
        (ChatTarget("private", BOB), BOB, False, "未授权的私聊"),
    ]
    for target, uid, want, label in cases:
        ok, why = plugin.check_access(target, uid)
        check(f"{label}", ok == want, f"allow={ok} want={want} why={why[:60]}")

    # ---- 与权限等级无关：超级用户也不能越界 ----
    ok, why = plugin.check_access(ChatTarget("group", 99999), SUPERUSER)
    check("超级用户在未绑定的群里也用不了", not ok, f"allow={ok}")
    # 小窗地址（private + 来源群）：授权与否看**这个小窗的主人**，
    # 而不是发消息的人 —— 否则群里任何人都能借别人的小窗地址越权。
    ok, why = plugin.check_access(ChatTarget("private", BOB, GROUP), BOB)
    check("未授权的小窗地址被拒绝", not ok, f"allow={ok}")
    ok, why = plugin.check_access(ChatTarget("private", ALICE, GROUP), ALICE)
    check("已授权用户的小窗可用", ok, f"allow={ok}")

    # ---- 未绑定的群 ----
    # 注意走不了 WS：框架的 `_on_ws_event()` 在 EventBus **之前**就把未绑群的
    # 群消息整条丢了（`GroupsBinding.is_bound`）。所以这里直接调插件入口，
    # 验证的是插件的**第二道闸**（插件自派发 / 直接 cmds.handle 的路径）。
    from napcat.models import MessageEvent, Sender

    def group_event(text, gid, uid=BOB):
        return MessageEvent(
            message_type="group", group_id=gid, user_id=uid,
            message=[{"type": "text", "data": {"text": text}}],
            plain_text=text, message_id=1,
            sender=Sender(user_id=uid, nickname="测试", role="member"))

    check("框架层确实会丢弃未绑群的消息（这是第一道闸）",
          not h.client.groups.is_bound(99999), "客户端认为 99999 已绑定")
    check("框架层认可已绑群", h.client.groups.is_bound(GROUP),
          "客户端认为 GROUP 未绑定")

    h.clear()
    plugin._denied_notice.clear()
    await plugin._handle_message(group_event("未绑群说话//", 99999))
    await asyncio.sleep(0.3)
    check("未绑群：插件不回复", not h.replies, f"{h.replies}")
    check("未绑群：插件不调模型", h.llm.calls == 0, f"calls={h.llm.calls}")

    # 未绑群 + 没 @机器人：连回绝说明都不发（否则群里聊两句就被刷屏）
    h.clear()
    plugin._denied_notice.clear()
    await plugin._handle_message(group_event("没@机器人//", 99999))
    await asyncio.sleep(0.3)
    check("未绑群且未 @ 时完全静默", not h.replies, f"{h.replies}")

    # 未绑群 + @机器人：给回绝说明（命令也要回绝，且不回显命令内容）
    h.clear()
    plugin._denied_notice.clear()
    ev_cmd = group_event("[@10001] /aichat current", 99999)
    await plugin._handle_message(ev_cmd)         # 对话路径（会被命令前缀挡住）
    await h.client.cmds.handle(ev_cmd)           # 命令路径
    try:
        await h.wait_for_replies(1, timeout=5.0)
    except TimeoutError:
        pass
    texts = [reply_text(r) for r in h.replies]
    check("未绑群的命令被回绝", any("没有开通" in t for t in texts), f"{texts}")
    check("回绝不泄露命令内容", not any("ai_chat ——" in t for t in texts), f"{texts}")

    # ---- 群白名单没配时，回落到框架的 bound_groups ----
    saved_groups = plugin.cfg["binding"]["groups"]
    try:
        plugin.cfg["binding"]["groups"] = []
        ok, _ = plugin.check_access(ChatTarget("group", GROUP), ALICE)
        check("binding.groups 未配时复用框架 bound_groups", ok,
              "应当放行（框架已绑该群）")
        ok, _ = plugin.check_access(ChatTarget("group", 99999), ALICE)
        check("复用兜底同样拦未绑群", not ok, "应当拒绝")
    finally:
        plugin.cfg["binding"]["groups"] = saved_groups

    # ---- 未授权的私聊：同样回绝 ----
    h.clear()
    await h.private("私聊未授权//", user_id=BOB, settle=0.8)
    texts = [reply_text(r) for r in h.replies]
    check("未授权私聊收到回绝说明", texts and "没有开通" in texts[0], f"{texts}")
    check("未授权私聊不调模型", h.llm.calls == 0, f"calls={h.llm.calls}")

    # ---- 已绑定的目标正常服务 ----
    h.clear()
    await h.group("已绑群说话//", at_bot=True, user_id=ALICE, settle=1.0)
    check("已绑群正常服务", bool(h.replies) and h.llm.calls == 1,
          f"replies={len(h.replies)} calls={h.llm.calls}")

    # ---- empty_means=allow：空名单 = 不限制（显式选择才生效）----
    h.client.groups.clear()
    plugin.cfg["binding"] = {"groups": [], "users": [], "empty_means": "allow",
                             "notify_denied": True, "notify_cooldown": 0}
    ok, _ = plugin.check_access(ChatTarget("group", 99999), ALICE)
    check("empty_means=allow 时不再拦", ok, "应当放行")
    ok, _ = plugin.check_access(ChatTarget("private", BOB), BOB)
    check("empty_means=allow 对私聊同样放行", ok, "应当放行")
    restore_bound()


async def test_binding_group_vs_private(h: Harness):
    """服务范围的两条独立通道：群开通 = 全群可用；私聊按人开通。"""
    ChatTarget = plugin_module("store").ChatTarget

    print("\n[33] 服务范围：群与私聊是两条独立通道")
    plugin = h.plugin
    saved = dict(plugin.cfg["binding"])
    try:
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [ALICE],
                                 "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 0}
        # 群：开通了就是全群可用（不按人过滤）—— 每人各有自己的会话
        ok, _ = plugin.check_access(ChatTarget("group", GROUP), ALICE)
        check("已开通群：成员可用", ok, "应当放行")
        ok, _ = plugin.check_access(ChatTarget("group", GROUP), BOB)
        check("已开通群：**其他人也可用**（群是整体开通）", ok,
              "应当放行：群里每人有自己的会话")
        # 未开通的群：谁都不行
        ok, why = plugin.check_access(ChatTarget("group", 99999), ALICE)
        check("未开通群：即使人在私聊白名单里也不行", not ok, f"allow={ok}")
        check("说明指向 groups 名单", "groups" in why, why[:80])
        # 私聊：按人开通
        ok, _ = plugin.check_access(ChatTarget("private", ALICE), ALICE)
        check("已开通的人：私聊可用", ok, "应当放行")
        ok, why = plugin.check_access(ChatTarget("private", BOB), BOB)
        check("未开通的人：私聊不行", not ok, f"allow={ok}")
        check("说明指向 users 名单", "users" in why, why[:80])
        # 群开通不影响私聊
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [],
                                 "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 0}
        ok, _ = plugin.check_access(ChatTarget("group", GROUP), BOB)
        check("群开通与私聊白名单互不影响（群仍可用）", ok, "应当放行")
        ok, why = plugin.check_access(ChatTarget("private", BOB), BOB)
        check("群开通不自动开放私聊", not ok, f"allow={ok}")
    finally:
        plugin.cfg["binding"] = saved


async def test_binding_cooldown(h: Harness):
    """回绝提示要有冷却，不能刷屏。"""
    ChatTarget = plugin_module("store").ChatTarget

    print("\n[34] 回绝提示冷却")
    plugin = h.plugin
    saved = dict(plugin.cfg["binding"])
    try:
        plugin.cfg["binding"] = {"groups": [], "users": [], "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 600}
        plugin._denied_notice.clear()
        h.clear()
        target = ChatTarget("private", BOB)
        check("首次回绝要提示", plugin._access_denied(target, BOB), "应回绝")
        await asyncio.sleep(0.2)
        first = len(h.replies)
        check("首次确实发了提示", first >= 1, f"{first}")
        for _ in range(3):
            plugin._access_denied(target, BOB)
        await asyncio.sleep(0.3)
        check("冷却期内不再重复提示", len(h.replies) == first,
              f"{len(h.replies)} != {first}")

        # notify_denied=False → 静默拒绝（不回任何话）
        plugin.cfg["binding"]["notify_denied"] = False
        plugin._denied_notice.clear()
        h.clear()
        plugin._access_denied(target, BOB)
        await asyncio.sleep(0.3)
        check("notify_denied=false 时静默拒绝", not h.replies, f"{h.replies}")
    finally:
        plugin.cfg["binding"] = saved


async def test_state_migration(h=None):
    """旧 state.json（含 scope + bindings 键）要能被读成新模型。"""
    import tempfile, json as _json
    SessionStore = plugin_module("store").SessionStore

    print("\n[35] 旧状态迁移：scope → owner_kind，bindings → focus")
    tmpdir = Path(tempfile.mkdtemp(prefix="migrate-",
                                   dir=str(ROOT / "tests" / ".test-tmp")))
    sdir = tmpdir / "sessions"
    sdir.mkdir(parents=True)
    legacy = {
        "version": 1,
        "seq": 2,
        "sessions": {
            "s0001": {"id": "s0001", "name": "老的群会话", "scope": "group",
                      "owner_id": 30001, "owner_name": "老群", "message_count": 3},
            "s0002": {"id": "s0002", "name": "老的私聊会话", "scope": "user",
                      "owner_id": 20001, "owner_name": "爱丽丝", "message_count": 1},
        },
        # 旧键名：chat key → session id（没有 @user: 段）
        "bindings": {"group:30001": "s0001", "private:20001": "s0002"},
    }
    (sdir / "state.json").write_text(_json.dumps(legacy, ensure_ascii=False),
                                     encoding="utf-8")
    (sdir / "s0001.jsonl").write_text(
        _json.dumps({"role": "user", "blocks": [{"type": "text", "text": "老的"}],
                     "metadata": {}, "ts": ""}, ensure_ascii=False) + "\n",
        encoding="utf-8")

    st = SessionStore(str(tmpdir))
    g = st.get("s0001")
    u = st.get("s0002")
    check("群会话的 scope 迁移成 owner_kind=group",
          g is not None and g.owner_kind == "group", f"{g and g.owner_kind}")
    check("私聊会话的 scope 迁移成 owner_kind=user",
          u is not None and u.owner_kind == "user", f"{u and u.owner_kind}")
    check("旧记录默认没有受邀名单",
          g is not None and g.guests == [], f"{g and g.guests}")
    check("旧 bindings 键仍能被读到（不丢历史配置）",
          len(st.focus_table()) == 2, f"{st.focus_table()}")
    check("历史照旧可读", [m.text for m in st.history("s0001")] == ["老的"],
          f"{[m.text for m in st.history('s0001')]}")
    # 新写法写盘用的是 focus 键
    st.set_focus(__import__("plugins.ai_chat.store", fromlist=["ChatTarget"])
                 .ChatTarget("group", 30001), 20001, "s0001")
    saved = _json.loads((sdir / "state.json").read_text(encoding="utf-8"))
    check("落盘用新键名 focus", "focus" in saved, f"{sorted(saved)}")
    check("焦点键带 user 段",
          any("@user:20001" in k for k in saved["focus"]), f"{saved['focus']}")
    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_shared_context(h: Harness):
    """同一会话下，参与者共享完全相同的上下文（含彼此说的话）。"""
    print("\n[36] 共享会话：上下文对所有参与者一致")
    h.clear()
    sid = (await establish(h, user_id=ALICE, group_id=GROUP)).id
    await h.group("Alice 开场//", at_bot=True, user_id=ALICE, settle=1.2)
    await h.wait_for_replies(1)

    # 邀请 Bob，并让他自己切过来
    h.clear()
    await h.group(f"/aichat invite {sid} {BOB}", at_bot=True, user_id=ALICE, settle=1.2)
    await h.wait_for_replies(1)
    check("邀请成功", "已邀请" in reply_text(h.replies[0]), reply_text(h.replies[0])[:120])
    h.clear()
    await h.group(f"/aichat attach {sid} --force=true", at_bot=True, user_id=BOB,
                  settle=1.2)
    text = await h.wait_reply()
    check("attach 提示这是共用会话（上下文共享）", "共享" in text, text[:200])
    check("两人焦点指向同一个会话",
          _current_session(h, user_id=ALICE, group_id=GROUP).id == sid
          and _current_session(h, user_id=BOB, group_id=GROUP).id == sid,
          "焦点不一致")

    # Bob 说话：他的请求里必须包含 Alice 之前说的话，并带说话人标注
    h.clear()
    await h.group("Bob 接着说//", at_bot=True, user_id=BOB, settle=1.2)
    await h.wait_for_replies(1)
    req = h.llm.requests[-1]["messages"]
    user_msgs = [m["content"] for m in req if m["role"] == "user"]
    check("共享上下文：Bob 看得到 Alice 的话",
          any("Alice 开场" in u for u in user_msgs), f"{user_msgs}")
    check("两人各自的话都带说话人标注",
          any(u.startswith("[用户20001]") and "Alice 开场" in u for u in user_msgs)
          and any(u.startswith("[用户20002]") and "Bob 接着说" in u for u in user_msgs),
          f"{user_msgs}")
    check("系统提示说明会话是共用的",
          "共享" in req[0]["content"], req[0]["content"][:400])

    # Alice 说话：也应看到 Bob 的话（双向共享）
    h.clear()
    await h.group("Alice 又问//", at_bot=True, user_id=ALICE, settle=1.2)
    await h.wait_for_replies(1)
    user_msgs2 = [m["content"] for m in h.llm.requests[-1]["messages"]
                  if m["role"] == "user"]
    check("共享上下文：Alice 也看得到 Bob 的话",
          any("Bob 接着说" in u for u in user_msgs2), f"{user_msgs2}")
    check("历史是同一份（不是各自副本）",
          [m.text for m in h.plugin.store.history(sid) if m.role == "user"]
          == ["Alice 开场", "Bob 接着说", "Alice 又问"],
          f"{[m.text for m in h.plugin.store.history(sid) if m.role == 'user']}")


async def test_solo_session_not_marked_shared(h: Harness):
    """只有自己用的会话不该被标注成共用（前缀与系统提示都别多话）。"""
    print("\n[37] 独享会话不标共用")
    await establish(h)
    h.clear()
    await h.private("我一个人说//", settle=1.2)
    await h.wait_for_replies(1)
    req = h.llm.requests[-1]["messages"]
    check("独享会话不加说话人前缀",
          not any(m["content"].startswith("[") for m in req if m["role"] == "user"),
          f"{[m['content'] for m in req if m['role'] == 'user']}")
    check("独享会话的系统提示不提「共用」",
          "共享" not in req[0]["content"], req[0]["content"][:300])


async def test_private_binding_bootstrap(h: Harness):
    """回归：私聊白名单为空时，超级用户仍必须能开出第一条私聊。

    真实故障（死锁）：服务范围管理命令只能在私聊用，而 `users` 为空 + `deny` 时
    私聊又被服务范围挡住 —— 于是永远开不出第一条私聊，提示还叫人「联系管理员」，
    而管理员正是执行命令的那个人。
    """
    print("\n[38] 私聊开通的死锁：add 必须能自举")
    plugin = h.plugin
    saved = dict(plugin.cfg["binding"])
    try:
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [],
                                 "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 0}
        plugin.config.set("binding.users", [])
        plugin._denied_notice.clear()

        # 死锁现场：私聊不可用
        ChatTarget = plugin_module("store").ChatTarget
        ok, _ = plugin.check_access(ChatTarget("private", SUPERUSER), SUPERUSER)
        check("私聊白名单为空时确实用不了", not ok, "本该被拒")

        # 但 bind 必须能执行（不受服务范围限制）
        h.clear()
        await h.private("/aichat add user", user_id=SUPERUSER, settle=1.0)
        texts = [reply_text(r) for r in h.replies]
        check("开通命令本身不被服务范围拦",
              any("已开通" in t for t in texts), f"{texts}")
        check("没有多余的「未开通」回绝", not any("⛔" in t for t in texts), f"{texts}")
        check("第一条私聊已开通", SUPERUSER in plugin.cfg["binding"]["users"],
              f"{plugin.cfg['binding']['users']}")
        ok2, _ = plugin.check_access(ChatTarget("private", SUPERUSER), SUPERUSER)
        check("开通后私聊可用", ok2, "仍被拒")

        # targets 同样豁免（否则运维看不到范围）
        h.clear()
        await h.private("/aichat targets", user_id=SUPERUSER, settle=0.9)
        text = await h.wait_reply()
        check("targets 也豁免服务范围", "服务范围" in text, text[:120])

        # 普通用户仍然不能用（权限层拦截，与服务范围豁免无关）
        h.clear()
        await h.private("/aichat add user 20002", user_id=BOB, settle=0.7)
        check("普通用户依旧不能用 add",
              "未知命令" in await h.wait_reply(), "不该放行")
    finally:
        plugin.cfg["binding"] = saved
        plugin.config.set("binding.users", saved.get("users", []))



async def test_unbound_is_silent(h: Harness):
    """未开通的聊天**什么都不回**（连「未开通」也不提示）。

    这是默认行为，且与「键缺失」等价 —— 曾经写成
    `binding.get("notify_denied", True)`，于是老配置/精简配置因为没有这个键
    而退回「要提示」，在别人的群里凭空冒出一条机器人消息。
    """
    from napcat.models import MessageEvent, Sender

    print("\n[39] 未开通的聊天完全静默")
    plugin = h.plugin
    saved = dict(plugin.cfg["binding"])
    try:
        # 故意**不带** notify_denied（模拟老配置/精简配置）
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [SUPERUSER],
                                 "empty_means": "deny"}
        h.client.groups.clear()
        h.client.groups.bind_many([GROUP])
        plugin._denied_notice.clear()

        def group_event(text, gid, uid):
            return MessageEvent(
                message_type="group", group_id=gid, user_id=uid,
                message=[{"type": "text", "data": {"text": text}}],
                plain_text=text, message_id=1,
                sender=Sender(user_id=uid, nickname="测试", role="member"))

        # 未开通的群：说话 / 命令 都不回（框架层本来就会丢弃未绑群的消息，
        # 这里直接调插件入口验证第二道闸也是静默的）
        for label, ev in (
            ("未开通群·说话", group_event("你好//", 99999, BOB)),
            ("未开通群·命令", group_event("[@10001] /aichat current", 99999, BOB)),
        ):
            h.clear()
            plugin._denied_notice.clear()
            await plugin._handle_message(ev)
            await h.client.cmds.handle(ev)
            await asyncio.sleep(0.5)
            check(f"{label} 不回任何消息", not h.replies, f"{h.replies}")

        # 未开通的私聊：说话 / 命令 都不回
        for label, cmd in (("未开通私聊·说话", "你好//"),
                           ("未开通私聊·命令", "/aichat current")):
            h.clear()
            plugin._denied_notice.clear()
            await h.private(cmd, user_id=BOB, settle=1.0)
            await asyncio.sleep(0.5)
            check(f"{label} 不回任何消息", not h.replies, f"{h.replies}")
            check(f"{label} 也不调模型", h.llm.calls == 0, f"calls={h.llm.calls}")

        # 显式打开时才提示（调试用），且带冷却
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [SUPERUSER],
                                 "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 0}
        h.clear()
        plugin._denied_notice.clear()
        await h.private("你好//", user_id=BOB, settle=1.0)
        check("显式 notify_denied=true 时才提示",
              any("没有开通" in reply_text(r) for r in h.replies), f"{h.replies}")

        # 未开通不会影响已开通的地址（私聊要有自己的私聊会话）
        plugin.cfg["binding"] = {"groups": [GROUP], "users": [SUPERUSER],
                                 "empty_means": "deny"}
        await establish(h, user_id=SUPERUSER)
        h.clear()
        await h.private("我自己用//", user_id=SUPERUSER, settle=1.2)
        check("已开通的私聊照常服务", bool(h.replies) and h.llm.calls >= 1,
              f"replies={len(h.replies)} calls={h.llm.calls}")
    finally:
        plugin.cfg["binding"] = saved
        h.client.groups.clear()
        h.client.groups.bind_many([GROUP, OTHER_GROUP])



async def test_maxtokens_command(h: Harness):
    print("\n[40] /aichat maxtokens：按会话改 max_tokens")
    await establish(h)
    h.clear()
    await h.private("/aichat maxtokens", settle=0.9)
    text = await h.wait_reply()
    check("maxtokens 显示状态", "max_tokens" in text and "全局默认" in text, text[:200])
    base = h.plugin.llm.max_tokens

    h.clear()
    await h.private("默认一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("默认用全局 max_tokens", h.llm.last_max_tokens == base,
          f"{h.llm.last_max_tokens} vs {base}")

    h.clear()
    await h.private("/aichat maxtokens 2048", settle=0.9)
    text = await h.wait_reply()
    check("设置成功", "已" in text and "2048" in text, text[:200])
    sid = _current_session(h).id
    check("落库到 session.meta",
          h.plugin.store.get(sid).meta.get("max_tokens") == 2048,
          f"{h.plugin.store.get(sid).meta}")
    check("状态卡标为会话设置", "会话设置" in text, text[:300])

    h.clear()
    await h.private("再来一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("payload 真的用 2048", h.llm.last_max_tokens == 2048,
          f"{h.llm.last_max_tokens}")

    h.clear()
    await h.private("/aichat maxtokens 8k", settle=0.9)
    await h.wait_for_replies(1)
    check("8k 简写 = 8192",
          h.plugin.store.get(sid).meta.get("max_tokens") == 8192,
          f"{h.plugin.store.get(sid).meta}")

    h.clear()
    await h.private("/aichat maxtokens off", settle=0.9)
    await h.wait_reply()
    check("off 清除覆盖", "max_tokens" not in (h.plugin.store.get(sid).meta or {}),
          f"{h.plugin.store.get(sid).meta}")
    h.clear()
    await h.private("清掉后一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("清除后回到全局", h.llm.last_max_tokens == base, f"{h.llm.last_max_tokens}")

    h.clear()
    await h.private("/aichat maxtokens 很多", settle=0.9)
    text = await h.wait_reply()
    check("非法值有明确提示", "不认识" in text, text[:150])

    h.clear()
    await h.private("/aichat maxtokens 10", settle=0.9)
    text = await h.wait_reply()
    check("低于下界被拒", "64" in text, text[:150])

    h.clear()
    await h.private("/aichat maxtokens 2048 --session=nope-xyz", settle=0.9)
    text = await h.wait_reply()
    check("指定不存在的会话有提示", "找不到会话" in text, text[:150])


async def test_timeout_command(h: Harness):
    print("\n[41] /aichat timeout：按会话改单次调用时长上限")
    await establish(h)
    h.clear()
    await h.private("/aichat timeout", settle=0.9)
    text = await h.wait_reply()
    check("timeout 显示状态", "时长上限" in text and "全局默认" in text, text[:200])
    check("状态卡解释 stall_timeout 与它的区别", "stall_timeout" in text, text[:400])
    base = float(h.plugin.llm.timeout)

    h.clear()
    await h.private("默认一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("默认用全局 timeout", h.llm.last_timeout == base, f"{h.llm.last_timeout} vs {base}")

    h.clear()
    await h.private("/aichat timeout 600", settle=0.9)
    text = await h.wait_reply()
    check("设置成功", "已" in text and "600" in text, text[:200])
    sid = _current_session(h).id
    check("落库到 session.meta",
          h.plugin.store.get(sid).meta.get("timeout") == 600.0,
          f"{h.plugin.store.get(sid).meta}")

    h.clear()
    await h.private("设完再一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("本会话调用真的用 600", h.llm.last_timeout == 600.0, f"{h.llm.last_timeout}")

    h.clear()
    await h.private("/aichat timeout 10m", settle=0.9)
    await h.wait_for_replies(1)
    check("10m = 600 秒",
          h.plugin.store.get(sid).meta.get("timeout") == 600.0,
          f"{h.plugin.store.get(sid).meta}")

    h.clear()
    await h.private("/aichat timeout 5", settle=0.9)
    text = await h.wait_reply()
    check("低于下界被拒", "10" in text, text[:150])

    h.clear()
    await h.private("/aichat timeout abc", settle=0.9)
    text = await h.wait_reply()
    check("非法值有提示", "不认识" in text, text[:150])

    h.clear()
    await h.private("/aichat timeout off", settle=0.9)
    await h.wait_reply()
    check("off 清除覆盖", "timeout" not in (h.plugin.store.get(sid).meta or {}),
          f"{h.plugin.store.get(sid).meta}")
    h.clear()
    await h.private("清掉后一轮//", settle=1.0)
    await h.wait_for_replies(1)
    check("清除后回到全局", h.llm.last_timeout == base, f"{h.llm.last_timeout}")

    h.clear()
    await h.private("/aichat timeout 600 --session=nope-xyz", settle=0.9)
    text = await h.wait_reply()
    check("指定不存在的会话有提示", "找不到会话" in text, text[:150])


async def test_plain_text_formatting(h: Harness):
    print("\n[45] 文本模式：Markdown 残留发出去前被清掉")
    md = plugin_module("markdown")
    cases = [
        ("**重点**：注意", "重点：注意"),
        ("> 引用的话", "引用的话"),
        ("### 小标题", "小标题"),
        ("看[文档](https://example.com/a)", "看文档（https://example.com/a）"),
        ("用 `pip install x`", "用 pip install x"),
        ("~~删掉~~ 和 *斜体* 和 __粗__", "删掉 和 斜体 和 粗"),
        ("转义 \\*字面星号\\*", "转义 *字面星号*"),
    ]
    for src, want in cases:
        got = md.to_plain_text(src)
        check(f"to_plain_text({src!r})", got == want, f"{got!r} != {want!r}")
    # 公式必须一字不动（里面的 * _ \frac 不能被当成 Markdown）
    math_src = "公式 $a*b = c$ 和 $$\\frac{a}{b}$$ 不动"
    check("公式整段保护", md.to_plain_text(math_src) == math_src,
          md.to_plain_text(math_src))
    check("分隔线换成实线", md.to_plain_text("---") == "─" * 10)

    # —— 端到端：桩回一段带 Markdown 的话，文本模式下发出的必须是降噪后的 ——
    await establish(h)
    h.llm.script_text = "> **注意**：先看[文档](https://example.com/a)。\n用 `pip install x` 安装。"
    h.clear()
    await h.private("来一段//", settle=1.0)
    await h.wait_for_replies(1)
    text = await h.wait_reply()
    check("发出的话里没有 ** 残留", "**" not in text, text[:200])
    check("没有行首 > 残留", not any(ln.startswith(">") for ln in text.splitlines()), text[:200])
    check("链接改成了 标题（url）", "文档（https://example.com/a）" in text, text[:200])
    check("反引号去掉了", "`" not in text, text[:200])

    # —— 混合路径：正文 + 行内公式（公式出图，文本部分同样要降噪）——
    h.llm.script_text = "**重点**：看公式 $x^2$ 的结果。"
    h.clear()
    await h.private("再一段//", settle=1.5)
    await h.wait_for_replies(1)
    text = await h.wait_reply()
    check("图文混合时文本段也降噪", "**" not in text, text[:200])
    check("公式部分没退化成源码", "$x^2$" not in text, text[:200])
    h.llm.script_text = "好的，我收到了。"


async def test_unified_help(h: Harness):
    """帮助统一走框架 `CommandManager` 的 /help，插件不再自绘一份。"""
    from napcat.models import parse_event

    print("\n[20] 统一帮助（/help）")
    cmds = h.client.cmds

    async def help_of(text, *, uid, group_id=None):
        payload = (h._msg_payload("group", text, user_id=uid, group_id=group_id,
                                 at_bot=True) if group_id is not None
                   else h._msg_payload("private", text, user_id=uid))
        res = await cmds.handle(parse_event(payload))
        return res if isinstance(res, str) else str(res)

    # ---- 子命令清单由框架从注册信息渲染（用法行来自 Schema，不是手写）----
    text = await help_of("/help aichat", uid=SUPERUSER)
    check("列出全部子命令",
          all(f"/aichat {n}" in text for n in
              ("new", "list", "attach", "detach", "current",
               "rename", "delete", "compact", "invite", "guests", "think")),
          text[:400])
    check("已删除的 switch / stats 不再出现",
          "/aichat switch" not in text and "/aichat stats" not in text, text[:400])
    check("超级用户才看得到 targets/add/del/model/config/reload",
          all(f"/aichat {n}" in text for n in ("targets", "add", "del",
                                               "model", "config", "reload")),
          text[:600])
    check("子命令清单只给名字与介绍（不摊参数）",
          "/aichat new" in text and "/aichat new <name>" not in text, text[:400])
    check("子命令清单标注权限 + 作用域（但不摊参数）",
          "[权限：" in text and "作用域：" in text and "/aichat new <name>" not in text,
          text[:600])
    # 权限与范围改在第三级详情行呈现
    d_private_sub = await help_of("/help aichat add", uid=SUPERUSER)
    check("仅私聊范围在第三级详情里标注",
          "仅私聊" in d_private_sub and "权限：superuser" in d_private_sub,
          d_private_sub.replace("\n", " / ")[:160])

    # ---- 已删除的重复命令 ----
    check("已删除 /aichat help（改用统一 /help）",
          "/aichat help" not in text, text[:600])
    check("已删除 /aichat info（与 current 重复）",
          "/aichat info" not in text, text[:600])
    for gone in ("/aichat help", "/aichat info"):
        page = await help_of(gone, uid=SUPERUSER)
        # 子命令名写错 = 用法不正确：框架重定向到命令组帮助页（列着现存的子命令）
        check(f"{gone} 已不存在（回命令组帮助页，页里没有它）",
              page.startswith("📖 /aichat") and gone not in page, page[:120])

    # ---- 单命令详情：参数表由 Schema 生成 ----
    detail = await help_of("/help aichat.new", uid=SUPERUSER)
    check("详情列出参数与默认值",
          "attach" in detail and "默认" in detail, detail[:400])
    check("详情给出键值传参写法", "--attach=" in detail, detail[:500])
    # 空格写法必须与点号写法等价（子命令先注册 → 父命令是框架的占位 handler）
    detail_space = await help_of("/help aichat new", uid=SUPERUSER)
    check("`/help aichat new` 与 `/help aichat.new` 等价",
          detail_space == detail, detail_space[:200])

    # ---- 结束符等「非命令」信息挂在命令组描述上 ----
    check("命令组描述提到结束符", "//" in text, text[:300])
    check("命令组描述提到服务范围", "开通" in text, text[:300])
    check("命令组描述说明「每人各有会话 + 受邀参与」",
          "invite" in text and "attach" in text, text[:400])

    # ---- 普通用户：看不到 superuser 子命令，但看得到公开子命令 ----
    text_user = await help_of("/help aichat", uid=BOB, group_id=GROUP)
    check("普通用户看得到公开子命令", "/aichat new" in text_user, text_user[:300])
    # 只看**子命令清单**里的行 —— 命令组描述里会提到 add/del/targets（那是给
    # 超级用户看的提示），用整段文本判断会误报。
    # 取子命令**名**（`/aichat xxx` 的第一个 token），别用子串匹配 ——
    # 描述里出现「模型」之类的词会造成误报。
    names_user = {ln.strip().split()[1] for ln in text_user.splitlines()
                  if ln.strip().startswith("/aichat ") and len(ln.split()) > 1}
    check("普通用户看得到公开子命令", {"new", "attach", "invite"} <= names_user,
          f"{sorted(names_user)}")
    check("普通用户看不到仅私聊的管理命令",
          not ({"add", "del", "targets", "model", "config", "reload"} & names_user),
          f"{sorted(names_user)}")
    # 未开通的地址：整块命名空间都不出现
    h.client.groups.clear()
    saved_b = dict(h.plugin.cfg["binding"])
    try:
        h.plugin.cfg["binding"] = {"groups": [], "users": [], "empty_means": "deny",
                                   "notify_denied": False, "notify_cooldown": 0}
        t2 = await help_of("/help aichat", uid=BOB)
        check("未开通的地址看不到命名空间", "未知命令" in t2, t2[:120])
        t3 = await help_of("/help aichat", uid=SUPERUSER)
        check("超级用户在未开通时仍看得到（否则无从开通）",
              "📖 /aichat" in t3, t3[:120])
        check("超级用户的帮助里含开通命令",
              "/aichat add" in t3 and "/aichat targets" in t3, t3[:600])
    finally:
        h.plugin.cfg["binding"] = saved_b
        h.client.groups.bind_many([GROUP, OTHER_GROUP])
    check("普通用户看不到 model", "/aichat model" not in text_user, text_user[:400])

    # ---- 未开通的地址：帮助里整块命名空间都不出现 ----
    h.client.groups.clear()
    plugin = h.plugin
    saved = dict(plugin.cfg["binding"])
    try:
        plugin.cfg["binding"] = {"groups": [], "users": [ALICE],
                                 "empty_means": "deny",
                                 "notify_denied": True, "notify_cooldown": 0}
        text_blocked = await help_of("/help", uid=BOB, group_id=99999)
        check("未开通群的 /help 不列出 aichat", "/aichat" not in text_blocked,
              text_blocked[:300])
        check("未开通群问详情按「未知命令」处理",
              "未知命令" in await help_of("/help aichat", uid=BOB, group_id=99999),
              "应当隐藏")
        check("别的插件的命令不受影响（/ping）",
              "/ping" in await help_of("/help", uid=BOB),
              "不该被牵连")
        # 已授权私聊照旧
        check("已授权私聊仍看得到 aichat",
              "/aichat" in await help_of("/help", uid=ALICE), "应当可见")
    finally:
        plugin.cfg["binding"] = saved
        h.client.groups.bind_many([GROUP, OTHER_GROUP])


# ==================== 驱动 ====================

async def establish(h: Harness, *, user_id=ALICE, group_id=None):
    """建一个会话并把它设为焦点（会话不再自动创建，所以每个用例都要显式建）。

    `group_id` 给了就在那个群里建**群会话**（位置决定作用地），
    否则在私聊里建个人的私聊会话。
    """
    h.clear()
    if group_id is None:
        await h.private("/aichat new 预热会话", user_id=user_id, settle=1.0)
    else:
        await h.group("/aichat new 预热群会话", at_bot=True,
                      user_id=user_id, group_id=group_id, settle=1.0)
    try:
        await h.wait_for_replies(1, timeout=5.0)
    except TimeoutError:
        pass
    h.clear()
    return _current_session(h, user_id=user_id, group_id=group_id)


async def attach_to(h: Harness, sid: str, *, user_id=BOB, group_id=None):
    """让另一个人切到指定会话（会话已存在时用；不再有自动创建兜底）。"""
    if group_id is None:
        await h.private(f"/aichat attach {sid} --force=true", user_id=user_id,
                        settle=1.0)
    else:
        await h.group(f"/aichat attach {sid} --force=true", at_bot=True,
                      user_id=user_id, group_id=group_id, settle=1.0)
    try:
        await h.wait_for_replies(1, timeout=5.0)
    except TimeoutError:
        pass
    h.clear()


def _fake_private(user_id: int):
    _FakeEvent = plugin_module("main")._FakeEvent
    ChatTarget = plugin_module("store").ChatTarget
    return _FakeEvent(ChatTarget("private", user_id), {"user_id": user_id}, "")


def _current_session(h: Harness, *, user_id=ALICE, group_id=None):
    """这个人在这个位置（私聊 / 某个群）的当前会话。"""
    ChatTarget = plugin_module("store").ChatTarget
    target = (ChatTarget("group", group_id) if group_id is not None
              else ChatTarget("private", user_id))
    return h.plugin._current_session(target, user_id)


ROOT_TMP = ROOT / "tests" / ".test-tmp" / "ai-chat-runs"


async def run_case(fn, root: Path):
    """每个用例一个**全新实例**：独立 data_dir / 会话库 / 暂存区。

    共用实例会让用例互相污染（上一个用例建的会话、累计的回复条数都会影响
    下一个用例的断言），排查成本远高于多起一次客户端。
    """
    tmp = Path(tempfile.mkdtemp(prefix=fn.__name__ + "-", dir=str(root)))
    h = Harness(tmp)
    try:
        await h.start()
        await fn(h)
    except Exception as e:  # noqa: BLE001
        import traceback
        globals()["FAILED"] = globals()["FAILED"] + 1
        FAILURES.append(f"{fn.__name__}: {e!r}")
        print(f"  ❌ {fn.__name__} 抛异常: {e!r}")
        traceback.print_exc()
    finally:
        try:
            await h.stop()
        except Exception:
            pass
        shutil.rmtree(tmp, ignore_errors=True)


async def main():
    ROOT_TMP.mkdir(parents=True, exist_ok=True)
    # 纯单元用例（不依赖客户端）
    await run_case(test_compaction_split_safety, ROOT_TMP)
    await run_case(test_reasoning_unit, ROOT_TMP)
    await run_case(test_config_seed_unit, ROOT_TMP)
    await run_case(test_prompt_tables_and_formulas, ROOT_TMP)
    await run_case(test_data_dir_resolution, ROOT_TMP)
    await run_case(test_state_migration, ROOT_TMP)

    for fn in (
        test_private_flow, test_group_trigger, test_command_messages_not_dialogue,
        test_session_lifecycle, test_group_session_is_per_group,
        test_invite_flow, test_shared_context,
        test_solo_session_not_marked_shared, test_wrapping,
        test_tool_calling, test_plot_tool, test_fetch_image_tool,
        test_fetch_page_lists_images, test_last_round_answers_without_tools,
        test_tool_permission,
        test_role_command, test_role_permission, test_role_persistence,
        test_compaction,
        test_compaction_refuses_bad_summary,
        test_target_commands, test_session_access_guard, test_model_and_current,
        test_recall_removes_staged, test_detach_drops_staged, test_staging_persistence,
        test_staging_shared_across_windows, test_window_key_migration,
        test_submit_while_busy, test_terminator_waits_for_ingest,
        test_paragraph_labels, test_quote_uses_label,
        test_quote_chain_labels, test_quote_falls_back_to_text,
        test_multi_part_reply, test_reply_text_as_context,
        test_reply_media_submits_text,
        test_paging_long_reply, test_llm_error_surfaces,
        test_quote_chain, test_quote_ai_image, test_md_command,
        test_reply_index_survives_reload,
        test_delete_and_rename, test_schema_error_feedback,
        test_llm_retry, test_llm_timeout_hint, test_escaped_prefix_paste,
        test_sse_parse, test_llm_stream_cutoff, test_partial_reply_on_cutoff,
        test_llm_stream_wiring,
        test_think_command, test_maxtokens_command, test_timeout_command,
        test_plain_text_formatting,
        test_reasoning_dialect_via_config,
        test_config_template_seeding,
        test_no_deferred_relative_imports, test_no_coupling_with_other_plugins,
        test_binding_allowlist, test_binding_group_vs_private, test_binding_cooldown,
        test_private_binding_bootstrap, test_unbound_is_silent,
        test_unified_help,
    ):
        await run_case(fn, ROOT_TMP)

    print(f"\n{'=' * 50}\n通过 {PASSED} / 失败 {FAILED}")
    if FAILURES:
        print("失败项：")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAILED else 0


def _ev_at():
    from napcat.models import MessageEvent
    return MessageEvent(
        message_type="group", group_id=GROUP, user_id=ALICE,
        message=[{"type": "at", "data": {"qq": str(BOT_ID)}}])


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
