"""`/aichat history` 与 `/aichat hget` 的端到端测试 —— 真起 OneBot WS 服务端。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_history_command.py

打桩边界与本仓库其它端到端用例如出一辙：**不打桩框架**，事件以原始 OneBot JSON
从 WS 推进去，走完 `parse_event → EventBus → 插件 → ApiClient → WS` 全链路；
只把「外部 LLM」换成本文件里的桩，以及把 `history.chromium` 指向一个**自己写的
假浏览器**（保证 CI/无 Chromium 的机器上也能跑，且出图逻辑与真浏览器共用同一条
代码路径）。

真 Chromium 的渲染质量在 `test_history_render.py` 里单独验证。
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import struct
import sys
import tempfile
import zlib
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
CAROL = 29999          # 不在 binding.users 里（用来验「受邀例外」）
DAVE = 28888           # 不在名单里、也没被邀请
GROUP = 30001
TMP_ROOT = ROOT / "tests" / ".test-tmp" / "history-cmd"


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


def has_answer_line(text: str) -> bool:
    """清单里有没有「答」那一行。

    ⚠️ 不能直接搜 `"答：" in text`：页脚的「看某条的完整**问答**：/aichat hget …」
    里就含「答：」两个字，会把「没有回答行」误判成有（实测踩过）。
    只认行首（可带缩进）的 `答：` / `答:`。
    """
    return any(re.match(r"\s*答[:：]", line) for line in str(text).splitlines())


async def free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ==================== 假浏览器 ====================

def _png(w: int, h: int) -> bytes:
    """造一张真 PNG（纯色）—— 用来当假浏览器的产物。"""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    row = b"\x00" + b"\x33\x66\x99" * w
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(row * h))
            + chunk(b"IEND", b""))


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
        path = a.split("=", 1)[1]
        with open(path, "wb") as f:
            f.write(png(720, 300))
if "--dump-dom" in args:
    # 从 HTML 里量一个高度出来（真浏览器是 JS 量，这里是等价替身）
    src = ""
    for a in args:
        if a.startswith("file://"):
            with open(a[7:], encoding="utf-8") as f:
                src = f.read()
    n = len(re.sub(r"<[^>]+>", "", src))
    height = max(200, min(12000, 120 + n // 3))
    sys.stdout.write("<html><body><div id=\\"__size__\\">H=%d</div></body></html>" % height)
'''


def make_fake_chrome(path: Path) -> Path:
    path.write_text(FAKE_CHROME_SRC, encoding="utf-8")
    path.chmod(0o755)
    return path


# ==================== LLM 桩 ====================

class _StubResult:
    """复刻 `llm.ChatResult` 的**被用到的那部分**接口。

    ⚠️ `to_blocks()` 不是可选的：`_run_turn` 用它把回复落库（见
    `assistant_message(result.text, tool_calls=result.to_blocks()[len(...):])`），
    少了它就直接 AttributeError —— 表现为「问了没答、历史里什么都没有」。
    """

    def __init__(self, text="", tool_calls=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.usage: dict = {}
        self.model = "stub-model"
        self.finish_reason = "stop"
        self.raw: dict = {}

    @property
    def has_tool_calls(self):
        return bool(self.tool_calls)

    def to_blocks(self):
        from plugins.ai_chat.segments import Block
        blocks = []
        if self.text:
            blocks.append(Block.text_block(self.text))
        return blocks


class StubLlm:
    """最小 LLM 桩：每次回复一句可辨识的文本。

    ⚠️ 插件调的是 `async chat(messages, *, tools, max_tokens, reasoning)`，
    不是 `complete()` —— 桩的方法名/签名对不上就会走真 client（未配置 → 报错），
    症状是「历史里一对问答都没有」。
    """

    def __init__(self):
        self.calls = 0
        self.reply_text = "好的，我收到了。"
        #: 置为 True 时返回一段 Markdown（md 模式用例用）
        self.markdown_reply = ""
        self.requests: list[dict] = []
        self.summary_calls = 0
        self.reasoning_seen: list = []
        self.last_reasoning_params: dict = {}
        self.last_max_tokens = 0

    #: 摘要请求被识别时返回的内容（测试可以改它来模拟模型跑偏）
    summary_reply: str = ""

    async def chat(self, messages, *, tools=None, max_tokens=None, reasoning=None):
        self.calls += 1
        self.requests.append({"messages": messages, "tools": tools})
        self.reasoning_seen.append(reasoning)
        joined = " ".join(str(m.get("content", "")) for m in messages)
        if "目录编辑" in joined or "问答对" in joined:
            self.summary_calls += 1
            if self.summary_reply:
                return _StubResult(text=self.summary_reply)
            # 默认：按输入逐条给一段**与内容相关**的提问概括（取问题前 12 字）。
            # 内容相关是刻意的 —— 否则测试只能断言占位串，验证不了「摘要来自内容」。
            #
            # ⚠️ 只在「问答对开始…结束」之间找编号：系统提示里带一个
            # `1. 问：图书馆国庆开吗？` 的示例，扫描全文会把**示例**当成
            # 待摘要的数据，于是清单里出现与真实内容无关的摘要（实测踩过）。
            import re as _re
            body = joined
            if "提问开始" in joined:
                body = joined.split("提问开始", 1)[1].split("提问结束", 1)[0]
            # 现在的输入是 `编号. 提问`（不再带回答）
            pairs = _re.findall(r"^\s*(\d+)\. ([^\n]*)", body, _re.M)
            lines = []
            for sid, question in pairs:
                head = " ".join(str(question).split())[:12]
                lines.append(f"{sid}|问:{head}")
            return _StubResult(text="\n".join(lines))
        if self.markdown_reply:
            return _StubResult(text=self.markdown_reply)
        return _StubResult(text=f"{self.reply_text}（第 {self.calls} 次）")

    def build_payload(self, messages, **kwargs):
        return {"messages": messages}


# ==================== Harness ====================

class Harness:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.replies: list[dict] = []
        self.clients: list = []
        self.client: NapCatClient | None = None
        self.plugin = None
        self.llm = StubLlm()

    async def start(self):
        plugin_dir = self.tmp / "plugins"
        plugin_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PLUGIN_DIR, plugin_dir / "ai_chat", ignore=shutil.ignore_patterns("__pycache__", "data", "data.*", "tests",
                                      "docs", ".git", "README.md", "LICENSE",
                                      ".gitignore"))
        fake = make_fake_chrome(self.tmp / "fake-chrome")
        self.fake_chrome = fake
        cfg = {
            "ws_url": f"ws://127.0.0.1:{await free_port()}", "token": "",
            "plugin_dir": str(plugin_dir), "superusers": [SUPERUSER],
            "command_prefixes": ["/"], "bound_groups": [GROUP],
            "all_groups": False, "api_delay": 0.05,
            "reconnect_interval": 0.2, "watchdog_enabled": False,
            "restart_enabled": False,
            "plugins": {
                "ai_chat": {
                    "data_dir": str(self.tmp / "ai_chat_data"),
                    "llm": {"base_url": "http://stub/v1", "api_key": "sk-stub",
                            "model": "stub-model",
                            "reasoning": {"dialect": "effort", "default": "off",
                                          "per_model": {}, "params": {},
                                          "max_tokens_floor": 0,
                                          "compaction": "off"}},
                    "reply": {"quote": False},
                    "log_verbose": False,
                    "binding": {"groups": [GROUP], "users": [ALICE, BOB, SUPERUSER],
                                "notify_cooldown": 0},
                }
            },
        }
        cfg_path = self.tmp / "config.json"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        self.cfg_path = cfg_path

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
                    if action in ("send_private_msg", "send_group_msg", "send_msg"):
                        self.replies.append({"action": action, "params": params})
                    if echo is None:
                        continue
                    data = None
                    if action == "get_login_info":
                        data = {"user_id": BOT_ID, "nickname": "测试机器人"}
                    elif action == "get_msg":
                        data = None
                    await ws.send(json.dumps({"status": "ok", "retcode": 0,
                                              "echo": echo, "data": data}))
            except Exception:
                pass

        self.port = int(cfg["ws_url"].rsplit(":", 1)[1])
        self.server = await serve(handler, "127.0.0.1", self.port)
        self.client = NapCatClient(config_path=str(cfg_path))
        started = asyncio.Event()
        self.client.on_ws_connected(lambda: started.set())
        self.runner = asyncio.create_task(self.client.start())
        await asyncio.wait_for(started.wait(), timeout=15)
        self.plugin = self.client.plugins.plugins.get("ai_chat")
        if self.plugin is None:
            raise RuntimeError(f"ai_chat 插件没加载：{list(self.client.plugins.plugins)}")
        self._install_stub()
        self._install_search_stub()
        # 假浏览器：走的是与真浏览器完全相同的渲染代码路径
        self.plugin.config.set("history.chromium", str(fake))
        self.plugin.reload_config()
        self._install_stub()          # reload 会 reconfigure 回真 client，再装一次
        await asyncio.sleep(0.3)

    def _install_search_stub(self):
        """把搜索客户端换成假的（测试绝不联网）。

        直接换掉 `main` 模块里的 `WebClient` 名字，而不是往插件里塞一个实例 ——
        这样**插件自己的构造路径**（读配置、拼参数）仍然走真代码，被替换的只是
        「发 HTTP 请求」这一步。
        """
        # ⚠️ 必须打补丁在**插件自己加载的那份** main 上，不能写
        # `from plugins.ai_chat import main`：插件是以合成包名从临时副本加载的，
        # 仓库路径下的那份是**另一个模块对象** —— 补丁会打在没人用的副本上，
        # 于是测试静默走了真网络（实测：报「被搜索引擎限流」而不是用桩）。
        import sys as _sys
        main_mod = _sys.modules[type(self.plugin).__module__]
        self.search_queries: list[str] = []

        class _FakeWeb:
            def __init__(self, **kw):
                self.kw = kw
                # ⚠️ 不要在构造时缓存 hits/fail：用例是在**建好插件之后**才改这些
                # 开关的（如 h.search_fail = True），缓存住就读不到新值 ——
                # 表现为「设了失败却返回成功结果」。

            async def search(self, query, *, count=5):
                from plugins.ai_chat.search import SearchOutcome, SearchResult
                self_outer.search_queries.append(query)
                if self_outer.search_fail:
                    return SearchOutcome(error="搜索失败：连不上搜索引擎")
                hits = self_outer.search_hits[:max(1, int(count))]
                return SearchOutcome(results=[
                    SearchResult(title=t, url=u, snippet=sn) for t, u, sn in hits])

            async def read_page(self, url, *, limit=4000, max_images=0):
                # ⚠️ 签名要跟 `plugins/ai_chat/search.py:SearchClient.read_page` 对齐：
                # 插件当年给 read_page 加了 `max_images`（把页面图片直链附在正文后面，
                # 供模型调用 fetch_image），这个桩没跟着改 —— 于是 fetch_page 一调就
                # TypeError，三条断言红了很久而没人发现（工具失败被吞成一句人话）。
                self_outer.page_fetches.append(url)
                self_outer.page_max_images.append(int(max_images))
                if self_outer.page_fail:
                    return "读取失败：拒绝访问内网或本机地址（127.0.0.1）"
                return f"来源：{url}\n\n这是网页正文。关键词：图书馆国庆开放。"

            async def close(self):
                pass

        self_outer = self
        self.search_hits = [
            ("图书馆国庆开放安排 - 官网", "https://lib.example.edu/notice",
             "10月1日至3日闭馆，4日起自习区 7:00-23:00 开放。"),
            ("校历：国庆假期", "https://www.example.edu/calendar",
             "国庆假期为10月1日至10月7日。"),
        ]
        self.search_fail = False
        self.page_fail = False
        self.page_fetches: list[str] = []
        self.page_max_images: list[int] = []      # 每次抓页请求的图片直链上限
        self._real_web_client = main_mod.WebClient
        main_mod.WebClient = _FakeWeb

    def patch_search_config(self, **values) -> None:
        """改 `plugins.ai_chat.search` 里的键并重载（模拟运维手改配置）。"""
        import json as _json
        cfg_path = self.tmp / "config.json"
        raw = _json.loads(cfg_path.read_text(encoding="utf-8"))
        raw["plugins"]["ai_chat"].setdefault("search", {}).update(values)
        cfg_path.write_text(_json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        self.plugin.reload_config(from_disk=True)

    def _install_stub(self):
        """把桩挂到 llm 实例上。"""
        self.plugin.llm.chat = self.llm.chat
        self.plugin.llm.requests = self.llm.requests
        self.plugin.compactor.llm = self.plugin.llm

    async def stop(self):
        try:
            import sys as _sys
            main_mod = _sys.modules.get(type(self.plugin).__module__)
            if main_mod is None:
                raise RuntimeError("插件模块不在 sys.modules 里")
            if getattr(self, "_real_web_client", None) is not None:
                main_mod.WebClient = self._real_web_client
        except Exception:
            pass
        try:
            await self.client.stop()
        except Exception:
            pass
        self.runner.cancel()
        self.server.close()
        await self.server.wait_closed()

    # ---- 事件注入 ----

    async def _send_raw(self, payload: dict, settle: float = 0.4):
        await self.clients[-1].send(json.dumps(payload, ensure_ascii=False))
        await asyncio.sleep(settle)

    def _payload(self, kind, text, *, user_id, group_id=None, at_bot=False,
                 message_id=0, role="member"):
        segs = []
        if at_bot:
            segs += [{"type": "at", "data": {"qq": str(BOT_ID)}},
                     {"type": "text", "data": {"text": " "}}]
        segs.append({"type": "text", "data": {"text": text}})
        payload = {
            "post_type": "message", "message_type": kind, "sub_type": "normal",
            "message_id": message_id or (abs(hash(text)) % 100000 + 1),
            "user_id": user_id, "message": segs, "raw_message": text,
            "self_id": BOT_ID, "time": 1700000000,
            "sender": {"user_id": user_id, "nickname": f"用户{user_id}",
                       "card": "", "role": role},
        }
        if group_id is not None:
            payload["group_id"] = group_id
        return payload

    async def private(self, text, *, user_id=ALICE, settle=0.5, message_id=0):
        await self._send_raw(self._payload("private", text, user_id=user_id,
                                           message_id=message_id), settle)

    async def group(self, text, *, user_id=ALICE, group_id=GROUP, at_bot=True,
                    settle=0.5, message_id=0):
        await self._send_raw(self._payload("group", text, user_id=user_id,
                                           group_id=group_id, at_bot=at_bot,
                                           message_id=message_id), settle)

    def clear(self):
        """清空观测窗口。

        ⚠️ 新增计数器必须在这里一起清 —— 漏了 `summary_calls` 时，上一段用例
        留下的计数会让「本轮不该调模型」的断言假失败（实测：报「调了 1 次」，
        而那一次是上半段用例调的）。
        """
        self.replies.clear()
        self.llm.calls = 0
        self.llm.requests.clear()
        self.llm.summary_calls = 0
        self.llm.reasoning_seen.clear()
        if hasattr(self, "search_queries"):
            self.search_queries.clear()
            self.page_fetches.clear()

    def texts(self) -> list[str]:
        out = []
        for rec in self.replies:
            msg = rec["params"].get("message")
            if isinstance(msg, str):
                out.append(msg)
                continue
            out.append("".join(s.get("data", {}).get("text", "")
                               for s in (msg or [])
                               if s.get("type") == "text"))
        return out

    def image_segments(self) -> list[dict]:
        out = []
        for rec in self.replies:
            msg = rec["params"].get("message")
            for seg in (msg or []):
                if isinstance(seg, dict) and seg.get("type") == "image":
                    out.append(seg)
        return out

    def image_files(self) -> list[str]:
        paths = []
        for seg in self.image_segments():
            raw = str(seg.get("data", {}).get("file") or "")
            if raw.startswith("file://"):
                from urllib.parse import unquote, urlparse
                paths.append(unquote(urlparse(raw).path))
            elif raw:
                paths.append(raw)
        return paths

    async def wait_images(self, count: int, timeout=15.0):
        """等到累积出现 count 张图片段。

        ⚠️ 别用固定 sleep：渲染是外部进程、每张图是**独立一条** send，
        慢机器上「图还没发完就断言」会得到一个假的 0。
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.image_segments()) < count:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"等 {count} 张图超时（已有 {len(self.image_segments())} 张；"
                    f"文本={self.texts()}）")
            await asyncio.sleep(0.1)

    async def wait_images_or_text(self, needle: str, timeout=15.0):
        """等到某个文本出现（渲染失败这类没有图的情况用它）。"""
        deadline = asyncio.get_event_loop().time() + timeout
        while not any(needle in t for t in self.texts()):
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(f"等「{needle}」超时；文本={self.texts()}")
            await asyncio.sleep(0.1)

    async def wait_replies(self, count: int, timeout=8.0):
        deadline = asyncio.get_event_loop().time() + timeout
        while len(self.replies) < count:
            if asyncio.get_event_loop().time() >= deadline:
                raise TimeoutError(
                    f"等 {count} 条回复超时（已有 {len(self.replies)}：{self.texts()}）")
            await asyncio.sleep(0.05)


# ==================== 场景准备 ====================

async def _say(h: Harness, question: str, *, group=True, user_id=ALICE,
               settle=0.6):
    """让某个会话里真的产生一对问答（走完整链路：暂存 → 提交 → 落库）。"""
    if group:
        await h.group(question, user_id=user_id, settle=settle)
        await h.group("//", user_id=user_id, at_bot=True, settle=settle)
    else:
        await h.private(question, user_id=user_id, settle=settle)
        await h.private("//", user_id=user_id, settle=settle)


async def _new_session(h: Harness, name="历史测试", *, group=True, user_id=ALICE):
    if group:
        await h.group(f"/aichat new {name}", user_id=user_id, settle=1.0)
    else:
        await h.private(f"/aichat new {name}", user_id=user_id, settle=1.0)


# ==================== 用例 ====================

async def test_history_empty(h: Harness):
    print("\n[1] 空历史要说人话（不是空白）")
    await _new_session(h)
    h.clear()
    await h.group("/aichat history", settle=0.8)
    text = "\n".join(h.texts())
    check("提示还没有问答", "还没有可查看的问答" in text, text[:200])
    check("给出下一步（怎么产生问答）", "收尾" in text or "提问" in text, text[:200])
    check("不出图", not h.image_segments(), f"{len(h.image_segments())} 张")


async def test_history_needs_session(h: Harness):
    print("\n[2] 没有当前会话时给建会话提示（不自动创建）")
    h.clear()
    await h.group("/aichat history", settle=0.8)
    text = "\n".join(h.texts())
    check("提示先建会话", "new" in text, text[:200])


async def test_history_list(h: Harness):
    print("\n[3] history 列清单（纯文字，不出图）")
    await _new_session(h)
    for i in range(1, 4):
        await _say(h, f"第 {i} 个问题是什么")
    h.clear()
    await h.group("/aichat history", settle=1.0)
    text = "\n".join(h.texts())
    check("列出 3 对", "共 3 对问答" in text, text[:300])
    check("有编号 1", "1. " in text, text[:300])
    check("有编号 3", "3. " in text, text[:300])
    check("带提问人", f"用户{ALICE}" in text, text[:300])
    check("每条只有「问」一行（不给回答）",
          "问：" in text and not has_answer_line(text), text[:300])
    check("提示怎么看图", "hget" in text, text[:300])
    check("history 本身不出图", not h.image_segments(), f"{len(h.image_segments())} 张")
    check("history 没走主对话（只为摘要调了一次模型）",
          h.llm.calls == h.llm.summary_calls and h.llm.summary_calls == 1,
          f"calls={h.llm.calls} summary={h.llm.summary_calls}")


async def test_history_limit(h: Harness):
    print("\n[4] --limit 控制条数")
    await _new_session(h)
    for i in range(1, 5):
        await _say(h, f"限制测试问题 {i}")
    h.clear()
    await h.group("/aichat history --limit=2", settle=1.0)
    text = "\n".join(h.texts())
    check("只显示最近 2 对", "共 4 对问答" in text and "显示第 3-4 对" in text,
          text[:300])
    # 清单里是摘要（内容相关但已加工），所以断言摘要里的关键内容
    check("列出的是最后两对", "限制测试问题 3" in text and "限制测试问题 4" in text,
          text[:400])
    check("没有列出更早的", "限制测试问题 1" not in text, text[:400])


async def test_hget_single(h: Harness):
    print("\n[5] hget 单张：文字头 + 一张图")
    await _new_session(h)
    for i in range(1, 3):
        await _say(h, f"出图测试问题 {i}")
    h.clear()
    await h.group("/aichat hget 2", settle=1.0)
    await h.wait_images(1)
    texts = h.texts()
    imgs = h.image_files()
    check("先发文字头", texts and "第 2 对" in texts[0], f"{texts}")
    check("文字头带会话名", texts and "历史测试" in texts[0], f"{texts}")
    check("文字头带总数", texts and "共 2 对" in texts[0], f"{texts}")
    check("出 1 张图", len(imgs) == 1, f"{len(imgs)} 张")
    check("图真的存在", all(Path(p).exists() for p in imgs), f"{imgs}")
    check("是 PNG", all(Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
                        for p in imgs), f"{imgs}")
    check("图片段引用本地文件（不是外链）",
          all(s.get("data", {}).get("file", "").startswith("file://")
              for s in h.image_segments()), f"{h.image_segments()}")


async def test_hget_single_only(h: Harness):
    print("\n[6] hget 一次只取一条（区间被明确拒绝）")
    await _new_session(h)
    for i in range(1, 5):
        await _say(h, f"单条测试问题 {i}")

    # 区间 → 拒绝，并给出可操作的替代写法
    h.clear()
    await h.group("/aichat hget 2-3", settle=1.0)
    texts = "\n".join(h.texts())
    check("区间被拒绝", "一次只能取一条" in texts, texts[:200])
    check("给出可复制的替代写法", "hget 2" in texts, texts[:200])
    check("区间不出图", not h.image_files(), f"{h.image_files()}")

    # 单条 → 正常
    h.clear()
    await h.group("/aichat hget 3", settle=1.0)
    await h.wait_images(1)
    check("单条正常出 1 张", len(h.image_files()) == 1, f"{len(h.image_files())}")
    check("文字头写单条编号", "第 3 对" in h.texts()[0], f"{h.texts()[:1]}")

    # 越界 / 非数字
    h.clear()
    await h.group("/aichat hget 99", settle=1.0)
    texts = "\n".join(h.texts())
    check("越界给出总数", "没有第 99 对" in texts and "共 4 对" in texts, texts[:200])
    check("越界不出图", not h.image_files(), f"{h.image_files()}")

    h.clear()
    await h.group("/aichat hget abc", settle=1.0)
    texts = "\n".join(h.texts())
    check("非数字给出用法", "hget" in texts and "看不懂" in texts, texts[:200])


async def test_history_uses_model_summaries(h: Harness):
    print("\n[6.5] 清单里的问/答是模型写的短摘要，不是原文引用")
    await _new_session(h)
    await _say(h, "图书馆国庆期间到底开不开门，我想知道具体安排")
    await _say(h, "顺便问一下奖学金评审什么时候截止")
    h.clear()
    await h.group("/aichat history", settle=1.5)
    text = "\n".join(h.texts())
    check("调了模型写摘要", h.llm.summary_calls == 1,
          f"摘要调用 {h.llm.summary_calls} 次")
    check("清单用摘要（带加工痕迹）", "问:图书馆国庆期间" in text, text[:400])
    check("清单里没有回答那行", not has_answer_line(text), text[:400])
    # 原文是「…我想知道具体安排」，摘要只取前 12 字 —— 出现全文才说明没走摘要
    check("原文没被整段引用进来",
          "我想知道具体安排" not in text, text[:400])
    check("第二对也有摘要", "问:顺便问一下奖学金" in text, text[:400])
    check("摘要请求走压缩那一档思考强度（默认关）",
          h.llm.reasoning_seen and h.llm.reasoning_seen[-1] in (None, "off"),
          f"{h.llm.reasoning_seen[-1:]!r}")

    # 同一个清单再看一遍 → 命中缓存，不再调模型
    h.clear()
    await h.group("/aichat history", settle=1.2)
    text2 = "\n".join(h.texts())
    check("再列一次不再调模型（缓存命中）", h.llm.summary_calls == 0,
          f"又调了 {h.llm.summary_calls} 次（第一次已消耗 1 次）")
    check("缓存结果一致",
          "问:图书馆国庆期间" in text2 and "问:顺便问一下奖学金" in text2,
          text2[:300])


async def test_history_summary_fallback(h: Harness):
    print("\n[6.6] 摘要失败要退到截断并说明（不能假装是摘要）")
    await _new_session(h)
    await _say(h, "会不会退化成截断原文的测试问题")
    h.llm.summary_reply = "模型这次完全跑偏了，没有任何编号"
    h.clear()
    await h.group("/aichat history", settle=1.5)
    text = "\n".join(h.texts())
    check("仍然给出清单（没有整条失败）", "1. " in text, text[:300])
    check("退到截断原文", "会不会退化成截断" in text, text[:400])
    check("明确说明是截断而不是摘要", "截断原文" in text, text[:400])
    h.llm.summary_reply = ""
    # 缓存里已存了这次的截断结果；换个会话来验证「关掉摘要」路径
    h.clear()
    h.plugin.config.set("history.summarize", False)
    h.plugin.reload_config()
    await _new_session(h, name="无摘要会话")
    await _say(h, "关掉摘要之后也应该能列出历史")
    h.clear()
    await h.group("/aichat history", settle=1.2)
    text = "\n".join(h.texts())
    check("summarize=false 时列出截断文本", "关掉摘要之后" in text, text[:300])
    check("summarize=false 时不调模型", h.llm.summary_calls == 0,
          f"调了 {h.llm.summary_calls} 次")


async def test_commands_not_sent_to_ai(h: Harness):
    print("\n[7] 这两条是指令，不转给 AI")
    await _new_session(h)
    await _say(h, "先来一对问答")
    h.clear()
    await h.group("/aichat history", settle=1.2)
    # ⚠️ 这里**不能**再断言 calls==0：清单的摘要本身就要调模型（产品要求）。
    # 要卡的是「这条指令没有被当成对话送进主循环」—— 即只发生了摘要调用，
    # 没有多轮的主对话调用。
    check("history 只产生摘要调用，没走主对话",
          h.llm.calls == h.llm.summary_calls, f"calls={h.llm.calls}")
    h.clear()
    await h.group("/aichat hget 1", settle=2.0)
    check("hget 没调模型", h.llm.calls == 0, f"calls={h.llm.calls}")


async def test_history_survives_reload_and_aligns_with_get(h: Harness):
    print("\n[8] 编号在重载后仍与 get 对齐（同一套算法）")
    await _new_session(h)
    for i in range(1, 4):
        await _say(h, f"对齐测试问题 {i}")
    h.clear()
    await h.group("/aichat history", settle=1.0)
    listed = "\n".join(h.texts())
    # 重载一次（会重建运行时对象），编号不能变
    h.plugin.reload_config()
    await asyncio.sleep(0.3)
    # ⚠️ 两次 hget 之间**不能** clear：那会把第一张图也从观测窗口里抹掉
    h.clear()
    await h.group("/aichat hget 1", settle=1.0)
    await h.wait_images(1)
    await h.group("/aichat hget 2", settle=1.0)
    await h.wait_images(2)
    check("重载后仍能出图", len(h.image_files()) == 2, f"{len(h.image_files())} 张")
    check("编号 1..3 都在清单里",
          all(f"{i}. " in listed for i in (1, 2, 3)), listed[:300])


async def test_hget_render_failure_is_visible(h: Harness):
    print("\n[9] 渲染失败必须让用户看见（不能静默）")
    await _new_session(h)
    await _say(h, "会渲染失败的一对")
    # 把浏览器换成一个一定失败的
    bad = h.tmp / "bad-chrome"
    bad.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bad.chmod(0o755)
    h.plugin.config.set("history.chromium", str(bad))
    h.plugin.reload_config()
    h.clear()
    await h.group("/aichat hget 1", settle=1.0)
    await h.wait_images_or_text("渲染")
    texts = "\n".join(h.texts())
    check("用户看到渲染失败", "渲染" in texts and ("失败" in texts or "没有产出" in texts),
          texts[:300])
    check("不抛穿成空白", bool(texts), f"{texts}")


async def test_config_disabled(h: Harness):
    print("\n[10] history.enabled=false 时明确回绝")
    await _new_session(h)
    await _say(h, "关掉出图功能前的一对")
    h.plugin.config.set("history.enabled", False)
    h.plugin.reload_config()
    h.clear()
    await h.group("/aichat hget 1", settle=1.5)
    texts = "\n".join(h.texts())
    check("提示功能已关闭", "关闭" in texts, texts[:200])
    check("关闭时不出图", not h.image_files(), f"{h.image_files()}")


async def test_web_search_tools_registered(h: Harness):
    print("\n[12] 联网搜索工具已注册且可用")
    names = set(h.plugin.tools.names())
    check("注册了 web_search", "web_search" in names, f"{sorted(names)}")
    check("注册了 fetch_page", "fetch_page" in names, f"{sorted(names)}")

    await _new_session(h, name="搜索测试")
    h.clear()
    from plugins.ai_chat.store import ChatTarget
    session = h.plugin.store.get("s0001")
    ctx = h.plugin._tool_context(ChatTarget("group", GROUP), session,
                                 {"user_id": ALICE, "nickname": "甲"})
    out, err = await h.plugin.tools.call("web_search",
                                         {"query": "图书馆国庆", "count": 2}, ctx)
    check("搜索桩被真的调用了", h.search_queries == ["图书馆国庆"],
          f"{h.search_queries}")
    check("结果回喂给模型（带标题与链接）",
          "图书馆国庆开放安排" in str(out) and "链接" in str(out), str(out)[:160])
    check("结果不含错误标记", not err, str(err))

    h.clear()
    out, err = await h.plugin.tools.call("fetch_page",
                                        {"url": "https://lib.example.edu/notice"}, ctx)
    check("fetch_page 抓取并回喂正文", "网页正文" in str(out), str(out)[:120])
    check("抓取地址被记录", h.page_fetches == ["https://lib.example.edu/notice"],
          f"{h.page_fetches}")
    check("抓页时请求了页面图片直链（max_images>0）",
          bool(h.page_max_images) and h.page_max_images[-1] > 0, str(h.page_max_images))


async def test_web_search_failure_is_visible(h: Harness):
    print("\n[13] 搜索失败要能被告知（不是假装成功）")
    from plugins.ai_chat.store import ChatTarget
    await _new_session(h, name="搜索失败")
    h.search_fail = True
    session = h.plugin.store.get("s0001")
    ctx = h.plugin._tool_context(ChatTarget("group", GROUP), session,
                                 {"user_id": ALICE, "nickname": "甲"})
    out, err = await h.plugin.tools.call("web_search", {"query": "x"}, ctx)
    check("失败时结果里带失败说明", "失败" in str(out), str(out)[:120])
    h.search_fail = False

    h.page_fail = True
    out, err = await h.plugin.tools.call("fetch_page", {"url": "http://127.0.0.1/x"}, ctx)
    check("抓内网地址被拒并说明原因", "失败" in str(out) and "内网" in str(out),
          str(out)[:120])
    h.page_fail = False


async def test_search_disabled_config(h: Harness):
    print("\n[14] search.enabled=false 时不注册搜索工具，也不提供客户端")
    h.plugin.config.set("search.enabled", False)
    h.plugin.reload_config()
    await h.plugin.on_disable()
    await h.plugin.on_enable()
    names = set(h.plugin.tools.names())
    check("关掉后没有 web_search", "web_search" not in names, f"{sorted(names)}")
    from plugins.ai_chat.store import ChatTarget
    session = h.plugin.store.list(limit=1)
    if session:
        ctx = h.plugin._tool_context(ChatTarget("group", GROUP), session[0],
                                     {"user_id": ALICE})
        check("上下文里的 web 客户端为 None", ctx.extras.get("web") is None,
              f"{ctx.extras.get('web')!r}")
    h.plugin.config.set("search.enabled", True)
    h.plugin.reload_config()
    await h.plugin.on_disable()
    await h.plugin.on_enable()


async def test_markdown_mode_sends_image(h: Harness):
    print("\n[15] //md 触发 Markdown 模式：回答渲染成图片")
    await _new_session(h, name="MD会话")
    h.llm.markdown_reply = "# 国庆安排\n\n图书馆**10月1日至3日闭馆**。\n\n- 自习区 7:00-23:00\n"
    h.clear()
    await h.group("国庆图书馆开吗//md", settle=1.0)
    await h.wait_images(1)
    check("收到 1 张图", len(h.image_files()) == 1, f"{len(h.image_files())} 张")
    imgs = h.image_files()
    check("图真的存在且是 PNG",
          all(Path(p).exists() and Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
              for p in imgs), f"{imgs}")
    check("没有把 Markdown 原文当文本发出来",
          not any("# 国庆安排" in t for t in h.texts()), f"{h.texts()}")
    # 系统提示必须切到 Markdown 档
    prompts = [str(r["messages"][0].get("content", "")) for r in h.llm.requests]
    check("系统提示切到 Markdown 档（整篇可用 Markdown）",
          any("Markdown 模式" in p and "整篇" in p for p in prompts),
          f"{prompts[-1][:200] if prompts else ''}")
    check("Markdown 档里「其它 Markdown 不要用」被显式豁免",
          any("限制本轮不适用" in p for p in prompts),
          f"{prompts[-1][:260] if prompts else ''}")
    check("系统提示不再自相矛盾地说分段",
          not any("分成多条" in p for p in prompts), "还留着分段说明")
    h.llm.markdown_reply = ""


async def test_plain_mode_not_markdown(h: Harness):
    print("\n[16] 普通 `//` 提交：纯文字仍走文字（不整段出图）")
    await _new_session(h, name="纯文本会话")
    # 注意：普通模式下**可渲染块**（表格/公式/列表/`###`）也会出图，这是有意的；
    # 这里验的是「整段没有可渲染块时，就老老实实发文字」。
    h.llm.markdown_reply = "这一段是普通回答，没有任何 Markdown 结构。\n"
    h.clear()
    await h.group("普通提交//", settle=1.0)
    check("不发图", not h.image_segments(), f"{len(h.image_segments())} 张")
    check("原文当文本发出（未渲染）",
          any("这一段是普通回答" in t for t in h.texts()), f"{h.texts()}")
    prompts = [str(r["messages"][0].get("content", "")) for r in h.llm.requests]
    check("系统提示没有 Markdown 档",
          not any("Markdown 模式" in p for p in prompts), "提示串档了")
    h.llm.markdown_reply = ""


async def test_markdown_render_failure_falls_back(h: Harness):
    print("\n[17] Markdown 出图失败要退回纯文本（内容不能丢）")
    await _new_session(h, name="MD失败")
    bad = h.tmp / "bad-chrome2"
    bad.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bad.chmod(0o755)
    h.plugin.config.set("history.chromium", str(bad))
    h.plugin.reload_config()
    h.llm.markdown_reply = "# 标题\n\n正文内容不能被丢掉。"
    h.clear()
    await h.group("给我看看//md", settle=1.5)
    await h.wait_images_or_text("失败")
    texts = "\n".join(h.texts())
    check("明确说明出图失败", "失败" in texts, texts[:200])
    check("原文仍然发出（没丢内容）", "正文内容不能被丢掉" in texts or "标题" in texts,
          texts[:300])
    h.llm.markdown_reply = ""


async def test_history_lists_question_summary_only(h: Harness):
    print("\n[18] 清单只给提问摘要（回答完全不进清单）")
    await _new_session(h, name="摘要会话")
    await _say(h, "图书馆国庆期间到底开不开门")
    h.clear()
    await h.group("/aichat history", settle=1.5)
    text = "\n".join(h.texts())
    check("问那行是模型概括", "问:图书馆国庆期间" in text, text[:400])
    check("清单里没有「答」那行", not has_answer_line(text), text[:400])
    check("回答的正文没泄漏进清单", "好的，我收到了" not in text, text[:400])
    check("给出看全文的入口", "hget" in text, text[:400])


async def test_rich_blocks_render_inline(h: Harness):
    print("\n[19] 文本模式下表格/公式出图并**插回消息体**")
    await _new_session(h, name="富文本会话")
    # 模型在**普通（非 md）模式**下吐出表格 + LaTeX —— 真实会话里就是这样
    h.llm.markdown_reply = (
        "先说结论。\n\n"
        "| 日期 | 状态 |\n| --- | --- |\n| 10-01 | 闭馆 |\n| 10-04 | 开放 |\n\n"
        "推导：\n\n" + r"$$T(n) = 2T(n/2) + O(n)$$" + "\n\n就这些。")
    h.clear()
    await h.group("给我讲讲//", settle=1.5)
    await h.wait_images(2)
    texts = h.texts()
    imgs = h.image_files()
    check("出了 2 张图（表格 + 公式）", len(imgs) == 2, f"{len(imgs)}")
    check("图真的存在且是 PNG",
          all(Path(p).exists() and Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
              for p in imgs), f"{imgs}")
    joined = "\n".join(texts)
    check("前后文字仍在（没被图挤掉）",
          "先说结论" in joined and "就这些" in joined, f"{texts}")
    check("表格 Markdown 没当纯文本发出去",
          "| --- |" not in joined and "| 日期 |" not in joined, f"{texts}")
    check("LaTeX 没当纯文本发出去",
          "$$" not in joined and "T(n) = 2T(n/2)" not in joined, f"{texts}")
    # 顺序：文字 → 表格图 → 文字 → 公式图 → 文字
    kinds: list[str] = []
    for rec in h.replies:
        msg = rec["params"].get("message")
        for seg in (msg or []):
            if isinstance(seg, dict):
                kind = "img" if seg.get("type") == "image" else "txt"
                if not kinds or kinds[-1] != kind:
                    kinds.append(kind)
    check("图文顺序为 文-图-文-图-文",
          kinds == ["txt", "img", "txt", "img", "txt"], f"{kinds}")
    h.llm.markdown_reply = ""


async def test_no_rich_blocks_stays_text(h: Harness):
    print("\n[20] 没有表格/公式时仍是普通纯文本（不出多余图）")
    await _new_session(h, name="纯文本会话2")
    h.llm.markdown_reply = "就是一句普通回答，没有任何表格或公式。"
    h.clear()
    await h.group("随便说说//", settle=1.2)
    check("不出图", not h.image_segments(), f"{len(h.image_segments())} 张")
    check("文本正常发出",
          any("普通回答" in t for t in h.texts()), f"{h.texts()}")
    h.llm.markdown_reply = ""


async def test_rich_render_failure_falls_back_to_text(h: Harness):
    print("\n[21] 表格/公式出图失败 → 退回纯文本（内容不能丢）")
    await _new_session(h, name="富文本失败")
    bad = h.tmp / "bad-chrome3"
    bad.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bad.chmod(0o755)
    h.plugin.config.set("history.chromium", str(bad))
    h.plugin.reload_config()
    h.llm.markdown_reply = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n表格在上面。"
    h.clear()
    await h.group("来张表//", settle=1.5)
    texts = "\n".join(h.texts())
    check("退回纯文本（表格 Markdown 原文仍在）",
          "| A | B |" in texts or "| --- |" in texts, f"{texts[:200]}")
    check("正文说明仍在", "表格在上面" in texts, f"{texts[:200]}")
    h.llm.markdown_reply = ""


async def test_inline_math_in_text_run(h: Harness):
    print("\n[22] 正文里的行内公式也出图（不是只发表格）")
    await _new_session(h, name="行内公式")
    # 截图里的形态：表格 + 正文段落里带 $...$
    # 公式放在**普通段落**里：放在列表项里的话，整段列表会作为一块出图，
    # 公式就不再单独出图了（那是对的行为，见 [25]）。
    h.llm.markdown_reply = (
        "要点：常数函数通常并入幂函数（$x^{0}$），所以有的教材说「五大类」。\n"
        "比如 $y=e^{\\sin x}\\cdot\\ln(1+x)$ 也是初等函数。\n")
    h.clear()
    await h.group("讲讲初等函数//", settle=1.5)
    await h.wait_images(2)
    texts = "\n".join(h.texts())
    imgs = h.image_files()
    check("两个行内公式各出一张图", len(imgs) == 2, f"{len(imgs)}")
    check("LaTeX 源码没当文字发出去",
          "$x^{0}$" not in texts and "$y=e^{" not in texts, f"{texts[:200]}")
    check("前后文字仍在",
          "常数函数通常并入幂函数" in texts and "也是初等函数" in texts,
          f"{texts[:200]}")

    # 列表项里的公式：随整段列表出图（一块，而不是每行公式各出一张）
    h.clear()
    h.llm.markdown_reply = ("要点：\n"
                           "- 并入幂函数（$x^{0}$），有的教材说「五大类」\n"
                           "- 比如 $y=e^{x}$ 也是初等函数\n")
    await h.group("再讲讲//", settle=1.5)
    await h.wait_images(1)
    check("列表里的公式随列表出一张图", len(h.image_files()) == 1,
          f"{len(h.image_files())} 张")
    h.llm.markdown_reply = ""


async def test_inline_math_can_be_disabled(h: Harness):
    print("\n[23] inline_math=false 时行内公式原样发文字（可退回）")
    await _new_session(h, name="关行内公式")
    h.plugin.config.set("reply.inline_math", False)
    h.plugin.reload_config()
    h.llm.markdown_reply = "要点：（$x^{0}$）这里有个公式。"
    h.clear()
    await h.group("讲解一下//", settle=1.2)
    check("不出图", not h.image_segments(), f"{len(h.image_segments())} 张")
    check("公式原样留在文字里",
          "$x^{0}$" in "\n".join(h.texts()), f"{h.texts()}")
    h.plugin.config.set("reply.inline_math", True)
    h.plugin.reload_config()
    h.llm.markdown_reply = ""


async def test_search_api_migration(h: Harness):
    print("\n[24] 老部署自动接入 AnySearch（不覆盖用户配置）")
    from plugins.ai_chat.search import ANYSEARCH_URL
    p = h.plugin

    # ① 老部署形态：磁盘上三个键都是空串 → 应当被自动切到 AnySearch
    h.patch_search_config(api_url="", api_key="", api_tag="")
    check("老部署被迁移", p.adopt_search_api_default() is True, "没有迁移")
    p.reload_config(from_disk=True)
    check("内存里已是 AnySearch", p.cfg["search"]["api_url"] == ANYSEARCH_URL,
          p.cfg["search"]["api_url"])
    on_disk = json.loads((h.tmp / "config.json").read_text(encoding="utf-8"))
    check("落盘也是 AnySearch（重启不丢）",
          on_disk["plugins"]["ai_chat"]["search"]["api_url"] == ANYSEARCH_URL,
          on_disk["plugins"]["ai_chat"]["search"]["api_url"])

    # ② 用户自己配了服务 → 一个字都不能动
    h.patch_search_config(api_url="https://my.search/api")
    check("用户已配置时不迁移", p.adopt_search_api_default() is False, "不该迁移")
    p.reload_config(from_disk=True)
    check("用户配置原样保留", p.cfg["search"]["api_url"] == "https://my.search/api",
          p.cfg["search"]["api_url"])

    # ③ 只填了 key（说明他在用某个服务）→ 也不动
    h.patch_search_config(api_url="", api_key="tvly-xxx", api_tag="")
    check("只填 key 时不迁移", p.adopt_search_api_default() is False, "不该迁移")

    # 恢复默认，免得影响后续用例
    h.patch_search_config(api_url="", api_key="", api_tag="")
    p.adopt_search_api_default()


async def test_headings_and_lists_render(h: Harness):
    print("\n[25] 标题/列表/代码围栏也出图（不再原样发 ### 与 - ）")
    await _new_session(h, name="多块会话")
    h.llm.markdown_reply = (
        "先说明：\n\n"
        "### MiniMax 大模型迭代时间线\n\n"
        "| 时间 | 模型 |\n| --- | --- |\n| 2023 | abab5 |\n| 2025 | Text-01 |\n\n"
        "另外公司层面：\n"
        "- 成立于 2021 年 12 月，总部上海\n"
        "- 旗下产品：海螺 AI、星野\n\n"
        "```python\nprint(\"hello\")\n```\n")
    h.clear()
    await h.group("讲讲 MiniMax//", settle=1.5)
    await h.wait_images(4)          # 标题 + 表格 + 列表 + 代码
    texts = "\n".join(h.texts())
    imgs = h.image_files()
    check("四类块各出一张图", len(imgs) == 4, f"{len(imgs)} 张")
    check("图都真的存在且是 PNG",
          all(Path(p).exists() and Path(p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
              for p in imgs), f"{imgs}")
    # ⚠️ 回归：这些符号曾经原样发到聊天里（截图里的 `### MiniMax…`）
    check("没有残留 ### 标题符号", "###" not in texts, f"{texts[:200]}")
    check("没有残留列表符号", "\n- 成立于" not in texts, f"{texts[:200]}")
    check("没有残留代码围栏", "```" not in texts, f"{texts[:200]}")
    check("正文说明仍在", "先说明" in texts and "另外公司层面" in texts, f"{texts[:200]}")

    # 顺序：文字 → 标题 → 表格 → 文字 → 列表 → 代码
    # ⚠️ 要收集**完整**序列，不能只记「类型变化」—— 连续两张图会被折叠成一条，
    # 断言就变成 ['txt','img','txt','img'] 看着像顺序错了（实测踩过）。
    kinds: list[str] = []
    for rec in h.replies:
        for seg in (rec["params"].get("message") or []):
            if isinstance(seg, dict):
                kinds.append("img" if seg.get("type") == "image" else "txt")
    check("图文顺序保序", kinds == ["txt", "img", "img", "txt", "img", "img"],
          f"{kinds}")
    h.llm.markdown_reply = ""


async def test_invited_guest_can_use_session(h: Harness):
    print("\n[26] 受邀的未绑定用户能在**私聊**里用该会话")
    # ALICE 建会话并邀请未绑定的 CAROL
    await h.private("/aichat new 邀请测试", user_id=ALICE, settle=1.0)
    h.clear()
    await h.private(f"/aichat invite s0001 {CAROL}", user_id=ALICE, settle=1.0)
    check("邀请成功", "已邀请" in (h.texts() or [""])[0], f"{h.texts()}")

    h.clear()
    await h.private("/aichat attach s0001 --force=true", user_id=CAROL, settle=1.2)
    text = (h.texts() or [""])[0]
    check("受邀者能自己 attach（否则例外永远走不到）",
          "已切到" in text, f"{text[:80]}")

    h.clear()
    await h.private("你好我是被邀请的", user_id=CAROL, settle=0.8)
    await h.private("//", user_id=CAROL, settle=1.5)
    check("受邀者能正常对话（调了模型）", h.llm.calls >= 1, f"calls={h.llm.calls}")
    check("受邀者收到回复", bool(h.texts()), f"{h.texts()}")

    # 未受邀的未绑定用户仍然完全静默
    h.clear()
    await h.private("我没被邀请", user_id=DAVE, settle=0.8)
    await h.private("//", user_id=DAVE, settle=1.2)
    check("未受邀者不调模型", h.llm.calls == 0, f"calls={h.llm.calls}")
    check("未受邀者静默（不给任何提示）", not h.texts(), f"{h.texts()}")

    # 取消邀请 → 立刻失效
    h.clear()
    await h.private(f"/aichat uninvite s0001 {CAROL}", user_id=ALICE, settle=1.0)
    await h.private("取消后我还能说吗", user_id=CAROL, settle=0.8)
    await h.private("//", user_id=CAROL, settle=1.2)
    check("取消邀请后不再服务", h.llm.calls == 0, f"calls={h.llm.calls}")


async def test_guest_exception_is_private_only(h: Harness):
    print("\n[27] 群里**没有**受邀例外（仍按群绑定判）")
    other_group = 39998                      # 未开通的群
    # 群里建会话并邀请未绑定的 CAROL（invite 本身在群里可用）
    await h.group("/aichat new 群会话", user_id=ALICE, group_id=GROUP, settle=1.0)
    h.clear()
    await h.group(f"/aichat invite s0001 {CAROL}", user_id=ALICE,
                  group_id=GROUP, settle=1.0)
    check("群里可以邀请（命令本身不限于私聊）",
          "已邀请" in (h.texts() or [""])[0], f"{h.texts()}")

    from plugins.ai_chat.store import ChatTarget
    p = h.plugin
    bound = ChatTarget("group", GROUP)
    unbound = ChatTarget("group", other_group)
    check("已开通群：受邀者按群绑定放行（群开通=全群可用）",
          p.check_access(bound, CAROL)[0] is True, f"{p.check_access(bound, CAROL)}")
    ok, why = p.check_access(unbound, CAROL)
    check("未开通群：受邀也**不**放行", ok is False, f"{ok} {why}")
    check("未被开通的群里不认受邀身份",
          p._is_invited_guest(unbound, CAROL) is False, "群聊里不该有受邀例外")

    h.clear()
    await h.group("/aichat attach s0001 --force=true", user_id=CAROL,
                  group_id=other_group, settle=1.2)
    check("未开通群里连 attach 都被挡住", not h.texts(), f"{h.texts()}")


async def test_scope_respected(h: Harness):
    print("\n[11] 未开通服务的群里，这两条也按同一口径回绝")
    # 用一个没开通的群（不在 binding.groups 里）
    other = 39999
    h.clear()
    await h.group("/aichat history", group_id=other, settle=1.0)
    texts = "\n".join(h.texts())
    check("未开通群不响应（或明确回绝）",
          texts == "" or "开通" in texts or "未" in texts, f"{texts!r}")


async def run_case(fn):
    tmp = Path(tempfile.mkdtemp(prefix=fn.__name__ + "-", dir=str(TMP_ROOT)))
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
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    for fn in (test_history_empty, test_history_needs_session,
               test_history_list, test_history_limit, test_hget_single,
               test_hget_single_only, test_history_uses_model_summaries,
               test_history_summary_fallback, test_commands_not_sent_to_ai,
               test_history_survives_reload_and_aligns_with_get,
               test_hget_render_failure_is_visible, test_config_disabled,
               test_web_search_tools_registered, test_web_search_failure_is_visible,
               test_search_disabled_config, test_markdown_mode_sends_image,
               test_plain_mode_not_markdown, test_markdown_render_failure_falls_back,
               test_history_lists_question_summary_only,
               test_rich_blocks_render_inline, test_no_rich_blocks_stays_text,
               test_rich_render_failure_falls_back_to_text,
               test_inline_math_in_text_run, test_inline_math_can_be_disabled,
               test_search_api_migration, test_headings_and_lists_render,
               test_invited_guest_can_use_session,
               test_guest_exception_is_private_only,
               test_scope_respected):
        await run_case(fn)
    print(f"\n{'=' * 50}\n通过 {PASSED} / 失败 {FAILED}")
    if FAILURES:
        print("失败项：")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
