"""``ai_chat`` —— 把 DSH 的对话语义搬进 QQ。

功能
----
* **多会话**：可新建/命名/切换/解附。会话归属只有「谁建的」（某个人 / 某个群），
  不绑定聊天地址；**群里每个人各有自己的当前会话**（焦点 = 聊天地址 + 人）。
  别人要参与你的会话，需你 `/aichat invite`、再由对方自己 `attach`；
  同一个会话在群聊与私聊里都能用。
* **服务范围**：只对 `config.json → binding` 里开通的群/私聊提供服务，
  与权限等级无关（超级用户在未开通处也用不了）。「绑定」指的是**能不能用这项功能**，
  与会话无关。
* **自然对话**（不走命令形式）：私聊直接说话即可；群聊 @机器人 即为一条输入。
  **结束符 `#` 的含义是「这一大段提交上去」**：输入可能很长、超过单条消息的
  长度上限，用户就分几条发 —— 每一条先落在暂存区，最后发一个结束符把它们
  **按顺序拼成一整段**交给模型（直接按换行拼接，不加 `(1/N)` 之类的序号：
  分段只是客户端的限制，对模型来说它本来就是一段话）。
* **DSH 式消息包装**：历史不是纯字符串，而是 `{role, blocks[], metadata}`；
  只有人类真实输入才加 `[说话人 · 时间]` 前缀，摘要与工具结果不加
  （否则模型会把摘要当成某个人刚说的话）。
* **上下文压缩**（自主设定长度上限）：估算 token 压力到 `context_limit ×
  threshold_ratio` 时，把较老的一段交给模型压成「接手摘要」，保留最近
  `retain_ratio` 的原文；切点只落在 `tool_calls`/`tool` 配对安全的位置。
* **内置工具**：function calling（时间、历史检索、群成员/群信息、禁言、
  校内通知、天气）。工具执行前后都做权限校验。

命令
----
帮助统一走框架的 `/help`（`/help aichat` 看子命令、`/help aichat.new` 看参数），
本插件**不再自绘**一份帮助 —— 两套渲染迟早口径不一致。

全域（任何人都能用）：`/aichat new|list|switch|attach|detach|current|rename|
delete|compact|stats|think`。
超级用户且仅私聊：`/aichat targets|add|del|model|config|reload`。
结束符：行尾符号（默认 `//`，可配 `trigger.terminators`）。

配置
----
见 `config.json` → `plugins.ai_chat`，字段与默认值见 `DEFAULT_CONFIG`。
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from napcat import (
    ApiError, CommandExecutionError, GROUP_AT_HINT, GroupRecallEvent,
    MessageChain, MessageEvent, NapCatError, PluginBase, PluginRuntimeError,
    short_reason,
)

from .compaction import (
    CompactionPolicy, Compactor, DshWrapping, Usage,
    WrappingConfig, build_system_prompt, sanitize_history, tool_guidance,
)
from .custom_tools import register_config_tools
from .llm import LlmClient, err_text
from .reasoning import (
    DIALECTS, LEVELS, VALIDATION_HINT, ReasoningPolicy, ReasoningSpec,
    override_text, spec_from_text,
)
from .history import (SUMMARY_SYSTEM_PROMPT, SummaryCache, extract_exchanges,
                      fallback_summary, has_compaction, parse_question_summaries,
                      parse_range, reject_range, render_list, summarize_context,
                      summary_key)
from .markdown import (has_inline_math, split_inline_math,
                       split_rich_segments)
from .plot import PlotService, extract_markers, strip_markers
from .render import HistoryRenderer, RenderError, RenderOptions
from .search import ANYSEARCH_URL, WebClient
from .segments import (ROLE_ASSISTANT, Message, assistant_message, tool_message,
                       user_message)
from .store import (
    ChatTarget, OWNER_GROUP, OWNER_USER,
    SentReplyIndex, Session, SessionStore,
)
from .tools import (ToolContext, ToolRegistry, register_builtin_tools,
                    register_image_tool, register_plot_tool, register_web_tools)
from .webimg import WebImageService
from . import schemas as S

# ==================== 默认配置 ====================

DEFAULT_CONFIG: dict = {
    "enabled": True,
    # 数据目录（会话历史/绑定表/暂存区）。留空 = 插件包内 data/。
    # 装到只读目录时**必须**显式指定到可写位置。
    "data_dir": "",
    # ---- LLM（使用指定 API 地址与 key）----
    "llm": {
        "base_url": "",
        "api_key": "",
        "model": "",
        "max_tokens": 4096,
        "temperature": None,      # None = 不发送该字段（推理模型友好）
        # 单次调用的**总时长**上限（秒）。流式下它是兜底：到点就把已经生成的部分
        # 交出来（见 llm._chat_stream），不是「取消这次回答」。
        "timeout": 270,
        # SSE 流式请求。开着才能区分「模型在想」和「上游卡死」：
        # 非流式下两者都是「一个字节都没回来」，只能一刀切总超时。
        # 端点不认 `stream` 时会自动退回普通 JSON 解析，所以默认开。
        "stream": True,
        # 流式下**多久没有新数据**就判定卡死（秒）—— 这才是真正的掐断判据。
        # 只要 token 还在流就一直等（直到 timeout），所以可以给得比较小。
        "stall_timeout": 60,
        "extra_headers": {},
        # 失败自动重试：网络抖动、上游 5xx、限流、回了看不懂的东西 —— 都值得再试一次。
        # ⚠️ **4xx 不重试**：参数写错、key 失效，再试一百次也是同一个结果，
        # 只会让用户多等几秒才看到错误（判据见 main._is_retryable）。
        "max_retries": 2,          # 失败后最多**再试**几次（0 = 不重试）
        "retry_backoff": 1.5,      # 首次退避多少秒，之后翻倍
        "retry_max_wait": 8.0,     # 单次等待上限（秒）
        "retry_notice": True,      # 第一次重试时告诉用户一句（否则干等像没反应）
        # 思考强度：见 llm.reasoning.* （旧写法 llm.reasoning_effort 仍兼容）
        "reasoning": {
            "dialect": "effort",   # effort | thinking | enable_thinking | none | raw
            "default": "off",      # off = 不发思考参数；级别 / 预算 / thinking:on
            "per_model": {},       # {"deepseek-v4-flash": "off", "o4-mini": "high"}
            "params": {},          # dialect=raw 时原样并入 payload
            "max_tokens_floor": 0,  # >0：开思考时把 max_tokens 抬到这个下界
            "compaction": "off",   # 压缩摘要单独一档（摘要要短，思考是浪费）
        },
    },
    # ---- 网图配图（工具 fetch_image）----
    # 让模型能把网上的图片**配在回复里**发给用户（QQ 直接显示成图片）。
    # 依赖 `search.enabled`（下载复用搜索那套 HTTP 客户端与 SSRF 闸）。
    "web_image": {
        "enabled": True,
        "max_bytes": 8 * 1024 * 1024,   # 单张上限（超过就拒绝，不发半张图）
        "max_per_turn": 4,              # 一轮最多配几张（与 reply.chunk_max_images 对齐）
        "keep_files": 60,               # 落盘目录只留最近多少个文件
        "page_images": 6,               # fetch_page 顺带列出几条页面图片直链
    },
    # ---- 服务范围（白名单）----
    # 只对**列在这里**的群/私聊提供服务，其余一律回绝。
    # 与权限等级无关：超级用户在未绑定的群里也用不了。
    # 注意这与顶层的 bound_groups 是两件事 —— 那个控制「机器人监听哪些群」，
    # 是全局的，且**从不限制私聊**（空列表 = 监听所有群）。
    "binding": {
        "groups": [],          # 允许使用的群号
        "users": [],           # 允许使用的 QQ 号（私聊；群开通即全群可用，不按人）
        "empty_means": "deny",  # groups/users 为空时的语义：deny=谁都不用 allow=不限制
        # 未开通的聊天**默认什么都不回**（连「未开通」也不提示）：
        # 服务范围是运维决策，不该在别人的群里制造一条机器人消息。
        # 调试期想看到回绝原因，把它改成 true（并用 notify_cooldown 限频）。
        "notify_denied": False,
        "notify_cooldown": 300,  # 同一目标多久内只提示一次（0=每次都提示）
    },
    # ---- 触发 ----
    "trigger": {
        "private": True,           # 私聊直接对话
        "group_require_at": True,  # 群里必须 @机器人
        "ignore_bot_self": True,   # 忽略其它机器人/自己的消息
        "ignore_command_messages": True,  # 带命令前缀的消息交给命令系统
        "terminators": ["//"],
        # 结束符后紧跟这个后缀 → Markdown 模式（回答渲染成图片发出）。例：`//md`。
        # 留空则关闭这个模式。
        "markdown_suffix": "md",
        "max_staged_parts": 12,    # 单次输入最多分段数
        "max_staged_chars": 6000,  # 单次输入最多字符数
        "auto_submit_seconds": 0,  # >0 时静默这么久自动提交
    },
    # ---- 会话 ----
    "session": {
        # 会话**只由命令创建**（/aichat new），不会因为有人说话就自动建。
        # 这是刻意的：自动建会让「谁在什么情况下有了一个会话」变得不可预测，
        # 也会让 list 里堆出一串没人认领的会话。
        "name_template": "{who}",
        "history_max_messages": 400,
        "max_sessions": 200,
    },
    # ---- 包装 ----
    "wrapping": {
        "speaker_prefix": "auto",   # auto | always | never
        "include_timestamp": False,
        "system_prompt": "",        # 追加在默认人格之后
        "persona": "",              # 覆盖默认人格
    },
    # ---- 压缩（自主设定上下文长度限制）----
    "compaction": {
        "enabled": True,
        "context_limit": 16384,     # 估算 token 预算
        "threshold_ratio": 0.8,
        "retain_ratio": 0.16,
        "retain_tokens": 0,
        "chars_per_token": 1.6,
        "summary_max_chars": 1200,
        "min_head_messages": 2,
        "max_compactions_per_session": 0,
        "max_context_retries": 1,   # 被上游判「超长」时压缩后重试的次数
    },
    # ---- 输出 ----
    "reply": {
        "chunk_max_chars": 1200,
        "chunk_max_images": 4,
        "quote": False,             # 回复时是否引用触发消息
        # 正文里的行内公式（`$x^0$`）也出图：QQ 不渲染 LaTeX，留着就是源码。
        # 关掉则只在整段 Markdown 模式（//md）里渲染公式。
        "inline_math": True,
        # ⚠️ `error_prefix` 已退役：错误文案由框架 `ErrorReporter` 统一渲染
        # （分类 + 短原因 + 错误编号，群里 @ 触发者），插件不再自己拼前缀。
        "max_rounds": 6,            # 一轮对话最多几次模型调用（工具循环）
        "fallback_text": "（我调用完工具但没有要补充的内容）",
    },
    # ---- 工具 ----
    "tools": {
        "enabled": True,
        "builtin": True,
        "max_mute_seconds": 2592000,
        "custom": [],
    },
    # ---- 联网搜索（工具用）----
    "search": {
        "enabled": True,
        "backend": "360",           # 抓页面时的首选；另一个自动兜底（见 search.py）
        "count": 5,                 # 每次搜索默认几条
        # ---- 搜索 API（默认走 AnySearch）----
        # 抓公开搜索页在有些网络里会被反爬挡住（验证码/空壳/**无关结果**），
        # 所以默认走 AnySearch 的 HTTP API：干净、稳定、结果可引用，且**匿名可用**。
        # 显式留空（`""`）表示「只走抓页面那条路」—— 与「没配过」是两回事：
        # 没配过（键缺失）会被补上默认值，显式留空则原样尊重。
        "api_url": "https://api.anysearch.com/v1/search",
        "api_key": "",              # 留空=匿名；填 key 走认证（额度更高）
        "api_method": "POST",       # POST（默认）或 GET
        # 垂直检索：`<领域>.<子域>`，如 academic.search / code.snippet / finance.quote。
        # 留空=通用搜索。可用领域见 search.ANYSEARCH_DOMAINS。
        "api_tag": "",
        "api_fields": {},           # 可选：{"list_path": "data.results", "zone": "cn", ...}
        "timeout": 15,              # 单次请求超时（秒）
        "max_page_chars": 4000,     # fetch_page 回喂给模型的正文上限
        "user_agent": "",           # 留空用内置 UA
    },
    # ---- 函数图像（plot_function 工具画的图）----
    "plot": {
        "enabled": True,
        "width": 720,               # 画布宽度（px）
        # ⚠️ 高度**不再由配置决定**：等比例（两轴同标度）下高度由数据的纵向跨度定，
        # 上限 900px。留着这个键只是为了兼容旧配置，改它没有效果。
        "height": 460,
        "samples": 1200,            # 采样点数（越多曲线越顺，越慢）
        "grid": True,               # 是否画网格线
        # 是否把 title 排成图标题。默认 false —— 公式在聊天正文里已经有了，
        # 再画在图上既重复又占地方（多条曲线时的**图例**不受这个开关影响）。
        "show_title": False,
        # 刻度间隔（数据单位）：留空 = 自动选一个 1/2/5×10ⁿ 的间隔。
        # 指定后不再自动 —— 例如 x_step=1 就是「每 1 个单位一个刻度」。
        "x_step": None,
        "y_step": None,
        # 一个数据单位画多少像素。等比例下它同时决定图的大小；留空按 width 铺开。
        "unit_px": None,
        # 两轴统一标度 + 格子方正。**默认就是开的**（null = 开）；false 关掉，
        # 回到「纵轴按数据自适应、画布固定 460」。代价是纵轴可能盖不住曲线。
        "equal_aspect": None,
        # 坐标轴省略区间（如 "0..999"）：把这区间折成一小条并画波浪线。
        # 典型用法：曲线整体偏移很大（y=sin(x)+1000），中间一大段没内容。
        "omit_y": None,
        "omit_x": None,
        # 参数方程里 t 的范围（默认一个周期）与采样点数
        "t_min": 0.0,
        "t_max": 6.283185307179586,
        "t_samples": 1200,
    },
    # ---- 历史对话（/aichat history 列表 + /aichat hget 出图）----
    "history": {
        "enabled": True,
        "list_limit": 10,           # `history` 默认列最近几对（可用 --limit 覆盖）
        "summarize": True,          # 清单里的问答是否调模型写短摘要
        "summary_chars": 20,        # 每条摘要最多几个字
        "summary_read_chars": 400,  # 送给摘要模型的单侧阅读上限（防烧 token）
        "max_chars_per_side": 2000, # 图里单侧正文的截断长度
        "render_width": 720,        # 图片宽度（px）
        "render_timeout": 20,       # 单张渲染超时（秒）
        "keep_files": 40,           # 图片目录最多留几个文件
        "chromium": "",             # 留空自动探测；可写绝对路径
    },
    "log_verbose": False,
}


#: 已退休的配置键（支持点号路径，深层键写 `history.max_images_per_call`）。
#:
#: `seed_config_file` **只补缺、从不删键** —— 所以把键从模板里拿掉之后，老部署的
#: `config.json` 里还会留着它。这些键不会再生效，却会让人以为还能配（甚至以为
#: aichat 还在读别的插件的数据）—— 启动时主动清掉并在日志里说清楚。
#:
#: * `notices` / `weather`：曾经读**别的插件**的私有数据文件、借它的天气模块（已解耦）；
#: * `history.max_images_per_call`：`hget` 改成一次只取一条后不再需要。
RETIRED_CONFIG_SECTIONS = (
    "notices",
    "weather",
    "history.max_images_per_call",
)


def factory_defaults() -> dict:
    """**出厂默认配置**的深拷贝。

    `DEFAULT_CONFIG` 是模块级可变字典，运行期被任何代码就地改一下（测试、插件、
    未来的热更新路径），`seed_config_file()` 就会把**改过的值**当成默认写进
    用户的 `config.json` —— 一个测试改过的默认值会污染真实部署的配置。
    所以补配置模板时一律用这份快照，而不是活的 `DEFAULT_CONFIG`。
    """
    return json.loads(json.dumps(DEFAULT_CONFIG))


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：`override` 有的键覆盖，没有的沿用 `base` 的默认值。

    只认 dict 的递归 —— 列表整体覆盖（配置里写 `terminators: ["//"]`
    就是要替换掉默认值，而不是追加）。
    """
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ==================== 插件 ====================

class AiChatPlugin(PluginBase):
    name = "ai_chat"
    version = "1.0.0"
    description = "多会话 AI 对话（DSH 式消息包装 + 上下文压缩 + 内置工具）"
    author = "ncqqpy"

    # ---------------- 生命周期 ----------------

    async def on_load(self):
        # ⚠️ 先**从磁盘重读**再补模板：`ConfigManager` 只在进程启动时读一次文件，
        # 之后 `self.config.all()` 拿到的都是那一刻的快照。于是「手工改了
        # config.json → 改代码触发热重载」会拿到一堆旧值，看起来像配置没生效
        # （`/aichat reload --confirm=true` 才有用，而那条路要能收 QQ 消息）。
        # 插件加载本来就该以文件为准，所以这里补一次重读。
        if self._cfg_root is not None:
            try:
                self._cfg_root.reload()
            except Exception as e:  # noqa: BLE001 - 配置读不出来不该让插件起不来
                self.logger.warning("[ai_chat] 重读 config.json 失败: %s", err_text(e))
        # 先把配置模板补进 config.json（缺键才补），再清掉已退休的段，最后读回来自用
        self.seed_config_file()
        self.retire_config_sections()
        self.cfg: dict = deep_merge(factory_defaults(), self.config.all())
        self.adopt_search_api_default()
        self._data_dir = self._resolve_data_dir()
        os.makedirs(self._data_dir, exist_ok=True)

        self.store = SessionStore(
            self._data_dir, logger=self.logger,
            max_messages=int(self.cfg["session"]["history_max_messages"]),
            max_sessions=int(self.cfg["session"]["max_sessions"]))
        self.buffer = _StagingBuffer(os.path.join(self._data_dir, "staging.json"),
                                     logger=self.logger, max_parts=int(
                                         self.cfg["trigger"]["max_staged_parts"]))

        self.web = self._build_web_client()
        self.renderer = self._build_renderer()
        self.summaries = SummaryCache(
            os.path.join(self._data_dir, "history_summaries.json"),
            logger=self.logger)
        #: 机器人自己发出的消息 id → 原文在哪（见 `SentReplyIndex`）。
        #: 用户引用机器人发的**图片**时，靠它把原文找回来。
        self.sent_reply = SentReplyIndex(
            os.path.join(self._data_dir, "sent_replies.json"),
            logger=self.logger)
        #: 本轮回复发出的消息 id，按聊天地址分桶（`_send` 里追加）。
        #: 分桶是因为不同会话的对话是并发的 —— 一个全局列表会把两个人的 id 混在一起。
        self._sent_ids: dict[str, list] = {}

        self.reasoning = self._build_reasoning_policy()
        self.compaction_reasoning = ReasoningPolicy(
            dialect=self.reasoning.dialect,
            default=str(self.cfg["llm"]["reasoning"].get("compaction") or "off"),
            per_model=self.reasoning.per_model,
            params=self.reasoning.params,
            max_tokens_floor=self.reasoning.max_tokens_floor)
        self.llm = self._build_llm()
        self.policy = CompactionPolicy(
            max_tokens=int(self.cfg["compaction"]["context_limit"]),
            threshold_ratio=float(self.cfg["compaction"]["threshold_ratio"]),
            retain_ratio=float(self.cfg["compaction"]["retain_ratio"]),
            retain_tokens=int(self.cfg["compaction"]["retain_tokens"]),
            min_head_messages=int(self.cfg["compaction"]["min_head_messages"]),
            chars_per_token=float(self.cfg["compaction"]["chars_per_token"]),
            summary_max_chars=int(self.cfg["compaction"]["summary_max_chars"]),
            max_compactions_per_session=int(
                self.cfg["compaction"]["max_compactions_per_session"]))
        self.compactor = Compactor(self.policy, self.llm, logger=self.logger)
        self.wrapping = DshWrapping(WrappingConfig(
            speaker_prefix=self.cfg["wrapping"]["speaker_prefix"],
            include_timestamp=bool(self.cfg["wrapping"]["include_timestamp"])))

        self.tools = ToolRegistry(logger=self.logger)
        self._locks: dict[str, asyncio.Lock] = {}
        self._multi_user: dict[str, tuple[int, bool]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._inflight: set[str] = set()
        #: 这一轮还在跑时用户按了提交键 —— 跑完立刻接着提交。
        #: 没有它的话 `_schedule_submit` 会**静默丢掉**这次提交：用户发了内容又发了
        #: 结束符，却什么都不发生（不回答、不报错），看起来就是机器人死了（实测复现）。
        self._pending_submit: set[str] = set()
        #: 「这条会话正在消化一条消息」的计数（按 `conversation_key`）。
        #: ⚠️ 从收到消息到内容落进暂存区之间要 await 若干次 `get_msg`，而框架给
        #: **每次** API 调用都加了 `api_delay`（线上 1~5 秒）—— 引用一条消息、再追
        #: 引用链，几秒钟就过去了。用户完全可能在这几秒里把结束符按下去。
        #: 线上实测（2026-10-09 21:40）：内容 `21:40:39` 才入暂存区，用户 `21:40:38`
        #: 按的 `#` → 回了「暂存区是空的」，然后那段内容**卡在暂存区里**再也没提交。
        self._ingesting: dict[str, int] = {}
        #: 等「别人的消化」最多等多久。引用链最多 3 层 + 媒体探测 1 次 = 4 次
        #: `get_msg`，每次最坏 `api_delay` 上限 5 秒 → 留到 30 秒足够。
        self._INGEST_WAIT_MAX = 30.0
        self._denied_notice: dict[str, float] = {}   # 回绝提示冷却（按目标）
        self._visibility_filter = None               # /help 可见性过滤器
        self._bind_all_group: Optional[bool] = None   # 缓存的 bound_groups 判定
        #: 「当前焦点是不是受邀会话」的缓存：`uid -> ((会话id, 受邀名单快照), 结果)`。
        #: 快照里带上受邀名单，是为了让「同一个人先后被邀进两个会话」这类改名也能
        #: 自然失效 —— 只在名单或焦点真变了却还命中旧结果时才会出错。
        self._invited_cache: dict[int, tuple] = {}
        #: 函数图像服务：渲染出图 + 记录本会话产出过哪些文件（发图前要核对）
        self.plots = PlotService(self.renderer, dict(self.cfg.get("plot") or {}),
                                 logger=self.logger, data_dir=self._data_dir)
        #: 网图配图服务：下载 + 魔数校验 + 落盘白名单（同 plots 那套规矩）
        self.webimg = WebImageService(cfg=dict(self.cfg.get("web_image") or {}),
                                      logger=self.logger, data_dir=self._data_dir)

        if not self.llm.enabled:
            self.logger.warning(
                "[ai_chat] 未配置 llm.base_url / llm.model / llm.api_key，"
                "对话功能将不可用（命令仍可查看与管理会话）")

    async def on_enable(self):
        # 幂等：热重载回滚会在同一个实例上再走一次 on_enable，而 EventBus 不按
        # hid 去重 —— 先撤销自己上次注册的，避免 /events 里出现重复监听器。
        self._off_own_handlers()
        # 工具表同样要**先清空再重建**：`self.tools` 挂在实例上，不清就会累积 ——
        # 于是把 search.enabled 改成 false 之后 web_search 仍然可用（实测），
        # 回滚重启用一次也会留下上一版的工具。工具表必须严格等于当前配置。
        for stale in self.tools.names():
            self.tools.remove(stale)
        if self.cfg["tools"]["builtin"]:
            register_builtin_tools(self.tools)
            if self.cfg["search"]["enabled"]:
                register_web_tools(self.tools)
            if self.cfg["plot"]["enabled"]:
                register_plot_tool(self.tools)
            # 配图要联网下载，所以依赖搜索那套 HTTP 客户端（关掉搜索就没有它）
            if (self.cfg.get("web_image", {}).get("enabled", True)
                    and self.cfg["search"]["enabled"]):
                register_image_tool(self.tools)
        register_config_tools(self.tools, self.cfg["tools"].get("custom") or [],
                              logger=self.logger)

        @self.on_message()
        async def _on_any_message(event: MessageEvent):
            await self._handle_message(event)

        @self.on_event(GroupRecallEvent)
        async def _on_recall(event: GroupRecallEvent):
            await self._handle_recall(event)

        self._register_commands()

    async def on_disable(self):
        # 摘掉 /help 过滤器：热重载会换实例，留着旧闭包会让新实例的判定失效
        if self._visibility_filter is not None:
            try:
                self.cmds.remove_visibility_filter(self._visibility_filter)
            except Exception:  # noqa: BLE001
                pass
            self._visibility_filter = None
        await self.llm.close()
        await self._close_web()
        self.buffer.save()

    async def on_unload(self):
        await self.llm.close()
        await self._close_web()
        self.buffer.save()
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    async def _close_web(self) -> None:
        """关掉搜索用的 HTTP 会话。

        ⚠️ 热重载会换实例：不关的话每次重载都留一个 aiohttp 会话（连接池 + fd），
        重载几十次就是几十个泄漏 —— 而且它以 `Unclosed client session` 的形式
        出现在日志里，很容易被当成无关噪音忽略掉。
        """
        web = getattr(self, "web", None)
        if web is None:
            return
        try:
            await web.close()
        except Exception as e:  # noqa: BLE001 - 关不掉不该阻断卸载
            self.logger.warning("[ai_chat] 关闭搜索会话失败: %s", err_text(e))

    # ---------------- 配置 ----------------

    def config_template(self) -> dict:
        """完整配置模板（框架在加载时写进 config.json → plugins.ai_chat）。

        出厂默认值来自 `factory_defaults()`（`DEFAULT_CONFIG` 的深拷贝，见那里的
        说明：活的 `DEFAULT_CONFIG` 被就地改过时不能污染用户配置）。
        """
        return factory_defaults()

    def seed_config_file(self) -> bool:
        """把完整配置模板补进 `config.json` 的 `plugins.ai_chat` 段。

        为什么需要：`config.json` 里原本**没有**这个段，跑的全是代码内置默认值 ——
        运维打开文件看不到有哪些开关可改（「插件设置使用 config.json」也就无从谈起）。
        这里把有效配置原样落盘，让文件成为一份自解释的清单。

        语义（三条，缺一不可）：

        1. **只补缺键，绝不覆盖已有值** —— 升级插件新增配置项时，
           老部署会被自动补上，而用户改过的值一个都不动；
        2. **写的是「有效值」**（`DEFAULT_CONFIG` 深合并用户值的结果），
           所以文件里看到的就是真正生效的；
        3. **幂等**：补完没有差异就不写盘（避免每次启动都重写文件）。

        实现收敛在框架的 `PluginConfig.fill_defaults()`（深合并 + 单次写盘）：
        框架加载插件时已经会调一次，这里保留入口是给
        `/aichat config` 与「手工删了键再重读」这类运行期路径用的。

        返回是否真的写了盘。
        """
        report = self.config.fill_defaults(factory_defaults())
        if report.filled and self.logger:
            self.logger.info(
                "[ai_chat] 已把 %d 个缺失配置项写入 config.json（plugins.ai_chat）：%s",
                len(report.filled), ", ".join(report.filled[:12])
                + ("…" if len(report.filled) > 12 else ""))
        if report.conflicts and self.logger:
            self.logger.warning("[ai_chat] 配置项类型与模板不符（保留你的值）：%s",
                                ", ".join(report.conflicts))
        return report.changed

    def retire_config_sections(self) -> bool:
        """删掉已退休的配置段（见 `RETIRED_CONFIG_SECTIONS`）。

        `seed_config_file` 只补缺、从不删键 —— 所以光把键从模板里拿掉，
        老部署的 `config.json` 里还会留着 `notices.path` 指向**另一个插件的私有
        数据目录**。那行配置不会再生效，却会让人以为「aichat 还能读别人的数据」。
        这里一次性清掉并在日志里说清楚。

        返回是否真的动了配置。
        """
        current = self.config.all()
        if not isinstance(current, dict):
            return False
        merged = json.loads(json.dumps(current))      # 深拷贝，避免就地改到只读视图
        removed: list[str] = []
        for path in RETIRED_CONFIG_SECTIONS:
            parts = path.split(".")
            node = merged
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    break
            else:
                if parts[-1] in node:
                    del node[parts[-1]]
                    removed.append(path)
        if not removed:
            return False
        self.config.set_all(merged)
        if self.logger:
            self.logger.info("[ai_chat] 已从 config.json 移除失效配置键：%s",
                             ", ".join(removed))
        return True

    def _resolve_data_dir(self) -> str:
        """数据目录（会话历史 / 绑定表 / 暂存区）。规则见 `_resolve_relative`。"""
        return self._resolve_relative(self.cfg.get("data_dir"), default="data")

    def _config_dir(self) -> str:
        """`config.json` 所在目录（取框架 ConfigManager 的路径；拿不到就退回 CWD）。"""
        path = getattr(self._cfg_root, "_path", None)
        if path:
            return os.path.dirname(os.path.abspath(str(path)))
        return os.path.abspath(os.getcwd())

    def _resolve_relative(self, configured: str, *, default: str = "data") -> str:
        """本插件数据目录的相对路径口径（`data_dir` 用）。

        * **裸名字**（`data`）→ 相对**插件包**（插件自带的数据就放插件里）；
        * **显式相对**（`./data/ai_chat`、`../shared`）→ 相对 **config.json
          所在目录**（从顶层配置写路径时这才是直觉预期）。
        """
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        text = str(configured or "").strip()
        if not text:
            return os.path.normpath(os.path.join(plugin_dir, default))
        text = os.path.expanduser(text)
        if os.path.isabs(text):
            return os.path.normpath(text)
        base = self._config_dir() if (text.startswith(("./", "../"))
                                      or os.sep in text) else plugin_dir
        return os.path.normpath(os.path.join(base, text))

    # ---------------- 历史清单的短摘要 ----------------

    async def _summarize_exchanges(self, session: Session,
                                   shown: list) -> tuple[dict, str]:
        """给清单里的**用户提问**生成短摘要，返回 `({编号: 概括}, 提示语)`。

        只概括提问，AI 的回答既不概括也**不进清单**：目录的用途是
        「我之前问过什么」，而回答往往很长且一条里覆盖多个要点，
        压成 20 字会失真、原样贴进来会把编号冲散 —— 要看全文走 `hget`。

        设计取舍（三条都不是可选项）：

        1. **只给要显示的那几条摘要**（`shown`），不是整个会话 —— 列 10 条就只算
           10 条，翻页也永远只有一次调用的开销；
        2. **按内容指纹缓存**（`history_summaries.json`）：`history` 是翻页查看的
           命令，没有缓存的话每看一遍都要重新烧一遍 token；
        3. **失败/关掉都退到截断**，并且提示语里说明是降级 —— 悄悄给一份原文截断、
           让用户以为那是摘要，比明说「摘要不可用」更糟。
        """
        hcfg = self.cfg["history"]
        chars = max(4, int(hcfg["summary_chars"]))
        if not hcfg["summarize"] or not shown:
            return {}, ""

        # 1) 先取缓存命中
        keys = {ex.index: summary_key(ex) for ex in shown}
        done: dict[int, str] = {}
        missing = []
        for ex in shown:
            hit = self.summaries.get(session.id, keys[ex.index])
            if hit is not None:
                done[ex.index] = hit
            else:
                missing.append(ex)

        # 2) 没命中的一次批量问模型（不是每条一次调用）
        if not self.llm.enabled:
            return (self._truncated_summaries(shown, chars),
                    "（未配置 LLM，下面的问是**截断**而非摘要）")
        if missing:
            got, err = await self._ask_summaries(missing, chars)
            fresh: dict[str, str] = {}
            degraded = 0
            for ex in missing:
                summary = got.get(ex.index) or ""
                if not summary:
                    summary = fallback_summary(ex, chars)
                    degraded += 1
                done[ex.index] = summary
                fresh[keys[ex.index]] = summary
            if fresh:
                self.summaries.put(session.id, fresh)
            if degraded:
                note = f"（{degraded} 条摘要没生成，这些是截断原文）"
                if err:
                    note += f"：{err}"
                return done, note
        return done, ""

    @staticmethod
    def _truncated_summaries(exchanges: list, chars: int) -> dict:
        """全部退到截断（未配置 LLM 时用）。"""
        return {ex.index: fallback_summary(ex, chars) for ex in exchanges}

    async def _ask_summaries(self, exchanges: list,
                             chars: int) -> tuple[dict, str]:
        """一次调用给多条**提问**写摘要。返回 `({编号: 概括}, 错误说明)`。"""
        ask = summarize_context(
            exchanges, read_chars=int(self.cfg["history"]["summary_read_chars"]))
        prompt = (
            "下面是若干条用户提问。请为**每一条**写一句概括，"
            f"每条不超过 {chars} 个字，只概括提问本身（不要写答案）。\n"
            "输出格式**严格**为每行：`编号|概括`，编号与输入一致，"
            "不要写别的行、不要加序号或引号。\n"
            "---- 提问开始 ----\n"
            f"{ask}\n"
            "---- 提问结束 ----\n"
            "请输出。"
        )
        try:
            result = await self.llm.chat(
                [{"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                reasoning=self._summary_reasoning())
        except Exception as e:  # noqa: BLE001 - 摘要失败不该让 list 失败
            self.logger.warning("[ai_chat] 历史摘要生成失败: %s", err_text(e))
            return {}, err_text(e)
        return parse_question_summaries(result.text or "",
                                        wanted=[e.index for e in exchanges],
                                        limit=chars)

    def _summary_reasoning(self):
        """摘要请求的思考强度（复用压缩那一档：要求短、密，思考链是浪费）。"""
        return override_text(self.compaction_reasoning, self.llm.model)

    def _build_web_client(self) -> WebClient:
        """按当前配置构造联网搜索客户端（懒建 HTTP 会话，构造不发请求）。"""
        scfg = self.cfg["search"]
        return WebClient(
            timeout=float(scfg["timeout"]),
            backend=str(scfg["backend"] or "360"),
            api_url=str(scfg.get("api_url") or ""),
            api_key=str(scfg.get("api_key") or ""),
            api_method=str(scfg.get("api_method") or "POST"),
            api_tag=str(scfg.get("api_tag") or ""),
            api_fields=dict(scfg.get("api_fields") or {}),
            user_agent=str(scfg.get("user_agent") or ""),
            logger=self.logger)

    def adopt_search_api_default(self) -> bool:
        """把「从没配过搜索 API」的老部署切到 AnySearch。

        为什么需要：`seed_config_file` 只补**缺键**，不会覆盖已有值 ——
        老配置里 `api_url` 是空串（当年那栏的默认值就是空），于是新默认值永远
        进不去，用户看到的就是「搜索还是不好用」。

        判定必须**同时**要求三个键都空：只要用户给 `api_url` 填过东西、
        或填过 key/tag，就说明他有意识地在用某个服务，绝不能替他改。
        """
        # ⚠️ 读的是 `self.cfg`（已与默认深合并的有效配置），**不是**
        # `self.config.get("search")` —— 后者在插件代理上的键就是
        # `plugins.ai_chat.search`，而用户配置里没有这个整段键时返回 None，
        # 于是这个方法永远在第一道判断就退出、迁移静默不生效（实测踩过）。
        sec = self.cfg.get("search")
        if not isinstance(sec, dict):
            return False
        if any(str(sec.get(k) or "").strip()
               for k in ("api_url", "api_key", "api_tag")):
            return False
        # ⚠️ 迁移目标必须是**写死的常量**，不能读 `self.cfg["search"]["api_url"]`：
        # 老部署磁盘上那个键是空串，`deep_merge` 会用空串**覆盖**模板默认值，
        # 于是「从配置里取默认值」取到的还是空串、迁移静默失败（实测踩过）。
        url = ANYSEARCH_URL
        self.config.set("search.api_url", url)
        if self.logger:
            self.logger.info(
                "[ai_chat] 检测到从未配置过搜索 API，已接入 AnySearch：%s"
                "（想用自己的服务改 search.api_url 即可）", url)
        return True

    def _build_llm(self) -> LlmClient:
        """按当前配置造 LLM 客户端。

        单独一个方法（和 `_build_renderer` / `_build_web_client` 同理）：配置项
        要真的接到对象上才生效，而「加了配置键、代码没读」在本仓库出现过好几次
        （`plot.unit_px`、`trigger.auto_submit_seconds`）—— 抽出来就能在测试里
        直接断言 `_build_llm().stream`（见 `test_llm_stream_wiring`）。
        """
        llm = self.cfg["llm"]
        return LlmClient(
            base_url=llm["base_url"], api_key=llm["api_key"], model=llm["model"],
            max_tokens=llm["max_tokens"], temperature=llm["temperature"],
            timeout=llm["timeout"], stream=bool(llm.get("stream", True)),
            stall_timeout=llm.get("stall_timeout", 60),
            extra_headers=llm["extra_headers"], logger=self.logger,
            reasoning_policy=self.reasoning)

    def _build_renderer(self) -> HistoryRenderer:
        """按当前配置构造历史出图渲染器。

        ⚠️ `reload_config()` 也必须调用它 —— 否则 `history.chromium` /
        `render_width` / `render_timeout` 这些改完**不会生效**（配置热重载
        偏偏只重建 llm/policy/wrapping，症状是「改了配置没反应」）。
        渲染器只在**用到时**才去找 Chromium（构造不发进程），所以没装浏览器的
        部署照样能用除 hget 以外的全部功能。
        """
        hcfg = self.cfg["history"]
        return HistoryRenderer(
            os.path.join(self._data_dir, "history_img"),
            options=RenderOptions(
                width=int(hcfg["render_width"]),
                timeout=float(hcfg["render_timeout"]),
                max_chars_per_side=int(hcfg["max_chars_per_side"]),
                chrome=str(hcfg.get("chromium") or ""),
                keep_files=int(hcfg["keep_files"])),
            logger=self.logger)

    def _build_reasoning_policy(self) -> ReasoningPolicy:
        """把配置里的 `llm.reasoning` 变成运行时策略。

        兼容旧写法：没有 `llm.reasoning` 时退回单个 `llm.reasoning_effort` 字符串。
        """
        llm = self.cfg["llm"]
        spec = llm.get("reasoning") or {}
        legacy = llm.get("reasoning_effort")
        if not isinstance(spec, dict):
            spec = {}
        dialect = str(spec.get("dialect") or "effort").strip().lower()
        if dialect not in DIALECTS:
            self.logger.warning("[ai_chat] 未知 reasoning.dialect=%r，按 effort 处理",
                                dialect)
            dialect = "effort"
        default = spec.get("default")
        if default in (None, ""):
            # 旧配置只有 reasoning_effort 时以它为准；全新配置时 llm.reasoning.default
            # 已经带了 "off"，不会走到这里
            default = legacy if legacy not in (None, "") else "off"
        per_model = spec.get("per_model")
        params = spec.get("params")
        try:
            floor = int(spec.get("max_tokens_floor") or 0)
        except (TypeError, ValueError):
            floor = 0
        return ReasoningPolicy(
            dialect=dialect,
            default=default,
            per_model=per_model if isinstance(per_model, dict) else {},
            params=params if isinstance(params, dict) else {},
            max_tokens_floor=max(0, floor))

    @staticmethod
    def _session_reasoning(session: Session) -> str:
        """会话级思考强度（`/aichat think` 写进 meta）。空 = 用全局默认。"""
        if session is None:
            return ""
        value = (session.meta or {}).get("reasoning")
        return str(value) if value not in (None, "") else ""

    def _set_session_reasoning(self, session: Session, value: str) -> None:
        with self.store._lock:
            if value:
                session.meta["reasoning"] = value
            else:
                session.meta.pop("reasoning", None)
            session.updated_at = time.time()
            self.store._save_locked()

    def effective_reasoning(self, session: Optional[Session]) -> ReasoningSpec:
        """当前生效的思考强度（会话覆盖 → per-model → 全局默认）。"""
        return self.reasoning.spec_for(self.llm.model,
                                       self._session_reasoning(session))

    def reload_config(self, *, from_disk: bool = False) -> dict:
        """把配置重新应用到运行中的实例。

        ⚠️ 默认**不重读文件**：`ConfigManager` 只在构造时读一次 `config.json`，
        所以这里的「reload」指的是「按当前内存里的配置重建运行时对象」——
        它服务的是 `/aichat think --global_=true` 这类「自己改自己」的路径
        （那些改动已经通过 `PluginConfig.set()` 同步进内存与文件）。

        `from_disk=True` 才会真的重读文件，用于「手工编辑了 config.json」。
        """
        if from_disk and self._cfg_root is not None:
            try:
                self._cfg_root.reload()
            except Exception as e:  # noqa: BLE001
                self.logger.warning("[ai_chat] 重读 config.json 失败: %s", err_text(e))
        self.cfg = deep_merge(factory_defaults(), self.config.all())
        llm = self.cfg["llm"]
        self.reasoning = self._build_reasoning_policy()
        self.compaction_reasoning = ReasoningPolicy(
            dialect=self.reasoning.dialect,
            default=str(llm["reasoning"].get("compaction") or "off"),
            per_model=self.reasoning.per_model, params=self.reasoning.params,
            max_tokens_floor=self.reasoning.max_tokens_floor)
        self.llm.reconfigure(
            base_url=llm["base_url"], api_key=llm["api_key"], model=llm["model"],
            max_tokens=int(llm["max_tokens"]), temperature=llm["temperature"],
            timeout=llm["timeout"], extra_headers=llm["extra_headers"],
            stream=bool(llm.get("stream", True)),
            stall_timeout=llm.get("stall_timeout", 60),
            reasoning_policy=self.reasoning)
        self.compactor.bind_reasoning(self.compaction_reasoning)
        comp = self.cfg["compaction"]
        self.policy.max_tokens = int(comp["context_limit"])
        self.policy.threshold_ratio = float(comp["threshold_ratio"])
        self.policy.retain_ratio = float(comp["retain_ratio"])
        self.policy.chars_per_token = float(comp["chars_per_token"])
        self.wrapping.config = WrappingConfig(
            speaker_prefix=self.cfg["wrapping"]["speaker_prefix"],
            include_timestamp=bool(self.cfg["wrapping"]["include_timestamp"]))
        # 出图参数也可能被改（chromium 路径、宽度、超时、keep_files）——
        # 不重建的话这些改动要重启进程才生效，而其它配置是热生效的，很难查。
        self.renderer = self._build_renderer()
        self.web = self._build_web_client()
        # 画图服务持有渲染器与 plot 段配置，两者都换了就地更新 —— 否则改了
        # plot.width 之后出图仍是旧宽度，症状与上面 `_build_renderer` 注释里
        # 说的是同一类（改了配置没反应）。
        self.plots.renderer = self.renderer
        self.plots.cfg = dict(self.cfg.get("plot") or {})
        self.webimg.cfg = dict(self.cfg.get("web_image") or {})
        return self.cfg

    # ---------------- 聊天地址 / 会话解析 ----------------

    @staticmethod
    def _target_of(event: MessageEvent) -> ChatTarget:
        if event.message_type == "group" and event.group_id:
            return ChatTarget("group", int(event.group_id))
        return ChatTarget("private", int(event.user_id),
                          int(event.group_id) if event.group_id else None)

    def _current_session(self, target: ChatTarget, user_id: int) -> Optional[Session]:
        """这个人在这个聊天里的当前会话（焦点）。"""
        session = self.store.focus_of(target, user_id)
        if session is not None and session.archived:
            return None
        return session

    def _session_for(self, target: ChatTarget, event: MessageEvent) -> Optional[Session]:
        """这个人现在该用哪个会话 —— **只认已有焦点，绝不自动创建**。

        没有会话时返回 None，由调用方提示「先用 /aichat new 建一个」。
        """
        return self._current_session(target, int(event.user_id))

    def _no_session_hint(self, target: ChatTarget, uid: int) -> str:
        """没有当前会话时给出的下一步（会话不会自动创建，所以这句很关键）。"""
        if not target.is_group:
            return ("还没有选中的会话。/aichat new <名称> 新建，"
                    "或 /aichat list 看已有的再 /aichat attach <会话>。")
        available = self.store.list(owner_kind=OWNER_GROUP, owner_id=target.chat_id,
                                    limit=self.store.max_sessions)
        if not available:
            return ("本群暂无会话。/aichat new <名称> 新建"
                    "（群里默认群会话）；"
                    "参与别人建的会话需所有者先 /aichat invite <会话> <你的QQ>，"
                    "再 /aichat attach <会话>。")
        usable = [x for x in available
                  if uid in set(x.guests)
                  or self.perm.get_group_role(target.chat_id, uid) in ("owner", "admin")]
        if usable:
            return ("还没有选中本群的会话。/aichat attach <会话> 切过去：\n"
                    + "\n".join(f"  {x.id}「{x.name}」" for x in usable[:5]))
        return ("本群已有会话，但都不对你开放（群会话需被邀请）：\n"
                + "\n".join(f"  {x.id}「{x.name}」" for x in available[:5])
                + "\n让所有者发 /aichat invite <会话> <你的QQ>，"
                  "你再用 /aichat attach <会话> 接入。")

    def _sessions_of_chat(self, target: ChatTarget) -> list[Session]:
        """**当前聊天所包含的会话**（`list` 的默认口径）。

        * 群里 → 本群的群聊会话；
        * 私聊 → 你自己的私聊会话 + 别人邀请你参与的私聊会话。
        """
        if target.is_group:
            return self.store.list(owner_kind=OWNER_GROUP, owner_id=target.chat_id,
                                   limit=self.store.max_sessions)
        uid = int(target.chat_id)
        own = self.store.list(owner_kind=OWNER_USER, owner_id=uid,
                              limit=self.store.max_sessions)
        invited = self.store.list(owner_kind=OWNER_USER, guest_id=uid,
                                  limit=self.store.max_sessions)
        seen = {x.id for x in own}
        return own + [x for x in invited if x.id not in seen]

    def _display_name(self, event: MessageEvent) -> str:
        sender = getattr(event, "sender", None)
        if sender is None:
            return str(event.user_id)
        return (getattr(sender, "card", "") or getattr(sender, "nickname", "")
                or str(event.user_id)).strip()[:24]

    # ---------------- 会话可用性（归属 + 受邀名单） ----------------

    @staticmethod
    def _session_type_label(session: Session) -> str:
        """给人看的会话类型（同时是「能在哪里用」的判据）。"""
        return "群聊会话" if session.owner_kind == OWNER_GROUP else "私聊会话"

    def _matches_context(self, session: Session, target: ChatTarget) -> bool:
        """会话类型与聊天类型是否一致 —— **群聊只准用群聊会话，私聊只准用私聊会话**。"""
        if target.is_group:
            return session.owner_kind == OWNER_GROUP
        return session.owner_kind == OWNER_USER

    @staticmethod
    def _is_creator(session: Session, user_id: int) -> bool:
        """是不是**建这个会话的人**。

        群会话归群所有（`owner_id` 是群号），所以「所有者 == 本人」这个判据对它
        永远不成立 —— 群会话的创建者必须单独记一笔，否则他建完就用不了/
        管不了自己刚建的会话（实测：建完说话被拒「需要被邀请」）。
        """
        creator = session.meta.get("created_by")
        return creator is not None and int(creator) == int(user_id)

    def _may_attach(self, session: Session, target: ChatTarget,
                    user_id: int) -> tuple[bool, str]:
        """这个人能不能在**这个聊天里**用该会话。所有 attach/switch/自动路径共用。

        两道判据，先类型后归属：

        1. **类型必须匹配**（推翻了早先「一个会话群聊私聊都能用」的设定）：
           群聊里只能用群聊会话，私聊里只能用私聊会话 —— 两个空间不互通，
           避免群里的上下文被谁顺手带进一对一窗口，反之亦然。
        2. **归属 / 受邀**：所有者本人、或被邀请过的人。群聊会话尤其要注意：
           它归某个群所有，**不等于全群可用**，成员仍需被邀请。

        超级用户**不豁免**：服务范围由 `binding` 决定，但「能不能用别人的会话」
        仍受这两条约束 —— 否则群主就能随手翻别人的私聊。
        """
        if session.archived:
            return False, f"会话 {session.id}「{session.name}」已归档，不能使用"
        kind = self._session_type_label(session)
        if not self._matches_context(session, target):
            if target.is_group:
                return False, (f"会话 {session.id}「{session.name}」是{kind}，"
                               f"不能在群里使用。群里只能用群聊会话："
                               f"/aichat new <名称> 新建（群里默认群会话）")
            return False, (f"会话 {session.id}「{session.name}」是{kind}，"
                           f"不能在私聊里使用。私聊里只能用私聊会话："
                           f"/aichat new <名称>")
        uid = int(user_id)
        if session.owner_kind == OWNER_USER and uid == int(session.owner_id):
            return True, ""
        if self._is_creator(session, uid):
            return True, ""
        if uid in set(session.guests):
            return True, ""
        if session.owner_kind == OWNER_GROUP and int(session.owner_id) == target.chat_id:
            return False, (f"会话 {session.id}「{session.name}」属于本群，"
                           f"群成员需被邀请才能使用："
                           f"/aichat invite {session.id} {uid}")
        return False, (f"会话 {session.id}「{session.name}」归"
                       f"{session.owner_label}所有，你不在受邀名单里。"
                       f"需所有者发 /aichat invite {session.id} {uid} 邀请你参与")

    def _may_manage(self, session: Session, user_id: int) -> tuple[bool, str]:
        """谁能管理会话（改名 / 邀请 / 删除）：只有所有者。

        群归属的会话由群主或管理员代管（群里总得有人能管），个人会话只有本人。
        """
        uid = int(user_id)
        if session.owner_kind == OWNER_GROUP:
            # 群主/管理员、以及**建这个会话的人**都能管
            if self.perm.is_superuser(uid) or self._is_creator(session, uid):
                return True, ""
            role = self.perm.get_group_role(int(session.owner_id), uid)
            if role in ("owner", "admin"):
                return True, ""
            creator = session.meta.get("created_by")
            return False, (f"会话 {session.id} 归群 {session.owner_id} 所有，"
                           f"只有群主/管理员"
                           + (f"或创建者（{creator}）" if creator else "")
                           + "能管理")
        if uid == int(session.owner_id):
            return True, ""
        return False, f"会话 {session.id} 属于 {session.owner_id}，只有所有者能管理"

    # ---------------- 服务范围（绑定白名单） ----------------

    @staticmethod
    def _to_ids(raw) -> set[int]:
        out: set[int] = set()
        for v in (raw or []):
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                continue          # 配置里写错的值忽略掉，不让一个笔误废掉整条白名单
        return out

    def _bound_ids(self, key: str) -> set[int]:
        return self._to_ids((self.cfg.get("binding") or {}).get(key))

    def _bound_groups(self) -> set[int]:
        """群白名单。

        优先用 `binding.groups`；没配就复用框架的 `bound_groups`。后者是
        「机器人监听哪些群」（见 `GroupBinding`），**空列表 = 监听所有群**，
        语义与白名单相反，所以只在它非空时借用来兜底 —— 免得为了同一件事
        在两处维护两份群号，漏配一边就静默没有服务。
        """
        own = self._bound_ids("groups")
        if own:
            return own
        try:
            inherited = self._to_ids(self._groups.list_bound()
                                     if self._groups is not None else [])
        except Exception:  # noqa: BLE001
            inherited = set()
        return inherited          # 空 = 框架也没限定，交给 empty_means 决定

    def _empty_means(self) -> str:
        return str((self.cfg.get("binding") or {}).get("empty_means") or "deny").lower()

    def check_access(self, target: ChatTarget, user_id: int) -> tuple[bool, str]:
        """这个聊天地址是否在服务范围内。返回 `(允许, 回绝原因)`。

        **与权限等级无关**：不做 `is_superuser` 豁免 —— 服务范围是运维决策，
        不是权限决策。超级用户在未绑定的群里同样用不了。

        **群与私聊是两条独立通道**：群开通了就是全群可用（群里每人各有自己的
        会话，不需要按人授权）；私聊才是按人开通。两者互不影响 ——
        群开通不会自动开放私聊，反之亦然。
        """
        groups = self._bound_groups()
        users = self._bound_ids("users")
        empty = self._empty_means()
        cfg_hint = "config.json → plugins.ai_chat.binding"

        if target.is_group:
            # 注意：正常的 WS 投递路径上，框架的 `_on_ws_event()` 已经先按
            # `bound_groups` 把未绑群的群消息**整条丢弃**了（不会到插件）。
            # 这里再判一次是第二道闸：插件自己派发的事件、直接调 `cmds.handle()`
            # 的场景（测试、未来别的入口）同样要被拦住。
            if not groups and empty == "deny":
                return False, (f"本群没有开通 AI 对话服务（{cfg_hint}.groups 为空）。"
                               "联系管理员把本群加入白名单。")
            if groups and int(target.chat_id) not in groups:
                return False, (f"本群（{target.chat_id}）没有开通 AI 对话服务。"
                               f"联系管理员把它加到 {cfg_hint}.groups。")
            return True, ""

        # 私聊 / 群临时会话（小窗）
        # ⚠️ 授权与否看**这个地址的主人**，不是发消息的人。`private` 地址的
        # `chat_id` 就是主人；正常投递路径上二者相同（NapCat 只会把该小窗的消息
        # 报给创建者），但伪造/转发的场景下不校验就等于让任何人借用别人的小窗。
        if int(user_id) != int(target.chat_id):
            return False, (f"这个私聊地址属于 {target.chat_id}，"
                           f"你（{user_id}）不能在它上面使用服务。")
        if users and int(user_id) in users:
            return True, ""
        if not users and empty != "deny":
            return True, ""
        # ---- 受邀例外（**只在私聊里生效**）----
        # 会话所有者可以 `/aichat invite` 一个不在白名单里的人，被邀请者 attach 后
        # 就能在**私聊**里用这个会话 —— 这是「按会话授权」，所有者对自建会话的
        # 可见性负责。
        #
        # ⚠️ 群里**没有**这条例外：群是公共场所，判断标准必须是群绑定本身
        # （`groups`），否则任何人都能在群里把人拉进来用 —— 群这道闸就形同虚设。
        # 所以受邀未绑定的人在群里仍然被拦。
        #
        # 判据是**当前焦点**：只有此刻正在用那个被邀请的会话才放行；不 attach
        # 就只是名单上的一个号码，不构成任何服务权限。
        invited = self._invited_session_of(int(user_id))
        if invited is not None:
            return True, ""
        if not users:
            return False, (f"私聊没有开通 AI 对话服务（{cfg_hint}.users 为空）。"
                           "联系管理员把你加入白名单。")
        return False, (f"你（{user_id}）没有开通 AI 对话服务。"
                       f"联系管理员把你加到 {cfg_hint}.users；"
                       f"或让某个会话的所有者 /aichat invite 你参与他的会话。")

    #: 「认领会话」的动作：受邀的未绑定用户必须能用它们建立/解除焦点，
    #: 否则服务范围的受邀例外无法生效（见命令守卫里的说明）。
    CLAIM_COMMANDS = frozenset({"attach", "detach"})

    def _is_invited_guest(self, target: ChatTarget, user_id: int) -> bool:
        """这个人在这类聊天里，是不是「被邀请参与某个会话」的客人。

        只在**私聊**里成立：群里没有受邀例外（见 `check_access`）。
        与 `_invited_session_of` 的差别是：这里**不要求已有焦点** ——
        它问的是「有没有资格去 attach」，所以直接看受邀名单。
        """
        if target.is_group:
            return False
        uid = int(user_id)
        if int(target.chat_id) != uid:
            return False          # 伪造/转发到别人的小窗
        for session in self.store.list(guest_id=uid, limit=self.store.max_sessions):
            if uid in set(session.guests) and not session.archived:
                return True
        return False

    def _invited_session_of(self, user_id: int) -> Optional[Session]:
        """这个人的**当前焦点**是不是一个「他被邀请参与」的会话。

        带缓存：`check_access` 在每条消息、每个命令上都会被问到，而它要遍历
        会话列表 —— 群里消息密集时那点开销也会累起来。缓存只在 `invite` /
        `uninvite` 时清（那才是这个答案会变的唯一时刻）。
        """
        uid = int(user_id)
        key = self.store.focus_key(ChatTarget("private", uid), uid)
        focus_now = self.store.focus_table().get(key) or ""
        signature = (focus_now, self.store.guests_of(focus_now) if focus_now else ())
        hit = self._invited_cache.get(uid)
        if hit is not None and hit[0] == signature:
            return self.store.get(hit[1]) if hit[1] else None
        found: Optional[Session] = None
        if focus_now:
            session = self.store.get(focus_now)
            if session is not None and uid in set(session.guests):
                found = session
        self._invited_cache[uid] = (signature, found.id if found else "")
        return found

    def _access_denied(self, target: ChatTarget, user_id: int, *,
                       text: str = "") -> bool:
        """不在服务范围内就回绝（带冷却，避免刷屏）。返回是否已回绝。

        `text` 是原始消息：若它是**管理服务范围本身**的命令（`/aichat add`/`del`/`targets` 等），
        则放行 —— 否则 users 为空时私聊连 bind 都用不了，永远开不出第一条私聊。
        """
        if text and self._is_scope_admin_command(text):
            return False
        allowed, reason = self.check_access(target, user_id)
        if allowed:
            return False
        binding = self.cfg.get("binding") or {}
        # ⚠️ 默认值必须是 False：只把「显式写了 notify_denied: true」当成要提示。
        # 写成 `.get("notify_denied", True)` 时，**缺这个键**（老配置、手工精简过的
        # 配置、测试夹具）就会退回「要提示」—— 于是「未开通一律静默」失效。
        if not binding.get("notify_denied", False):
            return True
        cooldown = int(binding.get("notify_cooldown") or 0)
        now = time.time()
        last = self._denied_notice.get(target.conversation_key, 0.0)
        if cooldown > 0 and now - last < cooldown:
            return True
        self._denied_notice[target.conversation_key] = now
        self._schedule_deny_notice(target, reason)
        return True

    @staticmethod
    def _namespace() -> str:
        """命令命名空间（`/aichat`）。

        ⚠️ 与 `self.name`（插件名 `ai_chat`，即目录名）**不是**同一个字符串 ——
        命名空间里没有下划线。两者写混会让 `_commands.get(...)` 静默返回 None，
        表现为子命令「挂不上」，而没有任何报错。
        """
        return "aichat"

    def _mount_scope_command(self, sub: str, func, *, schema=None,
                             description: str = "") -> None:
        """把「服务范围管理」子命令挂到 `/aichat` 命名空间下（**豁免服务范围判定**）。

        为什么不走 `self.register_command`：`_register_commands` 里给每个命令都套了
        一层服务范围守卫，而这几个命令正是用来**管理服务范围本身**的 ——
        它们必须能在「尚未开通」的私聊里执行，否则 users 为空时连 add 都用不了，
        永远开不出第一条私聊（实测死锁）。

        这里按框架注册子命令时的**同一套字段**手工构造，保证 `/help` 的渲染口径
        （权限、范围、参数表）与其它子命令完全一致。
        """
        from napcat.command import CommandHandler

        ns = self._namespace()
        parent = self.cmds._commands.get(ns)
        if parent is None:
            self.logger.warning(
                "[ai_chat] 挂载子命令 %s 失败：命名空间 /%s 还不存在", sub, ns)
            return
        handler = CommandHandler(
            sub, func, schema=schema,
            description=description,
            permission="superuser",      # 与服务范围无关，但仍限超级用户
            scope="private",             # 仅私聊
            permission_manager=self.cmds._perm_manager)
        handler.parent_name = ns
        parent.subcommands[sub] = handler
        parent.description = self._root_description()

    def _is_scope_admin_command(self, text: str) -> bool:
        """这条消息是不是「管理服务范围」的命令（bind/unbind/bindings/reload）。"""
        parser = getattr(self.cmds, "parser", None)
        raw = (text or "").lstrip()
        if parser is None or not hasattr(parser, "skip_prefix"):
            return False
        rest = parser.skip_prefix(raw)
        if not rest:
            return False
        parts = rest.split()
        ns = self._namespace()
        if len(parts) < 2 or parts[0].lower() != ns:
            return False
        known = getattr(self.cmds, "_commands", {}) or {}
        parent = known.get(ns)
        sub = parts[1].lower()
        if parent is not None and sub not in getattr(parent, "subcommands", {}):
            return False
        return sub in self.SCOPE_EXEMPT

    def _schedule_deny_notice(self, target: ChatTarget, reason: str) -> None:
        """后台发回绝说明。

        ⚠️ 必须**异步发**：这个判定跑在命令处理器里（`_register_commands` 的
        函数是同步的），同步 await 会变成「协程没被 await」的静默失败。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._send(target, MessageChain().text(
            f"⛔ {reason}")))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _lock(self, sid: str) -> asyncio.Lock:
        lock = self._locks.get(sid)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[sid] = lock
        return lock

    # ---------------- 消息入口 ----------------

    def _is_bot_self(self, event: MessageEvent) -> bool:
        self_id = getattr(self.client, "self_id", 0) if self.client else 0
        return bool(self_id) and int(event.user_id) == int(self_id)

    def _command_prefixes(self) -> list[str]:
        """框架配的命令前缀（`/`、`#` …）—— 只维护一份口径。"""
        parser = getattr(self.cmds, "parser", None)
        prefixes = list(getattr(parser, "prefixes", None)
                        or getattr(self.cmds, "prefixes", None) or [])
        return [str(p) for p in prefixes if p]

    def _escape_targets(self) -> list[str]:
        """用户可能想「转义掉」的记号：命令前缀 + 本插件的结束符。

        两者都要算：`#` 在本部署里**既是命令前缀又是结束符**，而用户加反斜杠
        时的心理模型只是「这个符号别被当记号」。只认命令前缀的话，结束符单独
        配成别的（测试环境就是 `//`）时这套兜底就静默失效了。
        """
        out = self._command_prefixes()
        out += [str(t) for t in (self.cfg["trigger"].get("terminators") or []) if t]
        return out

    def _unescape_prefix(self, text: str) -> str:
        """脱掉行首那个「为了不当命令」而加的反斜杠：`\\#include` → `#include`。

        用户粘 C/C++ 代码时几乎必然自己补这个 `\\`（本部署里 `#` 既是命令前缀
        又是结束符）。可框架**不需要**它：`#include<bits/stdc++.h>` 不是合法
        命令名（`CommandParser._NAME_PATTERN` 只认标识符），命令分发本来就静默
        丢掉。留着反斜杠的后果是模型收到 `\\#include<bits/stdc++.h>`，转头答
        「第一行编译不过」—— 实测踩到，用户还以为是模型笨。

        只认「行首（允许前导空白）+ 一个反斜杠 + 紧随其后的记号」，
        别处一律不动：`$\\#$`、`\\dfrac` 这类 LaTeX 写法不会被误伤。
        """
        raw = text or ""
        stripped = raw.lstrip()
        if not stripped.startswith("\\"):
            return raw
        body = stripped[1:]
        if not any(body.startswith(p) for p in self._escape_targets()):
            return raw
        return raw[: len(raw) - len(stripped)] + body

    def _is_command_text(self, text: str) -> bool:
        """这条消息是不是**已注册命令**的调用。

        ⚠️ 两个坑都在这里踩过，别再退回去：

        1. 前缀挂在 `cmds.parser` 上，`CommandManager` 自己没有 `prefixes`
           属性 —— 取错对象会让判据恒为 False，于是 `/aichat list` 既被命令
           系统执行、又被当成对话内容塞进暂存区。
        2. **不能**用「首字符是命令前缀」当判据 —— 结束符 `//` 也以 `/` 开头，
           那样每次提交都会被当成命令丢掉（表现为：发了结束符，机器人毫无反应）。

        所以这里解析出命令名，再查它是否真的注册过。副作用是好的：
        `/help` 打错成 `/halp` 会被当成对话内容交给模型，比框架回一句
        「未知命令」更贴心。

        切分复用 `parser.skip_prefix()`：前缀长度、空白裁剪、集合归一化
        只维护一份，免得这里的判据和 `CommandManager.handle()` 的口径漂移。
        """
        parser = getattr(self.cmds, "parser", None)
        raw = (text or "").lstrip()
        if parser is not None and hasattr(parser, "skip_prefix"):
            rest = parser.skip_prefix(raw)
        else:                      # 兼容没有该方法的旧框架
            prefixes = list(getattr(parser, "prefixes", None)
                            or getattr(self.cmds, "prefixes", None) or [])
            hit = next((str(p) for p in sorted(prefixes, key=len, reverse=True)
                        if p and raw.startswith(str(p))), None)
            rest = raw[len(hit):].strip() if hit else None
        if not rest:
            return False
        head = rest.split()
        if not head:
            return False
        known = getattr(self.cmds, "_commands", {}) or {}
        return head[0].lower() in known

    async def _handle_message(self, event: MessageEvent):
        if not self.cfg.get("enabled", True):
            return
        if not isinstance(event, MessageEvent):
            return
        trigger = self.cfg["trigger"]
        if trigger["ignore_bot_self"] and self._is_bot_self(event):
            return

        target = self._target_of(event)
        if target.is_group:
            # 先判 @ 再判服务范围：未绑定的群不该因为「有人 @ 了机器人」就收到回绝说明，
            # 那等于把机器人的存在暴露给没开通服务的群。
            if trigger["group_require_at"] and not self._at_bot(event):
                return
        else:
            if not trigger["private"]:
                return

        # ⚠️ 从这里往下到内容落进暂存区，中间有若干次**带节流的** `get_msg`
        # （读引用、追引用链），所以先把自己标成「正在消化」—— 否则用户在这几秒里
        # 按下的结束符会看到空暂存区、回一句「暂存区是空的」，而内容随后入栈、
        # 永远卡住（根因与线上复现见 `_ingesting` 的注释）。
        ingest_key = target.conversation_key
        self._begin_ingest(ingest_key)
        try:
            text, media = await self._extract_text_async(event)
            if not text.strip() and not media:
                return
            if trigger["ignore_command_messages"] and self._is_command_text(text):
                # 命令系统自己会处理（群里还要求 @，这里的判据与它一致）
                return

            # ⚠️ 转义前缀要在剥结束符**之前**处理：用户写 `\#include…` 时那个 `\`
            # 是给命令系统看的（其实框架并不需要，见 `_unescape_prefix`），
            # 它绝不该出现在送给模型的正文里。
            text = self._unescape_prefix(text)
            # ⚠️ 结束符必须在**拼上引用内容之前**剥掉：结束符只可能出现在用户自己敲的
            # 那部分（引用正文是别人的历史消息）。先拼再剥的话，`//` 会被引文挤到
            # 中间，`_strip_terminator` 只认行尾 → 永远匹配不到 → 用户发的每一条
            # 都卡在暂存区不提交。
            text, finished, md_mode = self._strip_terminator(text)
            # 引用内容在**剥完结束符之后**才拼上去（见 _extract_text_async 的说明）
            text += await self._quoted_suffix(event)

            # 服务范围（白名单）—— 与权限等级无关，不做超级用户豁免。
            # ⚠️ 必须放在「是不是命令」的判定**之后**：管理服务范围本身的命令
            # （`/aichat add`/`del`/`targets` 等）必须能在未开通的私聊里执行，否则会死锁 ——
            # bind 只能在私聊用，而 users 为空时私聊又被这里挡住，
            # 永远开不出第一条私聊。
            if self._access_denied(target, int(event.user_id), text=text):
                return

            body = text
            if finished and not body.strip():
                # 用户只发了个结束符（没带正文）—— 三种情况要分清楚，不能一律
                # 回「暂存区是空的」：
                staged = self.buffer.peek(ingest_key).get("items")
                if not staged:
                    # 上一条可能**还在读引用**（框架给每次 API 调用加 1~5 秒节流，
                    # 引用链更久）—— 等它落进暂存区再判，别让用户白按一下。
                    staged = await self._await_pending_content(ingest_key)
                if not staged and ingest_key in self._inflight:
                    # 内容早就收到了、**正在回答**。这时说「暂存区是空的」会让用户
                    # 以为那段白发了（用户截图报的就是这个：他发完内容按了提交键，
                    # 等了几秒没动静又按一次，于是收到「先发内容」）。
                    await self._send(target, MessageChain().text(
                        "刚才那段已经收到了，正在回答中，稍等它说完～"
                        "（想接着补充就直接发内容，不用再按结束符。）"))
                    return
                if not staged:
                    await self._send(target, MessageChain().text(
                        "暂存区是空的：先发内容（群里需 @我），再发结束符提交。"))
                    return
                # 有暂存内容，这次只是来按提交键的 —— 直接提交，**不要**再往暂存区
                # 塞一个空段（空段会被当成一段真实输入）。
                self._schedule_submit(target)
                return

            self._stage(target, event, body, explicit_finish=finished, media=media,
                        md=md_mode)
        finally:
            self._end_ingest(ingest_key)

    async def _handle_recall(self, event: GroupRecallEvent):
        """撤回的消息要从暂存区删掉 —— 用户显然不想让它进上下文。"""
        if event.group_id is None or event.user_id is None:
            return
        key = ChatTarget("group", int(event.group_id)).key
        removed = self.buffer.drop_message(key, int(event.operator_id or 0),
                                           int(event.message_id or 0))
        if removed and self.cfg.get("log_verbose"):
            self.logger.info("[ai_chat] 撤回消息已从暂存区移除: %s", event.message_id)

    def _at_bot(self, event: MessageEvent) -> bool:
        self_id = getattr(self.client, "self_id", 0) if self.client else 0
        if not self_id:
            return True             # 还没拿到 self_id，放行（与框架口径一致）
        for seg in (event.message if isinstance(event.message, list) else []):
            if isinstance(seg, dict) and seg.get("type") == "at":
                qq = str((seg.get("data") or {}).get("qq", ""))
                if qq in (str(self_id), "all"):
                    return True
        return False

    #: 引用消息里算「多媒体」的类型 —— 本插件不做多模态，遇到就明确回绝
    #: 菊花链最多往上追几层。3 层足够覆盖「引用 + 接龙」的实际用法，
    #: 再多只是把一串半截话塞给模型，还多几次 get_msg 往返。
    _QUOTE_CHAIN_MAX = 3

    _MEDIA_SEGMENTS = frozenset({"image", "record", "video", "file", "forward",
                                 "json", "xml", "mface"})
    _MEDIA_LABELS = {"image": "图片", "record": "语音", "video": "视频",
                     "file": "文件", "forward": "合并转发", "json": "卡片",
                     "xml": "卡片", "mface": "表情"}

    async def _extract_text_async(self, event: MessageEvent) -> tuple[str, str]:
        """取这条消息的**自身文本**与引用情况，返回 `(自身文本, 多媒体类型名)`。

        ⚠️ 只返回「用户自己敲的那部分」：结束符只可能出现在这里，所以**必须**由
        调用方先剥结束符、再拼引用内容。曾经把引用内容在这里就拼进去，结果
        `//` 被引文挤到中间、`_strip_terminator`（只认行尾）永远匹配不到 ——
        表现为「引用了消息之后，怎么发都不提交」。

        ⚠️ 这里也**不回绝、不发消息**：输入要先经暂存区（用户可能还分几段发），
        回绝必须发生在**提交那一刻**，否则会和「结束符才提交」的协议错位。

        引用内容由调用方通过 `_quoted_suffix(event)` 单独取。
        """
        return self._extract_text(event), await self._reply_media(event)

    async def _reply_media(self, event: MessageEvent) -> str:
        """引用的那条是不是多媒体；是则回类型名（图片/语音/…），否则空串。

        ⚠️ 机器人**自己**发的图不算 —— 那种情况原文能找回来（见 `_own_original`），
        正文由 `_quoted_suffix` 补上，不该再回一句「暂不支持多模态」。
        """
        reply_id = self._reply_message_id(event)
        if reply_id is None:
            return ""
        _, media, _ = await self._read_quoted(reply_id)
        if media and self._own_original(reply_id) is not None:
            return ""
        return media

    def _own_original(self, message_id: int) -> Optional[str]:
        """机器人**自己**发的那条消息，当时生成的原文。

        用户要求：引用 AI 的回答时，含图片的用原文替代。因为 Markdown 模式的整篇
        回答是一张图、函数图与公式图也都是图 —— 它们在 QQ 那边**没有文字**，
        `get_msg` 读回来只有「一张图片」。可那段内容本来就是我们自己生成的，
        原文就躺在会话历史里（`SentReplyIndex` 记着它在哪一条）。

        不是自己发的、或索引里没有（重启前发的、超出保留条数）→ 返回 None，
        调用方退回原来的「不支持多模态」口径。
        """
        got = self.sent_reply.get(message_id)
        if got is None:
            return None
        sid, seq = got
        msg = self.store.find(sid, seq)
        text = (getattr(msg, "text", "") or "") if msg is not None else ""
        return text.strip() or None

    async def _quoted_suffix(self, event: MessageEvent) -> str:
        """把引用的内容拼成「引用上文」段落，**沿着菊花链一直往上读**。

        「菊花链」= 被引用的那条消息自己也是引用别人的（群聊里接龙式对话、以及
        用户引机器人回答、机器人回答里本来又带着上一轮的引用）。只读一层的话，
        模型看到的是半截话。这里最多往上追 `_QUOTE_CHAIN_MAX` 层，
        并且用 `seen` 挡住环（QQ 里理论上不该出现，真出现了也不能转不出来）。
        """
        cur = self._reply_message_id(event)
        if cur is None:
            return ""
        out: list[str] = []
        seen: set[int] = set()
        depth = 0
        while cur is not None and depth < self._QUOTE_CHAIN_MAX and cur not in seen:
            seen.add(cur)
            text, media, nested = await self._read_quoted(cur)
            if media:
                # 机器人自己发的图 → 用当时生成的原文替代（用户要求）
                own = self._own_original(cur)
                if own is None:
                    # 别人的图/语音：读不到内容，但**不能因此丢掉用户的问题** ——
                    # 留一句提示占住引文的位置，用户写的正文照常提交
                    out.append(f"\n[引用上文] 对方引用的是一条{media}"
                               f"（暂不支持多模态，读不到里面的内容）。")
                    break
                text = f"（这是机器人自己发的{media}，以下是它当时生成的原文）\n{own}"
            if not text:
                if depth == 0:
                    out.append("\n（引用的那条消息读不到内容）")
                break
            if depth == 0:
                out.append(f"\n[引用上文] 对方引用的那条消息内容如下：\n{text}")
            else:
                out.append(f"\n[引用上文] 上面那条消息自己又引用了：\n{text}")
            cur = nested
            depth += 1
        return "".join(out)

    @staticmethod
    def _reply_message_id(event: MessageEvent) -> Optional[int]:
        """消息里 `reply` 段指向的消息 id。"""
        if not isinstance(event.message, list):
            return None
        for seg in event.message:
            if isinstance(seg, dict) and seg.get("type") == "reply":
                data = seg.get("data") or {}
                raw = data.get("id") or data.get("message_id") or data.get("seq")
                try:
                    return int(raw)
                except (TypeError, ValueError):
                    return None
        return None

    async def _read_quoted(self, message_id: int) -> tuple[str, str, Optional[int]]:
        """读引用消息的内容 → `(文本, 多媒体类型名, 它自己引用的消息 id)`。

        文本与多媒体互斥：拿到文本返回 `(文本, "")`；引用的是图片/语音等返回
        `("", "图片")`。读不到（被撤回、超出保留范围、接口失败）返回 `("", "", None)`。

        第三个值是**菊花链的下一环** —— 被引用的那条自己可能也是引用别人的，
        调用方拿它继续往上读（见 `_quoted_suffix`）。
        """
        if self.api is None:
            return "", "", None
        try:
            data = await self.api.get_msg(int(message_id))
        except Exception as e:  # noqa: BLE001 - 读不到不该让整轮对话失败
            self.logger.warning("[ai_chat] 读取引用消息 %s 失败: %s",
                                message_id, err_text(e))
            return "", "", None
        msg = data
        if isinstance(data, dict) and isinstance(data.get("message"), (list, str)):
            msg = data["message"]
        elif isinstance(data, dict) and isinstance(data.get("data"), dict):
            inner = data["data"]
            msg = inner.get("message", inner)
        if isinstance(msg, str):
            if "[CQ:image" in msg or "[CQ:record" in msg or "[CQ:video" in msg:
                return "", "多媒体内容", None
            return msg.strip(), "", None
        if not isinstance(msg, list):
            return "", "", None
        parts: list[str] = []
        media = ""
        nested: Optional[int] = None
        sender = ""
        if isinstance(data, dict):
            sd = data.get("sender")
            if isinstance(sd, dict):
                sender = str(sd.get("card") or sd.get("nickname") or "")
        for seg in msg:
            if not isinstance(seg, dict):
                continue
            stype = str(seg.get("type") or "")
            sdata = seg.get("data") or {}
            if stype == "text":
                parts.append(str(sdata.get("text", "")))
            elif stype == "at":
                qq = str(sdata.get("qq", ""))
                if qq and qq != "all":
                    parts.append(f"@{qq}")
            elif stype == "reply":
                # 菊花链的下一环：这条消息自己引用了谁
                raw = sdata.get("id") or sdata.get("message_id") or sdata.get("seq")
                try:
                    nested = int(raw)
                except (TypeError, ValueError):
                    nested = None
            elif stype in self._MEDIA_SEGMENTS and not media:
                media = self._MEDIA_LABELS.get(stype, "多媒体内容")
        body = "".join(parts).strip()
        if body:
            # 同时带图带字：正文可用，把图的存在说清楚（不丢信息也不越界）
            note = f"（其中含{media}，未读取）" if media else ""
            return (f"{sender}：{body}{note}" if sender else f"{body}{note}"), "", nested
        if media:
            return "", media, nested
        return "", "", nested

    def _extract_text(self, event: MessageEvent) -> str:
        """把消息段转成给模型看的文本（只支持文字；其它类型转成说明）。"""
        message = event.message
        if isinstance(message, str):
            return message
        self_id = str(getattr(self.client, "self_id", 0) or "")
        parts: list[str] = []
        for seg in message or []:
            if not isinstance(seg, dict):
                continue
            stype = seg.get("type")
            data = seg.get("data") or {}
            if stype == "text":
                parts.append(str(data.get("text", "")))
            elif stype == "at":
                qq = str(data.get("qq", ""))
                if qq and qq not in (self_id, "all"):
                    parts.append(f"@{qq}")
            elif stype == "image":
                parts.append("[图片]")
            elif stype == "record":
                parts.append("[语音]")
            elif stype == "video":
                parts.append("[视频]")
            elif stype == "file":
                parts.append(f"[文件 {data.get('file', '')}]")
            elif stype == "face":
                parts.append("[表情]")
            elif stype == "reply":
                pass          # 引用内容由 _extract_text_async 单独读取并标注
            elif stype in ("json", "xml"):
                parts.append("[卡片消息]")
            elif stype == "forward":
                parts.append("[合并转发]")
        return "".join(parts)

    def _strip_terminator(self, text: str) -> tuple[str, bool, bool]:
        """剥离行尾结束符，返回 `(正文, 是否看到结束符, 是否 Markdown 模式)`。

        结束符 = 触发开关，**不是**命令。可以单独发，也可以粘在正文末尾
        （`你好//`、`你好 //`、`你好\\n//` 都算）。冒号等全角引号变体会一起认，
        免得用户被输入法坑。末尾的多余空行顺手清掉（粘贴时很常见）。

        **Markdown 模式**：结束符后面紧跟 `md`（配置 `trigger.markdown_suffix`）
        就进入 —— `你好//md`、`你好 //md` 都算。识别放在这里而不是命令系统里，
        因为它是结束符的**修饰**：`//` 仍是唯一的提交开关，`md` 只改变输出形态
        （回答渲染成图片）。
        """
        trigger = self.cfg["trigger"]
        terminators = trigger.get("terminators") or []
        suffix = str(trigger.get("markdown_suffix") or "").strip()
        body = text.rstrip()
        if not body:
            return text, False, False
        for term in terminators:
            term = str(term)
            if not term:
                continue
            variants = {term, term.replace('"', "”"), term.replace("/", "／")}
            for variant in variants:
                marker = variant + suffix if suffix else variant
                # ⚠️ 不能要求 `body.endswith(variant)`：用户写的是 `你好//md`，
                # 整串的结尾是 `md` 而不是 `//` —— 那样判会把 Markdown 模式
                # 整个漏掉（实测：用户看到的一直是纯文本）。判的是**候选尾巴**：
                # 带后缀的（`//md`）优先，再退回裸结束符（`//`）。
                for tail, is_md in ((marker, True), (variant, False)):
                    if not tail or not body.endswith(tail):
                        continue
                    head = body[: len(body) - len(tail)].rstrip()
                    return head, True, is_md
                # 后缀大小写变体（`//MD`）
                if suffix and body.endswith(variant + suffix.upper()):
                    cut = body[: len(body) - len(variant) - len(suffix)].rstrip()
                    return cut, True, True
        return text, False, False

    # ---------------- 暂存（多段发送） ----------------

    def _begin_ingest(self, key: str) -> None:
        """标记「这条会话正在消化一条消息」（见 `_ingesting` 的说明）。"""
        self._ingesting[key] = int(self._ingesting.get(key, 0)) + 1

    def _end_ingest(self, key: str) -> None:
        left = int(self._ingesting.get(key, 0)) - 1
        if left > 0:
            self._ingesting[key] = left
        else:
            self._ingesting.pop(key, None)

    async def _await_pending_content(self, key: str) -> list:
        """等**别的**消息落进暂存区，返回等到的那一刻暂存区里的条目。

        为什么要等：读引用要走 `get_msg`，而框架给**每次** API 调用都加了
        `api_delay`（线上 1~5 秒）并串行化 —— 引用一条消息、再追它的引用链，
        几秒钟很正常。用户「发内容 → 紧接着按结束符」时，结束符那条消息完全
        可能先跑到这里，此时暂存区**确实是空的**，但那不代表用户没发内容。
        线上实测（2026-10-09 21:40）：内容 21:40:39 才入暂存区，用户 21:40:38
        按的 `#` → 回了「暂存区是空的」，而那段内容随后入栈、再也没被提交。

        判据是 `_ingesting > 1`：结束符**自己**也算一次消化，所以「>1」= 还有
        别人在往里塞。别人一旦消化完（`_end_ingest`）就立刻停止等待 —— 空等到
        超时才说「暂存区是空的」，比早说更糟。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self._INGEST_WAIT_MAX)
        while True:
            items = self.buffer.peek(key).get("items") or []
            if items:
                return items
            if int(self._ingesting.get(key, 0)) <= 1:
                return []
            if loop.time() >= deadline:
                return []
            await asyncio.sleep(0.1)

    def _stage(self, target: ChatTarget, event: MessageEvent, body: str, *,
               explicit_finish: bool, media: str = "", md: bool = False) -> None:
        trigger = self.cfg["trigger"]
        item = {
            "user_id": int(event.user_id),
            "nickname": self._display_name(event),
            "card": (getattr(event.sender, "card", "") or "") if event.sender else "",
            # OneBot 每条消息都带发言人在群里的角色 —— 工具权限要用它，
            # 见 `_tool_context` 里关于缓存不可靠的说明
            "role": _role_value(getattr(event.sender, "role", "")) if event.sender else "",
            "text": body or "",
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "local_time": datetime.now().astimezone().strftime("%H:%M"),
            "message_id": int(event.message_id or 0),
            "group_id": int(event.group_id) if event.group_id else 0,
            # 这一段的引用目标是不是多媒体（提交时才回绝，见 _extract_text_async）
            "media": media,
            # 这一段是不是用 `//md` 提交的（Markdown 模式：回答渲染成图片）
            "md": bool(md),
        }
        state = self.buffer.append(target.conversation_key, item)
        finish = explicit_finish or self._overflow(state)
        if self.cfg.get("log_verbose"):
            self.logger.info("[ai_chat] 暂存 %s：%d 段%s", target.conversation_key,
                             len(state.get("items") or []),
                             " → 提交" if finish else "")
        if finish:
            self._schedule_submit(target)

    def _overflow(self, state: dict) -> bool:
        """暂存区超限 → 自动提交**已有内容**（不丢用户已发出的东西）。"""
        trigger = self.cfg["trigger"]
        items = state.get("items") or []
        if len(items) > int(trigger["max_staged_parts"]):
            return True
        total = sum(len(str(it.get("text") or "")) for it in items)
        return total > int(trigger["max_staged_chars"])

    def _schedule_submit(self, target: ChatTarget) -> None:
        key = target.conversation_key
        if key in self._inflight:
            # 上一轮还在跑 —— **记下来**，跑完接着提交。
            # ⚠️ 不能直接 return：那样用户「发内容 + 发结束符」之后什么都等不到，
            # 而后台连一行日志都没有（实测复现：模型调用次数一直是 0，
            # 内容永远卡在暂存区里，直到用户再按一次提交键）。
            self._pending_submit.add(key)
            return
        items = self.buffer.take(key)
        if not items:
            return
        self._inflight.add(key)
        task = asyncio.get_event_loop().create_task(
            self._submit_guarded(target, items))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _compose_input(self, items: list[dict]) -> str:
        """把多段输入合成一次用户输入 —— **直接按换行拼接，不加任何标记**。

        分段只是因为 QQ 单条消息有长度上限，用户把一大段话拆成几条发；
        对模型来说它本来就是**一整段输入**，所以不该塞 `(1/N)` `(2/N)` 这种序号：
        那些字符既不是用户写的，又会诱导模型在回复里模仿同样的编号
        （提示词里专门叮嘱过「不要写 (1/3) 这类分段标记」，这里更不该由我们自己造）。
        段落之间的换行保留 —— 那正是用户实际敲下的分段位置。
        """
        texts = [str(it.get("text") or "").strip() for it in items]
        texts = [t for t in texts if t]
        return "\n".join(texts)

    def _compose_meta(self, items: list[dict]) -> dict:
        first = items[0]
        last = items[-1]
        return {
            "user_id": int(last.get("user_id") or 0),
            "nickname": str(last.get("nickname") or ""),
            "card": str(last.get("card") or ""),
            "role": str(last.get("role") or ""),
            "local_time": str(last.get("local_time") or ""),
            "ts": str(first.get("ts") or ""),
            "message_ids": [int(it.get("message_id") or 0) for it in items],
            "group_id": int(last.get("group_id") or 0),
            #: 这一段输入由几条消息拼成。**只是诊断信息**（落进会话 jsonl，
            #: 排查「用户说发了却没人回」这类问题时靠它），不投影给模型 ——
            #: 用户看不到序号，模型也不该看到。
            "parts": len(items),
            # 本次输入是否 Markdown 模式：任一段带了 md 就算
            "md": any(bool(it.get("md")) for it in items),
        }

    # ---------------- 一轮对话 ----------------

    async def _submit_guarded(self, target: ChatTarget, items: list[dict]) -> None:
        key = target.conversation_key
        try:
            # ⚠️ 引用的是图片/语音时**不再整轮回绝**。用户要的是「把能提交的
            # 文字部分提交上去」：他写的问题本身是好的，只是引文读不到 ——
            # 那句「暂不支持多模态」现在作为**提示**跟在引文位置（见 `_quoted_suffix`），
            # 而不是把整条输入扔掉。原来那样做，用户问「这张图里是什么」会被直接
            # 打回，连问题都没进模型。
            text = self._compose_input(items)
            if not text:
                return
            meta = self._compose_meta(items)
            event = _FakeEvent(target, meta, text)
            session = self._session_for(target, event)
            if session is None:
                # 会话不再自动创建，所以这里的提示必须能一次说清「下一步做什么」：
                # 群里如果已经有群会话，多数人其实是要「切过去」而不是「新建一个」。
                await self._send(target, MessageChain().text(
                    self._no_session_hint(target, int(meta["user_id"]))))
                return
            ok, why = self._may_attach(session, target, int(meta["user_id"]))
            if not ok:
                await self._send(target, MessageChain().text(f"⚠️ {why}"))
                return
            async with self._lock(session.id):
                await self._run_turn(target, session, text, meta)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - 后台任务里绝不能有未捕获异常
            # 兜底路径：这里出错说明是**本插件自己的 bug**（正常的上游失败在
            # `_run_turn` 里已经分类处理）。反馈统一走框架的 ErrorReporter：
            # 带错误编号（用户报编号、我们 grep 日志就能对上），群里会 @ 提问的人。
            try:
                await self._send(target, self._error_chain(
                    target, int(items[0].get("user_id") or 0), e, where="AI 对话",
                    note=f"会话键 {key}"))
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._inflight.discard(key)
            if key in self._pending_submit:
                # 这一轮跑的时候用户又提交过 —— 现在补上（比如「发内容 + 结束符」
                # 正好撞上上一轮还没说完）
                self._pending_submit.discard(key)
                self._schedule_submit(target)

    async def _run_turn(self, target: ChatTarget, session: Session, text: str,
                        meta: dict) -> None:
        """执行一轮：追加用户消息 → 模型/工具循环 → 发送回复。"""
        started = time.time()
        message = user_message(
            text,
            origin=("user.group" if target.is_group else "user.private"),
            user_id=int(meta.get("user_id") or 0),
            nickname=str(meta.get("nickname") or ""),
            card=str(meta.get("card") or ""),
            local_time=str(meta.get("local_time") or ""),
            group_id=int(meta.get("group_id") or 0),
            message_ids=meta.get("message_ids") or [],
            parts=int(meta.get("parts") or 1),
        )
        self.store.append(session.id, message)

        history = self.store.history(session.id)
        history, dropped = _sanitize(history)
        if dropped:
            self.logger.warning("[ai_chat] 会话 %s 丢掉 %d 条无法配对的历史前缀",
                                session.id, dropped)
            self.store.replace_history(session.id, history)
        #: 本轮开始时的消息条数 —— 跑完模型循环后用它圈出**本轮新增**的消息，
        #: 从里面把画图工具产出的图片捞出来（见 `_plots_from_tools`）。
        history_len = len(history)

        # 一轮开始：重置「本轮配了几张网图」的计数（上限见 web_image.max_per_turn）
        self.webimg.begin_turn()

        usage = Usage()
        ctx = self._tool_context(target, session, meta)
        system_prompt = self._system_prompt(
            target, session, meta, ctx,
            multi_user=self._is_multi_user(session, history),
            md=bool(meta.get("md")))
        comp_cfg = self.cfg["compaction"]

        # 进模型之前先按压力压缩（不等上游报超长）
        if comp_cfg["enabled"]:
            report = await self.compactor.should_and_compact(
                session.id, history, store=self.store,
                system_tokens=len(system_prompt) / self.policy.chars_per_token,
                session_name=session.name, role=session.role)
            if report:
                history = report["history"]
                usage.compactions += 1
                usage.note(f"压缩：吸收 {report['absorbed']} 条")

        try:
            reply_text, usage = await self._model_loop(
                target, session, ctx, system_prompt, history, usage)
        except Exception as e:  # noqa: BLE001 - 失败要让用户看见原因
            # 上游/工具失败的统一出口（框架 ErrorReporter）：日志带错误编号，
            # 用户拿到「分类 + 短原因 + 编号」，群里还会 @ 提问的人 —— 以前这里
            # 只发一句 `error_prefix + 异常原文`，群里没 @ 谁，谁也不知道该找谁。
            hint = self._failure_hint(e)
            # 流式被掐断时上游往往已经吐了大半 —— 「答了一半」比「什么都没有」
            # 有用得多，所以先把半截发出去，再补一句它是半截。
            partial = str(getattr(e, "partial", "") or "").strip()
            if partial:
                quote_id = meta.get("message_ids", [None])[-1]
                self._sent_ids.pop(target.key, None)
                await self._send(target, MessageChain().text(
                    f"（上游中断了：{err_text(e)}。以下是已经生成的部分）"))
                await self._deliver_reply(target, session, partial,
                                          quote_id=quote_id, plots=[],
                                          markdown=bool(meta.get("md")))
                self._remember_sent(session, target, partial)
                return
            chain = self._error_chain(target, int(meta.get("user_id") or 0), e,
                                      where="AI 对话", detail=hint,
                                      note=f"会话 {session.id}")
            await self._send(target, chain)
            return

        # 图片从**工具结果**里取，不从模型的正文里取（见 `_plots_from_tools`）。
        # 放在这里而不是 `_send_plots` 里，是因为要拿「本轮新增的消息」——
        # `store.append` 只加不改，所以数一下长度差就够了。
        plot_images = self._plots_from_tools(session, since=history_len)

        usage.elapsed = time.time() - started
        if self.cfg.get("log_verbose"):
            self.logger.info("[ai_chat] 会话 %s 完成：%s", session.id, usage.summary())
        pending = self.buffer.peek(target.conversation_key)
        if pending.get("items"):
            # 对话进行期间又来了新段落 → 提示结束符还在等（否则用户以为没反应）
            await self._send(target, MessageChain().text(
                "（刚才那段已回复；暂存区里还有 %d 段新内容，发结束符继续）"
                % len(pending["items"])))
        quote_id = meta.get("message_ids", [None])[-1]
        # 清空本轮 id 桶再发：`_send` 会把这一轮真正发出去的消息 id 记进去，
        # 发完就能建立「消息 id → 本轮回答的原文」这层映射（引用图片时要用）。
        self._sent_ids.pop(target.key, None)
        await self._deliver_reply(target, session, reply_text,
                                  quote_id=quote_id, plots=plot_images,
                                  markdown=bool(meta.get("md")))
        self._remember_sent(session, target, reply_text)

    def _error_chain(self, target: ChatTarget, user_id: int, exc: BaseException,
                     *, where: str, detail: str = "", note: str = "") -> MessageChain:
        """把失败渲染成回给用户的消息：框架文案（带错误编号）+ 群里 @ 提问的人。

        为什么不用 `self.logger.error` 自己拼：错误文案、编号、节流是框架的
        `ErrorReporter` 管的（见 `napcat/errors.py`），插件各拼一份就会出现
        「同一个失败在群里以 N 种写法出现、没有编号、没法 grep 日志、不 @ 人」。
        投递仍由本插件自己做 —— 它才知道要发到哪个会话、要不要渲染成图片。
        """
        error = exc if isinstance(exc, NapCatError) else PluginRuntimeError(
            short_reason(exc), plugin=self.name, cause=exc)
        self.errors.log(error, plugin=self.name, what=where, note=note)
        text = self.errors.render(error, plugin=self.name, what=where)
        if detail:
            text += f"\n{detail}"
        chain = MessageChain()
        if target.is_group and user_id:
            # 群里 @ 触发的人：他才知道自己刚才问了什么、要不要重发
            chain.at(str(user_id)).text(" ")
        return chain.text(text)

    def _failure_hint(self, e: BaseException) -> str:
        """失败原因后面补一句**能照做**的说明（取不到就返回空串）。

        为什么要有：模型「想太久」时用户看到的只有 `LlmError: TimeoutError` ——
        他不知道是网断了、额度没了还是问题太难，也不知道自己能做什么。
        实测这类失败几乎都是**上游思考链太长**（一个 C++ 报错分析能让
        `deepseek-v4-flash` 连想 200 秒以上），所以提示指向「拆小问题 / 调大
        超时」这两条真正有用的动作。

        ⚠️ 流式之后（见 `llm._chat_stream`）「卡死」和「想得久」已经分开了：
        前者是 `stall_timeout` 内没有任何新数据，后者只是慢。所以这里要按**是哪一种**
        给不同的话 —— 说「拆小问题」对真正的网络卡顿毫无帮助。
        """
        msg = err_text(e).lower()
        if not any(k in msg for k in ("timeout", "timed out", "504", "超时",
                                      "停滞", "上限")):
            return ""
        cfg = self.cfg["llm"]
        tries = max(0, int(_coerce_int(cfg.get("max_retries"), 2)))
        tail = f"，已重试 {tries} 次" if tries else ""
        if "停滞" in err_text(e):
            stall = _coerce_float(cfg.get("stall_timeout"), 60.0)
            return (f"（上游 {stall:.0f} 秒没吐任何数据{tail}，像是它那边卡住了。"
                    "再发一次通常就好；连续出现就把 ai_chat.llm.stall_timeout 调大）")
        secs = _coerce_float(cfg.get("timeout"), 120.0)
        # ⚠️ 总时长上限**不重试**（同样的时间再跑一遍还是同样的结果），所以别写
        # 「已重试 N 次」—— 那不是它身上发生的事。判据看异常上的 `retryable`，
        # 不按文案猜（老的 TimeoutError 路径仍然是会重试的）。
        tried = ("（没有再重试：同样的输入再跑一遍还是同样的时间）"
                 if getattr(e, "retryable", True) is False else tail)
        return (f"（超过单次 {secs:.0f} 秒的上限{tried}；"
                "先把已经生成的部分发给你了。这类问题拆小一点问更快，"
                "也可以调大 ai_chat.llm.timeout）")

    def _last_reply_text(self, session: Session) -> str:
        """会话里**最后一条**有正文的 AI 回复（`/md` 用）。"""
        for m in reversed(self.store.history(session.id)):
            if m.role == ROLE_ASSISTANT:
                text = (m.text or "").strip()
                if text:
                    return text
        return ""

    def _last_reply_seq(self, session: Session) -> int:
        for m in reversed(self.store.history(session.id)):
            if m.role == ROLE_ASSISTANT and (m.text or "").strip():
                return int(m.seq or 0)
        return 0

    def _remember_sent(self, session: Session, target: ChatTarget,
                       reply_text: str) -> None:
        """把本轮发出去的消息 id 记进索引，指向**本轮回答**那条会话消息。

        指向消息本身而不是拷一份原文：原文已经在会话 jsonl 里了。
        """
        ids = self._sent_ids.pop(target.key, [])
        if not ids or not str(reply_text or "").strip():
            return
        seq = 0
        for m in reversed(self.store.history(session.id)):
            if m.role == ROLE_ASSISTANT and m.text.strip():
                seq = int(m.seq or 0)
                break
        if seq:
            self.sent_reply.add(ids, session.id, seq)

    async def _deliver_reply(self, target: ChatTarget, session: Session, text: str, *,
                             quote_id: Optional[int] = None,
                             plots: Optional[list[str]] = None,
                             markdown: bool = False) -> None:
        """按模式把回复发出去。四条路互斥，优先级写在下面。

        单独抽成方法是为了**可测**：这段「发什么、发几条」的决策原先散在
        `_run_turn` 里，只能靠端到端用例覆盖，而它恰恰是最容易漏一条支路的地方
        （Markdown 模式下函数图像一度被整个丢掉）。
        """
        if markdown:
            # Markdown 模式下正文整段出图，但**函数图像仍要单独发**：它是另一张图，
            # 只把 Markdown 那张发出去会让用户以为画图没生效（工具明明成功调用了）。
            await self._send_markdown(target, session, text, quote_id=quote_id)
            if plots:
                await self._send_plot_images(target, plots, quote_id=quote_id)
            return
        if await self._send_plots(target, session, text, quote_id=quote_id,
                                  images=plots):
            return
        if await self._send_rich(target, session, text, quote_id=quote_id):
            return
        # 没有表格/公式 → 普通纯文本路径
        await self._send_text(target, text, quote_id=quote_id)

    def _take_plot_images(self, text: str) -> tuple[list[str], str]:
        """把回复里的画图标记换成图片路径，返回 `(图片路径, 去掉标记的正文)`。

        **只认本实例真的渲染过的路径**：标记是工具返回的，但正文是模型生成的 ——
        模型完全可以（被用户或网页内容诱导）自己写一个 `[[plot:/etc/passwd]]`。
        拿模型给的字符串去当图片发，等于把任意文件读出来发进群里。所以这里对着
        `self.plots.rendered` 核对，只放行确实由本插件产出过的文件。

        ⚠️ 不额外判 `os.path.exists`：渲染器会按 `keep_files` 定期清理旧图，
        判存在会让「图被清理掉了」变成**静默丢图**（正文照发、图不见了）。
        渲染成功那一刻文件一定在，路径也是我们自己记下的；真出了意外，
        `_send` 会报错，比静默丢掉更容易发现。
        核对通过的路径**不移除**记录：同一张图在一条回复里引用两次是合理的。
        """
        found = extract_markers(text or "")
        if not found:
            return [], text or ""
        allowed: list[str] = []
        for path, _title in found:
            if path in self.plots.rendered:
                if path not in allowed:
                    allowed.append(path)
            elif self.cfg.get("log_verbose"):
                self.logger.warning("[ai_chat] 忽略不是本插件产出的图片标记: %s", path)
        return allowed, strip_markers(text or "")

    def _plots_from_tools(self, session: Session,
                          since: int = 0) -> list[str]:
        """从**工具结果**里取出本轮生成的图片路径。

        为什么不从模型正文里取（原先的做法，实测在真实模型上完全不工作）：
        标记是工具写在返回文本里的，正文是模型自己生成的 —— 模型**不会原样转述**
        工具结果，它会用自己的话概括（「画好了，图在下面」），标记就此消失。
        实测会话记录：

            [tool]  RESULT:'…[[plot:/…/h-xxx.png|一次函数 y = 2x + 1]]'
            [assistant] TEXT:'画好了，是一条过 (0, 1) 的直线…'   ← 标记没了

        图和标记都在我们自己的工具结果里，那就从这里取 —— 模型说什么都不影响。

        `since` 是本轮开始时的消息条数：只看新增的那些，免得把**上一轮**的图
        又发一遍（`store.append` 只加不改，数长度差就够）。
        """
        out: list[str] = []
        for msg in self.store.history(session.id)[since:]:
            for block in msg.to_dict().get("blocks") or []:
                if block.get("type") != "tool_result":
                    continue
                # block 的字段名是 `text`（见 `segments.tool_message`），不是
                # OpenAI 那套 `content` —— 写错就是静默取不到图（实测踩过）
                body = block.get("text") or ""
                for path, _title in extract_markers(str(body)):
                    # 白名单：只放行**我们自己产出过**的文件（画图或下载），
                    # 模型编一个 [[plot:/etc/passwd]] 在这里就被挡掉
                    if path in out:
                        continue
                    if path in self.plots.rendered or path in self.webimg.rendered:
                        out.append(path)
        return out

    async def _send_plots(self, target: ChatTarget, session: Session, text: str, *,
                          quote_id: Optional[int] = None,
                          images: Optional[list[str]] = None) -> bool:
        """带了函数图像就发图（正文按普通模式发），否则返回 `False`。

        图**单独一条消息**发：QQ 的图片节点在正文里插队会让文字断成两截，
        而函数图本来就是「附在回复后面」的东西。

        `images` 由调用方从工具结果里取好传进来（见 `_plots_from_tools`）；
        正文里若**也**留着标记（模型偶尔会照抄），一并去掉，避免两个来源重复发图。
        """
        from_text, body = self._take_plot_images(text)
        if images is None:
            images = from_text
        else:
            # 工具结果与正文都提到同一张图时只发一次；正文里的标记仍然要清掉
            images = list(dict.fromkeys(images + from_text))
        if not images:
            return False
        if body.strip():
            # 正文照旧走富文本/纯文本那条路（表格公式该出图还是要出图）
            if not await self._send_rich(target, session, body, quote_id=quote_id):
                await self._send_text(target, body, quote_id=quote_id)
        await self._send_plot_images(
            target, images,
            quote_id=quote_id if not body.strip() else None)
        return True

    async def _send_plot_images(self, target: ChatTarget, images: list[str], *,
                                quote_id: Optional[int] = None) -> None:
        """把若干张已渲染好的图作为**单独一条**消息发出去。"""
        if not images:
            return
        chain = MessageChain()
        if self.cfg["reply"]["quote"] and quote_id:
            chain.reply(seq=int(quote_id))
        for path in images:
            chain.image(file=Path(path).as_uri(), name=os.path.basename(path))
        await self._send(target, chain)

    async def _send_markdown(self, target: ChatTarget, session: Session,
                             text: str, *, quote_id: Optional[int] = None) -> None:
        """Markdown 模式：把回答渲染成图片发出（渲染失败退回纯文本）。

        **为什么失败要退回而不是只报错**：用户已经等到回答了，此时因为
        「没装浏览器」把内容丢掉是最糟的结果 —— 内容本身没问题，只是形态退化。
        所以先明确说一句这是降级，再把原文按普通模式发出去。
        """
        body = (text or "").strip()
        if not body:
            await self._send_text(target, body, quote_id=quote_id)
            return
        title = f"{session.name} · Markdown"
        try:
            path = await self.renderer.render_markdown(body, title=title)
        except RenderError as e:
            self.logger.warning("[ai_chat] Markdown 渲染失败，退回纯文本: %s", e)
            await self._send(target, MessageChain().text(
                f"（Markdown 出图失败，下面是原文：{e}）"))
            await self._send_text(target, body, quote_id=quote_id)
            return
        except Exception as e:  # noqa: BLE001 - 渲染是外部进程，什么都可能
            self.logger.error("[ai_chat] Markdown 渲染异常: %s\n%s",
                              err_text(e), _tb())
            await self._send(target, MessageChain().text(
                f"（Markdown 出图异常，下面是原文：{err_text(e)}）"))
            await self._send_text(target, body, quote_id=quote_id)
            return
        chain = MessageChain()
        if self.cfg["reply"]["quote"] and quote_id:
            chain.reply(seq=int(quote_id))
        chain.image(file=Path(path).as_uri(), name=os.path.basename(path))
        await self._send(target, chain)

    async def _chat_with_retry(self, messages, *, tools, reasoning,
                               target: Optional[ChatTarget] = None):
        """调模型；**可重试的失败**按配置自动重试，其余立刻抛出。

        重试放在**单次模型调用**这一层，而不是整轮对话：一轮里可能已经跑过工具
        （搜索、画图），整轮重来会把这些副作用再做一遍；而失败几乎总是「这一次
        HTTP 没成」，重发这一次就够了，工具结果与历史都原封不动。

        上层还有一条**上下文超长**的重试（压缩后重来），那条路要换掉整个请求体，
        所以这里遇到超长错误**直接抛出**，不掺和。
        """
        cfg = self.cfg["llm"]
        tries = max(0, int(_coerce_int(cfg.get("max_retries"), 2)))
        backoff = max(0.0, _coerce_float(cfg.get("retry_backoff"), 1.5))
        cap = max(0.0, _coerce_float(cfg.get("retry_max_wait"), 8.0))
        notice = bool(cfg.get("retry_notice", True))
        told = False
        for attempt in range(tries + 1):
            try:
                return await self.llm.chat(messages, tools=tools, reasoning=reasoning)
            except Exception as e:  # noqa: BLE001 - 分类交给 _is_retryable
                if attempt >= tries or not _is_retryable(e) \
                        or _looks_like_context_overflow(e):
                    raise
                wait = min(backoff * (2 ** attempt), cap) if backoff else 0.0
                self.logger.warning(
                    "[ai_chat] 模型调用失败（第 %d/%d 次重试，%.1fs 后）：%s",
                    attempt + 1, tries, wait, err_text(e))
                if notice and not told and target is not None:
                    told = True
                    try:
                        await self._send(target, MessageChain().text(
                            f"⚠️ 上游暂时不通，正在自动重试（最多 {tries} 次）…"))
                    except Exception:  # noqa: BLE001 - 提示发不出去不该影响重试
                        pass
                if wait:
                    await asyncio.sleep(wait)
        raise AssertionError("unreachable")     # pragma: no cover

    async def _model_loop(self, target: ChatTarget, session: Session, ctx: ToolContext,
                          system_prompt: str, history: list[Message],
                          usage: Usage) -> tuple[str, Usage]:
        """模型 ↔ 工具循环，返回最终要发给用户的文本。"""
        reply = self.cfg["reply"]
        comp_cfg = self.cfg["compaction"]
        max_rounds = max(1, int(reply["max_rounds"]))
        retries_left = max(0, int(comp_cfg["max_context_retries"]))
        last_text = ""
        multi_user = self._is_multi_user(session, history)

        for round_no in range(1, max_rounds + 1):
            usage.rounds += 1
            # ⚠️ **最后一轮不给工具**：逼它把话说出来。否则模型很容易把 6 轮
            # 全花在搜/读上，用户最后只收到一句「已达单轮上限」——线上实测
            # （用户问「给出这几种封装的图片」，模型连读 6 轮网页，一张图没配上、
            # 一个字没说）。留一轮专门用来回答，是「至少给个交代」的保证。
            last_round = round_no >= max_rounds
            schemas = (self.tools.schemas(ctx)
                       if (self.cfg["tools"]["enabled"] and not last_round) else [])
            # 最后一轮在系统提示里明说「不许再调工具」：模型看不到工具清单时
            # 往往会自己写一段「我需要再查一下」，等于白费这一轮
            prompt = system_prompt
            if last_round and max_rounds > 1:
                prompt += ("\n\n（本轮工具调用次数已用尽：请**直接用已有信息回答**，"
                           "不要再要求调用工具。信息不足就说明哪部分没查到。）")
            api_messages = self.wrapping.build_request(
                prompt, history, in_group=target.is_group,
                multi_user=multi_user)
            try:
                result = await self._chat_with_retry(
                    api_messages, tools=schemas or None,
                    reasoning=self._session_reasoning(session) or None,
                    target=target)
            except Exception as e:  # noqa: BLE001
                if retries_left > 0 and _looks_like_context_overflow(e) \
                        and comp_cfg["enabled"]:
                    retries_left -= 1
                    report = await self.compactor.should_and_compact(
                        session.id, history, store=self.store, force=True,
                        session_name=session.name, role=session.role)
                    if report:
                        history = report["history"]
                        usage.compactions += 1
                        usage.note("上游报超长 → 压缩后重试")
                        continue
                raise

            if self.cfg.get("log_verbose") and self.llm.last_reasoning:
                self.logger.info("[ai_chat] 思考参数: %s", self.llm.last_reasoning)
            usage.prompt_tokens += int((result.usage or {}).get("prompt_tokens") or 0)
            usage.completion_tokens += int(
                (result.usage or {}).get("completion_tokens") or 0)
            assistant = assistant_message(result.text, tool_calls=result.to_blocks()[len(
                [b for b in result.to_blocks() if b.type == "text"]):] or None)
            history.append(assistant)
            self.store.append(session.id, assistant)
            if result.text.strip():
                last_text = result.text.strip()

            if not result.has_tool_calls:
                return last_text, usage

            # ---- 执行工具 ----
            for call in result.tool_calls:
                usage.tool_calls += 1
                if self.cfg.get("log_verbose"):
                    self.logger.info("[ai_chat] 工具调用 %s(%s)", call.name,
                                     call.arguments)
                out_text, is_error = await self.tools.call(call.name, call.arguments, ctx)
                tool_msg = tool_message(call.id, out_text, name=call.name,
                                        is_error=is_error)
                history.append(tool_msg)
                self.store.append(session.id, tool_msg)

        # 轮次用尽：把已有的正文发出去，并说明为何停下
        note = f"（已达单轮上限 {max_rounds} 次模型调用，先说到这里）"
        return (f"{last_text}\n\n{note}" if last_text else note), usage

    def _is_multi_user(self, session: Session, history: list[Message]) -> bool:
        """会话里是否出现过多个人（决定要不要加「某某说」前缀）。

        只扫人类输入消息，结果按消息条数缓存 —— 每轮全量扫历史没必要。
        """
        cached = self._multi_user.get(session.id)
        if cached and cached[0] == session.message_count:
            return cached[1]
        seen: set[int] = set()
        for m in history:
            if m.origin in ("user.private", "user.group"):
                uid = m.metadata.get("user_id")
                if uid:
                    seen.add(int(uid))
                if len(seen) > 1:
                    break
        multi = len(seen) > 1
        self._multi_user[session.id] = (session.message_count, multi)
        return multi

    def _tool_context(self, target: ChatTarget, session: Session,
                      meta: dict) -> ToolContext:
        """组装工具上下文。

        群角色的来源有讲究：OneBot 在**每条消息**的 `sender.role` 里就带了这个
        发言人在群里的角色，而框架的 `PermissionManager._group_roles` 缓存只在
        「管理员变更」通知里被动更新 —— 对启动前就已经是管理员的成员，缓存里
        什么都没有。只看缓存会把群管理员误判成普通成员（工具被静默下掉）。
        所以两条都看：事件自带的角色优先，缓存兜底。
        """
        group_name = ""
        can_moderate = False
        user_id = int(meta.get("user_id") or 0)
        is_super = bool(self.perm.is_superuser(user_id))
        if target.is_group:
            role = str(meta.get("role") or "") or \
                self.perm.get_group_role(target.chat_id, user_id)
            can_moderate = is_super or role in ("owner", "admin")
        return ToolContext(
            session_id=session.id, session_name=session.name,
            user_id=user_id, nickname=str(meta.get("nickname") or ""),
            chat_kind=target.kind, chat_id=target.chat_id,
            group_id=target.group_id or (target.chat_id if target.is_group else None),
            group_name=group_name,
            is_superuser=is_super, can_moderate=can_moderate,
            store=self.store, api=self.api, cfg=self.config,
            extras={"max_mute_seconds": int(self.cfg["tools"]["max_mute_seconds"]),
                    "group_id": target.chat_id if target.is_group else None,
                    # 联网搜索：客户端与正文上限都从上下文传，工具模块不持有它们
                    "web": self.web if self.cfg["search"]["enabled"] else None,
                    "max_page_chars": int(self.cfg["search"]["max_page_chars"]),
                    # 读网页时附几条图片直链（找图靠它，见 search.page_images）
                    "max_page_images": int(
                        self.cfg.get("web_image", {}).get("page_images") or 6),
                    # 函数图像：同理，工具只调服务，不持有渲染器
                    "plot": self.plots if self.cfg["plot"]["enabled"] else None,
                    # 网图配图：服务 + 下载用的 HTTP 客户端（web 在上面已经按开关取过）
                    "webimg": (self.webimg
                               if self.cfg.get("web_image", {}).get("enabled", True)
                               else None)})

    def _system_prompt(self, target: ChatTarget, session: Session, meta: dict,
                       ctx: ToolContext, *, multi_user: bool = False,
                       md: bool = False) -> str:
        extra: list[str] = []
        configured = str(self.cfg["wrapping"]["system_prompt"] or "").strip()
        if configured:
            extra.append(configured)
        # 「是否多人共用」按**历史里实际出现过几个人**判断，而不是按归属种类：
        # 共享有两个来源 —— 群归属的会话被多人用，或个人会话邀请了别人参与。
        # 只看归属会漏掉后者（那是邀请制的主要用法）。
        if multi_user or session.guests:
            owner = session.owner_label
            extra.append(
                f"这个会话是共用的：上下文对参与者**完全共享**（归{owner}所有）。"
                "参与者之间彼此可见对方说过的话，不要假设所有消息都来自同一个人；"
                "正文前的方括号标注了每句话是谁说的。")
        return build_system_prompt(
            persona=str(self.cfg["wrapping"]["persona"] or ""),
            # 会话角色（`/aichat role`）：走系统提示，**不进消息历史** ——
            # 所以上下文压缩（只处理历史消息）永远碰不到它，这就是「角色不被压缩」
            # 的实现方式。每轮原样重建。
            role=str(getattr(session, "role", "") or ""),
            extra_sections=extra,
            session_name=session.name, session_id=session.id,
            chat_kind=target.kind,
            chat_label=f"群 {target.chat_id}" if target.is_group else f"用户 {target.chat_id}",
            speaker=ctx.at_label, user_id=ctx.user_id,
            tool_names=[t.name for t in self.tools.visible(ctx)]
            if self.cfg["tools"]["enabled"] else [],
            # 指引按「本轮有什么能力」拼：没开搜索就不讲搜索，Markdown 模式
            # 才讲 Markdown —— 提示里讲不存在的工具会让模型空调用一轮。
            tool_guidance=tool_guidance(
                web=bool(self.cfg["search"]["enabled"]),
                # 画图指引同理：`tools.enabled=False` 时工具根本没暴露给模型，
                # 讲了它也调不到，只会多一轮空转。
                plot=bool(self.cfg["plot"]["enabled"]
                          and self.cfg["tools"]["enabled"]),
                # 配图指引同理：搜索或配图任一关掉，模型根本调不到 fetch_image
                image=bool(self.cfg.get("web_image", {}).get("enabled", True)
                           and self.cfg["search"]["enabled"]
                           and self.cfg["tools"]["enabled"]),
                markdown=bool(md)),
            markdown=bool(md))

    # ---------------- 外发 ----------------

    async def _send_rich(self, target: ChatTarget, session: Session, text: str, *,
                         quote_id: Optional[int] = None) -> bool:
        """文本模式下把表格/公式**挑出来渲染成图，插回消息体**。

        返回是否真的走了这条路（`False` = 没有富文本块，调用方按纯文本发）。

        为什么要做：QQ 不渲染 Markdown，表格发出去是一堆竖线、`\\frac{a}{b}`
        发出去就是一串反斜杠。系统提示里虽然要求纯文本，但模型**确实会**用表格和
        LaTeX（真实会话里有），所以与其假装它不会，不如把这些块挑出来出图。

        **保序**是硬要求：图文段按原顺序拼进 MessageChain，图必须出现在它原本
        所在的位置；一次渲染失败就整体退回纯文本（宁可排版差，也不要半截）。
        """
        segs = split_rich_segments(text or "")
        kinds = {k for k, _ in segs}
        want_inline = bool(self.cfg["reply"].get("inline_math", True))
        # 表格 / 公式 / 标题 / 列表 / 代码围栏都算「可渲染块」—— 它们发原样都是
        # 一堆符号（`###`、竖线、反斜杠），所以都挑出来出图。
        has_block = bool(kinds & {"table", "math", "heading", "list", "code"})
        # 只有行内公式（没有整块表格/公式）时也要走这条路，否则那些 `$x^0$`
        # 会原样发成 LaTeX 源码 —— 这正是用户截图里报的那个问题。
        has_inline = want_inline and any(
            kind == "text" and has_inline_math(content) for kind, content in segs)
        if not (has_block or has_inline):
            return False
        chain = MessageChain()
        if self.cfg["reply"]["quote"] and quote_id:
            chain.reply(seq=int(quote_id))
        try:
            for kind, content in segs:
                if kind == "text":
                    # ⚠️ 正文里还可能有**行内**公式（`$x^{0}$`）：只挑整块表格/公式
                    # 的话，这些会原样发成 LaTeX 源码（实测截图：「常数函数通常并入
                    # 幂函数（$x^{0}$）」）。所以文本段再切一层，把公式也换成小图，
                    # 与文字**保序**交错进同一条消息。
                    if want_inline:
                        for sub_kind, sub in split_inline_math(content):
                            if sub_kind == "math":
                                ipath = await self.renderer.render_inline_math(sub)
                                chain.image(file=Path(ipath).as_uri(),
                                            name=os.path.basename(ipath))
                            else:
                                chain.text(sub + "\n")
                    else:
                        chain.text(content + "\n")
                    continue
                path = await self.renderer.render_rich(kind, content)
                chain.image(file=Path(path).as_uri(), name=os.path.basename(path))
        except RenderError as e:
            self.logger.warning("[ai_chat] 表格/公式出图失败，退回纯文本: %s", e)
            await self._send_text(target, text, quote_id=quote_id)
            return True
        except Exception as e:  # noqa: BLE001 - 渲染是外部进程，什么都可能
            self.logger.error("[ai_chat] 表格/公式出图异常: %s\n%s",
                              err_text(e), _tb())
            await self._send_text(target, text, quote_id=quote_id)
            return True
        # 用现成的配额切分（长文本分页 + 图片配额），图文顺序保持不变
        chunks = chain.split(
            max_chars=int(self.cfg["reply"]["chunk_max_chars"]),
            max_images=int(self.cfg["reply"]["chunk_max_images"]))
        await self._send(target, chunks)
        return True

    async def _send_text(self, target: ChatTarget, text: str, *,
                         quote_id: Optional[int] = None) -> None:
        """把模型输出按配额分段发送（长回复不会被 QQ 截断或刷屏）。"""
        body = (text or "").strip()
        if not body:
            body = str(self.cfg["reply"]["fallback_text"])
        chain = MessageChain()
        if self.cfg["reply"]["quote"] and quote_id:
            chain.reply(seq=int(quote_id))
        chain.text(body)
        chunks = chain.split(
            max_chars=int(self.cfg["reply"]["chunk_max_chars"]),
            max_images=int(self.cfg["reply"]["chunk_max_images"]))
        await self._send(target, chunks)

    async def _send(self, target: ChatTarget,
                    payload: Any) -> None:
        """发送一段或多段消息。断线时停止后续分段（不刷一堆失败）。

        顺带把每条消息的 `message_id` 收进 `_sent_ids[聊天地址]` ——
        用户之后可能**引用**其中任意一条（尤其是图片那条），
        见 `SentReplyIndex` 的说明。
        """
        chains = payload if isinstance(payload, list) else [payload]
        for chain in chains:
            try:
                if target.is_group:
                    resp = await self.api.send_group_msg(
                        group_id=str(target.chat_id), message=chain)
                else:
                    kwargs = {}
                    if target.group_id:
                        kwargs["group_id"] = str(target.group_id)
                    resp = await self.api.send_private_msg(
                        user_id=str(target.chat_id), message=chain, **kwargs)
                mid = _message_id_of(resp)
                if mid:
                    self._sent_ids.setdefault(target.key, []).append(mid)
            except ApiError as e:
                self.logger.error("[ai_chat] 发送失败 [%s]: %s", e.retcode, e.message)
                return
            except Exception as e:  # noqa: BLE001
                self.logger.error("[ai_chat] 发送异常: %s", err_text(e))
                return

    # ---------------- 命令 ----------------

    #: 管理「服务范围」本身的命令：它们**不受服务范围限制**。
    #: 否则会死锁 —— `bind` 只能私聊用，而 users 为空时私聊又被服务范围挡住，
    #: 于是永远没有第一条私聊开通（实测就是如此）。
    SCOPE_EXEMPT = frozenset({"targets", "add", "del", "reload"})

    def _register_commands(self):
        """注册全部命令。

        除 `SCOPE_EXEMPT` 之外，所有命令都包一层「服务范围」判定：未开通的
        群/私聊连命令都用不了（与权限等级无关）。回绝时用**异步任务**发说明 ——
        命令处理器是同步函数，在里面 await 会变成「协程没被 await」的静默失败。
        """
        A = self._namespace()
        original_register = self.register_command

        def register_command(name, **kwargs):
            decorator = original_register(name, **kwargs)

            def wrap(func):
                # ⚠️ 参数名必须按被包函数的签名挑：框架是按**参数名**注入的
                # （packet / event / api / params），多传一个 `packet` 给
                # `async def cmd(event)` 会直接 TypeError，而它被框架吞成
                # 「命令没反应」。
                wanted = tuple(inspect.signature(func).parameters)

                # 从注册名里取出子命令名，判断是否属于「服务范围自管理」命令
                sub = name.split(".", 1)[1] if "." in name else ""
                exempt = sub in self.SCOPE_EXEMPT

                @functools.wraps(func)
                async def guarded(**kwargs):
                    event = kwargs.get("event")
                    if event is None:
                        return await func(**{k: v for k, v in kwargs.items()
                                             if k in wanted})
                    target = self._target_of(event)
                    # 「受邀的未绑定用户」必须能自己 attach/detach，否则会死锁：
                    # 服务范围的受邀例外判的是**当前焦点**，而焦点只能靠 attach
                    # 建立 —— 先把 attach 挡掉，那个例外就永远走不到（实测：
                    # 被邀请的人在私聊里发 attach 完全没有反应）。
                    # 放行的只是这两个动作；**能不能 attach 某个会话**仍由
                    # `_may_attach` 判（所有者 / 创建者 / 受邀者），不存在越权。
                    may_claim = (sub in self.CLAIM_COMMANDS
                                 and self._is_invited_guest(target, int(event.user_id)))
                    if not exempt and not may_claim \
                            and self._access_denied(target, int(event.user_id)):
                        return None
                    return await func(**{k: v for k, v in kwargs.items()
                                         if k in wanted})
                return decorator(guarded)
            return wrap

        self.register_command = register_command

        @self.register_command(f"{A}.new", schema=S.NewSchema, permission="",
                               scope="all",
                               description="新建会话（位置决定作用地：群里=群会话，私聊=私聊会话）")
        async def cmd_new(packet, params, event: MessageEvent):
            target = self._target_of(event)
            # `--owner` 曾经存在，后来被去掉（位置决定作用地）。老用户/旧文档
            # 还会带着它 —— 框架会把未知的 `--owner=x` 当**位置参数**塞进 args，
            # 于是被静默忽略、用户以为设置生效了。这里显式认出来并说清楚。
            if any(str(a).startswith("--owner") for a in (packet.args or [])) \
                    or any(k.startswith("owner") for k in (packet.kwargs or {})):
                return ("⚠️ `--owner` 已取消：**新建位置决定会话作用地** ——\n"
                        "  群里建 → 群会话（只能在本群用）\n"
                        "  私聊建 → 私聊会话（只能在私聊用）\n"
                        "想在哪里用就在哪里 /aichat new <名称>")
            # **新建的位置决定会话的作用地**（唯一规则，没有覆盖开关）：
            # 在群里建 → 群会话（归本群，只能在本群用）；
            # 在私聊建 → 私聊会话（归本人，只能在私聊用）。
            # 早先默认写死 user，于是「在群里 /aichat new 名称」建出私聊会话、
            # 随即在群里用不了；加 `--owner` 开关只是把坑挪了个位置，所以直接去掉。
            owner_kind = OWNER_GROUP if target.is_group else OWNER_USER
            owner_id = (target.chat_id if owner_kind == OWNER_GROUP
                        else int(event.user_id))
            # 群会话的所有者是**群**，owner_name 就别写成「建它的那个人」——
            # 列表里显示成「群 张三」会让人以为是张三的私有会话。
            owner_name = (str(target.chat_id) if owner_kind == OWNER_GROUP
                          else self._display_name(event))
            session = self.store.create(
                name=params.name.strip(), owner_kind=owner_kind, owner_id=owner_id,
                owner_name=owner_name)
            # 记下创建者：群会话归群所有，创建者要能继续用它、管它
            # （邀请/改名/删除），否则建完就失控。私聊会话也一并记，语义统一。
            session.meta["created_by"] = int(event.user_id)
            lines = [f"✅ 已新建会话 {session.describe()}"]
            if params.attach:
                self.store.set_focus(target, int(event.user_id), session.id)
                lines.append(f"已设为你在此处的当前会话（{target.label}），直接说话即可")
            else:
                lines.append(f"未使用。用 /aichat attach {session.id} 切过去")
            if owner_kind == OWNER_GROUP:
                lines.append("（本群会话：只能在本群用；"
                             f"群成员需你 /aichat invite {session.id} <QQ号> 才能参与）")
            else:
                lines.append("（私聊会话：只能在私聊用；"
                             f"想邀请别人一起用：/aichat invite {session.id} <QQ号>）")
            return "\n".join(lines)

        @self.register_command(f"{A}.list", schema=S.ListSchema, permission="",
                               scope="all",
                               description="列出**当前聊天**里的会话")
        async def cmd_list(params, event: MessageEvent):
            target = self._target_of(event)
            uid = int(event.user_id)
            uid_arg = _arg_list_id(getattr(params, "user", None))
            try:
                if uid_arg is not None:
                    # 指定了人 → 只有超级用户能查别人（查自己不需要特权）
                    if uid_arg != uid and not self.perm.is_superuser(uid):
                        return "⚠️ 只有超级用户可以查看别人的会话"
                    return self._list_for_person(uid_arg, target)
                sessions = self._sessions_of_chat(target)
                return self._render_session_list(sessions, target, uid)
            except Exception as e:  # noqa: BLE001
                # 列表是纯读取，失败几乎总是数据问题（state.json 半损坏等）。
                # 明确回一句，别让用户以为「命令不存在/机器人没反应」。
                self.logger.error("[ai_chat] list 渲染失败: %s\n%s", err_text(e), _tb())
                raise CommandExecutionError(
                    f"读取会话列表失败：{err_text(e)}（数据目录 "
                    f"{self._data_dir}，详见日志）") from None

        @self.register_command(f"{A}.listall", schema=S.ListAllSchema,
                               permission="superuser", scope="private",
                               description="列出全部会话（含各群/各人的；仅私聊）")
        async def cmd_listall(params):
            include_archived = bool(getattr(params, "archived", False))
            sessions = self.store.list(include_archived=include_archived,
                                       limit=self.store.max_sessions)
            if not sessions:
                return "还没有任何会话"
            lines = [f"📋 全部会话（{len(sessions)}）"]
            for x in sessions:
                who = ("群 " + str(x.owner_id) if x.owner_kind == OWNER_GROUP
                       else f"用户 {x.owner_id}")
                focus = self.store.attachments(x.id)
                lines.append(f"  {x.describe()}")
                lines.append(f"      归属：{who}"
                             f" · 受邀 {len(x.guests)} 人"
                             f" · 使用中 {len(focus)} 处")
            lines.append("\n（超级用户视图；普通用户用 /aichat list 只看当前聊天）")
            return "\n".join(lines)

        # ---- 历史对话：`history` 列清单（文字），`hget` 出图 ----
        # 两者共用**同一套编号**（都调 extract_exchanges），否则用户拿着 list 里
        # 的编号去 get 会看到另一对问答。
        @self.register_command(f"{A}.history", schema=S.HistorySchema, permission="",
                               scope="all",
                               description="列出当前会话的历史提问（只给问题摘要，看全文用 hget）")
        async def cmd_history(params, event: MessageEvent):
            target = self._target_of(event)
            session = self._session_for(target, event)
            if session is None:
                return self._no_session_hint(target, int(event.user_id))
            messages = self.store.history(session.id)
            exchanges = extract_exchanges(messages)
            if not exchanges:
                base = (f"「{session.name}」还没有可查看的问答"
                        f"（一问一答才算数）。")
                if has_compaction(messages):
                    return base + "\n历史已压缩过，老对话不在编号里。"
                return base + f"\n直接在本会话里提问，用 {self._term_hint()} 收尾。"
            limit = max(1, int(getattr(params, "limit", None)
                               or self.cfg["history"]["list_limit"]))
            shown = exchanges[-limit:]
            first, last = shown[0].index, shown[-1].index
            head = (f"📜 {session.id} {session.name} · 共 {len(exchanges)} 对问答"
                    f"（显示第 {first}-{last} 对）")
            tail = (f"看完整问答：/aichat hget <编号>（如 /aichat hget {last}，"
                    f"一次一条，出图）")
            if has_compaction(messages):
                tail += "\n（历史压缩过：被摘要吸收的老问答不在编号里）"

            # 清单只给**提问**的模型短摘要（不是原文引用），**不给回答**：
            # 目录要能一眼看出「我问过什么」，原文开头几十字做不到这点；
            # 而回答动辄上千字、一条里覆盖多个要点，放进目录只会把编号冲散
            # —— 要看全文走 `hget` 出图（见 history.format_exchange）。
            summaries, note = await self._summarize_exchanges(session, shown)
            if note:
                head += "\n" + note
            # 分页在**条与条之间**切，一条问答不会被劈成两半
            return render_list(shown, header=head + "\n" + tail,
                               summaries=summaries,
                               chunk_max_chars=int(self.cfg["reply"]["chunk_max_chars"]))

        @self.register_command(f"{A}.hget", schema=S.HGetSchema, permission="",
                               scope="all",
                               description="把某一条历史问答渲染成图片（一次一条）")
        async def cmd_hget(params, event: MessageEvent):
            target = self._target_of(event)
            session = self._session_for(target, event)
            if session is None:
                return self._no_session_hint(target, int(event.user_id))
            if not self.cfg["history"]["enabled"]:
                return "历史出图已被管理员关闭（history.enabled=false）"

            raw_n = getattr(params, "n", "")
            ranged = reject_range(raw_n)
            if ranged:
                return ranged          # 一次只取一条，区间直接说清楚
            exchanges = extract_exchanges(self.store.history(session.id))
            want, err = parse_range(raw_n, total=len(exchanges))
            if err:
                return err

            n = want[0]
            picked = next((x for x in exchanges if x.index == n), None)
            if picked is None:              # 理论上 parse_range 已经拦住了
                return f"没有第 {n} 对 —— 当前共 {len(exchanges)} 对"

            # ⚠️ 先渲染完，**再**发文字头。
            # 反过来的话，渲染失败时用户已经收到一句「🖼 第 3 对」然后什么都不来 ——
            # 一个没有下文的头比一句错误更让人困惑（实测踩过）。
            try:
                path = await self.renderer.render(
                    picked, session_name=session.name, total=len(exchanges))
            except RenderError as e:
                raise CommandExecutionError(f"渲染历史图片失败：{e}") from None
            except Exception as e:  # noqa: BLE001 - 渲染是外部进程，什么都可能
                self.logger.error("[ai_chat] hget 渲染异常: %s\n%s",
                                  err_text(e), _tb())
                raise CommandExecutionError(
                    f"渲染历史图片失败：{err_text(e)}") from None

            # 文字头：图片在聊天里是独立气泡，没有头用户不知道这是哪一段
            await self._send(target, MessageChain().text(
                f"🖼 {session.name} · 第 {n} 对（共 {len(exchanges)} 对）"))
            await self._send(target, MessageChain().image(
                file=Path(path).as_uri(), name=os.path.basename(path)))
            return None

        # ---- 顶层命令：把上一条 AI 回复转成 Markdown 图片 ----
        # ⚠️ **不在 `aichat` 命名空间下**（用户要求「一个没有子命令的命令」）：
        # 日常用得最多的一句话动作，值得少打几个字。
        @self.register_command(
            "md", permission="", scope="all",
            description="把上一条 AI 回复渲染成 Markdown 图片发过来")
        async def cmd_md(event: MessageEvent):
            target = self._target_of(event)
            session = self._session_for(target, event)
            if session is None:
                return self._no_session_hint(target, int(event.user_id))
            if not self.cfg["reply"].get("markdown", True):
                return "Markdown 出图已被管理员关闭（reply.markdown=false）"

            text = self._last_reply_text(session)
            if not text:
                return "这个会话里还没有可转的 AI 回复。"

            try:
                path = await self.renderer.render_markdown(
                    text, title=session.name)
            except RenderError as e:
                raise CommandExecutionError(f"渲染 Markdown 失败：{e}") from None
            except Exception as e:  # noqa: BLE001 - 渲染是外部进程，什么都可能
                self.logger.error("[ai_chat] md 渲染异常: %s\n%s",
                                  err_text(e), _tb())
                raise CommandExecutionError(
                    f"渲染 Markdown 失败：{err_text(e)}") from None

            # 先渲染完再发图：渲染失败时不该先发一句「🖼 …」然后没有下文
            await self._send(target, MessageChain().image(
                file=Path(path).as_uri(), name=os.path.basename(path)))
            # 图片本身也是一条「机器人发的消息」—— 记进索引，
            # 这样用户之后引用它时，取回的仍是 Markdown 原文（见 SentReplyIndex）
            self.sent_reply.add(self._sent_ids.pop(target.key, []),
                                session.id, self._last_reply_seq(session))
            return None

        @self.register_command(f"{A}.attach", schema=S.AttachSchema, permission="",
                               scope="all",
                               description="切换你在此处的当前会话")
        async def cmd_attach(params, event: MessageEvent):
            return self._do_switch(params.session, event)

        @self.register_command(f"{A}.detach", schema=S.DetachSchema, permission="",
                               scope="all", description="取消你在此处的当前会话")
        async def cmd_detach(params, event: MessageEvent):
            target = self._target_of(event)
            uid = int(event.user_id)
            current = self._current_session(target, uid)
            if current is None:
                return "你在此处没有当前会话"
            if params.session:
                wanted = self.store.resolve(params.session)
                if wanted is None:
                    return f"找不到会话：{params.session}"
                if wanted.id != current.id:
                    return (f"你在此处用的是 {current.id} {current.name}，"
                            f"不是 {wanted.id}；未做改动")
            # 暂存区里还有没提交的分段 → **直接丢弃**（不是拒绝解除）。
            # 理由：`detach` 的语义就是「我不在这里聊了」，留着那几段既不会提交、
            # 又会在下次 attach 回来时突然和历史拼在一起（那些分段往往正是用户
            # 想撤掉的半句话）。丢弃要说清楚丢了几段 —— 静默吞掉用户发过的内容
            # 比多问一句更糟。
            staged = self.buffer.peek(target.conversation_key).get("items") or []
            parts: list[str] = []
            if staged:
                self.buffer.clear(target.conversation_key)
                parts.append(f"暂存区里 {len(staged)} 段未提交内容已丢弃")
            sid = self.store.clear_focus(target, uid)
            self._multi_user.pop(sid, None)
            parts.append(f"已取消你在{target.label}的当前会话 "
                         f"{current.id}「{current.name}」")
            return "；".join(parts)

        @self.register_command(f"{A}.current", permission="", scope="all",
                               description="查看你在此处的当前会话与上下文用量")
        async def cmd_current(event: MessageEvent):
            return await self._status_text(event)

        @self.register_command(f"{A}.rename", schema=S.RenameSchema, permission="",
                               scope="all", description="重命名会话（仅所有者）")
        async def cmd_rename(params, event: MessageEvent):
            session = self.store.resolve(params.session)
            if session is None:
                return f"找不到会话：{params.session}"
            ok, why = self._may_manage(session, event.user_id)
            if not ok and not self.perm.is_superuser(event.user_id):
                return f"⚠️ {why}"
            new_name = " ".join(params.new_name or []).strip()
            if not new_name:
                return "新名称不能为空：/aichat rename <会话> <新名称>"
            if len(new_name) > S.NAME_MAX:
                return f"名称太长（最多 {S.NAME_MAX} 字）"
            self.store.rename(session.id, new_name)
            return f"会话 {session.id} 已更名为「{new_name}」"

        @self.register_command(f"{A}.role", schema=S.RoleSchema, permission="",
                               scope="all",
                               description="查看/设置会话里 AI 扮演的角色（不受压缩影响）")
        async def cmd_role(params, event: MessageEvent):
            session, text = self._split_role_args(params, event)
            if session is None:
                if params.session and not text:
                    # 只在「第一格被当成会话名、但查无此会话」且后面没有正文时
                    # 才报找不到 —— 否则那句话本来就是角色设定
                    return f"找不到会话：{params.session}"
                return "你在此处没有当前会话，也没有指定会话"
            is_super = self.perm.is_superuser(event.user_id)

            if params.clear or text:
                # 改设定 = 会话级配置，只给所有者/管理者（与 rename 同一把尺子）
                ok, why = self._may_manage(session, event.user_id)
                if not ok and not is_super:
                    return f"⚠️ {why}"
            else:
                # 只是**看**：能参与这个会话的人都能看。设定会直接影响他们收到的
                # 回复，看不到就成了「行为怪但查不出原因」。
                ok, why = self._may_attach(session, self._target_of(event),
                                           int(event.user_id))
                if not ok and not is_super:
                    return f"⚠️ {why}"

            if params.clear:
                if not session.role:
                    return f"会话 {session.id} 本来就没有角色设定"
                self.store.set_role(session.id, "")
                return f"已清除会话 {session.id} 的角色设定，恢复默认人格"
            if not text:
                if not session.role:
                    return (f"会话 {session.id} 还没设角色。\n"
                            f"设置：/aichat role <角色设定>\n"
                            f"例如：/aichat role 你是一位耐心的数学老师，"
                            f"多用生活例子解释概念")
                return (f"会话 {session.id} 的当前角色（{len(session.role)} 字）：\n"
                        f"———\n{session.role}\n———\n"
                        f"修改：/aichat role <新设定>　清除：/aichat role --clear=true")
            if len(text) > S.ROLE_MAX:
                return f"角色设定太长（{len(text)} 字，最多 {S.ROLE_MAX} 字）"
            self.store.set_role(session.id, text)
            return (f"已设定会话 {session.id} 的角色（{len(text)} 字）。\n"
                    f"这段设定会随每轮系统提示**原样**生效，"
                    f"并且**不参与上下文压缩**。\n"
                    f"查看：/aichat role　清除：/aichat role --clear=true")

        @self.register_command(f"{A}.delete", schema=S.DeleteSchema, permission="",
                               scope="all", description="删除会话（仅所有者，需 --yes=true）")
        async def cmd_delete(params, event: MessageEvent):
            session = self.store.resolve(params.session)
            if session is None:
                return f"找不到会话：{params.session}"
            ok, why = self._may_manage(session, event.user_id)
            if not ok and not self.perm.is_superuser(event.user_id):
                return f"⚠️ {why}"
            if not params.yes:
                return (f"确认删除会话 {session.id}「{session.name}」？"
                        f"它共有 {session.message_count} 条历史。"
                        f"确认发 /aichat delete {session.id} --yes=true")
            where = self.store.attachments(session.id)
            self.store.delete(session.id)
            self._multi_user.pop(session.id, None)
            tail = f"（原在 {len(where)} 处被使用）" if where else ""
            return f"已删除会话 {session.id}「{session.name}」{tail}"

        @self.register_command(f"{A}.compact", permission="", scope="all",
                               description="手动压缩当前（或指定）会话的上下文")
        async def cmd_compact(packet, event: MessageEvent):
            args = list(packet.args or [])
            target = self._target_of(event)
            uid = int(event.user_id)
            session = self._current_session(target, uid)
            if args and not args[0].startswith("-"):
                session = self.store.resolve(args[0]) or session
            if session is None:
                return "你在此处没有当前会话，也没有指定会话可压缩"
            ok, why = self._may_attach(session, target, uid)
            if not ok:
                return f"⚠️ {why}"
            if not self.compactor.client.enabled:
                return "未配置 LLM（llm.base_url / model / api_key），无法压缩"
            history = self.store.history(session.id)
            if len(history) < self.policy.min_head_messages:
                return f"会话 {session.id} 只有 {len(history)} 条历史，没什么可压缩的"
            before = self.policy.pressure(history)
            report = await self.compactor.should_and_compact(
                session.id, history, store=self.store, force=True,
                session_name=session.name, role=session.role)
            if not report:
                return (f"压缩未执行：{self.compactor.last_error or '没有可用的切点'}"
                        f"（历史 {len(history)} 条，压力 {before:.0%}）")
            return ("🗜 已压缩会话 %s：吸收 %d 条 → 检查点 %d 字，"
                    "保留最近 %d 条原文\n估算 token %d → %d（预算 %d）"
                    % (session.id, report["absorbed"], len(report["summary"]),
                       report.get("checkpoint", {}).get("retained", 0),
                       report["before_tokens"], report["after_tokens"],
                       self.policy.max_tokens))

        # ---- 邀请（会话所有者；群归属的会话由群主/管理员代管）----
        @self.register_command(f"{A}.invite", schema=S.InviteSchema, permission="",
                               scope="all",
                               description="邀请某人参与你的会话（对方需自己 attach）")
        async def cmd_invite(params, event: MessageEvent):
            return self._do_invite(params, event, invite=True)

        @self.register_command(f"{A}.uninvite", schema=S.InviteSchema, permission="",
                               scope="all", description="取消邀请（会同时摘掉对方焦点）")
        async def cmd_uninvite(params, event: MessageEvent):
            return self._do_invite(params, event, invite=False)

        @self.register_command(f"{A}.guests", permission="", scope="all",
                               description="查看会话的受邀名单")
        async def cmd_guests(packet, event: MessageEvent):
            target = self._target_of(event)
            uid = int(event.user_id)
            session = self._current_session(target, uid)
            args = list(packet.args or [])
            if args:
                session = self.store.resolve(args[0]) or session
            if session is None:
                return "你在此处没有当前会话，也没有指定会话"
            ok, why = self._may_attach(session, target, uid)
            if not ok:
                return f"⚠️ {why}"
            guests = self.store.guests_of(session.id)
            lines = [f"👥 会话 {session.id}「{session.name}」"]
            lines.append(f"所有者：{session.owner_label}")
            if not guests:
                lines.append("受邀名单：空（只有所有者能用）")
            else:
                lines.append(f"受邀名单（{len(guests)}）："
                             + "、".join(str(g) for g in guests))
            lines.append(f"邀请：/aichat invite {session.id} <QQ号>"
                         "（对方需自己 /aichat attach）")
            return "\n".join(lines)

        @self.register_command(f"{A}.model", schema=S.ModelSchema,
                               permission="superuser", scope="private",
                               description="查看/临时修改模型与上下文预算（不写回配置）")
        async def cmd_model(params):
            changed = []
            if params.model:
                self.llm.model = params.model
                changed.append(f"model={params.model}")
            if params.base_url:
                self.llm.base_url = params.base_url.rstrip("/")
                changed.append(f"base_url={self.llm.base_url}")
            if params.max_tokens:
                self.llm.max_tokens = int(params.max_tokens)
                changed.append(f"max_tokens={params.max_tokens}")
            if params.context_limit:
                self.policy.max_tokens = int(params.context_limit)
                changed.append(f"context_limit={params.context_limit}")
            head = ("🔧 已更新：" + "，".join(changed) + "\n（仅本次运行有效；"
                    "要持久化改 config.json → plugins.ai_chat.llm）"
                    if changed else "🔧 当前生效值")
            return (f"{head}\nmodel: {self.llm.model or '（未配置）'}\n"
                    f"base_url: {self.llm.base_url or '（未配置）'}\n"
                    f"api_key: {'已设置(%d 字符)' % len(self.llm.api_key) if self.llm.api_key else '（未配置）'}\n"
                    f"max_tokens: {self.llm.max_tokens}\n"
                    f"思考强度: {self.reasoning.describe(self.reasoning.spec_for(self.llm.model))}\n"
                    f"context_limit: {self.policy.max_tokens}"
                    f"（阈值 {self.policy.threshold_tokens:.0f}，"
                    f"保留 {self.policy.retain_target_tokens:.0f}）\n"
                    f"history_max_messages: {self.store.max_messages}")

        @self.register_command(f"{A}.think", schema=S.ThinkSchema, permission="",
                               scope="all",
                               description="查看/切换思考强度（推理模型的思考链开销）")
        async def cmd_think(params, event: MessageEvent):
            return self._do_think(params, event)

        # ---- 超级用户 + 仅私聊：绑定 ----
        @self.register_command(f"{A}.config", schema=S.ConfigSchema,
                               permission="superuser", scope="private",
                               description="查看/补齐 config.json 里的插件配置段（仅私聊）")
        async def cmd_config(params):
            report = self.config.fill_defaults(factory_defaults())
            if report.filled:
                return (f"⚙️ 已把 {len(report.filled)} 个缺失配置项写入 "
                        f"config.json → plugins.ai_chat\n"
                        f"补齐的键：{', '.join(report.filled[:12])}"
                        + ("…" if len(report.filled) > 12 else "")
                        + "\n（已有值一律未改动；改完用 "
                          "/aichat reload --confirm=true 生效）")
            return (f"⚙️ config.json → plugins.ai_chat 已完整"
                    f"（{len(DEFAULT_CONFIG)} 个顶层段，共 "
                    f"{len(_flatten(factory_defaults()))} 个配置项），无需补齐")

        @self.register_command(f"{A}.reload", schema=S.ReloadSchema,
                               permission="superuser", scope="private",
                               description="从 config.json 重读本插件配置（手工改配置后用）")
        async def cmd_reload(params):
            if not params.confirm:
                return ("确认从磁盘重读 config.json？"
                        "确认发 /aichat reload --confirm=true\n"
                        "（只影响本插件；手工改过的文件内容会覆盖内存里的值）")
            self.reload_config(from_disk=True)
            spec = self.reasoning.spec_for(self.llm.model)
            return (f"♻️ 已从 config.json 重读配置\n"
                    f"model: {self.llm.model or '（未配置）'}\n"
                    f"思考强度：{self.reasoning.describe(spec)}\n"
                    f"方言：{self.reasoning.dialect} · "
                    f"上下文预算：{self.policy.max_tokens}")

        # ---- 服务范围管理：targets / add / del（kind 选类型、target 给号码） ----
        # 这三个命令走**手工挂载**（见下方 _mount_scope_command 的说明），
        # 因为服务范围管理器必须豁免服务范围判定 —— 否则 users 为空时
        # 私聊连 add 都用不了，永远开不出第一条私聊（实测死锁）。
        async def cmd_targets(packet, event: MessageEvent):
            groups = sorted(self._bound_groups())
            users = sorted(self._bound_ids("users"))
            empty = self._empty_means()
            lines = ["🔓 服务范围（谁能用 AI 对话）", "", f"👥 群（{len(groups)}）"]
            lines += [f"  {g}" for g in groups] or ["  （未配置）"]
            lines += ["", f"👤 用户（{len(users)}）"]
            lines += [f"  {u}" for u in users] or ["  （未配置）"]
            if not groups and not users:
                lines += ["", f"⚠️ 名单为空，当前策略：{empty}"
                              + ("（谁都不可用）" if empty == "deny" else "（不限制）")]
            else:
                lines += ["", f"（名单为空时：{empty}）"]
            table = self.store.focus_table()
            lines += ["", f"💬 当前会话焦点（{len(table)} 处）"]
            if table:
                for key, sid in sorted(table.items()):
                    session = self.store.get(sid)
                    lines.append(f"  {_focus_label(key)} → "
                                 f"{sid}「{session.name if session else '?'}」")
            else:
                lines.append("  （还没有人设过；会话由 /aichat new 创建）")
            lines += ["", "💡 开通：/aichat add group|user [号]",
                      "     停止：/aichat del group|user [号]"]
            return "\n".join(lines)

        async def cmd_add(params, event: MessageEvent):
            kind = params.kind
            target = params.target
            if not target:
                if kind == "group" and event.group_id:
                    target = int(event.group_id)      # 群里 add group → 当前群
                elif kind == "user" and event.message_type == "private":
                    target = int(event.user_id)       # 私聊 add user → 自己
                else:
                    return "⚠️ 指定群号或 QQ 号：/aichat add group|user <号>"
            target = int(target)
            if target <= 0:
                return "⚠️ 目标必须是正整数"
            key = "groups" if kind == "group" else "users"
            ids = self._bound_ids(key)
            if target in ids:
                return f"{'群' if kind == 'group' else '用户'} {target} 已开通"
            ids.add(target)
            self.cfg["binding"][key] = sorted(ids)
            self.config.set(f"binding.{key}", sorted(ids))
            self.reload_config()
            return (f"✅ 已开通{'群' if kind == 'group' else '用户'} {target}"
                    f"（当前 {len(ids)} 个）\n"
                    f"写入 config.json → plugins.ai_chat.binding.{key}"
                    + ("\n群开通 = 全群可用；群里每个人各自有当前会话"
                       if kind == "group" else
                       "\n这个人现在可以在私聊里直接对话"))

        async def cmd_del(params, event: MessageEvent):
            kind = params.kind
            target = params.target
            if not target:
                if kind == "group" and event.group_id:
                    target = int(event.group_id)
                else:
                    return "⚠️ 指定群号/QQ号"
            target = int(target)
            key = "groups" if kind == "group" else "users"
            ids = self._bound_ids(key)
            if target not in ids:
                return f"⚠️ {target} 不在服务范围内（用 /aichat targets 查看）"
            ids.discard(target)
            self.cfg["binding"][key] = sorted(ids)
            self.config.set(f"binding.{key}", sorted(ids))
            self.reload_config()
            return (f"✅ 已停止{'群' if kind == 'group' else '用户'} {target} 的服务"
                    f"（剩余 {len(ids)} 个）\n"
                    f"已有会话与历史不受影响，只是这里用不了了")

        self._mount_scope_command("targets", cmd_targets,
                                  description="查看服务范围与各人当前会话（仅私聊）")
        self._mount_scope_command("add", cmd_add, schema=S.TargetAddSchema,
                                  description="开通服务：add group|user [号]（仅私聊）")
        self._mount_scope_command("del", cmd_del, schema=S.TargetDelSchema,
                                  description="停止服务：del group|user [号]（仅私聊）")

        # ---- 结束符命令（全域）----
        # 参数写错时的回执由**框架**统一负责（`handle()` 把 SchemaError 变成
        # 「⚠️ 用法不正确：… + 该命令的帮助页」）。插件此前自己注册 on_schema_error
        # 回调发提示：那个钩子只有一个槽位，两个插件都注册时后者覆盖前者，
        # 用户看到哪种报错取决于加载顺序 —— 已随框架接管删除。

        # 结束符的**命令形式**（`/end`）已退役：收尾只认行尾结束符（默认 `//`）。
        # 两条入口维护同一件事，行为迟早分叉（命令形式不吃 `//md` 的 Markdown 后缀、
        # 也不走 `_strip_terminator`），留一条即可。

        # ---- 帮助（显式注册，避免与框架 /help 的命令组展示口径不一致）----
        # 注册期结束，恢复原方法（命令包装只在本次注册内需要）
        self.register_command = original_register

        # 命令组本体的描述：统一 /help 会在 `/help aichat` 里显示它。
        # 父节点由框架在注册第一个子命令时自建（`register` 的 else 分支），
        # 这里只补描述 —— 自己注册父命令会让它带 parent_name，从而被
        # /help 的命名空间分支跳过。
        parent = self.cmds._commands.get(A)
        if parent is not None and parent.subcommands:
            parent.description = self._root_description()
            parent.scope = "all"

        # 让统一 /help 与服务范围同口径：未开通的地址在帮助里看不到本命名空间。
        # 没有这个钩子，框架的可见性只看 scope/permission —— 未绑定的群里
        # `/help` 仍会把 /aichat 的子命令清单列出来，等于暴露功能清单。
        cmds = self.cmds
        if hasattr(cmds, "add_visibility_filter"):
            if self._visibility_filter is not None:
                cmds.remove_visibility_filter(self._visibility_filter)

            def _visible(handler, event):
                """本命名空间（父节点 `/aichat` 与其子命令）是否对当前上下文可见。

                ⚠️ 两个例外，都必须放行，否则会出现「最需要帮助时看不到帮助」：

                1. **服务范围管理的子命令**（targets/add/del/reload）本身豁免范围判定；
                2. **父节点 `/aichat`**：只要调用者是超级用户就可见 —— 部署尚未开通时
                   正是最需要 `/help aichat` 告诉他「用 /aichat add 开通」的时刻。
                   未开通时把整个命名空间藏起来，等于把唯一的出路也藏了。
                """
                name = getattr(handler, "name", "") or ""
                held = getattr(handler, "parent_name", "") or ""
                if name != A and held != A:
                    return True          # 别的插件的命令不归我们管
                if event is None:
                    return True          # 没有上下文（直接调用渲染）时不拦
                uid = int(event.user_id)
                if name == A and self.perm.is_superuser(uid):
                    return True          # 父节点：超级用户始终看得到
                if held == A and name in self.SCOPE_EXEMPT:
                    return True          # 范围管理子命令：豁免
                return self.check_access(self._target_of(event), uid)[0]

            self._visibility_filter = _visible
            cmds.add_visibility_filter(_visible)

    def _render_session_list(self, sessions: list, target: ChatTarget,
                             uid: int) -> str:
        """渲染一个会话列表（带当前位置标记与状态标注）。"""
        where = "本群" if target.is_group else "你的私聊"
        if not sessions:
            if target.is_group:
                return (f"本群暂无会话。/aichat new <名称> 新建"
                        f"（群里默认群会话）；\n"
                        f"参与别人建的会话需所有者先 /aichat invite <会话> <你的QQ>，"
                        f"再 /aichat attach <会话>")
            return ("还没有私聊会话。/aichat new <名称> 新建。"
                    "\n（会话不会自动创建；群里建的会话属于群，不在私聊里显示）")
        current = self._current_session(target, uid)
        lines = [f"📋 {where}的会话（{len(sessions)}）"
                 + (f"，当前：{current.id} {current.name}" if current else "，当前：无")]
        for x in sessions:
            mark = "▶ " if current is not None and x.id == current.id else "  "
            tag = ""
            if x.owner_kind == OWNER_USER and int(x.owner_id) == uid:
                tag = ""
            elif uid in set(x.guests):
                tag = "（受邀）"
            else:
                tag = "（未受邀）"
            lines.append(f"{mark}{x.describe()}{tag}")
        lines.append("切换：/aichat attach <会话>　新建：/aichat new <名称>")
        lines.append(GROUP_AT_HINT if target.is_group else "")
        return "\n".join(ln for ln in lines if ln)

    def _list_for_person(self, user_id: int, target: ChatTarget,
                         viewed_by: int = 0) -> str:
        """超级用户视角：某个人名下的私聊会话 + 他所在群的群聊会话。"""
        own = self.store.list(owner_kind=OWNER_USER, owner_id=user_id,
                              limit=self.store.max_sessions)
        invited = self.store.list(owner_kind=OWNER_USER, guest_id=user_id,
                                  limit=self.store.max_sessions)
        seen = {x.id for x in own}
        own += [x for x in invited if x.id not in seen]
        lines = [f"📋 用户 {user_id} 的私聊会话（{len(own)}）"]
        for x in own:
            lines.append(f"  {x.describe()}"
                         + ("（受邀）" if user_id in set(x.guests) else ""))
        lines.append(f"  焦点：{self._focus_summary_of_user(user_id)}")
        return "\n".join(lines)

    def _focus_summary_of_user(self, user_id: int) -> str:
        keys = [k for k in self.store.focus_table()
                if k.endswith(f"@user:{int(user_id)}")]
        if not keys:
            return "（未设置）"
        return "、".join(_focus_label(k) for k in sorted(keys))

    def _do_switch(self, token: str, event: MessageEvent) -> str:
        """把会话设成「这个人在这个聊天里的焦点」。

        「都在哪些地方用它」由每个人各自决定：同一个人可以在群里用它、也可以在
        自己的私聊里用它；群里别人要用，得先被邀请、再自己 attach。

        **直接切换**：这里没有 `--force` 确认，也不要求先 `/aichat detach` ——
        切换是**覆盖焦点**，本来就不会丢任何东西；旧版本却要先拒一次、让用户加
        `--force=true` 再来一遍，纯属多一步。

        **暂存区**：里面有未提交的分段时**照切**，分段留在原地（暂存区按「聊天
        地址」存，不跟会话走）。所以回执里明说「N 段仍在暂存区、会并入新会话」——
        否则用户以为切走了、结果那几段跟进了新会话。
        """
        target = self._target_of(event)
        uid = int(event.user_id)
        session = self.store.resolve(token)
        if session is None:
            return (f"找不到会话：{token}。用 /aichat list 查看；"
                    f"新建用 /aichat new <名称>")
        ok, why = self._may_attach(session, target, uid)
        if not ok:
            return f"⚠️ {why}"
        existing = self._current_session(target, uid)
        where = f"（原 {existing.id}「{existing.name}」）" if existing is not None \
            and existing.id != session.id else ""
        self.store.set_focus(target, uid, session.id)
        staged = len(self.buffer.peek(target.conversation_key).get("items") or [])
        shared = (session.guests or session.owner_kind == OWNER_GROUP)
        note = ("\n⚠️ 这个会话是**共用**的：上下文与参与者完全共享，"
                "你说的和它答的，其他参与者都看得到"
                if shared and int(session.owner_id) != uid
                or (shared and session.owner_kind == OWNER_GROUP) else "")
        keep = (f"\nℹ️ 暂存区里还有 {staged} 段没提交的内容，"
                f"会并入这个会话（要丢掉先 /aichat detach）" if staged else "")
        return (f"✅ 你在此处的会话已切到 {session.id}「{session.name}」{where}"
                f"（{target.label}，所有者：{session.owner_label}）{note}{keep}\n"
                f"直接说话即可开始（群里先 @我；长内容可分段发，"
                f"最后用 {self._term_hint()} 收尾）")

    def _root_description(self) -> str:
        """命令组本体的描述 —— 统一 `/help aichat` 的首屏。

        这里说清**不是命令**的几件事：直接说话就能对话、结束符是什么、
        只对绑定的群/私聊开放、以及「群里每人各有会话 + 受邀才能参与」。
        子命令清单由框架自己渲染，不在这里重复。
        """
        terms = self.cfg["trigger"].get("terminators") or []
        shown = " / ".join(str(t) for t in terms) or "//"
        groups = self._bound_groups()
        users = self._bound_ids("users")
        empty = self._empty_means()
        def _n(vals, unit):
            if vals:
                return f"{len(vals)} 个{unit}"
            return "未限定" if empty == "allow" else "无"
        at = "先 @我，" if self.cfg["trigger"].get("group_require_at") else ""
        return (f"AI 对话。"
                f"不用命令也能对话：私聊直接说话，群里{at}长内容可分段发，"
                f"最后用 `{shown}` 收尾。")

    def _do_think(self, params, event: MessageEvent) -> str:
        """`/aichat think` —— 查看/切换思考强度。

        默认只改**当前会话**（「这件事要不要深度想」和具体会话绑在一起更合理）；
        `--global_=true` 才动全局默认（写回 config.json）。
        """
        target = self._target_of(event)
        uid = int(event.user_id)
        session = self._current_session(target, uid)
        raw = str(getattr(params, "level", "") or "").strip()
        use_global = bool(getattr(params, "global_", False))
        show = bool(getattr(params, "show", False))

        if not raw or show:
            return self._think_status(session, uid)

        spec = spec_from_text(raw)
        if spec.note.startswith("不认识"):
            return (f"不认识思考强度 {raw!r}。\n"
                    f"可用：off / {(' / '.join(LEVELS))} / 预算（如 4k）/ "
                    f"thinking:on|off / on|off\n"
                    f"（{VALIDATION_HINT}）")

        if use_global:
            self.cfg["llm"]["reasoning"]["default"] = raw
            self.config.set("llm.reasoning.default", raw)   # 内存 + 落盘
            self.reload_config()
            scope_desc = "全局默认（已写入 config.json）"
        elif session is None:
            return ("你在此处还没有会话，无法设置会话级思考强度。\n"
                    "先说话建会话，或加 --global_=true 改全局默认。\n"
                    + self._think_status(None, uid))
        else:
            self._set_session_reasoning(session, raw)
            scope_desc = f"会话 {session.id}「{session.name}」"

        return f"🧠 已将 {scope_desc} 的思考强度设为：{spec.label()}\n" \
               f"{self._think_status(session, uid)}"

    def _think_status(self, session: Optional[Session], user_id: int = 0) -> str:
        """思考强度状态卡：意图 → 实际会发的 payload。"""
        session_raw = self._session_reasoning(session) if session else ""
        effective = self.effective_reasoning(session)
        lines = ["🧠 思考强度"]
        lines.append("  全局默认："
                     + self.reasoning.describe(
                         self.reasoning.spec_for(self.llm.model)))
        if session is not None:
            from_model = (self.reasoning.per_model or {}).get(self.llm.model or "")
            source = ("会话设置" if session_raw
                      else (f"模型专用（{from_model}）" if from_model not in (None, "")
                            else "全局默认"))
            lines.append(f"  当前会话：{session.id}「{session.name}」（{source}）")
            lines.append(f"  实际生效：{self.reasoning.describe(effective)}")
        else:
            lines.append("  实际生效：同全局默认（此处无会话）")
        max_tokens = self.reasoning.min_max_tokens(effective, self.llm.max_tokens)
        note = ""
        if max_tokens > self.llm.max_tokens:
            note = f"（思考会把 max_tokens 从 {self.llm.max_tokens} 抬到下界）"
        lines.append(f"  max_tokens：{max_tokens}{note}")
        lines.append("  压缩摘要："
                     + self.compaction_reasoning.describe(
                         self.compaction_reasoning.spec_for(self.llm.model)))
        lines.append(f"\n  方言：{self.reasoning.dialect}"
                     "（换端点若报参数不识别，改 llm.reasoning.dialect）")
        lines.append(f"  切换：/aichat think <off|{'|'.join(LEVELS)}|4k|"
                     "thinking:on|off> [--global_=true]")
        if effective.wants_thinking:
            lines.append("  ⚠️ 思考 token 计入 completion：max_tokens 给小了会"
                         "「想完但没字数回答」（content 为空）")
        return "\n".join(lines)

    def _do_invite(self, params, event: MessageEvent, *, invite: bool) -> str:
        """邀请 / 取消邀请 —— 只有会话所有者（群归属的会话由群主/管理员代管）。

        邀请只授予「可以 attach」的资格；**对方要自己 attach** 才会真的用上。
        """
        target = self._target_of(event)
        uid = int(event.user_id)
        session = self.store.resolve(params.session)
        if session is None:
            return f"找不到会话：{params.session}"
        ok, why = self._may_manage(session, uid)
        if not ok and not self.perm.is_superuser(uid):
            return f"⚠️ {why}"
        guest = int(params.user_id)
        if guest == uid:
            return "不用邀请自己 —— 你是所有者，本来就能用"
        self._invited_cache.pop(guest, None)       # 名单变了，缓存的判定要失效
        if invite:
            if not self.store.invite(session.id, guest):
                return f"{guest} 已经在受邀名单里了"
            return (f"✅ 已邀请 {guest} 参与会话 {session.id}「{session.name}」\n"
                    f"对方需**自己**执行 /aichat attach {session.id} 才会用上"
                    f"（群里或私聊都可以）\n"
                    f"当前受邀名单：{self.store.guests_of(session.id)}")
        if not self.store.uninvite(session.id, guest):
            return f"{guest} 不在受邀名单里"
        return (f"✅ 已取消 {guest} 对会话 {session.id} 的邀请"
                f"（他基于这个会话的当前会话也一并摘掉）\n"
                f"当前受邀名单：{self.store.guests_of(session.id)}")

    def _split_role_args(self, params, event: MessageEvent):
        """把 `/aichat role` 的位置参数拆成 `(会话, 角色文本)`。

        这个命令的位置参数天生有歧义 ——「第一格」可能是会话名，也可能是角色设定
        的开头。三条用法都要成立：

            /aichat role                        看当前会话
            /aichat role 你是一只猫              设当前会话（「你是一只猫」是设定）
            /aichat role s0003 你是猫            指定会话 + 设定

        **判据是「第一格能不能解析成一个真会话」**，而不是「它长不长 / 像不像 id」：
        后者要靠猜，而猜错的代价是把用户的角色设定当成会话名（实测：
        `/aichat role 你是一只猫娘，说话要带喵` 报「找不到会话：你是一只猫娘…」）。

        能解析出来才当会话；解析不出来就整句当设定，落在当前会话上。
        """
        first = str(getattr(params, "session", "") or "").strip()
        rest = " ".join(getattr(params, "text", None) or []).strip()
        found = self.store.resolve(first) if first else None
        if found is not None:
            # 第一格是真的会话名 → 会话 = 它，剩下的才是设定
            return found, rest
        text = " ".join(x for x in (first, rest) if x).strip()
        session = self._current_session(self._target_of(event), int(event.user_id))
        if session is None:
            # 没有当前会话：把「找不到会话」的判断留给调用方 —— 它知道用户到底
            # 写没写东西，能给出比这里更准的回话
            return None, text
        return session, text

    async def _status_text(self, event: MessageEvent) -> str:
        target = self._target_of(event)
        uid = int(event.user_id)
        session = self._current_session(target, uid)
        pending = self.buffer.peek(target.conversation_key).get("items") or []
        if session is None:
            return (f"你在{target.label}还没有当前会话。\n"
                    f"/aichat new <名称> 新建；"
                    f"参与别人的会话需所有者 /aichat invite 你，"
                    f"再 /aichat attach <会话>。")
        history = self.store.history(session.id)
        tokens = sum(m.approx_tokens(self.policy.chars_per_token) for m in history)
        ratio = tokens / max(1, self.policy.max_tokens)
        bar_len = 12
        filled = min(bar_len, int(ratio * bar_len))
        bar = "█" * filled + "░" * (bar_len - filled)
        lines = [
            f"💬 会话 {session.id}「{session.name}」",
            f"归属：{session.owner_label}"
            + ("（你的）" if (session.owner_kind == OWNER_USER
                            and int(session.owner_id) == int(event.user_id))
               else ("（受邀参与）" if int(event.user_id) in set(session.guests)
                     else "")),
            f"受邀：{'、'.join(map(str, session.guests)) if session.guests else '（无，只有所有者能用）'}",
            f"位置：{target.label}（{target.key}）",
            f"历史：{len(history)} 条（累计 {session.message_count}）"
            f" · 压缩 {session.compact_count} 次",
            f"上下文：[{bar}] {int(tokens)}/{self.policy.max_tokens} 估算 token"
            f"（{ratio:.0%}，到 {self.policy.threshold_ratio:.0%} 自动压缩）",
        ]
        if pending:
            lines.append(f"暂存区：{len(pending)} 段待提交，发 {self._term_hint()} 提交")
        # 角色设定要在这里露出来：它每轮都影响回复，看不到会让人以为没生效
        if session.role:
            one_line = " ".join(session.role.split())
            lines.append(f"角色：{one_line[:80]}{'…' if len(one_line) > 80 else ''}"
                         f"（{len(session.role)} 字，/aichat role 查看全文）")
        else:
            lines.append("角色：（未设定，用默认人格）")
        if session.last_summary:
            lines.append(f"最近摘要：{session.last_summary[:120]}…")
        spec = self.effective_reasoning(session)
        lines.append(f"思考强度：{spec.label()}"
                     + ("（会话级）" if self._session_reasoning(session) else "（默认）")
                     + f" · 压缩摘要 {self.compaction_reasoning.spec_for(self.llm.model).label()}")
        # 掐断判据：流式之后「卡死」和「想得久」是两回事，用户排查超时时必须先
        # 知道现在用的是哪一套（详见 docs/ai-chat.md §5.2.2）
        if self.llm.stream:
            lines.append(f"掐断：流式 · 停滞 {self.llm.stall_timeout:.0f}s 判卡死 · "
                         f"单次总上限 {self.llm.timeout:.0f}s")
        else:
            lines.append(f"掐断：非流式 · 单次总超时 {self.llm.timeout:.0f}s"
                         "（开着 llm.stream 才能区分「想得久」和「卡死」）")
        if not self.llm.enabled:
            lines.append("⚠️ LLM 未配置（llm.base_url / model / api_key），无法对话")
        return "\n".join(lines)

    def _term_hint(self) -> str:
        terms = self.cfg["trigger"].get("terminators") or []
        shown = " / ".join(str(t) for t in terms) or "//"
        return f"`{shown}`"



def _arg_list_id(raw) -> Optional[int]:
    """把 `list --user=<QQ>` 的值解析成 int；非法/为空返回 None。"""
    if raw in (None, "", 0):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _focus_label(key: str) -> str:
    """把焦点键翻译成人话：`group:123@user:456` → `群 123 / 用户 456`。"""
    chat, _, who = key.partition("@user:")
    if chat.startswith("group:"):
        where = f"群 {chat[len('group:'):]}"
    else:
        body = chat[len("private:"):] if chat.startswith("private:") else chat
        gid = ""
        if "@group:" in body:
            body, g = body.split("@group:", 1)
            gid = f"（小窗自群 {g}）"
        where = f"与用户 {body} 的私聊{gid}"
    return f"{where} / 用户 {who}"


def _flatten(data: dict, prefix: str = "") -> list[str]:
    """列出所有叶子配置项的点号路径（用于「共 N 项」这类统计）。"""
    out: list[str] = []
    for key, value in (data or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.extend(_flatten(value, prefix=f"{path}."))
        else:
            out.append(path)
    return out


def _tb() -> str:
    import traceback
    return traceback.format_exc()


def _sanitize(history: list[Message]) -> tuple[list[Message], int]:
    return sanitize_history(history)


def _message_id_of(resp) -> int:
    """从发送接口的返回值里取 `message_id`（拿不到就 0）。

    返回值有两种形态：框架的 `SendMsgResult` 对象，或裸的 dict
    （测试桩与某些实现直接回 dict）。两种都认。
    """
    for getter in (lambda r: getattr(r, "message_id", None),
                   lambda r: (r or {}).get("message_id") if isinstance(r, dict) else None,
                   lambda r: ((r or {}).get("data") or {}).get("message_id")
                   if isinstance(r, dict) else None):
        try:
            got = getter(resp)
        except Exception:  # noqa: BLE001
            continue
        try:
            mid = int(got)
        except (TypeError, ValueError):
            continue
        if mid:
            return mid
    return 0


def _is_retryable(e: BaseException) -> bool:
    """这个失败值不值得再试一次。

    只认两件事：

    * `LlmError.retryable` —— 网络层异常（连不上、超时、连接被断）与
      408/409/425/429/5xx，以及「上游回了看不懂的东西」（见 `llm.py`）；
    * 直接漏出来的 `TimeoutError`（比如 `llm.py` 之外的等待超时）。

    ⚠️ **4xx 一律不重试**：参数写错、key 失效、模型名不存在 —— 再试多少次都是
    同一个结果，只会让用户多等几秒才看到错误。判据认的是 `retryable` 这个**标记**，
    而不是在这里按错误文本猜（那是 `llm.py` 的职责，它才知道每个失败的含义）。
    """
    flag = getattr(e, "retryable", None)
    if flag is not None:
        return bool(flag)
    return isinstance(e, (asyncio.TimeoutError, TimeoutError))


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _looks_like_context_overflow(exc: BaseException) -> bool:
    """上游是不是在抱怨「上下文太长」。

    各家的措辞不一样（context_length_exceeded / maximum context length /
    too many tokens / 400 invalid_request_error），只能按关键词认。
    """
    text = err_text(exc).lower()
    markers = ("context_length", "context length", "maximum context",
               "too many tokens", "reduce the length", "max_tokens is too large",
               "prompt is too long", "超过最大长度", "上下文长度")
    return any(m in text for m in markers)


def _role_value(role: Any) -> str:
    """把群角色统一成字符串。

    ⚠️ `GroupMemberRole.ADMIN` 是 `str` 枚举，但 `str(enum)` 得到的是
    `'GroupMemberRole.ADMIN'` —— 直接拿它比对 `"admin"` 永远不成立，
    群管理员的工具会被静默下掉（表现为「管理员也用不了禁言」）。
    """
    if role is None:
        return ""
    value = getattr(role, "value", role)
    text = str(value).strip()
    return text.rsplit(".", 1)[-1].lower()


class _FakeEvent:
    """给 `_display_name` / `_target_of` / `_can_use` 这类只读少量字段的方法复用。

    暂存区里只有纯 dict（JSON 往返过），直接用 `MessageEvent` 反序列化会把
    Sender 丢掉；这里做一个最小的鸭子类型，避免为了两三个字段去重建整个事件。
    """

    __slots__ = ("message_type", "user_id", "group_id", "message_id",
                 "sender", "plain_text", "message")

    def __init__(self, target: ChatTarget, meta: dict, text: str):
        self.message_type = target.kind
        self.user_id = int(meta.get("user_id") or 0)
        self.group_id = (target.chat_id if target.is_group
                         else (meta.get("group_id") or None))
        self.message_id = 0
        self.sender = _FakeSender(str(meta.get("nickname") or ""),
                                  str(meta.get("card") or ""))
        self.plain_text = text
        self.message = text


class _FakeSender:
    __slots__ = ("nickname", "card", "role", "user_id")

    def __init__(self, nickname: str, card: str = ""):
        self.nickname = nickname
        self.card = card
        self.role = None
        self.user_id = 0


class _StagingBuffer:
    """未提交输入的暂存区（落盘，重启不丢）。

    为什么落盘：用户可能在「@我 + 一长段 + 结束符」之间隔了几分钟，
    期间机器人重启（这个部署里 watchdog 会主动重启）—— 内存态会让整段输入消失。
    """

    def __init__(self, path: str, *, logger=None, max_parts: int = 12) -> None:
        self.path = path
        self.logger = logger
        self.max_parts = max(1, int(max_parts))
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = {k: v for k, v in data.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            self._data = {}
        self._migrate()

    def _migrate(self) -> None:
        """把老的「群小窗」键并到同一个人的主键上。

        暂存区现在按 `conversation_key` 存（`private:<uid>`），而早先按 `target.key`
        存 —— 群临时会话会写成 `private:<uid>@group:<gid>`。不并的话，用户在小窗里
        压栈的半截话会永远取不回来（键再也不会被用到）。
        """
        moved: dict[str, dict] = {}
        for key, state in list(self._data.items()):
            if "@group:" not in key or not key.startswith("private:"):
                continue
            plain = "private:" + key[len("private:"):].split("@group:", 1)[0]
            old_items = list(state.get("items") or [])
            self._data.pop(key, None)
            if not old_items:
                continue
            into = self._data.setdefault(plain, {"items": [], "since": time.time()})
            into["items"] = list(into.get("items") or []) + old_items
            into["items"] = into["items"][-self.max_parts * 2:]
            moved[plain] = into
        if moved:
            if self.logger:
                self.logger.info("[ai_chat] 暂存区已合并 %d 个群小窗键", len(moved))
            self.save()

    def save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError as e:
            if self.logger:
                self.logger.warning("[ai_chat] 暂存区落盘失败: %r", e)

    def peek(self, key: str) -> dict:
        return dict(self._data.get(key) or {"items": []})

    def append(self, key: str, item: dict) -> dict:
        state = self._data.setdefault(key, {"items": [], "since": time.time()})
        state.setdefault("items", []).append(item)
        state["updated"] = time.time()
        state["items"] = state["items"][-self.max_parts * 2:]
        self.save()
        return state

    def take(self, key: str) -> list[dict]:
        state = self._data.pop(key, None)
        if not state:
            return []
        self.save()
        return list(state.get("items") or [])

    def clear(self, key: str) -> None:
        if self._data.pop(key, None) is not None:
            self.save()

    def drop_message(self, key: str, operator_id: int, message_id: int) -> bool:
        """撤回：删掉暂存区里那条消息（只删自己发的，别人的撤回不越权）。"""
        state = self._data.get(key)
        if not state:
            return False
        items = state.get("items") or []
        kept = [it for it in items
                if not (int(it.get("message_id") or 0) == int(message_id)
                        and (not operator_id
                             or int(it.get("user_id") or 0) == int(operator_id)))]
        if len(kept) == len(items):
            return False
        state["items"] = kept
        if not kept and not state.get("summoned"):
            self._data.pop(key, None)
        self.save()
        return True

    def keys(self) -> list[str]:
        return list(self._data)
