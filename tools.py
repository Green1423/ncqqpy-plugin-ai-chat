"""内置工具（function calling）注册表 + 默认工具集。

设计约束（来自 DSH 的 tool 协议）：

* **工具的每一次调用与结果都要作为一条消息进历史**，不能只活在当前这一轮 ——
  否则下一轮模型会「记得自己问过、但不记得答案」，开始重复调用同一个工具。
  因此 `ToolResult.text` 必须自包含、可独立阅读。
* **工具失败不是对话失败**：任何异常都转成 `is_error=True` 的结果回喂给模型，
  由模型决定怎么向用户解释。直接把异常抛给用户是最差的选择（用户看到一个
  栈或者干脆没反应）。
* **权限在服务端判定**：模型看到的工具清单可以包含当前调用者无权使用的工具，
  但真正执行时必须再校验一次（`ctx.can_moderate`）—— 提示词注入不该变成提权。
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from napcat.tools import call_tool

from .llm import err_text
from .search import format_results

# 单个工具结果的字符上限 —— 超过就截断（避免一次 group_member_list 把上下文打爆）
MAX_RESULT_CHARS = 4000

# 工具结果里被截断时的提示语（要明确告诉模型「这是截断的」，否则它会当成全部事实）
TRUNCATED_HINT = "\n…（结果过长已截断，如需更多请缩小查询范围）"


def _as_text(value: Any) -> str:
    # `None` 表示"没有结果"，不是 JSON 的 null —— 早先漏了这一条，
    # 于是返回 None 的工具会把字面量 `null` **发给用户**（委派型工具直接把
    # 返回值发出去）。空就是空，交给调用方决定补什么话。
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=None, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class ToolContext:
    """一次工具调用可见的上下文。

    刻意**不**把 `api` / `plugin` 直接塞进来：工具只能通过这里的显式字段
    访问外部世界，这样工具的行为边界一眼可见，也方便测试时替换。
    """

    session_id: str = ""
    session_name: str = ""
    user_id: int = 0
    nickname: str = ""
    chat_kind: str = "private"          # private | group
    chat_id: int = 0
    group_id: Optional[int] = None
    group_name: str = ""
    is_superuser: bool = False
    can_moderate: bool = False          # 群主 / 管理员 / superuser
    store: Any = None                   # SessionStore
    api: Any = None                     # ApiClient
    cfg: Any = None                     # 插件配置代理
    extras: dict = field(default_factory=dict)

    @property
    def in_group(self) -> bool:
        return self.chat_kind == "group" and bool(self.chat_id)

    @property
    def at_label(self) -> str:
        if self.in_group:
            return f"{self.nickname or self.user_id}@{self.group_name or self.group_id}"
        return self.nickname or str(self.user_id)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Awaitable[Any]]
    group_only: bool = False
    requires_moderation: bool = False
    superuser_only: bool = False
    #: **委派型**：调用它等于「把这件事交给别的插件」，由那个插件自己回消息 ——
    #: 结果不回喂模型、AI 本轮也不说话（见 `main._model_loop` 的 `_DELEGATED`）
    delegates: bool = False
    #: 插件工具自带的「使用时机」提示（写进系统提示；为空就不写）
    hint: str = ""

    def schema(self) -> dict:
        """OpenAI `tools[]` 里的函数声明。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class ToolRegistry:
    """工具注册表。`tool()` 既支持装饰器，也支持直接构造。"""

    def __init__(self, logger=None) -> None:
        self.logger = logger
        self._tools: dict[str, Tool] = {}
        #: 动态工具来源（见 `set_provider`）
        self._providers: list = []

    # ---- 注册 ----

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, *, description: str,
             parameters: Optional[dict] = None, group_only: bool = False,
             requires_moderation: bool = False, superuser_only: bool = False):
        def decorator(func: Callable[..., Awaitable[Any]]):
            self.add(Tool(name=name, description=description,
                          parameters=parameters or {"type": "object", "properties": {}},
                          handler=func, group_only=group_only,
                          requires_moderation=requires_moderation,
                          superuser_only=superuser_only))
            return func
        return decorator

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    # ---- 动态来源（别的插件注册的技能）----

    def set_provider(self, provider) -> None:
        """挂一个**动态工具来源**：`provider(ctx) -> list[Tool]`。

        为什么是"每轮重算"而不是"加载时快照"：提供技能的插件会热重载、
        会被禁用/卸载 —— 快照下来的工具会指向已经死掉的 handler（调用时才炸，
        模型只会看到一句莫名其妙的异常）。

        ⚠️ **可以挂多个**：技能（命令映射）与插件工具（返回给模型）是两个
        独立来源，各自 `set_provider` 一次。早先只存一个，后挂的会把先挂的顶掉 ——
        表现为「注册了插件工具之后，所有技能工具都消失了」。
        """
        if provider is not None:
            self._providers.append(provider)

    @property
    def _provider(self):
        return self._providers[0] if self._providers else None

    def dynamic(self, ctx: ToolContext) -> list[Tool]:
        out: list[Tool] = []
        for provider in list(self._providers):
            try:
                got = provider(ctx)
            except Exception as e:  # noqa: BLE001 - 一个来源坏了不影响别的
                if self.logger:
                    self.logger.warning("[ai_chat] 动态工具取用失败: %r", e)
                continue
            out.extend(t for t in got or [] if isinstance(t, Tool))
        return out

    # ---- 可见性 ----

    def visible(self, ctx: ToolContext) -> list[Tool]:
        """当前上下文可用的工具（顺序稳定，便于 prompt cache 命中）。"""
        out = []
        for name in sorted(self._tools):
            t = self._tools[name]
            if t.group_only and not ctx.in_group:
                continue
            if t.requires_moderation and not (ctx.can_moderate or ctx.is_superuser):
                continue
            if t.superuser_only and not ctx.is_superuser:
                continue
            out.append(t)
        # 动态工具（别的插件的技能）：同名以内置为准，避免技能盖掉内置工具
        taken = {t.name for t in out}
        for t in self.dynamic(ctx):
            if t.name in taken or t.name in self._tools:
                continue
            if t.group_only and not ctx.in_group:
                continue
            if t.superuser_only and not ctx.is_superuser:
                continue
            out.append(t)
        return out

    def schemas(self, ctx: ToolContext) -> list[dict]:
        return [t.schema() for t in self.visible(ctx)]

    # ---- 执行 ----

    async def call(self, name: str, arguments: dict, ctx: ToolContext) -> tuple[str, bool]:
        """执行工具。返回 `(结果文本, 是否失败)` —— **永不抛异常**。"""
        tool = self._tools.get(name)
        if tool is None:
            # 也可能是别的插件注册的技能（动态工具）
            tool = next((t for t in self.dynamic(ctx) if t.name == name), None)
        if tool is None:
            avail = self.names() + [t.name for t in self.dynamic(ctx)]
            return (f"没有名为 `{name}` 的工具。可用工具：{', '.join(sorted(avail))}",
                    True)
        # 二次权限校验：模型可能被诱导去调用调用者无权使用的工具
        if tool.group_only and not ctx.in_group:
            return (f"工具 `{name}` 只能在群聊里使用。", True)
        if tool.requires_moderation and not (ctx.can_moderate or ctx.is_superuser):
            return (f"你没有使用 `{name}` 的权限（需要群主/管理员）。", True)
        if tool.superuser_only and not ctx.is_superuser:
            return (f"工具 `{name}` 仅超级用户可用。", True)
        # ⚠️ **两段 try，不能合成一段**：`TypeError` 只有在**调用那一刻**才是
        # "参数对不上"；`await` 之后抛的 TypeError 是工具内部的 bug。
        # 合在一起会把内部错误报成「参数错误」，于是模型以为是自己传错了参数、
        # 换个问法重试 —— 用户白等一轮，真正的 bug 被掩盖（线上 s0020 就这样）。
        try:
            result = tool.handler(ctx, **(arguments or {}))
        except TypeError as e:
            # 参数对不上（模型传错名字）—— 明确回喂，模型会自己改
            return (f"工具 `{name}` 参数错误：{e}", True)
        except Exception as e:  # noqa: BLE001 - 工具失败不该拖垮整轮对话
            if self.logger:
                self.logger.warning("[ai_chat] 工具 %s 执行失败: %r", name, e)
            return (f"工具 `{name}` 执行异常：{err_text(e)}", True)
        try:
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:  # noqa: BLE001 - 同上；这里**不是**参数问题
            if self.logger:
                self.logger.warning("[ai_chat] 工具 %s 执行失败: %r", name, e)
            return (f"工具 `{name}` 执行异常：{err_text(e)}", True)

        text = _as_text(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + TRUNCATED_HINT
        # ⚠️ 空结果就返回空串，**不要**在这里塞占位符：委派型工具会把
        # 返回值直接发给用户，塞了它就变成用户真的收到「（工具返回空结果）」
        # （线上就这么给用户发过一句废话）。占位符由模型侧自己补。
        return (text, False)


def _fmt_time(ts: datetime) -> str:
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return (f"{ts.strftime('%Y-%m-%d %H:%M:%S')} "
            f"{weekdays[ts.weekday()]}（时区 {ts.strftime('%Z') or '本地'}）")


#: 技能结果的上限 —— 与内置工具同一个量级（模型读不动更长的东西）
SKILL_MAX_CHARS = 3000

#: `CommandSchema` 的字段类名（去 Field）→ JSON Schema 类型
_FIELD_JSON_TYPE = {
    "bool": "boolean", "int": "integer", "float": "number", "qq": "integer",
    "duration": "integer", "choice": "string", "variadic": "string",
    "str": "string",
}


def command_tool_schema(handler) -> dict:
    """命令的 `CommandSchema` → 给模型看的 JSON Schema（**不另写一份**）。

    这是「技能 = 命令映射」的关键：参数定义只有一处（命令的 Schema），
    模型看到的工具参数就是从那儿翻译出来的，两边永远不会漂移。
    """
    props: dict = {}
    required: list[str] = []
    schema = getattr(handler, "schema", None)
    fields = schema.get_fields() if schema is not None else {}
    for fname, field in fields.items():
        type_name = type(field).__name__.replace("Field", "").lower()
        prop: dict = {"type": _FIELD_JSON_TYPE.get(type_name, "string")}
        choices = list(getattr(field, "choices", None) or [])
        if choices:
            prop["type"] = "string"          # ChoiceField：直接告诉模型可选值
            prop["enum"] = [str(c) for c in choices]
        desc = str(getattr(field, "description", "") or "")
        if desc:
            prop["description"] = desc
        props[fname] = prop
        if getattr(field, "required", False) and getattr(field, "default", None) is None:
            required.append(fname)
    out: dict = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def register_skill_tools(reg: ToolRegistry, *, skills, cfg, cmds,
                         logger=None) -> None:
    """把**别的插件注册的技能**接成工具 —— 技能是「命令映射」。

    提供方只写一行 `self.register_skill("search", command="插件名.命令")`；
    这里负责：解析出那条命令 → 用它的 Schema 生成工具参数 → 调用时交给
    `main._run_skill_command()`（合成真实事件走 `CommandManager.handle()`）。
    所以**权限/作用域/Schema 校验全部由命令系统判定**，AI 这条路绕不过去。

    `cfg` 可以是取值函数（配置热重载后立刻生效）。
    """
    def provider(ctx):
        conf = cfg() if callable(cfg) else cfg
        conf = conf if isinstance(conf, dict) else {}
        if not bool(conf.get("enabled", True)):
            return []
        allow = [str(x) for x in (conf.get("allow") or [])]
        deny = {str(x) for x in (conf.get("deny") or [])}
        out = []
        for skill in skills.list(exclude="ai_chat"):
            if allow and skill.name not in allow:
                continue
            if not allow and skill.name in deny:
                continue
            found = cmds.find_command(skill.command) if hasattr(
                cmds, "find_command") else None
            handler = found[1] if found else None
            if handler is None or getattr(handler, "disabled", False):
                continue          # 命令没了/被禁用 → 不暴露死工具
            out.append(_skill_tool(skill, handler, logger=logger))
        return out

    reg.set_provider(provider)


def _skill_tool(skill, handler, *, logger=None) -> Tool:
    """技能 → 工具（名字用 `plugin_skill`，参数由命令 Schema 翻译）。"""

    async def tool_handler(ctx, **_ignored):
        args = dict(_ignored)
        runner = (ctx.extras or {}).get("run_skill")
        if runner is None:
            return "（当前环境不支持触发插件命令）"
        try:
            text = await runner(skill, args)
        except Exception as e:  # noqa: BLE001 - 命令失败不该拖垮对话
            if logger:
                logger.warning("[ai_chat] 技能 %s 触发失败: %r", skill.name, e)
            raise RuntimeError(f"技能 {skill.name} 触发失败：{err_text(e)}") from e
        text = str(text or "")
        if len(text) > SKILL_MAX_CHARS:
            text = text[:SKILL_MAX_CHARS] + "…（已截断）"
        return text

    return Tool(
        name=skill.tool_name,
        description=(f"[{skill.plugin}] {skill.description or handler.description} "
                     f"（相当于用户发 {skill.command_line()}）").strip(),
        parameters=command_tool_schema(handler),
        handler=tool_handler,
        # 委派型：命令自己回消息，AI 不接话、也不转述
        delegates=True,
        # 「什么时候该用我」写进系统提示（与插件工具同一节）；
        # 不写就退回 `description` —— 至少让模型知道这工具是干什么的
        hint=str(getattr(skill, "prompt_hint", "") or ""),
    )


def register_plugin_tools(reg: ToolRegistry, *, tools, cfg, logger=None) -> None:
    """把**别的插件注册的工具**接进本插件的工具表 —— 与技能不同：

    * 工具自己带 handler 与 JSON Schema（不映射命令）；
    * 调用结果**返回给模型**（`delegates=False`），由 AI 组织答复；
    * 权限/作用域在**这里**判（`permission` / `scope`），不通过就不暴露。

    `cfg` 可以是取值函数（配置热重载后立刻生效）。
    """
    def provider(ctx):
        conf = cfg() if callable(cfg) else cfg
        conf = conf if isinstance(conf, dict) else {}
        if not bool(conf.get("enabled", True)):
            return []
        allow = [str(x) for x in (conf.get("allow") or [])]
        deny = {str(x) for x in (conf.get("deny") or [])}
        out = []
        for item in tools.list(exclude="ai_chat"):
            if allow and item.name not in allow:
                continue
            if not allow and item.name in deny:
                continue
            if not _tool_allowed(item, ctx):
                continue
            out.append(_plugin_tool(item, logger=logger))
        return out

    reg.set_provider(provider)


def _tool_allowed(item, ctx) -> bool:
    """工具声明的权限/作用域是否允许**此刻这次调用**。"""
    scope = str(getattr(item, "scope", "all") or "all").lower()
    if scope == "private" and ctx.in_group:
        return False
    if scope == "group" and not ctx.in_group:
        return False
    need = str(getattr(item, "permission", "") or "").lower()
    if not need:
        return True
    if need == "superuser":
        return bool(getattr(ctx, "is_superuser", False))
    if need == "admin":
        return bool(getattr(ctx, "is_superuser", False)
                    or getattr(ctx, "is_admin", False))
    return True


def _plugin_tool(item, *, logger=None) -> Tool:
    """工具 → 本插件的 `Tool`（结果**返回给模型**，非委派）。"""

    async def tool_handler(ctx, **_args):
        try:
            # ctx 只在工具自己声明了第二个参数时才传（见 `napcat.tools.call_tool`）：
            # 业务白名单（如「这个群允许用本插件吗」）需要知道**谁在调**，
            # 而 `permission` / `scope` 只能表达「超管 / 群聊」这种粗粒度。
            text = await call_tool(item, _args, ctx)
        except Exception as e:  # noqa: BLE001 - 工具失败不该拖垮对话
            if logger:
                logger.warning("[ai_chat] 工具 %s 调用失败: %r", item.name, e)
            raise RuntimeError(f"工具 {item.name} 调用失败：{err_text(e)}") from e
        return text or "（工具没有返回内容）"

    return Tool(
        name=item.tool_name,
        description=(f"[{item.plugin}] {item.describe()}"),
        parameters=item.parameters,
        handler=tool_handler,
        delegates=False,        # ★ 结果进模型上下文，不直接发给用户
        hint=str(getattr(item, "prompt_hint", "") or ""),
    )


def register_builtin_tools(reg: ToolRegistry) -> None:
    """注册内置工具集。

    这里只放**自给自足**的工具：数据来自本插件自己的会话库、或框架已经提供的
    `api`（群成员、禁言……）。任何「别的插件的知识源」都不在这里 ——
    要接外部 HTTP 接口请走 `tools.custom`，那是配置驱动的，不需要改代码。
    """

    @reg.tool("get_current_time",
              description="查询当前的日期、时间与星期。需要判断“今天/明天”或换算时间时先调用它。")
    async def _get_current_time(ctx: ToolContext) -> str:
        return _fmt_time(datetime.now().astimezone())

    @reg.tool("get_session_info",
              description="查询当前会话自身的元信息：名称、消息条数、已压缩次数、参与人。",
              parameters={"type": "object", "properties": {}})
    async def _get_session_info(ctx: ToolContext) -> dict:
        store = ctx.store
        info: dict[str, Any] = {
            "会话ID": ctx.session_id,
            "会话名称": ctx.session_name,
            "聊天类型": "群聊" if ctx.in_group else "私聊",
            "当前说话人": ctx.at_label,
            "历史消息条数": 0,
        }
        if store is not None:
            s = store.get(ctx.session_id)
            if s is not None:
                info["历史消息条数"] = s.message_count
                info["已压缩次数"] = s.compact_count
                info["最近一次摘要"] = s.last_summary or "（无）"
                info["附着地点"] = store.attachments(s.id) or ["（未附着）"]
        return info

    @reg.tool("search_history",
              description=("在当前会话（或指定会话）的历史对话里按关键词做子串检索，"
                           "返回命中消息及其上下文。想知道“之前聊过什么”时用它。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "query": {"type": "string", "description": "检索关键词"},
                      "session_id": {"type": "string",
                                     "description": "可选，限定在某个会话 id 内检索"},
                      "all_sessions": {"type": "boolean",
                                       "description": "是否检索全部会话（默认 false）"},
                      "limit": {"type": "integer", "description": "最多返回条数，默认 6"},
                  },
                  "required": ["query"],
              })
    async def _search_history(ctx: ToolContext, query: str = "",
                              session_id: str = "", all_sessions: bool = False,
                              limit: int = 6) -> Any:
        store = ctx.store
        if store is None:
            return "历史存储不可用"
        if not str(query).strip():
            return "请提供检索关键词 query"
        sid = session_id or ("" if all_sessions else ctx.session_id)
        # 只允许检索「自己参与过的会话」：群会话看群，私聊看本人
        owner = None if (all_sessions and ctx.is_superuser) else ctx.user_id
        hits = store.search(str(query), session_id=sid or None, owner_id=owner,
                            limit=max(1, min(int(limit or 6), 20)))
        if not hits:
            return f"没有找到包含「{query}」的历史消息"
        return hits

    @reg.tool("query_group_members",
              description=("查询群成员。可按关键词过滤（匹配昵称/群名片/QQ号），"
                           "返回成员的角色与名片。仅在群聊里可用。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "keyword": {"type": "string", "description": "可选过滤关键词"},
                      "limit": {"type": "integer", "description": "最多返回条数，默认 20"},
                  },
              },
              group_only=True)
    async def _query_group_members(ctx: ToolContext, keyword: str = "",
                                   limit: int = 20) -> Any:
        if ctx.api is None:
            return "API 不可用"
        members = await ctx.api.get_group_member_list(str(ctx.chat_id))
        kw = str(keyword or "").strip().lower()
        out = []
        for m in members or []:
            uid = getattr(m, "user_id", None)
            card = getattr(m, "card", "") or ""
            nick = getattr(m, "nickname", "") or ""
            role = getattr(m, "role", "") or "member"
            if kw and kw not in str(uid).lower() and kw not in card.lower() \
                    and kw not in nick.lower():
                continue
            out.append({"qq": uid, "名片": card, "昵称": nick, "角色": role})
            if len(out) >= max(1, min(int(limit or 20), 100)):
                break
        return out or "没有匹配的群成员"

    @reg.tool("query_group_info",
              description="查询当前群的信息：群名称、人数、自己的角色。仅在群聊里可用。",
              parameters={"type": "object", "properties": {}},
              group_only=True)
    async def _query_group_info(ctx: ToolContext) -> Any:
        if ctx.api is None:
            return "API 不可用"
        info = await ctx.api.get_group_info(str(ctx.chat_id))
        return {
            "群号": ctx.chat_id,
            "群名称": getattr(info, "group_name", "") or ctx.group_name,
            "人数": getattr(info, "member_count", None),
            "最高管理员": getattr(info, "max_member_count", None),
        }

    @reg.tool("mute_member",
              description=("禁言群成员（单位：秒；0 表示解除禁言）。仅群主/管理员可用。"
                           "禁言前应向用户说明原因。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "user_id": {"type": "integer", "description": "要禁言的 QQ 号"},
                      "duration": {"type": "integer", "description": "禁言秒数，0=解除"},
                      "reason": {"type": "string", "description": "禁言原因（会写进回复）"},
                  },
                  "required": ["user_id", "duration"],
              },
              group_only=True, requires_moderation=True)
    async def _mute_member(ctx: ToolContext, user_id: int, duration: int,
                           reason: str = "") -> Any:
        if ctx.api is None:
            return "API 不可用"
        if int(user_id) == int(ctx.user_id):
            return "不能禁言当前说话人自己"
        max_seconds = int((ctx.extras or {}).get("max_mute_seconds", 2592000) or 2592000)
        secs = max(0, min(int(duration or 0), max_seconds))
        await ctx.api.set_group_ban(group_id=str(ctx.chat_id),
                                    user_id=str(int(user_id)), duration=secs)
        return {"结果": "已解除禁言" if secs == 0 else f"已禁言 {secs} 秒",
                "目标": int(user_id), "原因": reason or "（未说明）"}

    @reg.tool("list_group_bans",
              description="列出本群当前被禁言的成员。仅群主/管理员可用。",
              parameters={"type": "object", "properties": {}},
              group_only=True, requires_moderation=True)
    async def _list_group_bans(ctx: ToolContext) -> Any:
        if ctx.api is None:
            return "API 不可用"
        try:
            data = await ctx.api.get_group_ban_list(str(ctx.chat_id))
        except Exception as e:  # noqa: BLE001
            return f"查询失败：{err_text(e)}"
        return data


def register_web_tools(reg: ToolRegistry) -> None:
    """注册联网搜索工具（`web_search` / `fetch_page`）。

    客户端从 `ctx.extras["web"]` 取 —— 工具模块**不持有** HTTP 客户端，
    这样测试能注入一个假客户端，而解析、截断、SSRF 判定仍是真代码在跑。

    为什么要两个工具（而不是「搜索顺便抓正文」）：搜索摘要经常只有几十字，
    而回答「XX 什么时候」「为什么」这类问题往往要读正文。拆开之后模型可以
    先搜、挑一两个链接再读，token 花在它自己判断有用的页面上 —— 也是「一个提问
    多段回答」的关键：多个来源各自成段，而不是把一堆摘要糊成一段。
    """

    @reg.tool("web_search",
              description=("联网搜索（实时信息用它）。返回编号、标题、链接与摘要。"
                           "问时效性内容（新闻、价格、版本、赛事、公告）或你"
                           "不确定的事实时先搜，不要凭记忆答。要读某条的完整内容，"
                           "用 fetch_page 传它的链接。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "query": {"type": "string", "description": "搜索关键词"},
                      "count": {"type": "integer",
                                "description": "返回几条，默认 5，最多 10"},
                  },
                  "required": ["query"],
              })
    async def _web_search(ctx: ToolContext, query: str = "",
                          count: int = 5) -> Any:
        web = (ctx.extras or {}).get("web")
        if web is None:
            return "联网搜索不可用（插件未启用搜索，或管理员把它关了）"
        limit = max(1, min(int(count or 5), 10))
        out = await web.search(str(query or ""), count=limit)
        if out.error and not out.results:
            return f"搜索失败：{out.error}"
        text = format_results(out.results)
        if out.error:
            text += f"\n（注意：{out.error}）"
        return text

    @reg.tool("fetch_page",
              description=("读取一个网页的正文（纯文本）。先用 web_search 拿到链接，"
                           "再对值得细看的链接调用它。只支持 http/https 公网地址。"
                           "**要找图就用它**：返回的末尾会列出页面里的图片直链，"
                           "挑相关的交给 fetch_image 下载（不要为了找图反复读页面）。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "url": {"type": "string", "description": "要读取的网页链接"},
                  },
                  "required": ["url"],
              })
    async def _fetch_page(ctx: ToolContext, url: str = "") -> Any:
        web = (ctx.extras or {}).get("web")
        if web is None:
            return "网页读取不可用（插件未启用搜索）"
        if not str(url or "").strip():
            return "请给出要读取的链接 url"
        return await web.read_page(
            str(url),
            limit=int((ctx.extras or {}).get("max_page_chars") or 4000),
            # 附上页面里的图片直链：模型没法从纯文本里「看见」图片，
            # 这是它找到 fetch_image 可用地址的唯一途径
            max_images=int((ctx.extras or {}).get("max_page_images") or 6))


def register_image_tool(reg: ToolRegistry) -> None:
    """注册 `fetch_image`：从网上下载一张图片，配在本轮回复里。

    为什么要这个工具：模型能搜、能读网页，但**发不出图** —— 它在正文里写
    `![](url)` 或贴个链接，QQ 里既不会显示图片、用户还得自己点开。有了它，
    「找几张 xx 的图片」这类要求才真的做得到。

    服务与 HTTP 客户端都从 `ctx.extras` 取（`webimg` / `web`），工具模块不持有
    它们 —— 与 `web_search` / `plot_function` 同一套路，测试能注入假的。
    """

    @reg.tool("fetch_image",
              description=("从网上下载一张图片，**附在本轮回复里**发给用户"
                           "（QQ 里直接显示成图片，比贴链接有用得多）。"
                           "url 必须是**图片直链**：http/https，指向 .png/.jpg/.gif/"
                           ".webp/.bmp 这类文件本身；网页地址用 fetch_page。"
                           "通常先用 web_search 找到图片页或直链再调它。"
                           "一轮最多配几张（默认 3 张），别为了凑数硬发。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "url": {"type": "string",
                              "description": "图片直链（http/https）"},
                      "caption": {"type": "string",
                                  "description": "这张图是什么，一句话"
                                                 "（会记在图片旁边，也方便你自己"
                                                 "在正文里对上号）"},
                  },
                  "required": ["url"],
              })
    async def _fetch_image(ctx: ToolContext, url: str = "",
                           caption: str = "") -> Any:
        service = (ctx.extras or {}).get("webimg")
        web = (ctx.extras or {}).get("web")
        if service is None:
            return "配图不可用（插件没开启「网图配图」）。"
        if web is None:
            return "配图不可用（下载图片需要联网能力，当前没开搜索）。"
        if not str(url or "").strip():
            return "请给出要下载的图片直链 url"
        return await service.fetch(web, str(url), caption=str(caption or ""))


#: 画图工具在回复里留下的占位标记。模型不会自己写这个标记 —— 它由工具**返回**，
#: 主流程在发消息前把标记换成真正的图片。
#: 一张图里最多画几条曲线。再多颜色就分不清了，而且模型也说不明白。
_PLOT_MAX_CURVES = 6

PLOT_MARKER_PREFIX = "[[plot:"
PLOT_MARKER_SUFFIX = "]]"


def register_plot_tool(reg: ToolRegistry) -> None:
    """注册函数图像工具 `plot_function`。

    渲染依赖从 `ctx.extras["plot"]` 取（一个 `PlotService`），工具模块不持有
    渲染器 —— 与 `web_search` 同一个套路，测试能注入假的。

    **为什么返回值里要带一个标记**：QQ 的消息链需要图片 URL，而工具结果只是一段
    文本，没法把图片直接塞回模型上下文（本项目客户端只发纯文本）。所以工具先把
    图渲出来、把路径写进标记，主流程再把它换成图片。
    """

    @reg.tool("plot_function",
              description=("画函数或方程的图像（坐标系 + 曲线），给用户看一张图。"
                           "用户要求「画函数图像 / 画个图看看」「这个函数长什么样」"
                           "「画出 y=…」「画个圆/椭圆/双曲线」「解这个方程的图形」"
                           "时必须用这个工具，不要只用文字描述形状。"
                           "expression 支持三种写法，自动识别："
                           "① 一元函数：`x^2-4`、`\\sin(2x)`、`1/x`；"
                           "② 方程（含 y 就是方程）：`x^2+y^2=25`（圆）、"
                           "`\\frac{x^2}{9}+\\frac{y^2}{4}=1`（椭圆）、`x^2-y^2=4`（双曲线）；"
                           "③ 参数方程：`x=\\cos(t), y=\\sin(t)`（用变量 t）。"
                           "LaTeX 与普通写法都认。"
                           "**要在一张图里画多条曲线就填 expressions 数组**"
                           "（求交点、比较增减快慢、函数配渐近线等都必须同图才看得出），"
                           "它们会依次配上不同颜色；只有一条时用 expression 即可。"),
              parameters={
                  "type": "object",
                  "properties": {
                      "expression": {
                          "type": "string",
                          "description": "函数或方程（只画一条时用这个）。"
                                         "一元函数如 x^2-4、sin(x)/x；"
                                         "方程如 x^2+y^2=25、x^2/9+y^2/4=1；"
                                         "参数方程如 x=cos(t), y=sin(t)",
                      },
                      "expressions": {
                          "type": "array",
                          "items": {"type": "string"},
                          "description": "**多条曲线画在同一张图里**（推荐用于"
                                         "求交点、比较两条函数）。按数组顺序依次配色："
                                         "第 1 条蓝色、第 2 条橙色、第 3 条绿色……"
                                         "工具回执里会写明每条的颜色，"
                                         "正文里按颜色说明哪条是哪条。"
                                         "例：[\"2*x+1\", \"-x+3\"]",
                      },
                      "x_min": {"type": "number",
                                "description": "横轴左端，默认 -10"},
                      "x_max": {"type": "number",
                                "description": "横轴右端，默认 10"},
                      "y_min": {"type": "number",
                                "description": "纵轴下端（一般不用给，会自动适配）"},
                      "y_max": {"type": "number",
                                "description": "纵轴上端（一般不用给，会自动适配）"},
                      "x_step": {
                          "type": "number",
                          "description": "横轴刻度间隔（数据单位），如 1 = 每 1 个单位"
                                         "一个刻度、0.5 = 每半个单位。"
                                         "用户抱怨刻度太密/太疏时用它，"
                                         "留空则自动选一个好看的间隔",
                      },
                      "y_step": {"type": "number",
                                 "description": "纵轴刻度间隔，含义同 x_step"},
                      "unit_px": {
                          "type": "number",
                          "description": "一个数据单位画多少像素（控制图的疏密），"
                                         "留空铺满画布",
                      },
                      "equal_aspect": {
                          "type": "boolean",
                          "description": "等比例：x/y 两个方向一个单位同样长"
                                         "（画圆、抛物线这类要看形状的图时用）。"
                                         "代价是纵轴可能盖不住曲线全高",
                      },
                      "omit_y": {
                          "type": "string",
                          "description": "纵轴省略（截断）一段，写成 `起..止`，"
                                         "如 `0..999`。用于「曲线整体偏移很大、"
                                         "中间一大段没有内容」的情况"
                                         "（例：y=sin(x)+1000 配 omit_y=0..999）——"
                                         "省略段会折成一小条并画波浪线。"
                                         "曲线仍按真实取值绘制",
                      },
                      "omit_x": {"type": "string",
                                 "description": "横轴省略一段，写法同 omit_y"},
                      "t_min": {"type": "number",
                                "description": "参数方程里 t 的起点，默认 0"},
                      "t_max": {"type": "number",
                                "description": "参数方程里 t 的终点，默认 2π"
                                               "（要画多圈就加大，如 4π）"},
                      "title": {"type": "string",
                                "description": "一行说明。**默认不出现在图上**"
                                               "（图里只有坐标系与曲线），"
                                               "所以别把要交代的信息只写在这里"},
                  },
                  # `expression` / `expressions` 二选一，运行时再校验
                  "required": [],
              })
    async def _plot_function(ctx: ToolContext, expression: str = "",
                             expressions: Any = None,
                             x_min: Any = None, x_max: Any = None,
                             y_min: Any = None, y_max: Any = None,
                             x_step: Any = None, y_step: Any = None,
                             unit_px: Any = None, equal_aspect: Any = None,
                             t_min: Any = None, t_max: Any = None,
                             omit_y: Any = None, omit_x: Any = None,
                             title: str = "") -> Any:
        service = (ctx.extras or {}).get("plot")
        if service is None:
            return "画图不可用（插件未启用函数图像）。"
        # 多条曲线：`expressions` 数组优先；只有一条时允许走 `expression`。
        # ⚠️ 两个都空要**明确回绝**，不能静默返回空 —— 模型会以为画好了。
        if isinstance(expressions, str):
            expressions = [expressions]
        items = [str(e).strip() for e in (expressions or []) if str(e or "").strip()]
        if not items:
            single = str(expression or "").strip()
            if not single:
                return ("请给出表达式：一条用 expression（如 x^2-4），"
                        "多条用 expressions 数组（如 [\"2*x+1\", \"-x+3\"]）")
            items = [single]
        if len(items) > _PLOT_MAX_CURVES:
            return f"一张图最多画 {_PLOT_MAX_CURVES} 条曲线，请减少到 {_PLOT_MAX_CURVES} 条以内"
        return await service.plot(
            items,
            opts={"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
                  "x_step": x_step, "y_step": y_step, "unit_px": unit_px,
                  "equal_aspect": equal_aspect,
                  "t_min": t_min, "t_max": t_max,
                  "omit_y": omit_y, "omit_x": omit_x},
            title=str(title or ""))
