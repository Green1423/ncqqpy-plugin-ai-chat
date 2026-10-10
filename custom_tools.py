"""由配置声明的 HTTP 工具（`plugins.ai_chat.tools.custom`）。

给运维一个不改代码就能挂工具的口子：把任意 HTTP 接口包成一个 function
暴露给模型。声明形如::

    {
      "name": "campus_bus",
      "description": "查询校车时刻表",
      "parameters": {
        "type": "object",
        "properties": {"line": {"type": "string", "description": "线路名"}},
        "required": ["line"]
      },
      "method": "GET",
      "url": "https://example.edu/bus?line={line}",
      "headers": {"X-Token": "..."},
      "timeout": 15,
      "max_chars": 2000,
      "response_path": "data.items"
    }

约定与安全边界：

* **URL/headers 里的 `{参数名}` 会被替换**，替换值做百分号编码 —— 模型给的参数
  不能借道注入查询串或额外请求头（换行、`&`、`#` 都被转义）。
* **POST 时未出现在 URL 里的参数走 JSON body**；`body_template` 可整段自定义。
* **只允许 http/https**，且强制超时 —— 一个跑偏的模型不该把机器人挂死。
* `response_path` 用点号取值（`data.items`），取不到就原样返回整个响应。
"""
from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import quote

import aiohttp

from .llm import err_text
from .tools import MAX_RESULT_CHARS, TRUNCATED_HINT, Tool, ToolContext, ToolRegistry

_ALLOWED_SCHEMES = ("http://", "https://")


def _subst(template: str, args: dict) -> str:
    """把 `{name}` 换成参数值（值做 URL 编码）。"""
    out = str(template)
    for key, value in (args or {}).items():
        out = out.replace("{" + str(key) + "}", quote(str(value), safe=""))
    return out


def _dig(data: Any, path: str) -> Any:
    """按 `a.b.c` 取嵌套值；`a.0.b` 也支持（列表下标）。"""
    cur = data
    for part in [p for p in str(path or "").split(".") if p]:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def build_http_tool(spec: dict, *, logger=None) -> Optional[Tool]:
    """把一条配置声明变成 `Tool`。声明不合法时返回 None（并记日志）。"""
    name = str(spec.get("name") or "").strip()
    url = str(spec.get("url") or "").strip()
    if not name or not url:
        if logger:
            logger.warning("[ai_chat] 自定义工具缺 name/url，已跳过: %r", spec)
        return None
    if not url.lower().startswith(_ALLOWED_SCHEMES):
        if logger:
            logger.warning("[ai_chat] 自定义工具 %s 的 url 必须是 http/https，已跳过", name)
        return None

    method = str(spec.get("method") or "GET").upper()
    headers = {str(k): str(v) for k, v in (spec.get("headers") or {}).items()}
    timeout = float(spec.get("timeout") or 15)
    max_chars = int(spec.get("max_chars") or 0) or MAX_RESULT_CHARS
    response_path = str(spec.get("response_path") or "").strip()
    body_template = spec.get("body_template")
    parameters = spec.get("parameters") or {"type": "object", "properties": {}}
    description = str(spec.get("description") or f"调用 {name} 接口")

    async def _handler(ctx: ToolContext, **kwargs: Any) -> str:
        target_url = _subst(url, kwargs)
        payload: dict = {}
        if method in ("POST", "PUT", "PATCH"):
            if isinstance(body_template, dict):
                # 模板里的 {name} 用原始值（body 里不需要 URL 编码）
                payload = json.loads(_subst_json(json.dumps(body_template,
                                                            ensure_ascii=False),
                                                 kwargs))
            else:
                # 没写 body_template：把「没进 URL 的参数」都塞进 body
                payload = {k: v for k, v in (kwargs or {}).items()
                           if f"{{{k}}}" not in url}
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout, connect=10)) as sess:
                async with sess.request(method, target_url, headers=headers,
                                        json=payload if payload else None) as resp:
                    text = await resp.text()
                    status = resp.status
        except Exception as e:  # noqa: BLE001 - 工具失败要回喂给模型，不能抛
            return f"请求失败：{err_text(e)}"
        if status >= 400:
            return f"HTTP {status}：{text[:400]}"
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            body = text
        else:
            got = _dig(data, response_path) if response_path else data
            body = got if got is not None else data
            if not isinstance(body, str):
                body = json.dumps(body, ensure_ascii=False)
        if len(body) > max_chars:
            body = body[:max_chars] + TRUNCATED_HINT
        if logger:
            logger.info("[ai_chat] 自定义工具 %s 调用成功（HTTP %s）", name, status)
        return body or "（接口返回空）"

    _handler.__name__ = f"custom_{name}"
    return Tool(name=name, description=description, parameters=parameters,
                handler=_handler,
                group_only=bool(spec.get("group_only")),
                superuser_only=bool(spec.get("superuser_only")),
                requires_moderation=bool(spec.get("requires_moderation")))


def _subst_json(template: str, args: dict) -> str:
    """在 JSON 文本里做替换（值用 JSON 编码，保证仍是合法 JSON）。"""
    out = str(template)
    for key, value in (args or {}).items():
        out = out.replace('"{' + str(key) + '}"',
                          json.dumps(value, ensure_ascii=False))
        out = out.replace("{" + str(key) + "}", json.dumps(value, ensure_ascii=False))
    return out


def register_config_tools(reg: ToolRegistry, specs: Any, *, logger=None) -> int:
    """注册配置里声明的全部 HTTP 工具，返回成功注册的数量。"""
    added = 0
    for spec in (specs or []):
        if not isinstance(spec, dict):
            continue
        if spec.get("enabled") is False:
            continue
        tool = build_http_tool(spec, logger=logger)
        if tool is None:
            continue
        if tool.name in reg:
            if logger:
                logger.warning("[ai_chat] 自定义工具 %s 与已有工具重名，已跳过", tool.name)
            continue
        reg.add(tool)
        added += 1
    return added
