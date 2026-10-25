"""多模态能力探测：这个模型到底收不收图片 / 音频 / 视频。

为什么要探测而不是「配置文件里写 true」：模型换了、网关键了口子，写死就会一直
400/500；反过来写死 false 又让换了视觉模型的人用不上。所以按
`(base_url, model)` 缓存一次探测结果，换了模型自动重探。

三种结论（**故意不把「服务器 500」当成「不支持」**）：

| 探测结果 | 记成 | 缓存多久 |
|---|---|---|
| 200 | 支持 | 7 天 |
| 4xx（明确的「不接受这种输入」） | 不支持 | 7 天 |
| 5xx / 超时 / 连不上 | **未知**（多半是上游的锅，不是能力问题） | 10 分钟内不重试 |

> 实测：本机网关对图片/音频 part 直接 500（vLLM 上游崩），对纯文本 200 ——
> 那就是「未知」，重试一阵仍 5xx 后才按不支持处理并记一笔警告。
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from .media import PROBE_IMAGE_PNG, probe_wav

#: 探测结果缓存 TTL
TTL_SUPPORTED = 7 * 86400
TTL_REJECTED = 7 * 86400
#: 5xx/超时之后多久再试
RETRY_UNKNOWN = 600
#: 连续多少次 5xx 之后按「不支持」处理（免得每轮都白打一次请求）
UNKNOWN_GIVE_UP = 3


def _part_for(kind: str) -> Optional[dict]:
    import base64
    if kind == "image":
        b64 = base64.b64encode(PROBE_IMAGE_PNG).decode()
        return {"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}}
    if kind == "record":
        b64 = base64.b64encode(probe_wav()).decode()
        return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
    return None


class CapabilityCache:
    """`data/capabilities.json`：`{"<base>|<model>": {"image": {...}}}`。"""

    def __init__(self, path: str, *, logger=None) -> None:
        self.path = str(path)
        self.logger = logger
        self._data: dict = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                got = json.load(f)
            if isinstance(got, dict):
                self._data = got
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            if self.logger:
                self.logger.warning("[ai_chat] 能力缓存写入失败: %r", e)

    @staticmethod
    def key(base_url: str, model: str) -> str:
        return f"{(base_url or '').rstrip('/')}|{model or ''}"

    def get(self, key: str, kind: str) -> Optional[bool]:
        rec = (self._data.get(key) or {}).get(kind)
        if not isinstance(rec, dict):
            return None
        at = float(rec.get("at") or 0)
        state = str(rec.get("state") or "")
        age = time.time() - at
        if state == "yes" and age < TTL_SUPPORTED:
            return True
        if state == "no" and age < TTL_REJECTED:
            return False
        return None

    def put(self, key: str, kind: str, *, state: str, why: str = "") -> None:
        rec = self._data.setdefault(key, {})
        old = rec.get(kind) if isinstance(rec.get(kind), dict) else {}
        fails = int(old.get("fails") or 0)
        if state == "unknown":
            fails += 1
            if fails >= UNKNOWN_GIVE_UP:
                state = "no"
                why = (why or "上游持续 5xx") + f"（连续 {fails} 次）"
                if self.logger:
                    self.logger.warning(
                        "[ai_chat] 多模态探测：%s 连续 %d 次 5xx，暂按不支持处理（%s）",
                        kind, fails, why)
            else:
                state = "unknown"
        else:
            fails = 0
        rec[kind] = {"state": state, "at": time.time(), "why": str(why or ""),
                     "fails": fails}
        self._data[key] = rec
        self._save()

    def describe(self, key: str) -> dict:
        """诊断用：`{kind: (state, why, age_seconds)}`。"""
        out = {}
        for kind, rec in (self._data.get(key) or {}).items():
            if not isinstance(rec, dict):
                continue
            out[kind] = (str(rec.get("state") or ""), str(rec.get("why") or ""),
                         int(time.time() - float(rec.get("at") or 0)))
        return out


async def probe(llm, kind: str, *, timeout: float = 25.0) -> tuple[str, str]:
    """发一次最小请求，判定模型收不收这种 part。

    返回 `("yes" | "no" | "unknown", 原因)`。
    """
    part = _part_for(kind)
    if part is None or not getattr(llm, "enabled", False):
        return "unknown", "没有可探测的客户端"
    payload = {
        "model": llm.model,
        "max_tokens": 4,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "ping"}, part]}],
    }
    headers = {"Authorization": f"Bearer {llm.api_key}",
               "Content-Type": "application/json"}
    headers.update(getattr(llm, "extra_headers", {}) or {})
    import aiohttp
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)) as s:
            async with s.post(f"{(llm.base_url or '').rstrip('/')}/chat/completions",
                              json=payload, headers=headers) as r:
                body = (await r.text())[:300]
                if r.status == 200:
                    return "yes", ""
                if 400 <= r.status < 500:
                    return "no", f"HTTP {r.status}：{body}"
                return "unknown", f"HTTP {r.status}：{body}"
    except Exception as e:  # noqa: BLE001 - 探测失败不该影响正常对话
        return "unknown", f"{type(e).__name__}"


async def supports(llm, cache: CapabilityCache, kind: str, *,
                   force: bool = False) -> bool:
    """`kind` 这一路上能不能传媒体（带缓存）。"""
    key = CapabilityCache.key(getattr(llm, "base_url", ""),
                             getattr(llm, "model", ""))
    if not force:
        got = cache.get(key, kind)
        if got is not None:
            return got
        rec = (cache._data.get(key) or {}).get(kind) or {}
        if rec.get("state") == "unknown" and \
                time.time() - float(rec.get("at") or 0) < RETRY_UNKNOWN:
            return False            # 刚失败过，别每轮都打一次
    state, why = await probe(llm, kind)
    cache.put(key, kind, state=state, why=why)
    return cache.get(key, kind) is True
