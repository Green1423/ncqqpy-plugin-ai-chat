"""语音转写（ASR）：把 QQ 语音变成文字，再交给模型。

为什么要转写而不是把音频塞进请求：

* 本机模型明确拒收音频（实测 `At most 0 audio(s) may be provided in one prompt`），
  塞进去整轮对话直接 400；
* 就算模型收音频，转成文字也更省 token、还能进历史被压缩/被引用。

**引擎由配置决定，插件不绑定任何一家**（`asr` 配置段）：

| `engine` | 跑什么 | 典型用途 |
|---|---|---|
| `cmd` | 一条外部命令（`{file}` 会被替换成 wav 路径） | 本地 sherpa-onnx / whisper.cpp / 将来的 RKNN 版本 |
| `http` | OpenAI 风格的 `/v1/audio/transcriptions` | 网关开了 whisper / qwen3-asr 之类 |
| 空 | 不转写（只用标号占位） | 还没装引擎 |

`engine` 留空时按配置自动选：有 `command` → cmd，有 `url` → http。
所以「装好引擎、把命令填进配置」就是全部工作，代码不用改。
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from typing import Optional

from .media import probe_wav


@dataclass
class AsrOptions:
    enabled: bool = True
    #: `""`（自动） / `cmd` / `http` / `api` / `none`
    #:   `api` = 交给 NapCat 的 `/fetch_ptt_text`（服务端自带语音转文字）
    engine: str = ""
    #: 外部命令模板；`{file}` 会被替换成 wav 路径
    command: str = ""
    #: HTTP 引擎：完整的 transcriptions 地址
    url: str = ""
    model: str = ""
    api_key: str = ""
    #: 走 `/fetch_ptt_text` 时传什么：`auto`（有 file 传 file）/ `file` / `message_id`
    api_mode: str = "auto"
    #: 单条语音的转写上限（秒）与结果上限（字）
    timeout: float = 60.0
    max_chars: int = 600

    @staticmethod
    def from_cfg(cfg: Optional[dict]) -> "AsrOptions":
        c = cfg if isinstance(cfg, dict) else {}
        def _f(key, default):
            try:
                return float(c.get(key, default) or default)
            except (TypeError, ValueError):
                return float(default)
        return AsrOptions(
            enabled=bool(c.get("enabled", True)),
            engine=str(c.get("engine") or "").strip().lower(),
            command=str(c.get("command") or "").strip(),
            url=str(c.get("url") or "").strip(),
            model=str(c.get("model") or "").strip(),
            api_key=str(c.get("api_key") or "").strip(),
            timeout=_f("timeout", 60.0),
            api_mode=str(c.get("api_mode") or "auto").strip().lower(),
            max_chars=int(c.get("max_chars") or 600),
        )

    def resolved_engine(self) -> str:
        """实际用哪个引擎（`""` = 自动判断）。"""
        if not self.enabled:
            return "none"
        if self.engine in ("cmd", "http", "api", "none"):
            return self.engine
        if self.command:
            return "cmd"
        if self.url:
            return "http"
        return "api"              # 交给 NapCat 的 /fetch_ptt_text
        return "none"

    def describe(self) -> str:
        eng = self.resolved_engine()
        if eng == "cmd":
            return f"cmd：{self.command.split()[0] if self.command else '（空）'}"
        if eng == "http":
            return f"http：{self.url or '（空）'}"
        if eng == "api":
            return "api：NapCat /fetch_ptt_text"
        return "未配置"


class Transcriber:
    """把 wav 转成文字。**永不抛异常**：失败返回 `("", 原因)`。"""

    def __init__(self, options: Optional[AsrOptions] = None, *, logger=None,
                 api=None, message_id: int = 0) -> None:
        self.opt = options or AsrOptions()
        self.logger = logger
        #: NapCat API 客户端（`api` 引擎用它的 `/fetch_ptt_text`）
        self.api = api
        #: 这条语音对应的消息 id（接口认 id 就不用再传文件）
        self.message_id = int(message_id or 0)

    # ---- 对外 ----

    async def transcribe(self, path: str) -> tuple[str, str]:
        eng = self.opt.resolved_engine()
        if eng == "none":
            return "", "没有配置语音转写引擎"
        if not path or not os.path.exists(path):
            return "", "音频文件不存在"
        try:
            text, why = "", ""
            if eng == "cmd":
                text = await self._run_cmd(path)
            elif eng == "api":
                text, why = await self._run_api(path)
            else:
                text = await self._run_http(path)
            # ⚠️ 回退：`api`（NapCat 的 /fetch_ptt_text）不通时，只要配了
            # `command` 或 `url` 就再试一次 —— 不然一条配置错就把语音全废了。
            if not text and eng == "api":
                if self.opt.command:
                    if self.logger:
                        self.logger.info("[ai_chat] api 引擎不行，回退 cmd")
                    text = await self._run_cmd(path)
                elif self.opt.url:
                    if self.logger:
                        self.logger.info("[ai_chat] api 引擎不行，回退 http")
                    text = await self._run_http(path)
            if not text and why:
                return "", why
        except Exception as e:  # noqa: BLE001 - 转写失败不该让整轮对话失败
            if self.logger:
                self.logger.warning("[ai_chat] 语音转写失败: %r", e)
            return "", f"{type(e).__name__}: {e}"
        text = " ".join(str(text or "").split()).strip()
        if len(text) > self.opt.max_chars:
            text = text[:self.opt.max_chars] + "…"
        return text, "" if text else "转写结果为空"

    # ---- 引擎 ----

    async def _run_cmd(self, path: str) -> str:
        """跑外部命令，取**最后一行非空输出**当转写（兼容各家的 CLI 输出）。"""
        if not self.opt.command:
            raise RuntimeError("asr.command 是空的")
        argv = [part.replace("{file}", path)
                for part in shlex.split(self.opt.command)]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), self.opt.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"转写超时（{self.opt.timeout:.0f}s）") from None
        if proc.returncode != 0:
            detail = (err or b"").decode("utf-8", "replace").strip()[:160]
            raise RuntimeError(f"命令退出码 {proc.returncode}：{detail or '无输出'}")
        lines = [ln.strip() for ln in (out or b"").decode("utf-8", "replace").splitlines()]
        lines = [ln for ln in lines if ln]
        return lines[-1] if lines else ""

    async def _run_api(self, path: str) -> tuple[str, str]:
        """走 NapCat 自带的 `/fetch_ptt_text`（服务端语音转文字）。

        为什么优先用它：不用另外装模型/跑 ffmpeg 推理，时效也好 ——
        NapCat 收到语音时就转过了。参数按 `api_mode` 决定：

        * `file`：传本地文件路径（我们下载下来的 amr/silk）；
        * `message_id`：传原始消息 id（服务端自己取那条语音）；
        * `auto`：两者都带上，服务端认哪个用哪个。
        """
        api = self.api
        if api is None:
            return "", "没有 API 客户端（api 引擎不可用）"
        # ⚠️ 默认**只传 `message_id`**：服务端的 `file` 指的是它自己的文件名，
        # 传本地路径（`/data/media/...amr`）大概率被拒；id 才是它认识的引用。
        # 没有 id 时才退回文件，`api_mode` 可以强制/两者都传。
        mode = str(self.opt.api_mode or "auto").lower()
        params: dict = {}
        if mode in ("auto", "message_id") and self.message_id:
            params["message_id"] = int(self.message_id)
        if mode in ("file", "both") and path:
            params["file"] = path
        if not params and path:
            params["file"] = path
        if not params:
            return "", "缺少语音文件/消息 id"
        try:
            data = await api._call("fetch_ptt_text", params)
        except Exception as e:  # noqa: BLE001 - 接口不可用就退回别的引擎
            if self.logger:
                self.logger.warning("[ai_chat] /fetch_ptt_text 失败（params=%s）：%s",
                                    list(params), e)
            return "", f"/fetch_ptt_text 调用失败：{e}"
        text = ""
        if isinstance(data, dict):
            for key in ("text", "transcription", "result", "msg"):
                if data.get(key):
                    text = str(data[key])
                    break
        elif isinstance(data, str):
            text = data
        text = " ".join(str(text or "").split())
        if not text:
            if self.logger:
                self.logger.info("[ai_chat] /fetch_ptt_text 返回里没有文字：%r",
                                 str(data)[:200])
            return "", "/fetch_ptt_text 没有返回文字"
        return text[: self.opt.max_chars], ""

    async def _run_http(self, path: str) -> str:
        if not self.opt.url:
            raise RuntimeError("asr.url 是空的")
        import aiohttp
        form = aiohttp.FormData()
        with open(path, "rb") as f:
            form.add_field("file", f.read(), filename=os.path.basename(path),
                           content_type="audio/wav")
        if self.opt.model:
            form.add_field("model", self.opt.model)
        headers = {}
        if self.opt.api_key:
            headers["Authorization"] = f"Bearer {self.opt.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.opt.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(self.opt.url, data=form, headers=headers) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}：{body[:160]}")
                try:
                    got = json.loads(body)
                except json.JSONDecodeError:
                    return body
                if isinstance(got, dict):
                    return str(got.get("text") or got.get("result") or "")
                return ""

    # ---- 自检（配好之后先跑一下，别等群里发语音才发现引擎坏了）----

    async def self_check(self) -> str:
        """用 1 秒静音跑一遍，返回人话结论。"""
        eng = self.opt.resolved_engine()
        if eng == "none":
            return ("未配置：`asr.engine=api`（NapCat 的 /fetch_ptt_text）或 "
                    "`asr.command` / `asr.url` 至少有一个可用。"
                    "填好后语音会自动转成文字交给模型。")
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(probe_wav())
            text, err = await self.transcribe(path)
            if err:
                return f"引擎 {self.opt.describe()} 不可用：{err}"
            return f"引擎 {self.opt.describe()} 可用（静音样本转写为 {text!r}）"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
