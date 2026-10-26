"""多模态媒体：落地、编号、按需附给模型。

模型侧只看到**标号**（`[图#3]` / `[视频#2]` / `[语音#1]`），默认拿不到媒体本身 ——
按需才附（见 `docs/ai-chat.md` §13）：

| 媒体 | 什么时候真的传给模型 |
|---|---|
| 图片 / 视频 | ① 用户**引用了**含它的消息；② 用户在正文里写了它的标号（`[图#3]`） |
| 语音 | **总是全传**（一轮里所有语音一起发，视为一条对话） |

为什么不默认全传：多模态 token 很贵，而且群聊里 90% 的图跟问题无关；
标号让用户/模型都能精确指认「就是这张」。

本模块只管「媒体对象」本身 —— 下载、缩放、编码成 OpenAI 的 content part。
「传不传」的判断在 `compaction.project`（投影）与 `main.py`（怎么收标号）。
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Optional

from .webimg import sniff_image

#: OneBot 段类型 → 中文标签（标号里用的那个字）。
#: **没有视频**：OpenAI 协议里没有标准视频 part，能收 `video_url` 的模型凤毛麟角，
#: 留一条半死功能只会让人踩坑（2026-10-24 按用户要求删掉整条 video 路径）。
KIND_LABELS = {"image": "图", "record": "语音"}
#: 中文标签 → 段类型
LABEL_KINDS = {v: k for k, v in KIND_LABELS.items()}
#: 哪些类型是「按需附」（语音不在里面：语音总是全传）。
#: 图片要**用户显式 `/aichat mark`** 才会传（见 `main._do_mark`）。
GATED_KINDS = frozenset({"image"})

#: 媒体标号：`[图#3]`。与段落编号 `[#3]` 同形不同字 —— 一眼能分清
#: 「这一段」和「这张图」，也不会互相误解析。
MARKER_RE = re.compile(r"\[(图|视频|语音)#(\d+)\]")

#: 音频能直接发的格式（OpenAI `input_audio.format` 只认这两种）。
#: QQ 语音常见的是 amr/silk，本机没有 ffmpeg 转不了 —— 那种就只能给说明。
AUDIO_FORMATS = {"wav": "wav", "mp3": "mp3", "m4a": "mp3"}


def marker(kind: str, num: int) -> str:
    """`("image", 3)` → `"[图#3]"`。"""
    return f"[{KIND_LABELS.get(kind, kind)}#{int(num)}]"


def parse_markers(text: str) -> list[tuple[str, int]]:
    """从正文里取出用户写的标号（按出现顺序去重）。

    只认**用户自己写的**：`main._ingest_media` 在拼上自动标号之前先扫一遍原文，
    否则「我们补上去的标号」会立刻把自己选中，等于默认全传。
    """
    out: list[tuple[str, int]] = []
    for label, raw in MARKER_RE.findall(str(text or "")):
        kind = LABEL_KINDS.get(label)
        try:
            num = int(raw)
        except (TypeError, ValueError):
            continue
        if kind and num > 0 and (kind, num) not in out:
            out.append((kind, num))
    return out


def strip_media_markers(text: str) -> str:
    """去掉自动补的标号（`hget` 卡片里不想看它 —— 它只是给模型看的定位符）。"""
    return MARKER_RE.sub("", str(text or "")).strip()


def _ext_of(name: str) -> str:
    base = os.path.basename(str(name or "")).split("?")[0]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


@dataclass
class MediaItem:
    """一个已落地的媒体。

    `num` 是**会话内**的稳定编号（1 起），`0` 表示临时物件（引用里拿到的副本，
    没编号 —— 它不属于「用户说的第几张图」）。
    """

    kind: str = "image"
    num: int = 0
    path: str = ""                  # 本地文件（绝对路径）
    mime: str = ""
    size: int = 0                   # 字节数（缩放后）
    ok: bool = True
    note: str = ""                  # 失败/降级原因（给用户看的那句话）
    #: 语音转写出来的文字（见 `asr.Transcriber`）—— 落进历史，模型直接读
    transcript: str = ""
    source: str = ""                # 原始 url / file（诊断用，不落历史）

    @property
    def is_audio(self) -> bool:
        return self.kind == "record"

    @property
    def gated(self) -> bool:
        return self.kind in GATED_KINDS

    def marker(self) -> str:
        return marker(self.kind, self.num) if self.num > 0 else ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "num": int(self.num), "path": self.path,
                "mime": self.mime, "size": int(self.size), "ok": bool(self.ok),
                "note": self.note, "transcript": self.transcript}

    @staticmethod
    def from_dict(d: dict) -> "MediaItem":
        return MediaItem(kind=str(d.get("kind") or "image"),
                         num=int(d.get("num") or 0),
                         path=str(d.get("path") or ""),
                         mime=str(d.get("mime") or ""),
                         size=int(d.get("size") or 0),
                         ok=bool(d.get("ok", True)),
                         note=str(d.get("note") or ""),
                         transcript=str(d.get("transcript") or ""))

    def describe(self) -> str:
        """一句话说明（给模型看的占位符补充说明，也是失败时的提示）。"""
        return f"{KIND_LABELS.get(self.kind, self.kind)}"

    # ---- 给模型的内容块 ----

    def payload_part(self) -> Optional[dict]:
        """转成 OpenAI content part；做不到就返回 None（附 `note` 说明原因）。"""
        if not self.ok or not self.path or not os.path.exists(self.path):
            return None
        try:
            with open(self.path, "rb") as f:
                raw = f.read()
        except OSError:
            return None
        if not raw:
            return None
        b64 = base64.b64encode(raw).decode()
        if self.is_audio:
            fmt = AUDIO_FORMATS.get(_ext_of(self.path))
            if not fmt:
                return None
            return {"type": "input_audio",
                    "input_audio": {"data": b64, "format": fmt}}
        mime = self.mime or "image/png"
        data_url = f"data:{mime};base64,{b64}"
        return {"type": "image_url", "image_url": {"url": data_url}}


@dataclass
class MediaStore:
    """媒体的落地与编码。

    文件按 `sha1(url)` 命名、放在 `<data>/media/` 下 —— 同一张图被引用两次
    只落一份（群里转发的图 url 常常完全相同）。
    """

    root: str
    max_bytes: int = 4 * 1024 * 1024
    max_side: int = 1568
    #: 落盘口径：`keep_files` 张之后按时间清理（跟 webimg 一个策略）
    #: 单张图片压缩目标（字节）：超过就降质/缩边重来（见 `_shrink_image`）
    max_image_bytes: int = 700_000
    keep_files: int = 200

    def __post_init__(self) -> None:
        self.dir = os.path.join(str(self.root), "media")
        try:
            os.makedirs(self.dir, exist_ok=True)
        except OSError:
            pass

    # ---- 下载 ----

    async def fetch(self, url: str, kind: str, *, num: int = 0,
                    session=None, max_side: Optional[int] = None) -> MediaItem:
        """下载一个媒体并落盘（图片会缩放）。失败返回 `ok=False` 的物件。"""
        item = MediaItem(kind=kind, num=int(num), source=str(url or ""))
        if not url:
            item.ok, item.note = False, "没有可下载的地址"
            return item
        try:
            raw = await self._fetch_bytes(url)
        except Exception as e:  # noqa: BLE001 - 网络问题不该炸掉整轮
            item.ok, item.note = False, f"下载失败：{type(e).__name__}"
            return item
        if not raw:
            item.ok, item.note = False, "下载到 0 字节"
            return item
        if len(raw) > self.max_bytes:
            item.ok = False
            item.note = f"文件太大（{len(raw) // 1024}KB > {self.max_bytes // 1024}KB）"
            return item
        mime = ""
        if kind == "image":
            try:
                got = sniff_image(raw)
                if got:
                    mime = got
            except Exception:  # noqa: BLE001
                mime = ""
            if kind == "image" and not mime:
                item.ok, item.note = False, "不是能识别的图片格式"
                return item
        data, ext = raw, ""
        if kind == "image":
            data, ext, mime = self._shrink_image(
                raw, mime, max_side=max_side)
            if data is None:
                item.ok, item.note = False, "图片解码失败"
                return item
        else:
            # 语音：wav/mp3 直接用；QQ 常见的 amr/silk 用 ffmpeg 转 wav。
            # 没有 ffmpeg 就没法转 —— 只能把原因说清楚（别把 amr 当 wav 发出去，
            # 上游只会回一句看不懂的解析错误）。
            src_ext = _ext_of(url) or "amr"
            if AUDIO_FORMATS.get(src_ext) is None:
                converted, note = self._to_wav(raw)
                if converted is None:
                    item.ok, item.note = False, note
                    return item
                data, ext = converted, ".wav"
                mime = "audio/wav"
            else:
                ext = "." + src_ext
        digest = hashlib.sha1((str(url) + f"|{kind}").encode()).hexdigest()[:16]
        path = os.path.join(self.dir, f"{kind}-{digest}{ext or ''}")
        try:
            with open(path, "wb") as f:
                f.write(data)
        except OSError as e:
            item.ok, item.note = False, f"落盘失败：{e}"
            return item
        item.path, item.mime, item.size = path, mime or item.mime, len(data)
        if session is not None and num > 0:
            self._remember(session, item)
        self.prune()
        return item

    async def _fetch_bytes(self, src: str) -> bytes:
        """取原始字节：http(s) 下载；`file://` 或**本地已存在路径**直接读。

        为什么需要后者：NapCat 给 `record` 段的常常是本地缓存文件名
        （`xxxx.amr`），而不是 URL —— 直接当 URL 用会 `InvalidUrlClientError`，
        线上就出现过「语音全部未取到」。
        """
        text = str(src or "")
        if text.startswith("file://"):
            text = text[len("file://"):]
        if text and not text.startswith(("http://", "https://")) \
                and os.path.exists(text):
            with open(text, "rb") as f:
                return f.read()
        return await self._download(src)

    async def _download(self, url: str) -> bytes:
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                chunks, total = [], 0
                async for chunk in r.content.iter_chunked(65536):
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise RuntimeError("超过大小上限")
                return b"".join(chunks)

    #: ffmpeg 可执行文件（第一次用到时探测，结果缓存；没有就记 None）
    _ffmpeg: Optional[str] = None
    _ffmpeg_checked: bool = False

    @classmethod
    def ffmpeg(cls) -> Optional[str]:
        """找 ffmpeg（找不到返回 None，**不抛异常**）。"""
        if cls._ffmpeg_checked:
            return cls._ffmpeg
        cls._ffmpeg_checked = True
        found = shutil.which("ffmpeg")
        cls._ffmpeg = found
        return found

    def _to_wav(self, raw: bytes) -> tuple[Optional[bytes], str]:
        """任意音频 → 16kHz 单声道 wav（`input_audio` 只认 wav/mp3）。

        ffmpeg 不在或转失败时返回 `(None, 原因)` —— 调用方把原因写进 `note`，
        模型/用户看到的是「语音格式转不了」，而不是一个上游 400。
        """
        exe = self.ffmpeg()
        if not exe:
            return None, "语音是 amr/silk 格式，需要 ffmpeg 转码（未安装）"
        src = os.path.join(self.dir, ".tmp-src.bin")
        dst = os.path.join(self.dir, ".tmp-out.wav")
        try:
            with open(src, "wb") as f:
                f.write(raw)
            import subprocess
            proc = subprocess.run(                     # noqa: S603 - 固定参数，无 shell
                [exe, "-y", "-loglevel", "error", "-i", src,
                 "-ar", "16000", "-ac", "1", dst],
                capture_output=True, timeout=25)
            if proc.returncode != 0 or not os.path.exists(dst):
                detail = (proc.stderr or b"").decode("utf-8", "replace")[:120]
                return None, f"ffmpeg 转码失败：{detail or proc.returncode}"
            with open(dst, "rb") as f:
                return f.read(), ""
        except Exception as e:  # noqa: BLE001 - 转码失败不该炸掉整轮
            return None, f"ffmpeg 转码异常：{type(e).__name__}"
        finally:
            for path in (src, dst):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _shrink_image(self, raw: bytes, mime: str, *,
                      max_side: Optional[int] = None) -> tuple[Optional[bytes], str, str]:
        """缩到 `max_side` 以内并转 JPEG —— 多模态接口对尺寸/体积都很敏感。

        没有 PIL 时原样返回（宁可大一点，也别丢功能）。
        """
        try:
            from PIL import Image
        except Exception:  # noqa: BLE001
            return raw, "", mime or "image/png"
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception:  # noqa: BLE001
            return None, "", ""
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        side = int(max_side or self.max_side)
        if side > 0 and max(img.size) > side:
            ratio = side / float(max(img.size))
            img = img.resize((max(1, int(img.width * ratio)),
                              max(1, int(img.height * ratio))))
        # 自适应压缩：单张超过 `target` 就降质/缩边重来（最多 3 轮）。
        # 为什么必须做：请求里图片是 base64 内联的，单张 1.2MB 的扫描页三张就
        # 能顶到 **413 request body too large**（上游直接拒，整轮失败）。
        target = int(getattr(self, "max_image_bytes", 700_000))
        data = b""
        for attempt in range(3):
            out = io.BytesIO()
            quality = 88 - attempt * 12          # 88 → 76 → 64
            img.convert("RGB").save(out, format="JPEG", quality=quality)
            data = out.getvalue()
            if len(data) <= target:
                break
            if attempt < 2:                      # 还太大 → 再缩 20%
                img = img.resize((max(1, int(img.width * 0.8)),
                                  max(1, int(img.height * 0.8))))
        return data, ".jpg", "image/jpeg"

    # ---- 缓存目录 ----

    def _remember(self, session, item: MediaItem) -> None:
        """记到会话上（编号自增 + 最近文件清单），随 state.json 落盘。"""
        meta = getattr(session, "meta", None)
        if not isinstance(meta, dict):
            return
        recent = list(meta.get("media_files") or [])
        recent.append(item.path)
        meta["media_files"] = recent[-int(self.keep_files):]

    def prune(self) -> None:
        """只保留最近的 `keep_files` 个文件（旧文件删掉，历史里的引用会失效）。"""
        try:
            names = [os.path.join(self.dir, n) for n in os.listdir(self.dir)]
        except OSError:
            return
        files = [p for p in names if os.path.isfile(p)]
        if len(files) <= self.keep_files:
            return
        files.sort(key=lambda p: os.path.getmtime(p))
        for path in files[:len(files) - self.keep_files]:
            try:
                os.remove(path)
            except OSError:
                pass


#: 探测用的最小样本（8×8 红点 PNG；音频是 0.1 秒静音 WAV）——
#: 只用来说明「这个模型收不收这种 part」，内容不重要。
PROBE_IMAGE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAI0lEQVR4nGP8z8Dwn4EIwESMolEF"
    "o0yjTKMMg0YBAwMDAAAhFwIBW8FHVwAAAABJRU5ErkJggg==")


def probe_wav() -> bytes:
    """0.1 秒 8kHz 单声道静音 WAV（探测音频用）。"""
    import struct
    rate, frames = 8000, 800
    data = struct.pack("<" + "h" * frames, *([0] * frames))
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
              + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
              + b"data" + struct.pack("<I", len(data)))
    return header + data
