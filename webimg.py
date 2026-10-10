"""让 AI 能把**网上的图片**配到消息里（工具 `fetch_image`）。

与 `plot.PlotService` 是同一套路数（工具产出文件 → 在白名单里记一笔 →
主流程从**工具结果**里取标记、把图作为单独一条消息发出去），区别只是
「图从哪来」：那边是 matplotlib 画出来的，这边是从 URL 下载的。

⚠️ 三条边界都是**安全边界**，不是洁癖 —— URL 来自模型，而模型读过不可信的
网页/搜索摘要，所以这里的输入等同于「陌生人给的地址」：

1. **只允许公网 http/https**：复用 `search.check_url()` 的 SSRF 闸，并且**每一跳
   跳转都重查**（自动跳转能绕过首跳检查，见 `search.WebClient.get_bytes`）；
2. **必须是真图片**：按**魔数**判定（PNG/JPEG/GIF/WebP/BMP），不看后缀、不信
   `Content-Type` —— 否则「下载这个 .png」就能把任意二进制塞进群里；
3. **只有我们下载过的文件才发得出去**：`rendered` 是白名单，模型自己编一个
   `[[plot:/etc/passwd]]` 发不出任何东西（与 PlotService 同一条规矩）。

另加三条工程约束：单张大小上限（默认 8MB）、一轮最多几张（默认 3）、
目录有界（只留最近 `keep_files` 个文件，防止 SD 卡被写满）。
"""
from __future__ import annotations

import hashlib
import os
import shutil
from typing import Optional

from .plot import marker_for
from .search import FetchBlocked, FetchTooLarge

#: 魔数 → 扩展名。**不看后缀**：`evil.png` 里可以是任何东西。
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)

#: 能安全发进 QQ 的类型。SVG/ICO/AVIF 之类一律不要 —— 要么不是位图，
#: 要么 QQ 客户端不认（发了等于白发）。
IMAGE_EXTS = frozenset({"png", "jpg", "gif", "webp", "bmp"})

_LABELS = {"png": "PNG", "jpg": "JPEG", "gif": "GIF", "webp": "WebP", "bmp": "BMP"}


def sniff_image(raw: bytes) -> str:
    """按魔数判断这是什么图片，返回扩展名；不是图片返回空串。"""
    if not raw:
        return ""
    for magic, ext in _IMAGE_MAGIC:
        if raw.startswith(magic):
            return ext
    # WebP 是 RIFF 容器：`RIFF....WEBP`
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp"
    return ""


def _human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.0f}KB"
    return f"{n}B"


class WebImageService:
    """下载 → 校验 → 落盘 → 记白名单。工具只调它，不自己碰文件与网络。"""

    def __init__(self, *, cfg: Optional[dict] = None, logger=None,
                 data_dir: str = "") -> None:
        self.cfg = dict(cfg or {})
        self.logger = logger
        self.data_dir = str(data_dir or "")
        #: 真的下载成功过哪些文件 —— 主流程只认这些路径（白名单）
        self.rendered: set[str] = set()
        #: 本轮已经配了几张（`begin_turn()` 清零）
        self._turn_paths: list[str] = []
        #: url → 路径：同一条 URL 重复调用不再下载第二次
        self._by_url: dict[str, str] = {}

    # ---- 路径 ----

    def _out_dir(self) -> str:
        base = self.data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data")
        return os.path.join(base, "web_img")

    def begin_turn(self) -> None:
        """一轮对话开始：重置「本轮配了几张」的计数。"""
        self._turn_paths = []

    # ---- 主流程 ----

    async def fetch(self, web, url: str, *, caption: str = "") -> str:
        """下载一张图并返回给模型的回执（成功时末尾带 `[[plot:…]]` 标记）。

        **失败一律返回文字说明、不带标记** —— 宁可告诉模型「下不下来 + 原因」，
        也不要发一个坏掉的图片节点（与 PlotService 的「渲染成功才留标记」一致）。
        """
        url = str(url or "").strip()
        if not url:
            return "请给出要下载的图片直链 url"
        if web is None:
            return "配图不可用（插件需要联网能力才能下载图片）"
        limit = max(1, int(self.cfg.get("max_bytes") or 8 * 1024 * 1024))
        per_turn = max(1, int(self.cfg.get("max_per_turn") or 3))

        # 同一轮里重复要同一张图：直接复用（模型经常在好几轮工具调用里提同一个链接）
        cached = self._by_url.get(url)
        if cached and os.path.exists(cached):
            if cached not in self._turn_paths:
                if len(self._turn_paths) >= per_turn:
                    return (f"这一轮已经配了 {per_turn} 张图（上限），"
                            f"这张就不再发了。")
                self._turn_paths.append(cached)
            return self._ok_reply(cached, caption, note="（这张图本轮已经下过，直接复用）")
        if len(self._turn_paths) >= per_turn:
            return (f"这一轮已经配了 {per_turn} 张图（上限 {per_turn}），"
                    f"这张就不再发了 —— 需要的话在下一条消息里再要。")

        try:
            raw, ctype = await web.get_bytes(url, max_bytes=limit)
        except FetchBlocked as e:
            return f"不能下载这个地址：{e}"
        except FetchTooLarge as e:
            return (f"这张图太大了（超过 {_human_size(e.limit)} 上限）。"
                    f"换一张小一点的，或者只把链接给用户让 ta 自己点开。")
        except Exception as e:  # noqa: BLE001 - 下载是外部世界，什么都可能
            if self.logger:
                self.logger.warning("[ai_chat] 配图下载失败 %s: %s", url[:120], e)
            return f"图片下载失败：{e}。可以改用文字描述，或换一个链接。"

        ext = sniff_image(raw)
        if not ext or ext not in IMAGE_EXTS:
            return (f"这个地址给的不是图片（Content-Type={ctype or '未知'}，"
                    f"前几字节 {raw[:8]!r}）。要读网页正文请用 fetch_page；"
                    f"配图请给图片直链（.png/.jpg/.gif/.webp）。")

        path = self._save(url, raw, ext)
        if path is None:
            return "图片保存失败（磁盘写入错误），这次就不配图了。"
        self._by_url[url] = path
        self._turn_paths.append(path)
        return self._ok_reply(path, caption,
                              note=f"（{_LABELS.get(ext, ext)} · {_human_size(len(raw))}）")

    def _ok_reply(self, path: str, caption: str, *, note: str = "") -> str:
        cap = str(caption or "").strip() or os.path.basename(path)
        return "\n".join([
            f"已把这张图配在本轮回复里：{cap} {note}".rstrip(),
            "（图片会作为**单独一条消息**附在回复后面，正文里不用再贴链接或 "
            "Markdown 图片语法；用一两句话说明这张图是什么即可。）",
            marker_for(path, cap),
        ])

    # ---- 落盘 ----

    def _save(self, url: str, raw: bytes, ext: str) -> Optional[str]:
        out_dir = self._out_dir()
        try:
            os.makedirs(out_dir, exist_ok=True)
            # 文件名按 URL 取哈希：同一条链接天然去重，也避免把远程文件名
            # （不可信输入）写进本地路径 —— `../../etc/x` 这类名字就被挡掉了。
            name = hashlib.sha1(url.encode("utf-8", "ignore")).hexdigest()[:20]
            path = os.path.join(out_dir, f"{name}.{ext}")
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(raw)
            os.replace(tmp, path)
        except OSError as e:
            if self.logger:
                self.logger.warning("[ai_chat] 配图落盘失败: %s", e)
            return None
        self.rendered.add(path)
        self.prune()
        return path

    def prune(self) -> None:
        """目录只留最近 `keep_files` 个文件（防止 SD 卡被慢慢写满）。"""
        keep = max(1, int(self.cfg.get("keep_files") or 60))
        out_dir = self._out_dir()
        try:
            entries = []
            for name in os.listdir(out_dir):
                path = os.path.join(out_dir, name)
                if name.endswith(".part"):
                    continue
                try:
                    entries.append((os.path.getmtime(path), path))
                except OSError:
                    continue
            if len(entries) <= keep:
                return
            entries.sort(reverse=True)
            for _mtime, path in entries[keep:]:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                except OSError:
                    continue
                self.rendered.discard(path)
        except OSError:
            return
