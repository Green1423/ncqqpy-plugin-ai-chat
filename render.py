"""把一对问答渲染成 PNG —— 只负责「数据 → 图」，不认识命令、不读会话。

为什么用 Chromium 而不是画图库：这台机器上**没有任何图像库**（`import PIL`
直接 ModuleNotFoundError），但装了 Chromium 152 和 Noto Sans CJK SC。用无头
浏览器截图等于把「排版」这件事交给一个真正的排版引擎 —— 中文换行、标点挤压、
长英文断词都不用自己写。

**两趟渲染**（关键，别简化成一趟）：

1. 第一趟 `--dump-dom`：让页面里的脚本把内容真实高度写进一个元素，正则取出来；
2. 第二趟 `--screenshot --window-size=W,H`：按量到的高度截图。

一趟做不到自适应高度 —— `--window-size` 就是最终图像尺寸，而页面里的
`window.resizeTo()` 在无头模式下**被忽略**（实测：请求 600×900 就拿回
600×900，底部一大片空白）。

**安全**：群友的昵称和消息正文会进 HTML，所以一律 `html.escape`；页面里不引任何
外部资源（`file://` 之外的 URL 一个都不出现），免得截图时去联网。
"""
from __future__ import annotations

import asyncio
import html
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .history import Exchange, clean_quoted, split_quoted
from .png import PAGE_PAD, crop_bottom_to_ink, crop_right_to_ink, crop_to_ink
from .markdown import (FRAGMENT_WIDTHS, build_fragment_html,
                       build_inline_math_html,
                       build_markdown_html, build_math_html,
                       build_table_html, katex_head, markdown_body_css,
                       markdown_to_html)

# ⚠️ Chromium 的**定位/启动/超时**统一由框架提供（`napcat.render`）：
# 三个插件各写一份，任何一份改了另外两份都不会跟着改（实测已分叉）。
# 框架模块只在**进程启动**时加载，而插件是热重载的 —— 老进程里还没有它，
# 直接 import 会把整个插件打挂，所以拿不到就按文件路径加载同一份实现。
try:                                        # 正常路径（重启后走这里）
    from napcat.render import (RenderError, find_chromium, file_uri)
    from napcat.render import run_chromium as _run_chromium
except ImportError:                         # 老进程：按路径加载，实现仍只有一份
    import importlib.util as _ilu
    import sys as _sys

    _spec = _ilu.spec_from_file_location(
        "napcat_render_boot",
        Path(__file__).resolve().parents[2] / "napcat" / "render.py")
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["napcat_render_boot"] = _mod
    _spec.loader.exec_module(_mod)
    RenderError = _mod.RenderError
    find_chromium = _mod.find_chromium
    file_uri = _mod.file_uri
    _run_chromium = _mod.run_chromium

_HEIGHT_RE = re.compile(r"H=(\d+)")

#: 截图时的**最小视口高度**。
#:
#: ⚠️ 这是个踩出来的硬约束：Chromium 在视口高度过小时会**整张画成空白** ——
#: 实测同一个页面，720x156、720x170 全白，720x200 起才画出来。而公式图、
#: 行内公式图恰恰只有几十像素高，正好落在空白区里（表现是「公式图全白」）。
#: 所以统一按这个下限渲染，再用 `png.crop_*` 按像素把多余留白裁掉 ——
#: 渲染大一点无害，渲染小了直接丢内容。
MIN_SHOT_HEIGHT = 240
#: 行内公式页上报的「宽x高」（用于紧裁）
_DIM_RE = re.compile(r"D=(\d+)x(\d+)")
#: 出图后**按内容右侧收口**的块类型。
#:
#: 表格不在此列：它的 CSS 是 `width: 100%`，本来就该铺满整页宽（收了反而窄）。
#: 其余块（公式/列表/代码/标题）内容多宽就出多宽 —— 页面报的宽度是整页宽，
#: 不收口就会在右边留一大片空白（实测：一个公式右边空了 561px）。
FRAGMENT_CROP_RIGHT = frozenset({"math", "list", "code", "heading"})

#: 整块出图时按**四边**紧裁的块。
#:
#: 公式是一张**居中排版、自身独立**的图：`.math-display` 居中 + `.wrap.tight`
#: 的 `margin:0 auto` 会把公式摆在视口中间，而 `crop_right_to_ink` 明说
#: 「左侧不动，保留排版起点」—— 于是出图左边白 56px、公式被挤到右边（实测）。
#: 列表/代码/标题的左边距是**排版的一部分**（项目符号缩进、代码对齐），不能切；
#: 公式没有这个语义，左右留白都是多余的。
FRAGMENT_CROP_ALL = frozenset({"math"})

#: 页面自报的「内容比视口宽，请把出图宽度放宽到 W」
_WIDE_RE = re.compile(r"W=(\d+)")




@dataclass
class RenderOptions:
    """渲染参数（来自 `config.json` 的 `history` 段）。"""

    width: int = 720
    timeout: float = 20.0
    max_height: int = 12000          # 兜底：内容再长也不超过这个像素高度
    max_chars_per_side: int = 2000   # 单侧正文截断
    chrome: str = ""                 # 留空则自动探测
    keep_files: int = 40             # 图片目录最多保留多少个文件
    #: 渲染的虚拟时间预算（毫秒）。它决定「等页面资源（CDN 字体、KaTeX）多久」——
    #: 等不够就会量出偏小的高度并把内容截断。0 = 立刻快照（会截断，别用）。
    virtual_time_budget: int = 6000
    #: 行内公式紧裁时四周保留的像素（公式上下需要的气口更小）
    inline_pad_x: int = 6
    inline_pad_y: int = 4
    #: 出图宽度的上限：超宽公式会把图放宽，但不能无限宽（QQ 里也看不清）
    max_width: int = 1600




def _truncate(text: str, limit: int) -> tuple[str, bool]:
    """按字符数截断；返回 `(文本, 是否截断过)`。"""
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit].rstrip(), True


def _css() -> str:
    """内联样式。字体挑本机有的，不引外部字体。"""
    return """
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; height: auto; background: #f4f5f7; }
    body {
      font-family: "Noto Sans CJK SC", "Source Han Sans SC",
                   "WenQuanYi Zen Hei", sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    .card {
      background: #fff; margin: 0; padding: 22px 24px 18px;
    }
    .head {
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid #e6e8eb; padding-bottom: 12px; margin-bottom: 18px;
    }
    .title { font-size: 19px; font-weight: 700; color: #1f2329; }
    .count { font-size: 14px; color: #8a9099; }
    .who {
      display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px;
    }
    .name { font-size: 17px; font-weight: 700; }
    .time { font-size: 13px; color: #9aa0a6; }
    /* ⚠️ **不要** `white-space: pre-wrap`：正文现在由 `markdown_to_html` 渲染，
       段落结构与换行由它负责（软换行会变成空格、段落之间是 `\n` + `<p>`）——
       再加 pre-wrap 会把段落之间那个换行**多渲染成一行**，整段变成双倍行距。
       长单词/长链接仍然要能断，所以留着 `word-break`。 */
    .q, .a {
      font-size: 18px; line-height: 1.72; color: #1f2329;
      word-break: break-word;
    }
    /* 首尾外边距去掉：留白由 .q/.a 自己的 padding 控制 */
    .q > *:first-child, .a > *:first-child { margin-top: 0; }
    .q > *:last-child, .a > *:last-child { margin-bottom: 0; }
    .q { background: #eef3ff; border-radius: 10px; padding: 12px 14px; }
    .a { padding: 2px 2px 0; }
    /* 引用**普通消息**：用户输入里贴着的那段引文（没有对应的问答可渲染） */
    .quote { margin: 8px 0 4px; padding: 8px 10px; background: #f2f3f6;
             border-left: 3px solid #c9ccd6; border-radius: 6px; }
    .quote-h { font-size: 12px; color: #7a7f8c; margin-bottom: 4px; }
    .quote > *:first-child { margin-top: 0; }
    .quote > *:last-child { margin-bottom: 0; }
    .quote-note { margin-top: 6px; font-size: 12px; color: #8a90a0; }
    /* 引用链：主问答之上的一块灰色上下文，视觉上明确「这是上文」 */
    .ctx { margin: 10px 0 14px; padding: 10px 12px; background: #f6f6f8;
           border-left: 3px solid #c9ccd6; border-radius: 8px; }
    .ctx-title { font-size: 12px; color: #7a7f8c; margin-bottom: 8px; }
    .ctx-item { margin: 8px 0; padding: 8px 10px; background: #fff;
                border-radius: 6px; }
    .ctx-head { font-size: 12px; color: #9096a3; margin-bottom: 4px; }
    .ctx-q { font-size: 13px; color: #3a3f4b; }
    .ctx-a { font-size: 13px; color: #5b6170; margin-top: 4px;
             padding-left: 10px; border-left: 2px solid #e3e5ea; }
    .ctx-q > *:first-child, .ctx-a > *:first-child { margin-top: 0; }
    .ctx-q > *:last-child, .ctx-a > *:last-child { margin-bottom: 0; }
    .ai { display: flex; align-items: baseline; gap: 8px; margin: 16px 0 8px; }
    .ai .name { font-size: 17px; font-weight: 700; color: #2b6cb0; }
    .tool {
      margin-top: 14px; padding-top: 10px; border-top: 1px dashed #e6e8eb;
      font-size: 14px; color: #8a9099;
    }
    .shot { margin: 10px 0 14px; }
    .shot figcaption { font-size: 12px; color: #888; margin-bottom: 4px; }
    .shot img { max-width: 100%; border-radius: 6px; display: block; }
    .cut { font-size: 13px; color: #c05621; margin-top: 6px; }
    /* 正文里的 Markdown（表格 / 列表 / 代码 / 公式）由**同一套**样式渲染 ——
       见 markdown.markdown_body_css 的说明，不要在这里另写一份。 */
""" + markdown_body_css()


def _esc(text: str) -> str:
    """转义**页头字段**（会话名 / 昵称 / 时间 / 工具小字）。

    ⚠️ 问答**正文**不再走这里 —— 正文由 `markdown_to_html()` 渲染（它自己负责
    转义，而且只转 `<`/`>`/`&`、不转引号：文本节点里引号没有危险）。这里的
    `quote=True` 是给要拼进属性的场合留的余量。
    """
    return html.escape(str(text or ""), quote=True)


def build_html(ex: Exchange, *, width: int = 720, session_name: str = "",
               total: int = 0, max_chars_per_side: int = 2000,
               context: Optional[list] = None,
               media: Optional[list] = None) -> str:
    """一对问答 → 完整 HTML 文档。

    ⚠️ 所有动态内容都必须过 `_esc()`。群友昵称和消息正文直接拼进 HTML 是一个
    自己给自己开的注入：注入 `<img src="file:///etc/...">` 会被 Chromium 读出来
    画进图里发到群里。
    """
    question, q_cut = _truncate(str(ex.question or ""), max_chars_per_side)
    answer, a_cut = _truncate(str(ex.answer or ""), max_chars_per_side)
    # ⚠️ 正文一律走**同一个** Markdown 渲染器（`markdown_to_html`），不再自己
    # `html.escape` 一把了事。早先那样做的后果：`$$…$$` 在历史卡片里就是一串
    # LaTeX 源码，而同一段回答在聊天里却是排好版的公式 —— 同一份内容两种样子。
    # 转义由 `markdown_to_html` 内部负责（它本来就是为「不可信文本进模板」写的），
    # 所以这里**不能**再 escape 一次，否则 `&lt;` 会被原样显示出来。
    # 引用段怎么显示，取决于被引用的是什么：
    #   * **一轮对话**（能在本会话历史里对上）→ 由下面的「引用链」区块展示整轮，
    #     正文里**不再重复**那段引文，只留一行指路；
    #   * **普通消息**（对不上任何问答，比如别人插的一句）→ 没有别处能看，
    #     就把内容原样贴进引用框里。
    q_body, q_quoted = split_quoted(question)
    chain_ctx = [x for x in (context or [])
                 if isinstance(x, Exchange) and x.index != ex.index]
    q_html = markdown_to_html(q_body)
    if q_quoted:
        if chain_ctx:
            q_html += ('<div class="quote-note">↳ 引用了上文：'
                       + "、".join(f"第 {int(x.index)} 对" for x in chain_ctx)
                       + "（见上方的「引用链」）</div>")
        else:
            body = clean_quoted(q_quoted) or q_quoted
            q_html += ('<div class="quote"><div class="quote-h">引用上文</div>'
                       + markdown_to_html(body) + "</div>")
    # ★ 提问附带的图片（hget 以前只画文字，图全丢了 —— 用户明确要求补上）。
    # 只在卡片里嵌**本地已存在**的文件；`file://` URI 由 Chromium 直接读。
    media_html = ""
    for item in (media or []):
        try:
            path = str((item or {}).get("path") or "")
            marker = str((item or {}).get("marker") or "")
        except AttributeError:
            continue
        if not path or not os.path.exists(path):
            continue
        media_html += (
            '<figure class="shot">'
            f'<figcaption>{_esc(marker)}</figcaption>'
            f'<img src="{Path(path).as_uri()}" alt="{_esc(marker)}">'
            '</figure>')
    a_html = markdown_to_html(answer)

    # 引用链（沿引用追溯出来的上文，最早在前）：每一对只给**问**与答案首段，
    # 否则一张卡片会被上文撑爆 —— 要看全文就 /aichat hget 那一对的编号。
    ctx_html = ""
    for old_ex in chain_ctx:
        q_prev, _ = _truncate(split_quoted(old_ex.question)[0], 240)
        a_prev, _ = _truncate(str(old_ex.answer or ""), 240)
        ctx_html += (
            '<div class="ctx-item">'
            f'<div class="ctx-head">第 {int(old_ex.index)} 对'
            + (f' · [#{int(old_ex.seq)}]' if getattr(old_ex, "seq", 0) else "")
            + f' · {_esc(old_ex.speaker or "")} {_esc(old_ex.time or "")}</div>'
            f'<div class="ctx-q">{markdown_to_html(q_prev)}</div>'
            f'<div class="ctx-a">{markdown_to_html(a_prev)}</div>'
            '</div>')
    ctx_block = ('<div class="ctx"><div class="ctx-title">引用链（沿引用追溯，'
                 '最早在前）</div>' + ctx_html + "</div>") if ctx_html else ""

    q_label = f" [#{int(ex.seq)}]" if getattr(ex, "seq", 0) else ""
    a_label = f" [#{int(ex.answer_seq)}]" if getattr(ex, "answer_seq", 0) else ""
    katex = katex_head(q_html + a_html + ctx_block)

    title = f"会话：{session_name}" if session_name else "历史对话"
    count = f"第 {ex.index} 对" + (f" / 共 {total} 对" if total else "")
    tool_line = ""
    if ex.tool_note:
        tool_line = f'<div class="tool">{_esc(ex.tool_note)}</div>'
    cut_lines = ""
    if q_cut or a_cut:
        cut_lines = ('<div class="cut">⚠️ 内容过长已截断，'
                     '原文见会话历史</div>')

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_css()}</style>{katex}</head>
<body>
<div class="card" style="width:{int(width)}px">
  <div class="head">
    <div class="title">{_esc(title)}</div>
    <div class="count">{_esc(count)}</div>
  </div>
  {ctx_block}
  <div class="who">
    <div class="name">{_esc(ex.speaker or "（未知）")}</div>
    <div class="time">{_esc(ex.time or "")}{_esc(q_label)}</div>
  </div>
  <div class="q">{q_html}</div>
  {media_html}
  <div class="ai"><div class="name">AI{_esc(a_label)}</div></div>
  <div class="a">{a_html}</div>
  {tool_line}
  {cut_lines}
</div>
<script>
  function measure() {{
    var b = document.body;
    if (b) {{ b.style.height = 'auto'; b.style.minHeight = '0'; }}
    var wrap = document.querySelector('.card') || document.querySelector('.md')
               || document.querySelector('.wrap');
    var h;
    if (wrap) {{
      var r0 = wrap.getBoundingClientRect();
      h = Math.ceil(r0.bottom - r0.top);
      var cs = window.getComputedStyle(wrap);
      h += (parseFloat(cs.marginTop) || 0) + (parseFloat(cs.marginBottom) || 0);
    }} else {{
      h = Math.max(document.documentElement.scrollHeight,
                   b ? b.scrollHeight : 0);
    }}
    // 留安全余量；高度写 title（截图里不可见，dump-dom 能读到）
    h += 24;
    document.title = 'H=' + h;
  }}
  // ⚠️ 必须等 **web 字体就位**再量：正文里的公式由 KaTeX 排版，它的字体是
  // 异步加载的，没换上来时公式更矮/更窄，量完再换就会把卡片**底部截掉**
  // （渲染那条链路量高度的坑，见 markdown._SIZE_SCRIPT 的同一处说明）。
  function waitFontsThenMeasure() {{
    var fonts = document.fonts;
    if (fonts && fonts.ready && fonts.ready.then) {{
      fonts.ready.then(measure).catch(measure);
    }} else {{
      measure();
    }}
  }}
  if (document.readyState === 'complete') {{ waitFontsThenMeasure(); }}
  else {{ window.addEventListener('load', waitFontsThenMeasure); }}
</script>
</body></html>
"""


class HistoryRenderer:
    """按需把问答对渲染成 PNG，落到 `out_dir`。

    并发、超时、清理都在这里收口 —— 渲染会 fork 一个 Chromium，**必须**有上限：
    群里连发 hget 而不设闸，就是一次 fork 炸弹。
    """

    def __init__(self, out_dir: str, *, options: Optional[RenderOptions] = None,
                 logger=None) -> None:
        self.out_dir = out_dir
        self.opt = options or RenderOptions()
        self.logger = logger
        #: 同时只允许一个 Chromium（多张图串行渲染；一张几十毫秒到几秒）
        self._gate = asyncio.Semaphore(1)
        self._chrome: Optional[str] = None
        self._pending_width = 0          # 页面自报的「需要多宽」（0 = 用默认宽度）
        self._force_width = 0            # 调用方指定的出图宽度（0 = 不指定）
        self._crop_right = False         # 出图后是否按内容右侧收口
        self._crop_all = False           # 出图后是否按内容四边收口（居中的独立块）
        self._seq = 0
        self._unavailable = ""      # 探测失败的原因（避免每次重试都刷日志）

    # ---- 探测 ----

    def chromium(self) -> Optional[str]:
        """可用则返回 Chromium 路径，否则 None 并把原因记在 `unavailable()`。"""
        if self._chrome:
            return self._chrome
        got = find_chromium(self.opt.chrome)
        if not got:
            self._unavailable = (
                "找不到可用的 Chromium（无头浏览器）。"
                "用 `history.chromium` 指定路径，或确保它在 PATH 里。")
            return None
        self._chrome = got
        return got

    def unavailable(self) -> str:
        return self._unavailable

    # ---- 渲染 ----

    async def render(self, ex: Exchange, *, session_name: str = "",
                     total: int = 0, context: Optional[list] = None,
                     media: Optional[list] = None) -> str:
        """渲染一对问答，返回 PNG 的绝对路径。失败抛 `RenderError`。"""
        chrome = self.chromium()
        if not chrome:
            raise RenderError(self._unavailable or "渲染器不可用")
        os.makedirs(self.out_dir, exist_ok=True)
        self._seq += 1
        stamp = f"{int(time.time() * 1000)}-{self._seq}"
        html_path = os.path.join(self.out_dir, f"h-{stamp}.html")
        png_path = os.path.join(self.out_dir, f"h-{stamp}.png")
        profile = os.path.join(self.out_dir, f"prof-{stamp}")

        doc = build_html(ex, width=self.opt.width, session_name=session_name,
                         total=total, max_chars_per_side=self.opt.max_chars_per_side,
                         context=context, media=media)
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(doc)
        except OSError as e:
            raise RenderError(f"写渲染临时文件失败：{e}") from e

        try:
            async with self._gate:
                height = await self._measure(chrome, html_path, profile)
                await self._shoot(chrome, html_path, png_path, profile, height)
        finally:
            # 无论成败都清理：临时 HTML 与 Chromium 的 profile 目录
            # （profile 一次能有几 MB，SD 卡上不能留）
            for path in (html_path, profile):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            raise RenderError("渲染没有产出图片（Chromium 未写出文件）")
        self.prune()
        return png_path

    async def render_markdown(self, text: str, *, title: str = "") -> str:
        """把一段 Markdown 渲染成 PNG，返回路径。失败抛 `RenderError`。

        与 `render()`（问答卡）共用同一套「两趟渲染 + 并发闸 + 超时 + 清理」，
        区别只是 HTML 从哪来：这里是 `markdown.build_markdown_html()`。

        ⚠️ **不裁切**（`crop=False`）：它出的是一整页，页边距是版面的一部分。
        裁切只归**文字模式**的块图用 —— 那些图要插在文字之间，必须紧贴内容。
        """
        return await self._render_html(
            build_markdown_html(text, width=self.opt.width, title=title),
            crop=False)

    async def render_rich(self, kind: str, content: str) -> str:
        """把一块 Markdown（表格 / 公式 / 标题 / 列表 / 代码）渲染成 PNG。

        `kind` 见 `markdown.find_rich_blocks`。公式单独走 `build_math_html`
        （它带 KaTeX 的 display 处理），其余都走 `build_fragment_html` ——
        那一支复用 `markdown_to_html`，所以块内的行内公式与行内代码也对。
        """
        # 公式块与其它块同一条路：`build_fragment_html` 走 `markdown_to_html`，
        # 而 `markdown_to_html` 现在会把 `$$…$$` 渲染成 `math-display`（由 KaTeX 补渲染），
        # 再由像素裁剪**紧贴**内容收口。
        # ⚠️ 不再用 `build_math_html`：那个页面是定宽 720 + 公式居中，出图后两侧各留
        # 220 多像素空白（实测），与「紧贴」的要求相反。它仍保留给「单公式」的
        # 独立用法（`test_math_and_table_docs` 在测）。
        doc = build_fragment_html(content, kind=kind,
                                  width=FRAGMENT_WIDTHS.get(kind, self.opt.width))
        # 标题这类「一两行的块」出图后按内容收窄：铺满整页宽会是一条很空的白带
        # （正文与表格该铺满，所以只对窄块这么做）。
        # 不指定出图宽度，让页面按内容自己报（报多少用多少），最后再按像素收口。
        # ⚠️ 曾经给标题指定 420：页面上的容器就有 420 宽，内容只占 200 时右边
        # 会白 200 多像素，而像素收口只在「内容比容器宽」时才生效 —— 白边就留下了。
        self._force_width = 0
        self._crop_right = kind in FRAGMENT_CROP_RIGHT
        self._crop_all = kind in FRAGMENT_CROP_ALL
        try:
            # 文字模式的块图**要裁**：它们插在文字之间，必须紧贴内容
            return await self._render_html(doc, crop=True)
        finally:
            self._force_width = 0
            self._crop_right = False
            self._crop_all = False


    async def render_inline_math(self, latex: str) -> str:
        """把**行内**公式渲染成紧凑小图（插在文字之间用）。

        与 `render_rich` 的区别是**紧裁**：先问页面「公式实际多宽多高」，
        再按那个尺寸截图 —— 否则一张 12px 高的公式会被裁成整页宽（720px）的白条，
        插在文字之间就是一大片空白（实测）。
        """
        doc = build_inline_math_html(latex)
        chrome = self.chromium()
        if not chrome:
            raise RenderError(self._unavailable or "渲染器不可用")
        os.makedirs(self.out_dir, exist_ok=True)
        self._seq += 1
        stamp = f"{int(time.time() * 1000)}-{self._seq}"
        html_path = os.path.join(self.out_dir, f"h-{stamp}.html")
        png_path = os.path.join(self.out_dir, f"h-{stamp}.png")
        profile = os.path.join(self.out_dir, f"prof-{stamp}")
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(doc)
        except OSError as e:
            raise RenderError(f"写渲染临时文件失败：{e}") from e
        width = int(self.opt.width)
        try:
            async with self._gate:
                out, _err = await self._run_chrome(
                    chrome, [f"--virtual-time-budget={int(self.opt.virtual_time_budget)}",
                             "--dump-dom", _file_uri(html_path)], profile)
                m = _DIM_RE.search(out or "")
                if m:
                    # 宽度取公式实际宽度，但不小于 40px、不超过整页宽
                    width = max(40, min(int(m.group(1)), int(self.opt.width)))
                wide = _WIDE_RE.search(out or "")
                if wide:
                    # 页面说「内容更宽」→ 放宽（超长行内公式不被切）
                    width = max(width, min(int(wide.group(1)), self.opt.max_width))
                height = max(24, int(m.group(2)) if m else 60)
                await self._run_chrome(
                    chrome, [f"--virtual-time-budget={int(self.opt.virtual_time_budget)}",
                             "--screenshot=" + png_path, "--hide-scrollbars",
                             # 同样不能低于 MIN_SHOT_HEIGHT：低了 Chromium 画空白，
                             # 之后按像素紧裁自然也就裁不出东西
                             f"--window-size={width},{max(height, MIN_SHOT_HEIGHT)}",
                             _file_uri(html_path)], profile)
        finally:
            for path in (html_path, profile):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            raise RenderError("渲染没有产出图片（Chromium 没有写出文件）")
        # 按**像素**紧裁留白：页面只能给「元素的盒子」，给不了「墨迹在哪」——
        # 实测按盒子裁出来的图右边多 42px、下边多 30px 空白。
        crop_to_ink(png_path, pad_x=self.opt.inline_pad_x,
                    pad_y=self.opt.inline_pad_y)
        self.prune()
        return png_path

    async def _render_html(self, doc: str, *, crop: bool = True) -> str:
        """把一份完整 HTML 文档渲染成 PNG（两趟：量高度 → 按高度截图）。"""
        chrome = self.chromium()
        if not chrome:
            raise RenderError(self._unavailable or "渲染器不可用")
        os.makedirs(self.out_dir, exist_ok=True)
        self._seq += 1
        stamp = f"{int(time.time() * 1000)}-{self._seq}"
        html_path = os.path.join(self.out_dir, f"h-{stamp}.html")
        png_path = os.path.join(self.out_dir, f"h-{stamp}.png")
        profile = os.path.join(self.out_dir, f"prof-{stamp}")
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(doc)
        except OSError as e:
            raise RenderError(f"写渲染临时文件失败：{e}") from e
        try:
            async with self._gate:
                height = await self._measure(chrome, html_path, profile)
                await self._shoot(chrome, html_path, png_path, profile, height)
        finally:
            for path in (html_path, profile):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            raise RenderError("渲染没有产出图片（Chromium 没有写出文件）")
        # 留白按**像素**收口：底部一定有预留（量高度的安全余量），
        # 右侧则只在「内容被缩放过 / 页面比内容宽」时才需要裁 ——
        # 页面报的宽度与内容实际宽度经常对不上（容器定宽 + em 缩放），
        # 与其继续跟布局较劲，不如直接看像素。宽度不动则保持原样（该铺满就铺满）。
        if not crop:
            # 整页图**不裁**。裁切是**文字模式**的需要：那些块图要插在文字之间，
            # 必须紧贴内容，否则一行字中间夹一条几十像素的白边。
            # 而 Markdown 模式 / 问答卡出的是一整「页」，它就该有自己完整的
            # 页边距 —— 按墨迹裁掉右下留白，会让每次出图的边距都不一样
            # （内容长短一变，页边距就跟着变），看着就不像一页文档了。
            self.prune()
            return png_path
        if getattr(self, "_crop_all", False):
            # 居中的独立块（公式）：左右都是多余留白，四边一起收
            crop_to_ink(png_path, pad_x=PAGE_PAD, pad_y=PAGE_PAD)
        elif (getattr(self, "_pending_width", 0)
                or getattr(self, "_crop_right", False)):
            crop_right_to_ink(png_path)
        else:
            crop_bottom_to_ink(png_path)
        self.prune()
        return png_path

    async def _measure(self, chrome: str, html_path: str, profile: str) -> int:
        """第一趟：让页面自己报高度。

        ⚠️ `--virtual-time-budget` 要**给足**并且在预算内完成截图：
        公式（KaTeX）与表格依赖 CDN 上的 CSS/字体，字体没到位时量出来的高度偏小，
        截图就会把内容**截断**（实测：表格只露出表头的一半）。预算设成 0 等于
        「立刻快照」，一定截断；给几秒让虚拟时钟走完资源加载才对。
        """
        out, err = await self._run_chrome(
            chrome, [f"--virtual-time-budget={int(self.opt.virtual_time_budget)}",
                     "--dump-dom", _file_uri(html_path)], profile)
        wide = _WIDE_RE.search(out or "")
        self._pending_width = (min(int(wide.group(1)), self.opt.max_width)
                               if wide else 0)
        m = _HEIGHT_RE.search(out or "")
        if not m:
            # 量不到就用一个保守高度，而不是直接失败 —— 图能出，只是可能留白
            if self.logger:
                self.logger.warning("[ai_chat] 渲染高度探测失败，退回默认高度: %s",
                                    (err or out or "")[-200:])
            return min(self.opt.max_height, 1200)
        return max(1, min(int(m.group(1)), self.opt.max_height))

    async def _shoot(self, chrome: str, html_path: str, png_path: str,
                     profile: str, height: int) -> None:
        """第二趟：按量到的高度截图。"""
        # 截图这一趟同样要给虚拟时间预算：给 0 的话会在字体/KaTeX 就绪前快照，
        # 内容被截断（与 _measure 同一个坑）。
        shot_h = max(int(height), MIN_SHOT_HEIGHT)
        # 页面报了「内容更宽」就按那个宽度出图（超宽公式不被切掉的唯一办法）
        shot_w = int(getattr(self, "_force_width", 0)
                     or getattr(self, "_pending_width", 0) or self.opt.width)
        await self._run_chrome(
            chrome, [f"--virtual-time-budget={int(self.opt.virtual_time_budget)}",
                     "--screenshot=" + png_path, "--hide-scrollbars",
                     f"--window-size={shot_w},{shot_h}",
                     _file_uri(html_path)], profile)

    async def _run_chrome(self, chrome: str, args: list[str],
                          profile: str) -> tuple[str, str]:
        """跑一次 Chromium。参数一律走 argv（不拼 shell），带超时。"""
        _rc, out, err = await _run_chromium(
            chrome, args, profile=profile, timeout=float(self.opt.timeout))
        return out, err

    # ---- 清理 ----

    def prune(self) -> None:
        """图片目录只保留最近 `keep_files` 个文件（防止 SD 卡被慢慢写满）。"""
        keep = max(1, int(self.opt.keep_files))
        try:
            entries = []
            for name in os.listdir(self.out_dir):
                path = os.path.join(self.out_dir, name)
                if name.startswith("prof-"):
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
                    pass
        except OSError:
            pass


#: 框架版 `file_uri` 的本地别名（调用点保持原名，改动面最小）。
_file_uri = file_uri
