"""Markdown → HTML（受支持子集，为出图服务）。

**为什么自己写**：这台机器上没有任何 Markdown 库（也没有 bs4/lxml），而渲染
Markdown 又必须做对一件事 —— **转义**。模型的输出、以及它从网页抓回来的内容
都会进模板，不转义就是自己给自己开的注入（注入 `<img src=file:///etc/...>` 会被
Chromium 读出来画进图里发到群里）。一个可控的、只认白名单语法的转换器，比引一个
大而全的库更适合这个场景。

**支持的子集**（够用即可，未支持的语法原样显示为纯文本，不会报错也不会吞内容）：

| 语法 | 说明 |
|------|------|
| `#` ~ `######` | 标题 |
| `**粗体**` / `*斜体*` / `` `行内代码` `` | 行内 |
| ` ``` ` 围栏代码块 | 保留缩进与换行，等宽字体 |
| `-` / `*` / `1.` 列表 | 无序/有序，最多两级缩进 |
| `> 引用` | 引用块 |
| `| a | b |` 表格 | 带表头分隔行（`---`） |
| `[文字](链接)` | 只保留文字（图里点不了链接，显示 URL 没意义） |
| `---` | 分隔线 |
| 空行 | 段落分隔 |

⚠️ 顺序很关键：先把 `` ` `` 里的内容摘出来占位（否则 `**` 会在代码里被当成粗体），
再转义，最后才拼标签 —— 三步顺序颠倒任何一个都会出注入或错排版。
"""
from __future__ import annotations

import html
import logging
import os
import re
from pathlib import Path
from typing import Optional

#: 代码块/行内代码的占位符标记（不会与正常文本冲突：用私用区字符）
_PH = "\ue000{}\ue001"
_PH_RE = re.compile("\ue000(\\d+)\ue001")

_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([\w+#-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
#: 表格分隔行。前后竖线可选、允许对齐冒号；**单列表格也要认** ——
#: 原实现强制「至少两组 `-`」，于是 `| 只有表头 |\n| --- |` 会掉成普通段落。
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", re.S)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_STRIKE_RE = re.compile(r"~~(?!\s)(.+?)(?<!\s)~~", re.S)
_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_AUTOLINK_RE = re.compile(r"(?<![\"'>=])(https?://[^\s<>\u4e00-\u9fff]+)")


#: 行内公式的占位符（与代码片段共用同一套 stash 机制）。
#: ⚠️ 必须在 `_inline` **之前**摘出来：LaTeX 里的 `_`、`*`、`[` 会被
#: 斜体/链接规则误伤（`x_0` 变斜体、`a[1]` 变链接文字）。
#: **裸 LaTeX**（没写定界符）里「不可能出现在正文中」的命令 —— 见到就当公式。
#:
#: 为什么需要它：模型明明被要求用 `$...$`，实际仍会直接写
#: `\frac{x^2}{25}+\frac{y^2}{9}=1` —— 原样发出去就是一串反斜杠花括号
#: （用户截图报过）。
#:
#: 判据是「命令后面必须跟着花括号参数」：这类命令的**边界明确**（配对花括号到哪儿
#: 就是哪儿），而 `\sin`、`\pi` 这种裸命令「该包到哪儿」只能猜 ——
#: 猜错的代价是把用户的正常文字变成图片，所以宁可漏。
_BARE_LATEX_CMDS = (
    "frac", "dfrac", "tfrac", "sqrt", "overline", "underline", "vec", "hat",
    "bar", "dot", "ddot", "text", "mathrm", "mathbf", "mathbb", "mathcal",
    "begin", "end", "binom", "pmatrix", "bmatrix", "sum", "prod", "int",
)
#: `\left(…\right)` / `\left[…\right]`：成对定界符，整体算一个公式
_BARE_LEFTRIGHT_RE = re.compile(
    r"\\left\s*[([|.]\s*.{0,160}?\\right\s*[)\]|.]"
    r"(?:\s*[\^_]\s*\{?[^{}\s]{1,12}\}?)?")
#: 反斜杠 + 一对花括号（转义写法 `\{x\}`）—— 也算公式
_BARE_BRACED_RE = re.compile(r"\\\{[^{}]{1,80}\\\}")
#: 命令名 + 后面的可选空白
_BARE_CMD_RE = re.compile(r"\\([A-Za-z]+)")


def _read_braced(s: str, i: int) -> int:
    """从 `s[i] == '{'` 开始按配对读到闭合处，返回闭合花括号之后的索引。

    必须按配对读：根号里套分式（两层花括号）这种写法，用「非花括号字符」那种
    正则只会吃到内层的第一段，把一条公式切成三段（实测踩过）。
    """
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return -1                          # 花括号不配对 → 不认


def wrap_bare_latex(text: str) -> str:
    """把**没写定界符**的 LaTeX 命令包成 `$...$`，让它走公式出图那条路。

    只在「这段文本里完全没有 `$` 或反斜杠括号定界符」时动手：一旦有定界符，
    说明作者是按规矩写的，剩下的反斜杠更可能是转义字符（百分号、下划线、换行），
    此时不动比乱动安全。
    """
    body = str(text or "")
    if not body or "$" in body or "\\(" in body:
        return body
    if "\\" not in body:
        return body                       # 连反斜杠都没有，不用看
    # `\left(…\right)` / `\left[…\right]` 先单独处理：它靠"成对定界符"界定，
    # 没有花括号参数，走不了下面的命令分支。
    body = _BARE_LEFTRIGHT_RE.sub(lambda m: f"${m.group(0)}$", body)
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        if body[i] == "$":                # 上一步包好的，原样跳过
            end = body.find("$", i + 1)
            end = n if end < 0 else end + 1
            out.append(body[i:end])
            i = end
            continue
        m = _BARE_CMD_RE.match(body, i)
        if m is None or m.group(1) not in _BARE_LATEX_CMDS:
            out.append(body[i])
            i += 1
            continue
        j = m.end()
        # 可选参数 `[3]`（`\sqrt[3]{x}`）
        while j < n and body[j] == " ":
            j += 1
        if j < n and body[j] == "[":
            close = body.find("]", j)
            j = close + 1 if close > 0 else j
        # 至少一个花括号参数（可嵌套），否则这个命令没有明确边界，不认
        args = 0
        while True:
            k = j
            while k < n and body[k] == " ":
                k += 1
            if k >= n or body[k] != "{":
                break
            end = _read_braced(body, k)
            if end < 0:
                break
            j, args = end, args + 1
        if args == 0:
            out.append(body[i:m.end()])
            i = m.end()
            continue
        # 把紧跟在公式后面的**同类项**一起纳入，别把它留在图片外面：
        # ① 算术符号后接数字/单字母（`+1`、`-2`、`^2`）；
        # ② 关系符后接数字/单字母（`=1`、`<3`、`=a`）——`\frac{a}{b}=1` 应该是
        #    **一整条等式**，只把分式画成图、`=1` 留在文字里很难看（实测截图）。
        # 中文与中文标点一律不碰，所以「…的公式，下一句」不会被吞。
        trail = j
        while trail < n:
            k = trail
            while k < n and body[k] in " \t":
                k += 1
            if k >= n or body[k] not in "+-=<>^":
                break
            m2 = re.match(r"[+\-=<>^]\s*(?:\d+(?:\.\d+)?|[a-zA-Z])\b",
                          body[k:])
            if not m2:
                break
            trail = k + m2.end()
        j = trail
        out.append(f"${body[i:j]}$")
        i = j
    wrapped = _BARE_BRACED_RE.sub(lambda m: f"${m.group(0)}$", "".join(out))
    # `$\frac{a}{b}$+$\frac{c}{d}$=1` 这种被切成两段的合成一个：
    # 用户看到的应该是**一整条等式**，不是两张小图中间夹着 `+`。
    # 只吃相邻公式之间的算术符号（含中文标点不碰）。
    return re.sub(r"\$([^$\n]{1,400}?)\$([\s+\-=<>]*)\$([^$\n]{1,400}?)\$",
                  lambda m: f"${m.group(1)}{m.group(2)}{m.group(3)}$", wrapped)


def _math_body(m) -> str:
    """取出公式内容 —— 三个分支（`$$…$$` / `$…$` / `\\(…\\)`）各占一个捕获组。"""
    for i in (1, 2, 3):
        got = m.group(i)
        if got is not None:
            return got
    return ""


def _stash_math(text: str, stash) -> str:
    """把 `$...$` / `\\(...\\)` 行内公式摘成占位符（内容原样交给 KaTeX）。"""
    return _MATH_INLINE_RE.sub(lambda m: stash(_math_body(m), block=False,
                                               math=True), text)


def _inline(text: str) -> str:
    """行内元素 → HTML。`text` 已是**转义后**的文本，且代码/公式已占位。"""
    out = _LINK_RE.sub(r"\1", text)          # 链接只留文字（图里点不了）
    out = _AUTOLINK_RE.sub(r"\1", out)
    out = _BOLD_RE.sub(r"<strong>\1</strong>", out)
    out = _STRIKE_RE.sub(r"<del>\1</del>", out)
    out = _ITALIC_RE.sub(r"<em>\1</em>", out)
    return out


def _restore(text: str, codes: list[str]) -> str:
    """把占位符换回代码 HTML（代码内容**已经**转义过）。"""
    def repl(m: re.Match) -> str:
        idx = int(m.group(1))
        return codes[idx] if 0 <= idx < len(codes) else ""
    return _PH_RE.sub(repl, text)


def _stash_inline(line: str, stash) -> str:
    """把一行的行内代码摘成占位符（其余内容原样返回）。"""
    return _INLINE_CODE_RE.sub(lambda m: stash(m.group(1), block=False), line)


def markdown_to_html(text: str) -> str:
    """Markdown（子集）→ HTML 片段。所有文本都已转义。"""
    codes: list[str] = []

    def stash(raw: str, *, block: bool, math: bool = False,
              display: bool = False) -> str:
        """把代码/公式存起来，返回占位符。

        `math=True` 时**不转义**并包成 KaTeX 能识别的行内公式节点 ——
        理由同 `_embed_tex`：LaTeX 里的 `&`、`<` 转义后会被 KaTeX 当字面量
        （`x < y` 会渲染成 `x &lt; y`）。这里只防 `</`（避免提前闭合标签）。
        """
        if math:
            safe = str(raw).replace("</", "<\\/")
            cls = "math-display" if display else "math-inline"
            codes.append(
                f'<span class="{cls}">{html.escape(safe, quote=False)}</span>')
            return _PH.format(len(codes) - 1)
        esc = html.escape(raw, quote=False)
        if block:
            codes.append(f"<pre><code>{esc}</code></pre>")
        else:
            codes.append(f"<code>{esc}</code>")
        return _PH.format(len(codes) - 1)

    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # ---- 第一趟：摘出代码与公式，其余行原样留着 ----
    # 两个顺序约束都不能反：
    # ① 行内代码必须在这一趟就摘掉：晚了 `**` 会在代码内容里被当成粗体，
    #    而代码里的 `<` 又已经转义 —— 结果是一段既错排版又多出标签的东西；
    # ② 行间公式 `$$…$$` 要**整段**处理：它常常单独占一行、公式在中间行，
    #    逐行套跨行正则永远匹配不上（实测：整块公式带着 `$$` 原样发给了用户）。
    stashed: list[str] = []
    i = 0
    total = len(lines)
    while i < total:
        line = lines[i]
        m = _FENCE_RE.match(line)
        if m:
            fence = m.group(1)
            body: list[str] = []
            i += 1
            while i < total and not lines[i].strip().startswith(fence):
                body.append(lines[i])
                i += 1
            i += 1                              # 跳过收尾的 ```
            stashed.append(stash("\n".join(body), block=True))
            continue
        # 行间公式（同一行内 `$$…$$` 起止）
        stripped = line.strip()
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            inner = stripped[2:-2]
            if "$$" not in inner:
                stashed.append(stash(inner.strip(), block=False,
                                     math=True, display=True))
                i += 1
                continue
        # 行间公式（单独的 `$$` 行，直到下一个单独的 `$$`）
        if stripped == "$$":
            body = []
            i += 1
            while i < total and lines[i].strip() != "$$":
                body.append(lines[i])
                i += 1
            i += 1                              # 跳过收尾的 $$
            stashed.append(stash("\n".join(body).strip(), block=False,
                                 math=True, display=True))
            continue
        # 普通行：代码 → 行内公式（顺序不能反：代码里的 $ 不是公式）
        stashed.append(_stash_math(_stash_inline(line, stash), stash))
        i += 1

    # ---- 第二趟：按块解析 ----
    out: list[str] = []
    para: list[str] = []

    def flush_para() -> None:
        if not para:
            return
        # ⚠️ 段内的**单换行要保留**（渲染成 `<br>`），不能折成空格。
        # Markdown 标准把软换行视作空格，那是为「手写文档自动折行」设计的；
        # 这里是聊天，模型敲的每一行都是**有意换的行**（「第一步…\n第二步…」），
        # 折成一行会让出图与聊天里看到的不一致（历史卡片这条链路尤其明显：
        # 它就是一问一答的转录，必须跟原话一样）。
        # 注意顺序：**先逐行转义再拼**，否则 `<br>` 会被自己转义掉。
        parts = [html.escape(ln.strip(), quote=False) for ln in para]
        para.clear()
        joined = "<br>".join(x for x in parts if x)
        if joined:
            out.append(f"<p>{_restore(_inline(joined), codes)}</p>")

    i = 0
    n = len(stashed)
    while i < n:
        line = stashed[i]
        raw = line.strip()

        if not raw:                              # 空行 → 段落分隔
            flush_para()
            i += 1
            continue

        # 代码块占位符自成一块
        if _PH_RE.fullmatch(raw):
            flush_para()
            out.append(_restore(raw, codes))
            i += 1
            continue

        m = _HEADING_RE.match(raw)
        if m:
            flush_para()
            level = len(m.group(1))
            out.append(f"<h{level}>{_restore(_inline(html.escape(m.group(2).strip(), quote=False)), codes)}</h{level}>")
            i += 1
            continue

        if _HR_RE.match(raw):
            flush_para()
            out.append("<hr>")
            i += 1
            continue

        # 表格：当前行含 | 且下一行是分隔行
        if "|" in raw and i + 1 < n and _TABLE_SEP_RE.match(stashed[i + 1].strip()):
            flush_para()
            header = [c.strip() for c in raw.strip("|").split("|")]
            rows: list[list[str]] = []
            i += 2
            while i < n and "|" in stashed[i] and stashed[i].strip():
                rows.append([c.strip() for c in stashed[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{_restore(_inline(html.escape(c, quote=False)), codes)}</th>"
                         for c in header)
            trs = "".join(
                "<tr>" + "".join(
                    f"<td>{_restore(_inline(html.escape(c, quote=False)), codes)}</td>"
                    for c in row) + "</tr>"
                for row in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
            continue

        m = _QUOTE_RE.match(raw)
        if m:
            flush_para()
            body: list[str] = [m.group(1)]
            i += 1
            while i < n and _QUOTE_RE.match(stashed[i].strip()):
                body.append(_QUOTE_RE.match(stashed[i].strip()).group(1))
                i += 1
            inner = " ".join(body).strip()
            out.append(f"<blockquote>{_restore(_inline(html.escape(inner, quote=False)), codes)}</blockquote>")
            continue

        # 列表（连续同类项合成一个 <ul>/<ol>）
        lm = _UL_RE.match(line) or _OL_RE.match(line)
        if lm:
            flush_para()
            ordered = bool(_OL_RE.match(line))
            items: list[str] = []
            while i < n:
                cur = stashed[i]
                mm = _OL_RE.match(cur) if ordered else _UL_RE.match(cur)
                if not mm:
                    break
                body = mm.group(3) if ordered else mm.group(2)
                # 续行（缩进的下一行）并进当前项
                i += 1
                while i < n and stashed[i].strip() and not (
                        _UL_RE.match(stashed[i]) or _OL_RE.match(stashed[i])
                        or _HEADING_RE.match(stashed[i].strip())):
                    body += " " + stashed[i].strip()
                    i += 1
                items.append(f"<li>{_restore(_inline(html.escape(body.strip(), quote=False)), codes)}</li>")
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        para.append(raw)
        i += 1

    flush_para()
    return "\n".join(out)


def markdown_body_css() -> str:
    """Markdown **元素**的样式：标题、段落、列表、代码、表格、引用、**公式**。

    ⚠️ 这一份是**唯一**的 owner。凡是「Markdown 渲染进 HTML 再出图」的地方
    （Markdown 模式整篇、片段块、问答卡 `hget`）都必须引它 —— 各自抄一套的后果是
    同一段 Markdown 在几个地方长得不一样，而且改一处漏两处。问答卡原先干脆没引，
    于是 `$$…$$` 在历史卡片里一直是一串源码（用户报过）。

    只放**元素级**规则；页面外壳（容器宽度、字号、`body` 字体）由调用方自己定 ——
    那些本来就该随场景不同。
    """
    return """
    h1, h2, h3, h4, h5, h6 { margin: 20px 0 10px; line-height: 1.4; }
    h1 { font-size: 26px; } h2 { font-size: 23px; }
    h3 { font-size: 21px; } h4, h5, h6 { font-size: 19px; }
    p { margin: 10px 0; }
    ul, ol { margin: 10px 0; padding-left: 26px; }
    li { margin: 4px 0; }
    code {
      font-family: "Noto Sans Mono", "DejaVu Sans Mono", monospace;
      font-size: 16px; background: #f2f3f5; padding: 1px 5px; border-radius: 4px;
    }
    pre {
      background: #f6f7f9; border: 1px solid #e6e8eb; border-radius: 8px;
      padding: 12px 14px; margin: 12px 0; overflow-x: auto;
    }
    pre code { background: none; padding: 0; font-size: 15px; line-height: 1.6; }
    blockquote {
      margin: 12px 0; padding: 8px 14px; border-left: 4px solid #c9d4e4;
      background: #f7f9fc; color: #44506b;
    }
    table { border-collapse: collapse; margin: 12px 0; width: 100%; }
    th, td { border: 1px solid #dfe3e8; padding: 7px 10px; text-align: left; }
    th { background: #f2f5fa; }
    hr { border: none; border-top: 1px solid #e6e8eb; margin: 18px 0; }
    strong { font-weight: 700; }
    /* 行内公式稍小一点：它要跟正文混排 */
    .math-inline { font-size: 0.95em; }
    /* 行间公式：居中；**不要** overflow-x:auto —— 出图不支持滚动，
       那样只会把右边切掉；宽度不够时由 _SIZE_SCRIPT 放宽整张图。
       按**自身宽度**排版：不给容器宽度限制，元素才量得出真实需要的宽度，
       也才不会在行内盒内部溢出（那样量宽度与截图都会被静默裁掉）。 */
    .math-display { display: block; margin: 14px 0; text-align: center;
                     font-size: 1.06em; width: max-content; max-width: none; }
    /* ⚠️ KaTeX 自带的 .katex-display 是 `overflow: auto` —— 它会把超宽公式
       裁在容器里，连**几何查询也一起裁**（getBoundingClientRect 只报容器宽），
       于是「页面自报需要多宽」那段永远测不到溢出、图照样被切（实测）。
       出图不支持滚动，所以这里把它关掉，让内容真的撑开，再由 _SIZE_SCRIPT 放宽整图。 */
    .math-display .katex-display { margin: 0; overflow: visible !important; }
"""


def _md_css(width: int) -> str:
    """Markdown 整篇出图的样式 = 页面外壳 + `markdown_body_css()`。"""
    css = f"""
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: #ffffff; }}
    body {{
      font-family: "Noto Sans CJK SC", "Source Han Sans SC",
                   "WenQuanYi Zen Hei", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .md {{
      width: {int(width)}px; padding: 24px 26px 20px;
      font-size: 18px; line-height: 1.75; color: #1f2329;
    }}
    .md > *:first-child {{ margin-top: 0; }}
""" + markdown_body_css()
    return f"{css}    {_MATH_FAILED_CSS}"


def katex_head(html_fragment: str, *, force: bool = False) -> str:
    """片段里有行内公式时才引 KaTeX（省一次 CDN 往返）。

    ⚠️ 公式渲染脚本必须**等 load**：外链脚本按文档顺序执行，但下载是异步的 ——
    标签排在前面的 `katex.min.js` 没下载完时 `typeof katex === 'undefined'`，
    渲染脚本会静默 return，公式全留成源码（实测踩过）。

    `force=True` 是给**明确知道这段就是公式**的调用方用的（`kind="math"`）：
    这一支靠「HTML 里有没有 `math-*` 类名」来嗅探，而嗅探一旦不中就是一整条
    KaTeX 都不引 —— 出图里只剩一行 LaTeX 源码，还长得像正常文字，
    从图上看不出是渲染失败（实测踩过，查了很久）。知道是公式就别靠嗅探。
    """
    if not force and 'class="math-inline"' not in html_fragment \
            and 'class="math-display"' not in html_fragment:
        return ""
    return (f'<link rel="stylesheet" href="{KATEX_CSS}">'
            f'<script src="{KATEX_JS}"></script>'
            f'<script>{_KATEX_INLINE_JS}</script>')


def as_display_math(latex: str) -> str:
    """一段**裸** LaTeX 源码 → 带 `$$` 定界符的行间公式（已经带了就原样返回）。

    为什么要这一步：`find_rich_blocks` 交出来的公式块内容是**去掉定界符**的
    纯 LaTeX（`m.group(1)`），而 `build_fragment_html` 那一支要把内容再喂回
    `markdown_to_html` 重新解析 —— 补不回定界符，它就只是个普通段落，
    `katex_head()` 嗅不到 `math-*` 类名、一份 KaTeX 都不引，出图就是
    **一行 16px 的 LaTeX 源码**（实测：244×33，墨迹 217×17，正是用户截图报的那张）。
    """
    body = str(latex or "").strip()
    if not body:
        return body
    if len(body) > 4 and body.startswith("$$") and body.endswith("$$"):
        return body
    if body.startswith("\\[") and body.endswith("\\]"):
        return body
    # 用「单独成行」的写法：多行的公式（cases / aligned）也走同一条分支
    return f"$$\n{body}\n$$"


def build_markdown_html(text: str, *, width: int = 720,
                        title: str = "") -> str:
    """Markdown 全文 → 完整 HTML 文档（带量高度脚本 + 行内公式渲染）。

    `markdown_to_html` 把 `$...$` 包成 `<span class="math-inline">`（原文在里面），
    这里再让 KaTeX 把它们渲染成真公式。**公式渲染失败不影响整篇** ——
    单个 span 出错就让它保持原文，其余照常出图。
    """
    body = markdown_to_html(text)
    head = f'<div class="md-title">{html.escape(title)}</div>' if title else ""
    katex = katex_head(body)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_md_css(width)}</style>{katex}</head>
<body>
<div class="md">
{head}
{body}
</div>
{_SIZE_SCRIPT}
</body></html>
"""


#: 把每个 `.math-inline` 交给 KaTeX 渲染；**逐个 try/catch** ——
#: 一个公式写错不该让整张图退回源码（整篇 fallback 是 11.4 那层的事）。
#: 取不到 KaTeX（没网）时它保持原文，用户至少看得到 LaTeX。
_KATEX_INLINE_JS = """
(function () {
  function render() {
    var nodes = document.querySelectorAll('.math-inline, .math-display');
    if (!nodes.length) { return; }
    if (typeof katex === 'undefined') {
      // KaTeX 没就绪（自带资源被删、CDN 不可达…）——**不能静默 return**：
      // 那样公式会以「看起来像正常文字」的样子发出去，用户根本不知道是渲染失败
      // （实测踩过：CSS/字体没回来时出图就是一行源码）。这里统一打上
      // `math-failed`，样式上退成等宽字体，和正常排版一眼可辨。
      for (var k = 0; k < nodes.length; k++) {
        nodes[k].className += ' math-failed';
      }
      return;
    }
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      try {
        var disp = el.className.indexOf('math-display') >= 0;
        katex.render(el.textContent, el, {displayMode: disp, throwOnError: true});
        el.className = (disp ? 'math-display' : 'math-inline') + ' katex-inline';
      } catch (e) {
        // 单个公式写错就让它留原文，不影响其它公式与整页排版
        el.className = (el.className.indexOf('math-display') >= 0
                        ? 'math-display' : 'math-inline') + ' math-failed';
      }
    }
  }
  // ⚠️ 必须等 load：脚本按文档顺序执行，这段在 <head> 里、排在 katex.min.js
  // **之后**，但那只是「标签顺序」—— 外链脚本还没下载完时 `typeof katex`
  // 就是 undefined，于是整段静默 return，公式全留成源码（实测踩过）。
  if (document.readyState === 'complete') { render(); }
  else { window.addEventListener('load', render); }
})();
"""


# ==================== 行内富文本块（表格 / 公式） ====================

#: 行间公式：`$$...$$` 或 `\[...\]`（单独成段时）
_MATH_BLOCK_RE = re.compile(r"(?ms)^\s*(?:\$\$|\\\[)\s*(.+?)\s*(?:\$\$|\\\])\s*$")
#: 行内公式：`$...$`（要求两侧不是数字/字母，避免把「价格 5$ 到 10$」当公式）
#: 行间公式 `$$...$$`（可跨行；DOTALL 让 `.` 匹配换行）。
#: 内容至少 1 个字符，避免把孤立的 `$$$$` 吃掉。
_MATH_DISPLAY_RE = re.compile(r"\$\$(.+?)\$\$", re.S)

#: 公式没渲染成功时的样式：等宽 + 红底，和正常排版一眼可辨。
#:
#: 为什么要专门做这个：出图发出去时用户只看到**一张图**，分不出
#: 「排版好的公式」和「没渲染成功的源码」—— 实测出现过 CSS/字体没回来、
#: 出图就是一行 `\frac{x^2}{16}+\frac{y^2}{4}=1` 的情况，用户以为功能坏了。
#: 现在失败会显式标红，一眼能看出是渲染失败而不是公式长这样。
#: （注意这两个模板都是 f-string，样式里的花括号要写成双份。）
_MATH_FAILED_CSS = """    .math-failed {
      font-family: "Noto Sans Mono", monospace; font-size: 15px;
      background: #fff1f0; color: #a33; padding: 0 4px; border-radius: 4px;
    }
    .math-display.math-failed { display: block; padding: 4px 6px; }
"""


#: 公式正文：**允许换行**，但最多 `_MATH_MAX_LINES` 行。
#:
#: ⚠️ 原先写的是 `[^$\n]{1,200}` —— 把换行排除在外，于是**跨行的 `$…$` 一个都
#: 认不出来**。而 `\begin{cases}` / `\begin{aligned}` 这类环境**必须**换行写：
#:
#: ```
#: 解方程组 $\begin{cases}
#: y = 2x + 1
#: y = -x + 3
#: \end{cases}$，可得：
#: ```
#:
#: 结果整串原样发给了用户（实测截图里 `$`、`\begin{cases}` 全是裸文本）。
#: 行数仍要设上限：不限的话，正文里孤零零一个 `$` 会和几十行之外的另一个 `$`
#: 配成一对，把中间整段话吞进公式。
#: 需要「一行一个式子」的 LaTeX 环境（`\\` 分行）。模型常直接用换行，
#: 见 `normalize_math_rows`。
_MULTIROW_ENVS = ("cases", "aligned", "align", "align*", "alignedat", "array",
                  "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix",
                  "Vmatrix", "smallmatrix", "gathered", "split", "subarray",
                  "dcases", "rcases")
_MULTIROW_RE = re.compile(r"\\begin\{(?:%s)\}" % "|".join(
    re.escape(e) for e in _MULTIROW_ENVS))

_MATH_MAX_LINES = 6
_MATH_BODY = (r"[^$\n]{0,200}" + r"(?:\n[^$\n]{1,200}){0,%d}" % (_MATH_MAX_LINES - 1))
#: 至少要有一个非空白字符，别把 `$ $` 这种噪声摘出来
_MATH_HAS_CONTENT = r"(?=[^$]*[^\s$])"

#: 行内公式 `$...$`，以及 LaTeX 的另一套标准定界符 `\(...\)`。
#:
#: 为什么还要认 `\(...\)`：**很多模型（和不少用户）就是这么写行内公式的**
#: （`\(\frac{1}{2}\)`、`\(\sin x\)`）。只认 `$...$` 的话这些会原样发出去，
#: 变成一串反斜杠括号。KaTeX 本身就收 `\(...\)`，摘出来直接交给它即可。
#:
#: 两条例外都是为「表格里的结果列」服务的 —— 那一列常常就是 `$1$`、`$0$`、`$x$`
#: 这种**单字符**公式，一刀切要求 2 个字符会整列漏掉（实测）。
#: 同时要挡住货币写法：`$5 到 $10`、`价格 $5 元` 这类**闭合 `$` 后面跟着数字**的
#: 明显不是公式，用 `(?!\d)` 排除掉。
_MATH_INLINE_RE = re.compile(
    # ① `$$…$$`：**先于单 `$` 匹配**，这样它出现在句子中间也能被认出来。
    #    原先只认「单独成段」，于是「椭圆：$$…$$」这种写法整串原样发出去
    #    （用户截图报过）。`$$` 本来就该跟 `$` 同等对待，只是排版更大。
    r"\$\$(" + _MATH_BODY + r")\$\$"
    r"|(?<![\w$])\$(" + _MATH_BODY + r")\$(?![\w$])(?!\d)"
    r"|\\\((" + _MATH_BODY + r")\\\)")


def _is_table_sep(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))


#: 列表行（无序 `- * +` 与有序 `1.` / `1)`）；`markdown_to_html` 也用它
_UL_LINE = re.compile(r"^\s*[-*+]\s+\S")
_OL_LINE = re.compile(r"^\s*\d+[.)]\s+\S")

#: 当「块」处理的标题级别：`###` 及以上（`#`/`##` 留给正文）
_HEADING_MIN_RE = re.compile(r"^#{3,6}\s+\S")

#: 一段列表至少要有这么多行才当「块」。
#: 只要一行的话，正文里随手一个 `- ` 开头（甚至破折号）都会被拉出来单独出图，
#: 反而把好好的段落切碎。
_MIN_LIST_ITEMS = 2


def find_rich_blocks(text: str) -> list[tuple[int, int, str, str]]:
    """找出正文里需要**单独出图**的块。

    返回 `[(起始行, 结束行, 类型, 内容)]`，类型为
    `"table"` / `"math"` / `"heading"` / `"list"` / `"code"`；
    行号是 `splitlines()` 的下标，**左闭右开**，按出现顺序排列。

    为什么要单独出图：QQ 不渲染 Markdown —— 表格发出去是竖线、公式是反斜杠、
    标题是一串 `#`（实测截图里 `### 标题` 就是原样发出去的）。提示词里虽然要求
    别用，但模型**确实会用**，所以与其假装它不会，不如把这些块挑出来渲染成图。
    """
    lines = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[int, int, str, str]] = []
    i = 0
    n = len(lines)
    while i < n:
        # 表格：当前行是表头（含 |）且下一行是分隔行
        if ("|" in lines[i] and i + 1 < n and _is_table_sep(lines[i + 1])
                and lines[i].strip()):
            start = i
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                i += 1
            out.append((start, i, "table", "\n".join(lines[start:i])))
            continue
        # 行间公式：单行 $$...$$ 或 \[...\]
        m = _MATH_BLOCK_RE.match(lines[i])
        if m:
            out.append((i, i + 1, "math", m.group(1).strip()))
            i += 1
            continue
        # 行内公式单独成行（模型常这么写）
        stripped = lines[i].strip()
        if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 2:
            out.append((i, i + 1, "math", stripped.strip("$").strip()))
            i += 1
            continue
        # 标题：`### 标题`。QQ 不渲染 `#`，留着就是一串井号（实测截图里就有）。
        # ⚠️ 只认 3 个及以上：`#` / `##` 与正文里的井号（`#1`、`C#`）太容易混，
        # 把它们当标题会误伤；而模型写小标题基本都用 `###`。
        if _HEADING_MIN_RE.match(stripped):
            out.append((i, i + 1, "heading", lines[i]))
            i += 1
            continue
        # 代码围栏：整段搬走（裸贴时缩进与换行才是对的）
        fm = _FENCE_RE.match(lines[i])
        if fm:
            fence = fm.group(1)
            start = i
            i += 1
            while i < n and not lines[i].strip().startswith(fence):
                i += 1
            i += 1                              # 跳过收尾的 ```
            out.append((start, min(i, n), "code", "\n".join(lines[start:min(i, n)])))
            continue
        # 列表：**连续**多项才算一块（单个 `- ` 开头多半只是正文里的破折号）
        if _UL_LINE.match(lines[i]) or _OL_LINE.match(lines[i]):
            ordered = bool(_OL_LINE.match(lines[i]))
            start = i
            count = 0
            while i < n and lines[i].strip() and (
                    _OL_LINE.match(lines[i]) if ordered else _UL_LINE.match(lines[i])):
                count += 1
                i += 1
            if count >= _MIN_LIST_ITEMS:
                out.append((start, i, "list", "\n".join(lines[start:i])))
                continue
            i = start + 1                        # 不够成块 → 当普通行
            continue
        i += 1
    return out


def split_rich_segments(text: str) -> list[tuple[str, str]]:
    """把正文切成 `[(类型, 内容)]`，类型为 `"text"` / `"table"` / `"math"`。

    相邻的文本会合并；空文本段会被丢掉。**保序**是硬要求 —— 图必须插在
    它原来所在的位置，否则用户读到的上下文就乱了。
    """
    body = str(text or "")
    blocks = find_rich_blocks(body)
    if not blocks:
        return [("text", body)] if body.strip() else []
    lines = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segs: list[tuple[str, str]] = []
    cursor = 0

    def add_text(chunk: str) -> None:
        chunk = chunk.strip("\n")
        if not chunk.strip():
            return
        if segs and segs[-1][0] == "text":
            segs[-1] = ("text", segs[-1][1] + "\n" + chunk)
        else:
            segs.append(("text", chunk))

    for start, end, kind, content in blocks:
        add_text("\n".join(lines[cursor:start]))
        segs.append((kind, content))
        cursor = end
    add_text("\n".join(lines[cursor:]))
    return segs


# ---- 文本模式：把 Markdown 压成纯文本 ------------------------------------

#: 文本模式里要**一字不改**的片段：数学公式（含 `*` `_` `\frac`，被当成强调/转义就改错意思了）
_PLAIN_PROTECT_RE = re.compile(
    r"\$\$.+?\$\$"
    r"|\\\[.+?\\\]"
    r"|(?<![\w$])\$[^$\n]{1,200}\$(?![\w$])(?!\d)"
    r"|\\\(.+?\\\)", re.S)
_BOLD2_RE = re.compile(r"__(?!\s)(.+?)(?<!\s)__", re.S)
#: Markdown 反斜杠转义（只处理这些字符；`\f` `\s` 这类是 LaTeX，不能动）
_MD_ESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~>|])")


def to_plain_text(text: str) -> str:
    r"""把 Markdown 正文压成「QQ 纯文本里读得懂」的样子（**不改写内容**）。

    文本模式（没加 `//md`）整段不出图，而 QQ 不渲染 Markdown —— 直接把模型输出
    发出去，用户看到的就是 `**重点**` 的星号、`> 引用` 的尖括号、
    `[文档](https://…)` 的原样链接、`` `code` `` 的反引号。

    这里只做**降噪**：去强调/删除线标记、链接改成 `标题（url）`、引用去掉 `>`、
    分隔线换成实线、标题去掉 `#`、行内代码去掉反引号、还原 `\*` 这类转义；
    顺带压掉行尾空白与 3 个以上连续空行（QQ 里会显得空得莫名其妙）。

    ⚠️ 数学公式（`$…$` / `$$…$$` / `\(…\)` / `\[…\]`）整段保护、一字不改：
    公式里的 `*` `_` `\frac` 如果被当成 Markdown 处理，语义直接变了。
    """
    body = str(text or "")
    if not body.strip():
        return body
    stash: list[str] = []

    def keep_math(m):
        stash.append(m.group(0))
        return _PH.format(len(stash) - 1)

    body = _PLAIN_PROTECT_RE.sub(keep_math, body)

    def keep_escape(m):
        # `\*` = 用户/模型想要的**字面星号**，先收起来，否则会被下面的强调规则
        # 当成斜体标记吃掉（实测 `\*星号\*` 会变成 `\星号\`）。
        stash.append(m.group(1))
        return _PH.format(len(stash) - 1)

    body = _MD_ESCAPE_RE.sub(keep_escape, body)

    def keep_code(m):
        stash.append(m.group(1))          # 行内代码：保留内容、去掉反引号
        return _PH.format(len(stash) - 1)

    body = _INLINE_CODE_RE.sub(keep_code, body)

    lines: list[str] = []
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if _HR_RE.match(line):
            lines.append("─" * 10)
            continue
        line = _HEADING_RE.sub(lambda m: m.group(2).strip(), line)
        line = _QUOTE_RE.sub(lambda m: m.group(1), line)
        lines.append(line)
    body = "\n".join(lines)

    body = _BOLD_RE.sub(lambda m: m.group(1), body)
    body = _BOLD2_RE.sub(lambda m: m.group(1), body)
    body = _STRIKE_RE.sub(lambda m: m.group(1), body)
    body = _ITALIC_RE.sub(lambda m: m.group(1), body)
    body = _LINK_RE.sub(
        lambda m: (m.group(1) if m.group(1).strip() == m.group(2).strip()
                   else f"{m.group(1)}（{m.group(2)}）"), body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = _PH_RE.sub(lambda m: stash[int(m.group(1))], body)
    return body.strip()


def has_rich_blocks(text: str) -> bool:
    return bool(find_rich_blocks(text))


def normalize_math_rows(tex: str) -> str:
    """多行环境里把**裸换行补成 `\\\\`**。

    KaTeX 在数学模式里**忽略换行**，而 LaTeX 的 `cases` / `aligned` / 矩阵要求
    行与行之间写 `\\\\`。模型却常常直接换行：

    ```
    $\\begin{cases}
    y = 2x + 1
    y = -x + 3
    \\end{cases}$
    ```

    不补的话渲染出来是一行挤三个式子（用户截图里的方程组就是这个）。这里只在
    **确实含多行环境**、且内容里有换行时才动手，普通公式一律原样。
    """
    body = str(tex or "")
    if "\n" not in body or not _MULTIROW_RE.search(body):
        return body
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if len(lines) < 2:
        return body
    out = lines[0]
    for prev, ln in zip(lines, lines[1:]):
        # `\end{...}` 前面不加；上一行本身就是 `\begin{...}`（环境名独占一行）时
        # 也不加 —— 否则公式开头会多出一个空行（KaTeX 会忽略，但源码不该脏）；
        # 已经有 `\\` 的更不重复加。
        bare_begin = bool(re.fullmatch(r"\\begin\{[^}]*\}", prev))
        sep = " " if (ln.startswith("\\end{") or bare_begin
                      or prev.endswith("\\\\")) else " \\\\ "
        out += sep + ln
    return out


def split_inline_math(text: str) -> list[tuple[str, str]]:
    """把一段**文本**按行内公式切成 `[("text", …), ("math", …), …]`，保序。

    为什么还要切一层：`split_rich_segments` 只挑出**整块**的表格/公式，
    正文里的 `$x^{0}$` 会原样留在 text 段里 —— 用户看到的还是 LaTeX 源码
    （实测截图：「常数函数通常并入幂函数（$x^{0}$）」）。这里把行内公式也挑出来，
    交给调用方渲染成小图插进消息体。
    """
    body = wrap_bare_latex(text)
    if not body:
        return []
    out: list[tuple[str, str]] = []
    cursor = 0
    for m in _MATH_INLINE_RE.finditer(body):
        before = body[cursor:m.start()]
        if before.strip():
            out.append(("text", before))
        out.append(("math", normalize_math_rows(_math_body(m)).strip()))
        cursor = m.end()
    tail = body[cursor:]
    if tail.strip():
        out.append(("text", tail))
    if not out:
        return [("text", body)] if body.strip() else []
    return out


def has_inline_math(text: str) -> bool:
    return bool(_MATH_INLINE_RE.search(wrap_bare_latex(text)))


# ==================== 表格 / 公式 单独出图的文档 ====================

#: 量内容真实高度的脚本（各出图文档共用一份）。
#:
#: 两个坑都实测踩过，所以这里写成一份、别处只 `{_SIZE_SCRIPT}` 引用：
#: ① 不能用 `getBoundingClientRect().height` —— 它是「视口内可见高度」，
#:    内容超出初始视口时偏小（表格实测 158 vs 真实 226），按它截图会截掉下半截；
#: ② 也不能直接用 `documentElement.scrollHeight` —— 视口默认高度会把它**撑大**
#:    （实测真实内容 158px 时它报 403px），图底部就留一大片空白。
#: 正确做法：html/body 设 `height:auto`，量**内容容器**，并把探针元素排除在外。
_SIZE_SCRIPT = """<script>
  (function () {
    var b = document.body;
    if (b) { b.style.height = 'auto'; b.style.minHeight = '0'; }
    var wrap = document.querySelector('.wrap') || document.querySelector('.card')
               || document.querySelector('.md') || document.querySelector('.inline-math');
    var mdEl = document.querySelector('.md') || document.querySelector('.wrap')
               || document.querySelector('.card');
    if (!mdEl) { document.title = 'H=240'; return; }

    function box(el) { return el.getBoundingClientRect(); }

    // ① 先量内容的**真实**边界。公式会在行内盒内部溢出，所以遍历后代取极值，
    //    不能只看容器宽度（实测：容器 668px，公式自然宽度 2703px）。
    function extremes() {
      var L = Infinity, R = -Infinity, T = Infinity, B = -Infinity;
      var nodes = mdEl.querySelectorAll('*');
      for (var i = 0; i < nodes.length; i++) {
        var r = box(nodes[i]);
        if (r.width === 0 && r.height === 0) { continue; }
        if (r.left < L) { L = r.left; }
        if (r.right > R) { R = r.right; }
        if (r.top < T) { T = r.top; }
        if (r.bottom > B) { B = r.bottom; }
      }
      var wr = box(mdEl);
      return { L: Math.min(L, wr.left), R: Math.max(R, wr.right),
               T: Math.min(T, wr.top), B: Math.max(B, wr.bottom) };
    }

    // ③ 等字体到位再量：KaTeX 的 web 字体没换上来时公式更矮，量完再换字体就会
    //    把底部切掉（实测：量到 279px，实际内容到 300+）。整个量测放进
    //    fonts.ready 里跑一遍。
    function measureAndReport() {
    var need = extremes().R - extremes().L;
    // ② 太宽就整体等比缩小。
    // ⚠️ 用**根字号**缩放而不是 CSS zoom：zoom 会让 getBoundingClientRect 的
    // 坐标空间乱掉，量出来的边界与实际画出来的对不上（实测：墨迹只占左上角
    // 344px，右侧全是空白）。KaTeX 的字号本身是 em 相对的，缩根字号就能等比缩小。
    var MAXW = 1160;   // 出图宽度上限（再宽在 QQ 里也看不清）
    var wide = false;
    if (need > MAXW) {
      var scale = MAXW / need;
      var rootPx = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      document.documentElement.style.fontSize = (rootPx * scale) + 'px';
      wide = true;
    }
    // ③ 缩放之后再取最终外框：宽高都用**缩放后**的真实边界，
    //    否则缩放后图还是按缩放前的宽度出，右边会留一大片空白（实测 403px）。
    var e = extremes();
    var h = Math.ceil(e.B - e.T) + 8;
    // ⚠️ 宽度取「容器宽」与「内容真实边界」的**较大者**。
    // 只取容器宽会**切掉**超出的内容（实测：列表/标题的墨迹伸到 106px 之外，
    // 而容器只报了那么多 —— 图右边就被裁了）。
    var wrapW = Math.ceil(e.R - e.L);
    var boxW = 0;
    try {
      var wb = (wrap || mdEl).getBoundingClientRect();
      boxW = Math.ceil(wb.width);
    } catch (err) { boxW = 0; }
    var w = Math.max(wrapW, boxW) + 12;
    // 缩放后按理内容就在 MAXW 内；但量测受字体/行内盒影响可能偏大，
    // 所以宽度**夹到 MAXW**（宁可多留点白，也不要出一张 9000px 的图）。
    if (w > MAXW) { w = MAXW; }
    // 独立的公式块是**居中**排的：容器占满整页宽，而公式本身可能只有一百多像素。
  // 这时按整页宽出图，公式两侧会各留一大片空白（实测右边白 41px）。
  // 所以单独量一次公式内容的左右边界，按它报宽度 —— 但**仅限整页只有这一个
  // 公式**的情况（见下面 onlyMath）。
  // ⚠️ 这条「按公式宽度出图」只对**整页就是一个公式**成立（单公式块出图）。
  // 整篇 Markdown 里出现一个 `$$…$$` 时，若也这么报，页面会按公式那几十像素
  // 去截图 —— 正文被**整片切掉**（实测：一篇 720px 的 Markdown 报成 W=61）。
  var allMath = document.querySelectorAll('.math-display');
  var onlyMath = allMath.length === 1 && mdEl.children.length === 1
                 && mdEl.children[0] === allMath[0];
  var mw = 0;
  var mnodes = onlyMath ? allMath : [];
  for (var q = 0; q < mnodes.length; q++) {
    var L = Infinity, R = -Infinity, kids = mnodes[q].querySelectorAll('*');
    for (var z = 0; z < kids.length; z++) {
      var kr = kids[z].getBoundingClientRect();
      if (kr.width === 0 && kr.height === 0) { continue; }
      if (kr.left < L) { L = kr.left; }
      if (kr.right > R) { R = kr.right; }
    }
    if (R > -Infinity && R - L > mw && R - L < MAXW) { mw = R - L; }
  }
  if (mw > 0) {
    w = Math.ceil(mw) + 12;
    // ⚠️ 量到公式本身更窄时，**必须**清掉外层的 wide：否则下面仍按「内容比视口宽」
    // 那条分支报出整页宽（实测：公式只要 200px，却因为容器被 `.math-display`
    // 撑到 744px 而报 W=744，右边留了 41px 空白）。
    wide = false;
  }

  var msg = 'H=' + Math.max(h + 24, 40);   // 再留一点余量
    if (wide) {
      // 内容真的比视口宽（超宽公式）：留足余量，宁可多留点白也不能切内容
      msg = 'W=' + Math.ceil(w + 24) + ' ' + msg;
    } else {
      // ⚠️ 内容**没**超宽时，报的就是「容器宽与内容边界的较大者」，**不再加余量**。
      // 原先无条件 `+12`/`+24`，那点余量在**裁切**那条路上看不见（右侧会被裁掉）；
      // 但 Markdown 模式不再裁切之后，它就变成图片右边多出来的一条白边 ——
      // 出图比配置的宽度还宽（实测配 720 出 756）。
      msg = 'W=' + Math.max(wrapW, boxW) + ' ' + msg;
    }
    document.title = msg;
    }

    function run() {
      var fonts = document.fonts;
      if (fonts && fonts.ready && fonts.ready.then) {
        fonts.ready.then(measureAndReport).catch(measureAndReport);
        // 兜底：字体接口挂住时也要出图
        setTimeout(function () {
          if (!document.title) { measureAndReport(); }
        }, 2500);
      } else {
        measureAndReport();
      }
    }
    if (document.readyState === 'complete') { run(); }
    else { window.addEventListener('load', run); }
  })();
</script>"""


#: 公式渲染用 KaTeX。**为什么不用本机那个 MathJax 2.x**：
#: `/usr/share/yelp/mathjax` 是给 yelp 用的老 2.x，接进来实测**渲染不出东西** ——
#: 公式节点被它处理掉、却没插入排版结果，出图只有标题（1.3KB 空白图）。
#: KaTeX 一个 CSS + 一个 JS 就够，实测排版正确。
#:
#: ⚠️ **必须优先用本插件自带的那份**（`katex/` 目录）。原先只挂 CDN，实测出现过
#: 「JS 回来了、CSS/字体没回来」的窗口：KaTeX 把公式转成了 HTML，但没有样式与字体，
#: 出图就是 `\frac{x^2}{16}+\frac{y^2}{4}=1` 这样**一串原样文字**
#: （用户截图报的就是这个，244×33 的图，逐像素吻合）。
#: 自带一份之后公式排版不再依赖网络；CDN 只作为兜底。
KATEX_VERSION = "0.16.9"
_CDN_BASE = f"https://cdn.jsdelivr.net/npm/katex@{KATEX_VERSION}/dist"
#: 插件包内自带的 KaTeX（`plugins/ai_chat/katex/`）
_KATEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "katex")


def _katex_local(name: str) -> Optional[str]:
    """自带资源的 `file://` URI；没有就返回 None。"""
    path = os.path.join(_KATEX_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return Path(path).as_uri()
    return None


def katex_assets() -> tuple[str, str]:
    """返回 `(css_url, js_url)` —— 优先自带，缺了才退回 CDN。"""
    css = _katex_local("katex.min.css") or f"{_CDN_BASE}/katex.min.css"
    js = _katex_local("katex.min.js") or f"{_CDN_BASE}/katex.min.js"
    return css, js


#: 兼容旧引用（页面模板里直接插值用）
KATEX_CSS = _katex_local("katex.min.css") or f"{_CDN_BASE}/katex.min.css"
KATEX_JS = _katex_local("katex.min.js") or f"{_CDN_BASE}/katex.min.js"

#: 启动时打一行日志说明用的是哪份 KaTeX。
#: **为什么要打**：公式渲染失败是「静默但可见」的 —— 出图里就是一行 LaTeX 源码，
#: 而后台没有任何报错。没有这行日志时，只能靠反推（踩过：查了两轮才发现
#: 服务用的是 CDN、而我本地用的是自带资源）。有它就能一眼看出服务吃的是哪份。
_LOGGER = logging.getLogger("napcat")
if KATEX_CSS.startswith("file://"):
    _LOGGER.info("[ai_chat] 公式排版使用自带 KaTeX：%s", _KATEX_DIR)
else:
    _LOGGER.warning("[ai_chat] 自带 KaTeX 不可用（缺 %s），公式排版将依赖 CDN —— "
                    "网络抖动时会退化成一行 LaTeX 源码", _KATEX_DIR)


def _math_css(width: int, pad: str = "20px 24px") -> str:
    """片段出图的样式 = 页面外壳 + `markdown_body_css()`（元素规则只有一份）。"""
    css = f"""
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; background: #ffffff; }}
    body {{
      font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{ width: {int(width)}px; padding: {pad}; }}
    /* 按内容宽度排（配合 margin:auto 居中）：短内容右边就不会留一大片空白 */
    .wrap.tight {{ width: fit-content; max-width: {int(width)}px; margin: 0 auto; }}
    .wrap.tight > * {{ margin-top: 0; margin-bottom: 0; }}
    .cap {{ font-size: 15px; color: #8a9099; margin-bottom: 10px; }}
    .math {{ font-size: 22px; color: #1f2329; overflow-x: auto;
             padding: 8px 2px; text-align: center; }}
    .raw {{
      font-family: "Noto Sans Mono", monospace; font-size: 16px;
      background: #f6f7f9; border: 1px solid #e6e8eb; border-radius: 8px;
      padding: 10px 12px; white-space: pre-wrap; word-break: break-word;
      display: none;
    }}
    /* 块级元素自带上下外边距（段落 10px、标题 20px），单块出图时会变成
       图片顶部/底部的一大段空白。块图只装**一个**片段，所以首尾外边距去掉，
       留白统一由 .wrap 的 padding 控制（目标是紧贴）。 */
    .wrap > *:first-child {{ margin-top: 0; }}
    .wrap > *:last-child {{ margin-bottom: 0; }}
""" + markdown_body_css()
    return f"{css}    {_MATH_FAILED_CSS}"


def _embed_tex(latex: str) -> str:
    """把 LaTeX 安全地塞进 `<script type="text/plain">`。

    ⚠️ **不能**用 `html.escape()`：LaTeX 里的 `&` 会被转成 `&amp;`，而
    `<script>` 的内容不被解析为 HTML 实体 —— KaTeX 拿到的是字面量 `&amp;`，
    渲染出来就是 `amp;b = 0`（实测：cases 环境全错）。
    这里只处理唯一真正危险的序列 `</script`（会提前闭合标签）。
    """
    return (str(latex or "")
            .replace("</script", "<\\/script")
            .replace("<!--", "<\\!--"))


#: 行内公式图的宽度（比整段图窄）：它是插在文字之间的，太宽会占满一行
INLINE_MATH_WIDTH = 560


def build_inline_math_html(latex: str, *, width: int = INLINE_MATH_WIDTH) -> str:
    """单个**行内**公式 → 紧凑的 HTML 文档（左对齐、窄边距、不居中）。

    与 `build_math_html`（整段公式，居中、带标题）的区别只在排版意图：
    这张图会被插在文字之间，所以要尽量少占地方、看起来像「一行公式」。
    """
    from html import escape
    safe = _embed_tex(latex)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX_CSS}">
<style>
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; height: auto; background: #ffffff; }}
  body {{ font-family: "Noto Sans CJK SC", sans-serif; }}
  .inline-math {{
    display: inline-block; width: auto; min-width: 1px;
    padding: 1px 2px; font-size: 20px; color: #1f2329;
    white-space: nowrap;
  }}
  .raw {{
    font-family: "Noto Sans Mono", monospace; font-size: 15px;
    padding: 2px 6px; background: #f6f7f9; border-radius: 4px;
    white-space: nowrap; display: none;
  }}
</style>
</head>
<body>
<div class="inline-math" id="m"></div>
<div class="raw" id="raw">{safe}</div>
<script type="text/plain" id="tex">{safe}</script>
<script src="{KATEX_JS}"></script>
<script>
  (function () {{
    function run() {{
      var tex = document.getElementById('tex').textContent;
      var target = document.getElementById('m');
      try {{
        if (typeof katex === 'undefined') {{ throw new Error('no katex'); }}
        katex.render(tex, target, {{displayMode: false, throwOnError: true}});
      }} catch (e) {{
        target.style.display = 'none';
        document.getElementById('raw').style.display = 'inline-block';
      }}
    }}
    // 等 load：外链脚本下载是异步的，早跑会拿到 undefined
    if (document.readyState === 'complete') {{ run(); }}
    else {{ window.addEventListener('load', run); }}
  }})();
</script>
<script>
  // 行内公式要**紧裁**：默认整页宽会把一张 12px 高的公式裁成 720px 宽的白条，
  // 插在文字之间就是一大片空白。所以这里把公式自身的外框尺寸写进 title。
  window.addEventListener('load', function () {{
    var el = document.getElementById('m');
    var raw = document.getElementById('raw');
    var box = (el && el.offsetWidth) ? el : raw;
    if (!box) {{ document.title = 'D=200x60'; return; }}
    // ⚠️ 宽要留足余量：`load` 触发时 KaTeX 的 web 字体可能**还没换上来**，
    // 量到的是回退字体的窄宽度，按它截图会把公式**右边切掉**（实测：
    // 一个正弦指数式公式的右括号会被裁掉）。
    // 页面宽度够宽不会影响观感，窄了却会丢内容 —— 所以宁可宽一点。
    var r = box.getBoundingClientRect();
    var w = Math.ceil(r.width * 1.25) + 32;
    var h = Math.ceil(r.height * 1.3) + 12;
    document.title = 'D=' + w + 'x' + h;
  }});
</script>
{_SIZE_SCRIPT}
</body></html>
"""


def build_math_html(latex: str, *, width: int = 720, caption: str = "") -> str:
    """一个公式 → 完整 HTML 文档。

    ⚠️ TeX 源码放在 `<script type="text/plain">` 里而不是当正文 ——
    LaTeX 里的 `<`、`&` 直接进 HTML 会被当成标签（`a < b` 会吃掉后面内容）。
    JS 再把它取出来交给 KaTeX。

    KaTeX 从 CDN 取；取不到时 JS 把公式**原样显示**（`.raw` 块）——
    宁可让用户看到 LaTeX 源码，也不能给一张空白图。
    """
    from html import escape
    safe = _embed_tex(latex)
    cap = f'<div class="cap">{escape(caption)}</div>' if caption else ""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="{KATEX_CSS}">
<style>{_math_css(width)}</style>
</head>
<body>
<div class="wrap">
{cap}
<div class="math" id="m"></div>
<div class="raw" id="raw">{safe}</div>
<script type="text/plain" id="tex">{safe}</script>
<script src="{KATEX_JS}"></script>
<script>
  (function () {{
    var tex = document.getElementById('tex').textContent;
    var target = document.getElementById('m');
    try {{
      if (typeof katex === 'undefined') {{ throw new Error('katex not loaded'); }}
      katex.render(tex, target, {{displayMode: true, throwOnError: true}});
    }} catch (e) {{
      // 渲染不出来（没网/CDN 挂了/LaTeX 有错）→ 显示源码，**不留空白**
      target.style.display = 'none';
      document.getElementById('raw').style.display = 'block';
    }}
  }})();
</script>
</div>
{_SIZE_SCRIPT}
</body></html>
"""


#: 各类型块出图时的宽度。标题单独出图时窄一点更像「小标签」，
#: 铺满整页宽会显得空；列表/代码/表格该铺满就铺满。
FRAGMENT_WIDTHS = {
    "heading": 640,
    "table": 720,
    "list": 720,
    "code": 900,
    "math": 720,
}


#: 各类块的**盒子留白**。默认值调得很小，目标是「图紧贴内容」。
#:
#: ⚠️ 但表格与 `\boxed` 这类**带边框**的块不能给 0：CSS 边框是画在盒子边界上的，
#: 0 留白会让边框正好压在图片最外一像素上，看着像被截断。给 6px 让边框完整。
#: 左右也统一走这个值 —— 出图后的右侧像素裁剪只负责去掉**多余**空白，
#: 不会再把已经收紧的留白吃掉。
FRAGMENT_PADDING = {
    "heading": "6px",
    "list": "6px",
    "code": "8px",
    "table": "6px",
    "math": "6px 10px",
}


def build_fragment_html(markdown_fragment: str, *, kind: str = "table",
                        width: int = 0, caption: str = "") -> str:
    """任意 Markdown 片段（表格 / 列表 / 代码围栏 / 标题 / 公式）→ 出图用 HTML。

    与 `build_math_html`（单个公式）并列：这一支走 `markdown_to_html`，
    所以**行内公式与行内代码都能一起渲染对**（表格单元格里的 `$x$` 一直靠它）。

    ⚠️ `kind="math"` 时内容**不是 Markdown 而是裸 LaTeX**（`find_rich_blocks`
    去掉了定界符），必须先补回 `$$` 再解析 —— 详见 `as_display_math`。
    """
    from html import escape
    src = as_display_math(markdown_fragment) if kind == "math" else markdown_fragment
    inner = markdown_to_html(src)
    width = int(width or FRAGMENT_WIDTHS.get(kind, 720))
    cap = f'<div class="cap">{escape(caption)}</div>' if caption else ""
    pad = FRAGMENT_PADDING.get(kind, FRAGMENT_PADDING["table"])
    # 表格要铺满整页宽；其余块按**内容宽度**排（`tight`）—— 固定 720 宽会让
    # 一段短列表右边空 100 多像素（实测），与「紧贴」相反。
    cls = "wrap tight" if kind != "table" else "wrap"
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_math_css(width, pad)}</style>{katex_head(inner, force=(kind == "math"))}</head>
<body>
<div class="{cls}">
{cap}
{inner}
</div>
{_SIZE_SCRIPT}
</body></html>
"""


def build_table_html(markdown_table: str, *, width: int = 720,
                     caption: str = "") -> str:
    """一个 Markdown 表格 → 完整 HTML 文档（复用 markdown_to_html 的表格转换）。"""
    from html import escape
    inner = markdown_to_html(markdown_table)
    cap = f'<div class="cap">{escape(caption)}</div>' if caption else ""
    # 表格单元格里常有行内公式（结果列尤其多），所以这里也要引 KaTeX
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{_math_css(width)}</style>{katex_head(inner)}</head>
<body>
<div class="wrap">
{cap}
{inner}
</div>
{_SIZE_SCRIPT}
</body></html>
"""
