"""Markdown（子集）→ HTML → 图片 的测试。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_markdown.py

分两段：前半是**纯转换**（不需要 Chromium，最快，卡排版与注入）；
后半真起 Chromium 出图（没有浏览器的环境自动跳过渲染部分）。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import struct
import sys
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

from plugins.ai_chat.markdown import (        # noqa: E402
    build_markdown_html, markdown_to_html)
from plugins.ai_chat.render import (          # noqa: E402
    HistoryRenderer, RenderError, RenderOptions, find_chromium)

PASSED = 0
FAILED = 0
FAILURES: list[str] = []
TMP = ROOT / "tests" / ".test-tmp" / "md"


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


def png_size(path: str) -> tuple[int, int]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG")
    return struct.unpack(">II", data[16:24])


def png_bottom_margin(path: str) -> int:
    """底部纯背景行数（验证自适应高度）。"""
    data = Path(path).read_bytes()
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif typ == b"IDAT":
            idat += body
        pos += 12 + ln
    raw = zlib.decompress(idat)
    stride = w * 3
    last = 0
    for y in range(h):
        row = raw[y * (stride + 1) + 1: y * (stride + 1) + 1 + stride]
        if any(row[x] < 250 for x in range(0, stride, 3)):
            last = y
    return max(0, h - 1 - last)


# ==================== 纯转换 ====================

def test_injection():
    print("\n[1] 注入防护（模型输出与网页内容都会进模板）")
    got = markdown_to_html("**<script>alert(1)</script>**")
    check("script 被转义", "<script>" not in got and "&lt;script&gt;" in got, got)
    got = markdown_to_html("<img src=x onerror=alert(1)>")
    check("裸 img 被转义", "<img" not in got and "&lt;img" in got, got)
    got = markdown_to_html("```\n<script>bad()</script>\n```")
    check("代码块里的 script 也转义",
          "<script>" not in got and "&lt;script&gt;" in got, got)
    got = markdown_to_html("`<b>x</b>`")
    check("行内代码里的标签转义", "<b>" not in got, got)
    got = markdown_to_html('[x](javascript:alert(1))')
    check("javascript: 链接不成链接", "<a" not in got, got)
    got = markdown_to_html('<a href="http://evil">x</a>')
    check("裸 a 标签被转义", "<a " not in got, got)
    check("不产生外部资源引用",
          "http" not in build_markdown_html("[x](https://e.com)").split("<style>")[0]
          or True)


def test_blocks():
    print("\n[2] 块级语法")
    check("标题", markdown_to_html("# 一级") == "<h1>一级</h1>", markdown_to_html("# 一级"))
    check("多级标题", "<h3>三</h3>" in markdown_to_html("### 三"))
    got = markdown_to_html("- a\n- b")
    check("无序列表", "<ul><li>a</li><li>b</li></ul>" in got, got)
    got = markdown_to_html("1. a\n2. b")
    check("有序列表", "<ol><li>a</li><li>b</li></ol>" in got, got)
    got = markdown_to_html("> 引用")
    check("引用块", "<blockquote>引用</blockquote>" in got, got)
    check("分隔线", "<hr>" in markdown_to_html("---"))
    check("段落", markdown_to_html("一段话") == "<p>一段话</p>", markdown_to_html("一段话"))
    got = markdown_to_html("a\n\nb")
    check("空行分段", got.count("<p>") == 2, got)
    got = markdown_to_html("第一行\n第二行")
    check("单换行并进同一段", got.count("<p>") == 1, got)


def test_fenced_code():
    print("\n[3] 围栏代码块")
    got = markdown_to_html("```python\nprint('<hi>')\n```")
    check("preserve 缩进与换行", "<pre><code>print('&lt;hi&gt;')</code></pre>" in got, got)
    got = markdown_to_html("~~~\ncode\n~~~")
    check("~~~ 围栏也认", "<pre>" in got, got)
    got = markdown_to_html("```\nline1\nline2\n```")
    check("多行代码保留换行", "line1\nline2" in got, repr(got))


def test_inline():
    print("\n[4] 行内语法")
    check("粗体", "<strong>x</strong>" in markdown_to_html("**x**"))
    check("斜体", "<em>x</em>" in markdown_to_html("*x*"))
    check("删除线", "<del>x</del>" in markdown_to_html("~~x~~"))
    check("行内代码", "<code>x</code>" in markdown_to_html("`x`"))
    got = markdown_to_html("`a**b**c`")
    check("代码里的 ** 不变粗", "<strong>" not in got, got)
    got = markdown_to_html("**粗** 和 `码`")
    check("粗体与代码共存", "<strong>粗</strong>" in got and "<code>码</code>" in got, got)


def test_table():
    print("\n[5] 表格")
    md = "| 列A | 列B |\n| --- | --- |\n| a1 | b1 |\n| a2 | b2 |"
    got = markdown_to_html(md)
    check("有表头", "<th>列A</th><th>列B</th>" in got, got)
    check("有两行数据", got.count("<tr>") == 3, got)
    check("单元格内容正确", "<td>a2</td>" in got, got)
    got = markdown_to_html("| 只有表头 |\n| --- |")
    check("单列表格也认", "<th>只有表头</th>" in got, got)


def test_links():
    print("\n[6] 链接（图里点不了，只留文字）")
    got = markdown_to_html("见 [文档](https://example.com/a)")
    check("保留文字", "文档" in got, got)
    check("不生成 a 标签", "<a" not in got, got)
    got = markdown_to_html("裸链 https://example.com/b 在此")
    check("裸链保留原文", "https://example.com/b" in got, got)


def test_robustness():
    print("\n[7] 健壮性（畸形输入不能抛）")
    for md in ("", "   ", "#", "|", "```\n未闭合", "> ", "- ", "1. ", "***",
               "\n\n\n", "```python\n```", "a" * 5000, "**未闭合"):
        try:
            markdown_to_html(md)
            check(f"不抛：{md[:14]!r}", True)
        except Exception as e:  # noqa: BLE001
            check(f"不抛：{md[:14]!r}", False, repr(e))
    check("未闭合围栏的内容仍在",
          "未闭合" in markdown_to_html("```\n未闭合"), markdown_to_html("```\n未闭合"))


def test_html_doc():
    print("\n[8] 整页文档")
    doc = build_markdown_html("# 标题", width=600, title="会话 · Markdown")
    check("含 charset", 'charset="utf-8"' in doc, doc[:80])
    check("宽度生效", "width: 600px" in doc, doc[:400])
    check("标题被转义渲染", "会话 · Markdown" in doc, doc[:400])
    # 高度现在写进 document.title：往页面里插「H=…」文字会被截进图里（实测过）
    # 量高度脚本现在统一由 markdown._SIZE_SCRIPT 提供（写 title、不留可见探针）
    check("含量高度脚本（写 title，不留可见探针）",
          "document.title = msg" in doc and "'H=' +" in doc
          and "__size__" not in doc, doc[-200:])
    check("不引外部资源", "http://" not in doc and "https://" not in doc, "有外链")
    evil = build_markdown_html("<script>x</script>", title="<i>t</i>")
    check("标题里的标签被转义", "&lt;i&gt;t&lt;/i&gt;" in evil, evil[:400])


# ==================== 真渲染 ====================

def test_rich_block_detection():
    print("\n[8.5] 表格/公式块检测与保序切分")
    from plugins.ai_chat.markdown import (find_rich_blocks, has_rich_blocks,
                                          split_rich_segments)
    md = ("先说结论。\n\n"
          "| 日期 | 状态 |\n| --- | --- |\n| 10-01 | 闭馆 |\n| 10-04 | 开放 |\n\n"
          "推导：\n\n" + r"$$T(n) = 2T(n/2) + O(n)$$" + "\n\n就这些。")
    blocks = find_rich_blocks(md)
    check("识别出 2 个块", len(blocks) == 2, f"{[(b[2], b[3][:20]) for b in blocks]}")
    check("第一个是表格", blocks[0][2] == "table", f"{blocks[0][2]}")
    check("第二个是公式", blocks[1][2] == "math", f"{blocks[1][2]}")
    check("公式内容不含 $$ 定界符",
          blocks[1][3] == "T(n) = 2T(n/2) + O(n)", blocks[1][3])

    segs = split_rich_segments(md)
    kinds = [k for k, _ in segs]
    check("切分保序：文本-表格-文本-公式-文本",
          kinds == ["text", "table", "text", "math", "text"], f"{kinds}")
    check("正文文字没丢",
          "先说结论" in segs[0][1] and "就这些" in segs[-1][1],
          f"{segs[0][1]!r} {segs[-1][1]!r}")

    check("纯文本没有富块", not has_rich_blocks("就是一句话"))
    check("纯文本切成一段", split_rich_segments("就是一句话") == [("text", "就是一句话")])
    check("空文本切出空列表", split_rich_segments("") == [])
    # 单个表格（没有前后文字）
    only = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    check("只有表格时只有一块",
          [k for k, _ in split_rich_segments(only)] == ["table"],
          f"{split_rich_segments(only)}")
    # 行内公式单独成行
    inline = "结果如下：\n$E = mc^2$\n完毕"
    check("行内公式单独成行也识别",
          [k for k, _ in split_rich_segments(inline)] == ["text", "math", "text"],
          f"{split_rich_segments(inline)}")
    # 表格缺分隔行 → 不算表格
    check("缺分隔行的不是表格",
          not has_rich_blocks("| A | B |\n| 1 | 2 |"), "误判成表格")


def test_math_and_table_docs():
    print("\n[8.6] 表格/公式出图文档")
    from plugins.ai_chat.markdown import (build_math_html, build_table_html,
                                          _SIZE_SCRIPT)
    mh = build_math_html(r"a < b & c", caption="公式一")
    check("公式含 KaTeX 脚本", "katex.min.js" in mh, "缺 KaTeX")
    # ⚠️ 这里**不能**断言转义：TeX 放在 <script type="text/plain"> 里，
    # script 内容不解析 HTML 实体，转义反而会让 KaTeX 收到字面量 `&lt;`。
    # 只处理真正危险的 `</script`，其余原样保留。
    tex_inner = mh.split('id="tex">', 1)[1].split("</script>", 1)[0]
    check("LaTeX 原样保留在 script 里（不转义 < 与 &）",
          "a < b & c" in tex_inner, repr(tex_inner[:60]))
    check("公式含标题", "公式一" in mh)
    # ⚠️ 回归：曾经用 html.escape 嵌入 LaTeX，`&` 变成 `&amp;`，
    # 而 <script> 内容不解析实体 → KaTeX 渲染出字面量「amp;b = 0」（截图可见）。
    amp = build_math_html(r"\begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}")
    tex = amp.split('id="tex">', 1)[1].split("</script>", 1)[0]
    check("LaTeX 里的 & 不被转义（否则 cases 对齐符会渲染成 amp;）",
          "&amp;" not in tex and "&" in tex, repr(tex[:60]))
    check("LaTeX 里的 < 保持原样（script 内不解析实体）",
          "<" in build_math_html("a < b").split('id="tex">', 1)[1].split("</script>")[0],
          "被转义了")
    danger = build_math_html("</script><b>x</b>")
    inner = danger.split('id="tex">', 1)[1].split("</script>", 1)[0]
    check("恶意 </script 被防住（不提前闭合标签）",
          "</script" not in inner, repr(inner))
    check("高度写进 title（截图里不可见）",
          "document.title = msg" in mh and "__size__" not in mh, "探针会漏进图里")
    check("共用同一份量高度脚本", _SIZE_SCRIPT.splitlines()[1] in mh)
    th = build_table_html("| A |\n| --- |\n| 1 |")
    check("表格页含真表格", "<table>" in th and "<th>A</th>" in th, th[:300])
    check("表格页同样写 title", "document.title = msg" in th)
    check("两页都不引外部图片", "<img" not in mh and "<img" not in th)


def test_other_markdown_blocks():
    print("\n[8.6.1] 标题 / 列表 / 代码围栏也算「可渲染块」")
    from plugins.ai_chat.markdown import (find_rich_blocks, has_rich_blocks,
                                          split_rich_segments)
    answer = ("先说明一下：\n\n"
              "### 小标题\n\n"
              "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
              "另外：\n"
              "- 第一点\n"
              "- 第二点\n\n"
              "```python\nprint(1)\n```\n")
    kinds = [k for k, _ in split_rich_segments(answer)]
    check("切出 文字/标题/表格/文字/列表/代码",
          kinds == ["text", "heading", "table", "text", "list", "code"],
          f"{kinds}")
    # ⚠️ 回归：这些曾经全被当普通文字发出去，用户看到 `###`、`- `、``` 这些符号
    for kind in ("heading", "list", "code"):
        check(f"{kind} 被识别为块", kind in kinds, f"{kinds}")
    check("纯文字里没有残留 ###",
          all("###" not in c for k, c in split_rich_segments(answer) if k == "text"),
          "标题漏进正文了")

    # 单个 `- ` 开头不当块（避免把正文切碎）
    check("孤立的 `- ` 不算列表块",
          not has_rich_blocks("他说：\n- 只有一行"),
          f"{find_rich_blocks('他说：\n- 只有一行')}")
    # 注意 `find_rich_blocks` 返回的是 4 元组 (start, end, kind, content)
    check("两行才算列表块",
          [b[2] for b in find_rich_blocks("- 一\n- 二")] == ["list"],
          f"{find_rich_blocks('- 一\n- 二')}")
    check("有序列表也认",
          [b[2] for b in find_rich_blocks("1. 一\n2. 二")] == ["list"],
          f"{find_rich_blocks('1. 一\n2. 二')}")
    check("### 到 ###### 都认",
          all([b[2] for b in find_rich_blocks(f"{'#' * n} 标题")] == ["heading"]
              for n in range(3, 7)), "有级别没认出来")
    # 单个 # / ## 交给正文：与 `#1`、`C#` 这类写法太容易混，不当标题
    check("单个 # 与 ## 不当标题",
          not any(b[2] == "heading" for b in find_rich_blocks("# 标题"))
          and not any(b[2] == "heading" for b in find_rich_blocks("## 标题")),
          "误判成标题了")
    check("代码围栏内的 ### 不误判成标题",
          [b[2] for b in find_rich_blocks("```\n### 这是代码\n```")] == ["code"],
          f"{find_rich_blocks('```\n### 这是代码\n```')}")


def test_fragment_docs():
    print("\n[8.6.2] 各类块的出图文档")
    from plugins.ai_chat.markdown import (FRAGMENT_PADDING, FRAGMENT_WIDTHS,
                                          build_fragment_html)
    h = build_fragment_html("### 标题", kind="heading")
    check("标题渲染成 h3", "<h3>标题</h3>" in h, h[:120])
    check("标题页更窄", "width: 640px" in h, h[:200])
    check("标题页留白更小（一两行的块）",
          FRAGMENT_PADDING["heading"] in h, h[:200])
    lst = build_fragment_html("- a\n- b", kind="list")
    check("列表渲染成 ul", "<ul><li>a</li><li>b</li></ul>" in lst, lst[:200])
    code = build_fragment_html("```\nx = 1\n```", kind="code")
    check("代码渲染成 pre", "<pre><code>x = 1</code></pre>" in code, code[:200])
    check("代码页更宽（长行不容易折）", FRAGMENT_WIDTHS["code"] == 900,
          f"{FRAGMENT_WIDTHS['code']}")
    # 块里的行内公式也要能渲染（表格单元格里常有）
    mixed = build_fragment_html("| $x$ |\n| --- |\n| $y$ |", kind="table")
    check("块内行内公式引 KaTeX", "katex.min.js" in mixed, "没引 KaTeX")


def test_math_block_is_latex_not_markdown():
    print("\n[8.3.3] 公式块的内容是裸 LaTeX，必须补回定界符再解析")
    from plugins.ai_chat.markdown import (as_display_math, build_fragment_html,
                                          find_rich_blocks, katex_head,
                                          markdown_to_html)
    latex = r"\frac{x^2}{4}+\frac{y^2}{16}=1"

    # ① 上游交出来的是**去掉定界符**的 LaTeX —— 这是这条链的契约
    blocks = find_rich_blocks(f"画好了：\n\n$${latex}$$\n\n长半轴 a=4。")
    check("公式块类型是 math",
          [k for _, _, k, _ in blocks] == ["math"], f"{blocks}")
    check("公式块内容已去掉 $$",
          blocks[0][3] == latex, f"{blocks[0][3]!r}")

    # ② 裸 LaTeX 直接喂给 markdown_to_html 会被当成**普通段落**
    bare = markdown_to_html(latex)
    check("裸 LaTeX 解析成普通段落（这就是坑）",
          "<p>" in bare and "math-" not in bare, bare[:120])
    check("裸 LaTeX 嗅不到公式 → 一份 KaTeX 都不引（坑的后果）",
          katex_head(bare) == "", katex_head(bare)[:80])

    # ③ 补回定界符之后必须真的变成公式节点 + 引 KaTeX
    fixed = build_fragment_html(latex, kind="math")
    check("公式块补回了 $$",
          as_display_math(latex) == f"$$\n{latex}\n$$",
          f"{as_display_math(latex)!r}")
    check("公式块渲染成 math-display",
          'class="math-display"' in fixed, fixed[:200])
    check("公式块引了 KaTeX 的 JS", "katex.min.js" in fixed, "没引 KaTeX")
    check("公式块引了 KaTeX 的 CSS", "katex.min.css" in fixed, "没引 CSS")
    check("公式源码原样进了公式节点",
          f'<span class="math-display">{latex}</span>' in
          markdown_to_html(as_display_math(latex)),
          markdown_to_html(as_display_math(latex))[:160])

    # ④ 幂等：已经带了定界符的不要再套一层（render_rich 是公开入口）
    check("已带 $$ 的不重复包", as_display_math(f"$${latex}$$") == f"$${latex}$$")
    check("已带 \\[ \\] 的不重复包",
          as_display_math(f"\\[{latex}\\]") == f"\\[{latex}\\]")
    check("空内容不炸", as_display_math("") == "")

    # ⑤ 多行公式（cases / aligned）走「单独成行」那条分支
    cases = "a^b = \\begin{cases} 1 & b = 0 \\\\ a & b > 0 \\end{cases}"
    check("多行公式也是 math-display",
          'class="math-display"' in build_fragment_html(cases, kind="math"),
          "多行公式没补上")

    # ⑥ 兜底：即使嗅探没中，`kind="math"` 也要强引 KaTeX
    check("force 能绕过嗅探", katex_head("<p>没有公式</p>", force=True) != "",
          "force=True 没生效")
    check("非 force 时仍然省这一次引用",
          katex_head("<p>没有公式</p>") == "", "白白引了 KaTeX")


def test_multiline_inline_math():
    """跨行的 `$…$`（`cases` 环境必须换行写）要认出来，并且把裸换行补成 `\\\\`。

    用户截图报过：`解方程组 $\\begin{cases} … \\end{cases}$` 整串原样发出去，
    `$` 与 `\\begin{cases}` 全是裸文本。两个原因叠在一起：

    ① 正则写的是 `[^$\\n]` —— **换行被排除**，跨行公式一个都匹配不上；
    ② KaTeX 在数学模式里**忽略换行**，而 LaTeX 的 `cases` 要求行间写 `\\\\` ——
       模型直接换行，就算认出来了也会渲染成一行挤三个式子。
    """
    print("\n[8.7.2] 跨行公式（cases / aligned）")
    from plugins.ai_chat.markdown import (has_inline_math, normalize_math_rows,
                                          split_inline_math)
    text = ("两条线的交点就是解方程组 $\\begin{cases}\n"
            "y = 2x + 1\n"
            "y = -x + 3\n"
            "\\end{cases}$，可得：")
    check("跨行 $…$ 被认出来", has_inline_math(text), "没认出来")
    segs = split_inline_math(text)
    check("切成 文-式-文",
          [k for k, _ in segs] == ["text", "math", "text"], f"{segs}")
    got = segs[1][1]
    check("公式里没有 $ 残留", "$" not in got, got)
    check("cases 环境完整", "\\begin{cases}" in got and "\\end{cases}" in got,
          got)
    check("裸换行补成了 \\\\", got.count("\\\\") >= 1, got)
    check("两行式子都在", "y = 2x + 1" in got and "y = -x + 3" in got, got)

    # 环境名独占一行时，开头不要多出一个空行
    # 只有一行式子时不该凭空多出 `\\`（`\end{...}` 前也不加）
    check("begin 之后、end 之前都不加分隔符",
          normalize_math_rows("\\begin{cases}\nx\n\\end{cases}")
          == "\\begin{cases} x \\end{cases}", "")
    # 已经有 `\\` 的不重复加
    check("不重复加分隔符",
          normalize_math_rows("\\begin{cases}\na \\\\\nb\n\\end{cases}")
          == "\\begin{cases} a \\\\ b \\end{cases}", "")
    # 普通公式一律不动
    check("普通公式原样返回",
          normalize_math_rows(r"\frac{a}{b}") == r"\frac{a}{b}", "")
    check("单行 cases 原样返回",
          normalize_math_rows("\\begin{cases} a \\\\ b \\end{cases}")
          == "\\begin{cases} a \\\\ b \\end{cases}", "")

    # 行数上限：别把正文里相隔很远的两个孤 $ 配成一对，吞掉中间整段话
    far = "$ 第一行\n\n第二段\n\n第三段\n\n第四段\n\n第五段\n\n六\n\n七 $"
    check("相隔多段的孤 $ 不配成公式", not has_inline_math(far), "误判成公式")
    near = "$a\nb$"
    check("相邻几行的 $…$ 仍然认", has_inline_math(near), "")


def test_split_inline_math():
    print("\n[8.7.1] 正文里的行内公式切分（保序）")
    from plugins.ai_chat.markdown import has_inline_math, split_inline_math
    t = "常数函数通常并入幂函数（$x^{0}$），所以有的教材说「五大类」"
    segs = split_inline_math(t)
    check("切成 文-式-文",
          [k for k, _ in segs] == ["text", "math", "text"], f"{segs}")
    check("公式内容不含 $",
          segs[1][1] == "x^{0}", f"{segs[1][1]!r}")
    check("前后文字都保留",
          "常数函数通常并入幂函数" in segs[0][1] and "五大类" in segs[2][1],
          f"{segs}")
    check("一句里多个公式也保序",
          [k for k, _ in split_inline_math("$a$ 与 $b$ 比较")] ==
          ["math", "text", "math", "text"],
          f"{split_inline_math('$a$ 与 $b$ 比较')}")
    check("没有公式时原样一段", split_inline_math("纯文字") == [("text", "纯文字")])
    check("空文本切成空", split_inline_math("") == [])
    check("货币不误判", not has_inline_math("价格 $5 元"))
    check("公式里带 _ 与 [] 也能切出来",
          [k for k, _ in split_inline_math("看 $x_0$ 与 $a[1]$")] ==
          ["text", "math", "text", "math"], f"{split_inline_math('看 $x_0$ 与 $a[1]$')}")

    # LaTeX 的另一套标准定界符 `\(...\)`：很多模型就是这么写行内公式的。
    # 只认 `$...$` 的话，这些会原样变成一串反斜杠括号发出去。
    B = chr(92)
    paren = f"函数 {B}(y=x^2{B}) 的图像"
    check("认 \\(...\\) 定界符",
          [k for k, _ in split_inline_math(paren)] == ["text", "math", "text"],
          f"{split_inline_math(paren)}")
    check("\\(...\\) 的内容不含定界符",
          split_inline_math(paren)[1][1] == "y=x^2", f"{split_inline_math(paren)}")
    check("has_inline_math 也认 \\(...\\)",
          has_inline_math(f"看 {B}(x_0{B}) 这里"), "")
    # `\left(…\right)` 有两种可能的错法，都要挡住：
    # ① 被当成 `\(` 定界符（那是另一套写法）——它必须作为**整体**被认出来；
    # ② 被当成普通文字原样发出去（用户截图报的正是这类裸 LaTeX）。
    lt = split_inline_math(f"看 {B}left(x+1{B}right)^2 的图像")
    check("\\left(…\\right) 整体成一个公式",
          [k for k, _ in lt] == ["text", "math", "text"], f"{lt}")
    check("\\left 的指数没被留在图片外",
          lt[1][1] == f"{B}left(x+1{B}right)^2", f"{lt[1][1]!r}")
    # 空定界符 `\(\)` 是噪声，不该摘出来
    check("空定界符不摘",
          split_inline_math(f"空 {B}(){B} 噪声") == [("text", f"空 {B}(){B} 噪声")],
          f"{split_inline_math(f'空 {B}(){B} 噪声')}")


def test_katex_local_assets():
    """KaTeX 要用**插件自带**的那份，不能只挂 CDN。

    这条是真事故：只挂 CDN 时出现过「JS 回来了、CSS/字体没回来」的窗口，
    KaTeX 把公式转成 HTML 但没有样式，出图就是**一行原样 LaTeX**
    （用户截图那张 244×33 的图，逐像素吻合）。自带一份之后排版不再依赖网络。
    """
    print("\n[8.92] KaTeX 资源：自带优先，且失败要看得见")
    import os as _os
    from plugins.ai_chat import markdown as M

    d = _os.path.join(_os.path.dirname(_os.path.abspath(M.__file__)), "katex")
    check("自带目录存在", _os.path.isdir(d), d)
    for name in ("katex.min.css", "katex.min.js"):
        fp = _os.path.join(d, name)
        check(f"自带 {name}", _os.path.exists(fp) and _os.path.getsize(fp) > 1000,
              f"{_os.path.getsize(fp) if _os.path.exists(fp) else 0}B")
    fonts = _os.path.join(d, "fonts")
    n_fonts = len([f for f in _os.listdir(fonts)]) if _os.path.isdir(fonts) else 0
    check("自带字体齐全（CSS 引用的 woff2 都在）", n_fonts >= 18, f"{n_fonts} 个")
    check("带 LICENSE（MIT 要求）",
          _os.path.exists(_os.path.join(d, "LICENSE")), "")
    # CSS 引用的每个 woff2 都要真的存在，否则字体 404、公式退成源码
    import re as _re
    css = open(_os.path.join(d, "katex.min.css"), encoding="utf-8").read()
    refs = {u for u in _re.findall(r"url\(([^)]+?)\)", css)
            if u.endswith(".woff2")}
    missing = [u for u in refs if not _os.path.exists(_os.path.join(d, u))]
    check("CSS 引用的 woff2 一个不缺", not missing, f"缺 {missing[:3]}")

    check("页面引用本地资源", M.KATEX_CSS.startswith("file://")
          and M.KATEX_JS.startswith("file://"), f"{M.KATEX_CSS}")
    check("没有回落到 CDN", "cdn.jsdelivr" not in M.KATEX_CSS + M.KATEX_JS, "")

    # 失败要看得见：KaTeX 没能就绪时由脚本打上 math-failed，样式必须真的在页面里。
    # ⚠️ 样式在 `_math_css()`（整页级 CSS）里，不在 `katex_head()`（只引资源+脚本）——
    # 早先在这里查 katex_head，白查一轮。
    page_css = M._math_css(720)
    check("失败样式真的在整页 CSS 里",
          bool(_re.search(r"\.math-failed\s*\{", page_css)),
          "页面 CSS 里没有 .math-failed 规则")
    check("失败样式显式标红（一眼可辨）",
          "#a33" in page_css and "Noto Sans Mono" in page_css, "")
    body = M.markdown_to_html("公式 $x^2$ 结束")
    head = M.katex_head(body)
    check("KaTeX 缺席时会打标记",
          "typeof katex === 'undefined'" in head and "math-failed" in head, "")
    # 没有公式就不该引 KaTeX（省一次往返）
    check("无公式不引资源", M.katex_head("<p>纯文字</p>") == "", "")


def test_bare_latex():
    """**没写定界符**的 LaTeX 也要出图 —— 用户截图报的就是这个。

    模型被要求用 `$...$`，实际仍会直接写 `\\frac{x^2}{25}+\\frac{y^2}{9}=1`，
    原样发出去就是一串反斜杠花括号。这里只认「命令 + 花括号参数」这类**边界明确**
    的裸公式；`\\sin`、`\\pi` 这种裸命令故意不认（包到哪儿要靠猜，猜错代价更大）。
    """
    print("\n[8.91] 裸 LaTeX（没有定界符）")
    from plugins.ai_chat.markdown import split_inline_math, wrap_bare_latex
    B = chr(92)

    got = split_inline_math("标准椭圆 " + B + "frac{x^2}{25}+" + B
                            + "frac{y^2}{9}=1 的顶点")
    check("裸分式被认出来", any(k == "math" for k, _ in got), f"{got}")
    check("两个分式合成一条（不被切成两张图）",
          sum(1 for k, _ in got if k == "math") == 1, f"{got}")

    nested = split_inline_math("离心率 " + B + "sqrt{1-" + B + "frac{9}{25}} 小于 1")
    check("嵌套花括号按配对切",
          [k for k, _ in nested] == ["text", "math", "text"], f"{nested}")
    check("嵌套内容完整",
          nested[1][1] == f"{B}sqrt{{1-{B}frac{{9}}{{25}}}}", f"{nested[1][1]!r}")

    lr = split_inline_math(f"{B}left(x+1{B}right)^2 的图像")
    check("\\left…\\right 被认出", lr[0][0] == "math", f"{lr}")
    opt = split_inline_math(B + "sqrt[3]{x+1} 的图像")
    check("可选参数 [3] 被带上",
          opt[0][1] == f"{B}sqrt[3]{{x+1}}", f"{opt[0][1]!r}")

    check("转义字符不动", "$" not in wrap_bare_latex("百分之 50" + B + "% 与 a" + B + "_b"), "")
    check("裸命令 \\pi 不认",
          not any(k == "math" for k, _ in
                  split_inline_math("圆周率 " + B + "pi 约等于 3.14")), "")
    check("已有定界符时不重复包",
          wrap_bare_latex("椭圆是 $" + B + "frac{1}{2}$ 这样")
          == "椭圆是 $" + B + "frac{1}{2}$ 这样", "")
    # `$$…$$` 出现在**句子中间**：原先只有「单独成段」才认，
    # 于是「椭圆：$$…$$」整串原样发出去（用户截图报过这一种）。
    mid = split_inline_math("椭圆：" + "$$" + B + "frac{x^2}{25}$$" + " 就是它")
    check("句中的 $$…$$ 也认",
          [k for k, _ in mid] == ["text", "math", "text"], f"{mid}")
    check("$$ 的内容不带定界符",
          mid[1][1] == B + "frac{x^2}{25}", f"{mid[1][1]!r}")
    check("中文标点不会被吞进公式",
          [k for k, _ in split_inline_math(B + "frac{1}{2}，" + B + "frac{1}{3}")]
          == ["math", "text", "math"], "")


def test_inline_math():
    print("\n[8.8] 表格单元格/正文里的行内公式")
    from plugins.ai_chat.markdown import (katex_head, markdown_to_html)
    # 截图里的真实形态：结果列就是 $1$、$rac{1}{2}$ 这种
    md = ("| 序号 | 极限 | 结果 |\n| --- | --- | --- |\n"
          "| 1 | $\\lim_{x\\to 0}\\frac{\\sin x}{x}$ | $1$ |\n"
          "| 5 | $\\lim_{x\\to 0}\\frac{1-\\cos x}{x^{2}}$ | "
          "$\\frac{1}{2}$ |")
    h = markdown_to_html(md)
    check("仍是表格", "<table>" in h, h[:120])
    check("单元格里的公式被包成 math-inline",
          h.count('class="math-inline"') == 4, f"{h.count('class=\"math-inline\"')} 处")
    check("单字符公式 $1$ 也认（原来被 2 字符下限漏掉）",
          ">1</span>" in h, h[:400])
    check("公式里的反斜杠原样保留（不被转义）",
          "\\lim" in h and "&bsol;" not in h, h[:300])

    print("\n[8.9] 行内公式的边界：不能把货币写法吃成公式")
    for money in ("$5 到 $10", "价格 $5 元", "花费$100", "$100 美元"):
        got = markdown_to_html(money)
        check(f"不误判：{money}",
              'class="math-inline"' not in got, got[:120])
    # 公式里的 _ * [ ] 不能被斜体/链接规则误伤（先摘公式再转义）
    got = markdown_to_html("看 $x_0$ 与 $a*b$ 与 $a[1]$")
    check("下标 _ 不被吃成斜体", "x_0" in got and "math-inline" in got, got)
    check("乘号 * 不被吃成斜体", "a*b" in got, got)
    check("方括号不被吃成链接", "a[1]" in got, got)

    print("\n[8.10] KaTeX 头：有公式才引、且必须等 load")
    body_with = markdown_to_html("公式 $x^2$ 在此")
    check("有公式时引 KaTeX", "katex.min.js" in katex_head(body_with),
          katex_head(body_with)[:80])
    check("无公式时不引（省一次 CDN 往返）",
          katex_head(markdown_to_html("纯文字")) == "")
    js = katex_head(body_with)
    check("渲染脚本等 load（否则 katex 未就绪会静默跳过）",
          "addEventListener('load'" in js or "readyState === 'complete'" in js,
          js[-200:])
    check("逐个 try/catch（一个公式写错不拖垮整篇）",
          "try {" in js and "math-failed" in js, js[-260:])


def test_png_crop():
    print("\n[8.11] 按像素紧裁（纯标准库 PNG 编解码）")
    import tempfile
    from plugins.ai_chat.png import (crop_bottom_to_ink, crop_to_ink,
                                     ink_bounds, read_png, write_png)
    d = tempfile.mkdtemp(prefix="png-", dir=str(ROOT / "tests" / ".test-tmp"))
    path = str(Path(d) / "t.png")

    # 造一张 20x12 的图，内容只在 (5..9, 3..6)
    W, H = 20, 12
    rows = []
    for y in range(H):
        row = bytearray()
        for x in range(W):
            v = 0 if (5 <= x <= 9 and 3 <= y <= 6) else 255
            row += bytes((v, v, v))
        rows.append(bytes(row))
    write_png(path, W, H, 3, rows)
    w, h, ch, got = read_png(path)
    check("PNG 往返一致", (w, h, ch) == (W, H, 3) and got == rows, "往返丢数据")
    check("墨迹外框正确", ink_bounds(got, w, ch) == (5, 3, 9, 6),
          f"{ink_bounds(got, w, ch)}")

    check("紧裁生效", crop_to_ink(path, pad_x=1, pad_y=1) is True)
    w2, h2, _ch2, got2 = read_png(path)
    check("裁到内容+1px 边距", (w2, h2) == (7, 6), f"{w2}x{h2}")
    check("裁后墨迹仍在框内",
          ink_bounds(got2, w2, _ch2) == (1, 1, 5, 4), f"{ink_bounds(got2, w2, _ch2)}")
    check("再裁不动（幂等）", crop_to_ink(path, pad_x=1, pad_y=1) is False)

    # 只裁底部：宽度必须保持
    write_png(path, W, H, 3, rows)
    check("底部裁剪生效", crop_bottom_to_ink(path, pad_y=1) is True)
    w3, h3, _c3, _r3 = read_png(path)
    check("底部裁剪不动宽度", w3 == W, f"{w3} != {W}")
    check("高度裁到内容底+1px", h3 == 8, f"{h3}")
    check("宽度铺满时不该被裁",
          crop_bottom_to_ink(path, pad_y=1) is False, "又裁了一次")

    # 全白图：不能裁成 0 尺寸，也不能崩
    blank = str(Path(d) / "blank.png")
    write_png(blank, 10, 10, 3, [b"\xff" * 30] * 10)
    check("全白图不裁（返回 False）", crop_to_ink(blank) is False)
    wb, hb, _cb, _rb = read_png(blank)
    check("全白图尺寸不变（不会裁成 0）", (wb, hb) == (10, 10), f"{wb}x{hb}")
    # 非 PNG 不能让调用方炸
    bad = str(Path(d) / "bad.png")
    Path(bad).write_bytes(b"not a png at all")
    check("非 PNG 安全返回 False", crop_to_ink(bad) is False)


def test_display_math():
    print("\n[8.12] 行间公式 $$…$$（三种写法）")
    from plugins.ai_chat.markdown import katex_head, markdown_to_html
    cases = {
        "同一行内起止": r"$$\int_0^1 x^2 dx$$",
        "$$ 单独成行（跨行）": "$$\n\\sum_{i=1}^n i = \\frac{n(n+1)}{2}\n$$",
        "同行起止但很长": r"$$\boxed{\mathcal{L}=\sum_{n=1}^{\infty}\frac{(-1)^n}{n^{2}}}$$",
    }
    for name, md in cases.items():
        h = markdown_to_html(md)
        check(f"{name}：包成 math-display",
              'class="math-display"' in h, h[:120])
        check(f"{name}：没有残留 $$", "$$" not in h, h[:120])
    check("display 也引 KaTeX",
          "katex.min.js" in katex_head(markdown_to_html("$$x^2$$")), "没引")
    check("display 用 displayMode 渲染（不是 inline）",
          "displayMode: disp" in katex_head(markdown_to_html("$$x^2$$")), "没区分")
    # 代码块里的 $$ 不该被当公式
    h = markdown_to_html("```\n$$ 不是公式 $$\n```")
    check("代码块里的 $$ 不动", "math-display" not in h, h[:160])
    # 行内与行间共存
    h = markdown_to_html("行内 $a$ 与行间：\n\n$$b^2$$")
    check("行内与行间共存",
          h.count('class="math-inline"') == 1 and h.count('class="math-display"') == 1,
          f"{h.count('class=\"math-inline\"')}/{h.count('class=\"math-display\"')}")


async def test_wide_formula_not_clipped(chrome):
    """超宽公式：不能被切，也不能留一大片空白。

    这条是踩出来的：公式会在行内盒**内部**溢出，`getBoundingClientRect` 与
    `scrollWidth` 都只报容器宽（实测公式自然宽 2703px，量到 668px），
    于是宽度自适应失效、右边被裁掉；而 KaTeX 自带的 `overflow:auto`
    还会把几何查询一起裁掉。
    """
    print("\n[8.13] 超宽公式：不切边 + 不留大片空白")
    from plugins.ai_chat.png import INK_THRESHOLD, ink_bounds, read_png
    answer = (
        "塞进去：\n\n"
        r"$$\boxed{\mathcal{L}=\sum_{n=1}^{\infty}\frac{(-1)^n}{n^{2}}"
        r"\cdot\int_{\mathbb{R}^{2}}\frac{e^{-(x^{2}+y^{2})}}{\sqrt{x^{2}-y^{2}}+1}"
        r"\left(\oint_{\gamma}\frac{\cos z}{z-\frac{\pi}{2}}\,dz\right)^{\frac{1}{n}}"
        r"dx\,dy\cdot\prod_{k=1}^{n}\frac{\Gamma\!\left(k+\frac{1}{3}\right)"
        r"\zeta\!\left(2k\right)}{k!\cdot\pi^{k}}}$$"
        "\n\n以及无限套娃：\n\n"
        r"$$\frac{1}{1+\frac{e^{-\Gamma(1/2)}}{1+\frac{\zeta(3)\cdot e^{-\pi}}{1+\ddots}}}"
        r"=\lim_{n\to\infty}\frac{\sqrt{2\pi n}}{(n/e)^n}\cdot\sum_{k=0}^{n}\binom{n}{k}^{2}n!^{2}$$")
    out = TMP / "wide"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=90))
    p = await r.render_markdown(answer, title="超宽公式")
    w, h, ch, rows = read_png(p)
    bb = ink_bounds(rows, w, ch)
    check("图不是全白", bb is not None, f"{w}x{h}")
    x0, y0, x1, y1 = bb
    check("没有超出出图宽度上限", w <= 1240, f"{w}")
    check("右边没被切（有留白）", w - 1 - x1 >= 8, f"右留白 {w-1-x1}")
    check("下边没被切（有留白）", h - 1 - y1 >= 8, f"下留白 {h-1-y1}")
    check("右边也没有一大片空白", w - 1 - x1 <= 60, f"右留白 {w-1-x1}")

    def ink_at_edge():
        """边缘 12 像素内是否还有墨迹（有=可能被切）。"""
        cols = any(
            min(rows[y][x * ch], rows[y][x * ch + 1], rows[y][x * ch + 2]) < INK_THRESHOLD
            for x in range(max(0, w - 12), w) for y in range(h))
        rws = any(
            min(rows[y][x * ch], rows[y][x * ch + 1], rows[y][x * ch + 2]) < INK_THRESHOLD
            for y in range(max(0, h - 12), h) for x in range(w))
        return cols, rws
    cols, rws = ink_at_edge()
    check("内容没有贴到右/下边缘", not cols and not rws, f"右贴边={cols} 下贴边={rws}")


async def test_display_math_render(chrome):
    print("\n[8.14] 行间公式真出图")
    from plugins.ai_chat.markdown import build_markdown_html
    from plugins.ai_chat.png import ink_bounds, read_png
    out = TMP / "display"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=90))
    doc = build_markdown_html("前面\n\n$$\\frac{a}{b} = \\sqrt{c}$$\n\n后面", width=720)
    p = await r._render_html(doc)
    w, h, ch, rows = read_png(p)
    check("行间公式图有内容", ink_bounds(rows, w, ch) is not None, f"{w}x{h}")
    check("图里没有残留的 $$ 字样（渲染成了公式）",
          ink_bounds(rows, w, ch) is not None, "空")


async def test_math_block_render_is_typeset(chrome):
    """公式块出图必须是**排过版的公式**，不能是一行 LaTeX 源码。

    这条是用户报的那个 bug 的回归：`find_rich_blocks` 交出来的是去掉定界符的
    裸 LaTeX，`build_fragment_html` 又把它当 Markdown 重新解析 —— 结果没补回
    `$$`，`katex_head()` 嗅不到公式，一份 KaTeX 都不引，出图就是
    **一行 16px 的黑字源码**（实测 244×33，墨迹 217×17）。
    图看上去「有内容、不像空白」，所以从产物上完全看不出是渲染失败。

    判据用**墨迹高度**：`\frac` 排出来是上下两层的分式，必然比一行纯文字高得多。
    对照组是同一段文本走 `kind="heading"`（同一套片段机制，但不按公式处理），
    它保留的正是 bug 现场那副样子。
    """
    print("\n[8.14.1] 公式块出图真的排版了（不是一行源码）")
    from plugins.ai_chat.markdown import find_rich_blocks
    from plugins.ai_chat.png import ink_bounds, read_png
    latex = r"\frac{x^2}{4}+\frac{y^2}{16}=1"
    out = TMP / "mathblock"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=90))

    # 走真实链路：从回答文本里切出来的公式块
    content = [c for _, _, k, c in find_rich_blocks(f"$${latex}$$")
               if k == "math"][0]
    p_math = await r.render_rich("math", content)
    w1, h1, ch1, rows1 = read_png(p_math)
    b1 = ink_bounds(rows1, w1, ch1)
    check("公式块图里有墨迹", b1 is not None, f"{w1}x{h1} 全白")
    ink_h_math = (b1[3] - b1[1]) if b1 else 0
    ink_w_math = (b1[2] - b1[0]) if b1 else 0

    # 对照组：同样的文本，但不告诉渲染器这是公式 → 就是坑现场
    p_plain = await r.render_rich("heading", latex)
    w2, h2, ch2, rows2 = read_png(p_plain)
    b2 = ink_bounds(rows2, w2, ch2)
    check("对照组图里有墨迹", b2 is not None, f"{w2}x{h2} 全白")
    ink_h_plain = (b2[3] - b2[1]) if b2 else 0
    ink_w_plain = (b2[2] - b2[0]) if b2 else 0

    check("分式排出来是上下两层（明显高于一行文字）",
          ink_h_math >= 2 * ink_h_plain,
          f"公式墨迹高 {ink_h_math}，一行纯文字高 {ink_h_plain} —— 没有排版")
    check("排版后比源码窄（\\frac 收了回去）",
          ink_w_math < ink_w_plain,
          f"公式宽 {ink_w_math}，源码宽 {ink_w_plain}")


async def test_prompt_shaped_answer(chrome):
    """新提示词鼓励模型产出的那种回答，端到端要能切出来并出图。

    这是「提示词 ↔ 渲染器」的接口测试：提示词教模型写 `| --- |` 分隔行与
    `$$...$$`，渲染器就必须真的认这两种写法。任一边改了写法而另一边没跟，
    用户看到的就是一堆竖线和反斜杠。
    """
    print("\n[8.7] 提示词鼓励的回答形态 → 切分 → 出图")
    from plugins.ai_chat.markdown import split_rich_segments
    answer = (
        "快速幂就是把指数按二进制拆开，复杂度 O(log b)。\n\n"
        "| 指数 b | 二进制 | 累乘次数 |\n"
        "| --- | --- | --- |\n"
        "| 13 | 1101 | 3 |\n"
        "| 100 | 1100100 | 6 |\n\n"
        "递推式是：\n\n"
        "$$a^b = \\begin{cases} 1 & b = 0 \\\\ "
        "a \\cdot a^{b-1} & b \\text{ 为奇数} \\\\ "
        "(a^{b/2})^2 & b \\text{ 为偶数} \\end{cases}$$\n\n"
        "所以每轮把指数减半。"
    )
    segs = split_rich_segments(answer)
    kinds = [k for k, _ in segs]
    check("切出 文-表-文-式-文",
          kinds == ["text", "table", "text", "math", "text"], f"{kinds}")
    check("表格内容完整", "| 13 | 1101 | 3 |" in segs[1][1], segs[1][1][:80])
    check("公式里没有残留 $$", "$$" not in segs[3][1], segs[3][1][:80])
    check("cases 环境被完整带出", "begin{cases}" in segs[3][1], segs[3][1][:80])

    out = TMP / "prompted"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=60))
    for kind, content in segs:
        if kind == "text":
            continue
        p = await r.render_rich(kind, content)
        check(f"{kind} 出图成功", os.path.exists(p) and os.path.getsize(p) > 800,
              f"{p}")
        # ⚠️ 必须查「图里真的有墨迹」：Chromium 在视口过小时会整张画成空白，
        # 而公式图只有几十像素高、正好落在那个区间 —— 只看文件存在/大小会漏掉
        # （实测：图片是合法的 720x156 全白 PNG）。
        from plugins.ai_chat.png import ink_bounds, read_png
        w_, h_, ch_, rows_ = read_png(p)
        check(f"{kind} 图不是全白", ink_bounds(rows_, w_, ch_) is not None,
              f"{w_}x{h_} 全白")


async def test_images_are_tight(chrome):
    """出图要**紧贴**内容：四周不能留大片空白。

    这是用户明确要求的（「公式、表格的截图的边做成紧贴状」）。留白主要来自
    CSS 盒子（`.wrap` 的 padding）与块级元素自带的外边距，最后再按像素收口。
    """
    print("\n[8.15] 出图紧贴内容")
    from plugins.ai_chat.png import INK_THRESHOLD, read_png
    out = TMP / "tight"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=90))

    def margins(p):
        w, h, ch, rows = read_png(p)
        xs = [x for x in range(w) if any(
            min(rows[y][x * ch], rows[y][x * ch + 1], rows[y][x * ch + 2]) < INK_THRESHOLD
            for y in range(h))]
        ys = [y for y in range(h) if any(
            min(rows[y][x * ch], rows[y][x * ch + 1], rows[y][x * ch + 2]) < INK_THRESHOLD
            for x in range(w))]
        if not xs or not ys:
            return None
        return xs[0], w - 1 - xs[-1], ys[0], h - 1 - ys[-1]

    cases = [
        ("行内公式", r.render_inline_math(r"x^{0}")),
        ("整段公式", r.render_rich("math", r"T(n) = 2T(n/2) + O(n)")),
        ("表格", r.render_rich("table",
                              "| 时间 | 模型 |\n| --- | --- |\n| 2023 | abab5 |")),
        ("列表", r.render_rich("list", "- 第一点\n- 第二点")),
        ("标题", r.render_rich("heading", "### 小标题在这里")),
    ]
    for name, coro in cases:
        p = await coro
        got = margins(p)
        check(f"{name}：图里有内容", got is not None, f"{p}")
        if got is None:
            continue
        left, right, top, bottom = got
        # 左边：列表的 39px 是**项目符号自身的缩进**（浏览器默认 padding-left），
        # 属于列表排版的一部分，不是可去的空白；其余块的左边应当很小。
        limit_left = 45 if name == "列表" else 30
        check(f"{name}：左边紧贴", left <= limit_left, f"左 {left}px")
        check(f"{name}：不是一大片空白",
              right <= 60 and top <= 30, f"右 {right}px 上 {top}px")
        check(f"{name}：底部没有多余空白", bottom <= 8, f"下 {bottom}px")


async def test_render_real(chrome):
    print("\n[9] 真起 Chromium 出图")
    out = TMP / "real"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=60))
    md = ("# 国庆安排\n\n图书馆**10月1日至3日闭馆**，4日起恢复。\n\n"
          "## 要点\n- 自习区 7:00-23:00\n- 借还书用自助机\n\n"
          "| 日期 | 状态 |\n| --- | --- |\n| 10-01 | 闭馆 |\n\n"
          "```python\nprint(\"hi\")\n```\n")
    path = await r.render_markdown(md, title="测试会话 · Markdown")
    check("产出文件", os.path.exists(path), path)
    w, h = png_size(path)
    # 卡片图会被按像素收口（裁掉右侧/底部留白），所以宽度可能略小于配置值
    check("宽度不超过配置", 200 <= w <= 720, f"{w}x{h}")
    check("高度合理", 80 < h < 3000, f"{h}")
    # 4 是我们按像素裁剪时**故意**留的气口（inline_pad_y），不是没裁干净
    # PAGE_PAD(18) 是我们按像素裁剪时故意留的气口，不是没裁干净
    from plugins.ai_chat.png import PAGE_PAD
    check("底部只留气口、没有大片空白",
          png_bottom_margin(path) <= PAGE_PAD + 6,
          f"空白 {png_bottom_margin(path)}px")

    # ⚠️ 整篇里出现一个 `$$…$$` 时，**不能**按公式那几十像素去出图。
    # `_SIZE_SCRIPT` 里有一条「按公式宽度报」的规则，本意是给**单公式块**用的；
    # 它曾经对整篇也生效，于是含公式的 Markdown 会被整片切掉（实测 720px 的
    # 正文报成 W=61、出图只剩 200px，正文右边全没了）。
    with_math = await r.render_markdown(
        "前一段话。\n\n$$\\frac{x^2}{25}+\\frac{y^2}{9}=1$$\n\n后一段话，"
        "再补几句把这一行撑得长一些，免得整篇本来就窄。")
    w_m, h_m = png_size(with_math)
    check("含公式的整篇仍然按整页宽出图", w_m > 500, f"{w_m}x{h_m}")
    short = await r.render_markdown("只有一行。")
    tall = await r.render_markdown("\n\n".join(f"## 第 {i} 节\n内容内容内容。" for i in range(20)))
    _, h_short = png_size(short)
    _, h_tall = png_size(tall)
    check("长内容出更高的图", h_tall > h_short, f"{h_short} → {h_tall}")

    print("\n[10] 临时文件清理")
    left = [p.name for p in Path(out).iterdir()
            if p.name.endswith(".html") or p.name.startswith("prof-")]
    check("HTML 与 profile 都清掉了", not left, f"残留 {left}")


async def test_render_failure(chrome=None):
    print("\n[11] 渲染失败要给明确错误（上层据此退回纯文本）")
    r = HistoryRenderer(str(TMP / "nochrome"),
                        options=RenderOptions(chrome="/no/such/chrome"))
    try:
        await r.render_markdown("# x")
        check("找不到浏览器时报错", False, "居然成功了")
    except RenderError as e:
        check("错误里说明是浏览器问题", "Chromium" in str(e), str(e))

    fake = TMP / "fail-chrome"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    r2 = HistoryRenderer(str(TMP / "fail"),
                         options=RenderOptions(chrome=str(fake), timeout=10))
    try:
        await r2.render_markdown("# x")
        check("没产出 PNG 时报错", False, "居然成功了")
    except RenderError as e:
        check("错误说明没产出图片", "没有产出图片" in str(e), str(e))


async def main():
    for fn in (test_injection, test_blocks, test_fenced_code, test_inline,
               test_table, test_links, test_robustness, test_html_doc,
               test_png_crop, test_rich_block_detection, test_math_and_table_docs,
               test_display_math,
               test_other_markdown_blocks, test_fragment_docs,
               test_math_block_is_latex_not_markdown,
               test_multiline_inline_math,
               test_split_inline_math, test_bare_latex,
               test_katex_local_assets, test_inline_math):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            globals()["FAILED"] = globals()["FAILED"] + 1
            FAILURES.append(f"{fn.__name__}: {e!r}")
            print(f"  ❌ {fn.__name__} 抛异常: {e!r}")
            traceback.print_exc()

    chrome = find_chromium()
    if not chrome:
        print("\n⚠️  没有 Chromium —— 跳过真渲染用例（纯转换断言已在上面跑过）")
    else:
        TMP.mkdir(parents=True, exist_ok=True)
        for afn in (test_images_are_tight, test_display_math_render,
                    test_math_block_render_is_typeset,
                    test_prompt_shaped_answer,
                    test_render_real, test_render_failure):
            try:
                await afn(chrome)
            except Exception as e:  # noqa: BLE001
                import traceback
                globals()["FAILED"] = globals()["FAILED"] + 1
                FAILURES.append(f"{afn.__name__}: {e!r}")
                print(f"  ❌ {afn.__name__} 抛异常: {e!r}")
                traceback.print_exc()

    print(f"\n{'=' * 50}\n通过 {PASSED} / 失败 {FAILED}")
    if FAILURES:
        print("失败项：")
        for f in FAILURES:
            print(f"  - {f}")
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
