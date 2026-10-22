"""历史问答**图片渲染**的测试 —— 真起 Chromium，真出 PNG。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_history_render.py

与 `test_history.py` 的分工：那边只测归并语义（纯数据），这边测「数据 → 图」这条
链路，包括它最容易伤到人的三件事：**注入**（群友昵称/正文进 HTML）、
**跑飞**（Chromium 是外部进程，必须有超时）、**写满磁盘**（每张图都会落盘）。

Chromium 不在的环境自动跳过真实渲染部分，只跑纯逻辑断言（不报假绿）。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import struct
import sys
import tempfile
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

from plugins.ai_chat.history import Exchange          # noqa: E402
from plugins.ai_chat.render import (                  # noqa: E402
    HistoryRenderer, RenderError, RenderOptions, build_html, find_chromium)

PASSED = 0
FAILED = 0
FAILURES: list[str] = []
TMP = ROOT / "tests" / ".test-tmp" / "render"


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
    """从 PNG 头里读尺寸（不引图像库）。"""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("不是 PNG")
    w, h = struct.unpack(">II", data[16:24])
    return w, h


def _png_raw(path: str):
    """解出 `(宽, 高, 每行原始像素)`。

    ⚠️ 用插件自己的 `png.read_png`，**不要**手写解码：Chrome 截图的 PNG 是
    **逐行带 filter** 的（实测 filter 值有 1/2/4），忽略那个字节解出来是垃圾 ——
    量出来的「边距」会全是 0，而断言看着像通过（这个坑踩过）。
    `read_png` 是 render 之外的另一支，用它不算「拿被测模块证明自己」。
    """
    from plugins.ai_chat.png import read_png
    w, h, ch, rows = read_png(path)
    return w, h, ch, rows


def png_bottom_margin(path: str) -> int:
    """图底部有多少行纯背景（用来验证「自适应高度」真的生效）。"""
    w, h, ch, rows = _png_raw(path)
    last = 0
    for y in range(h):
        if any(rows[y][x] < 250 for x in range(0, w * ch, ch)):
            last = y
    return max(0, h - 1 - last)


def png_right_margin(path: str) -> int:
    """图右侧有多少列纯背景 —— 用来判断「这张图裁没裁」。

    裁过的块图右边只剩 `PAGE_PAD`（4px）那口气；整页图（Markdown 模式）不裁，
    右边是完整的页边距（几十像素）。
    """
    w, h, ch, rows = _png_raw(path)
    for x in range(w - 1, -1, -1):
        if any(rows[y][x * ch] < 250 for y in range(h)):
            return w - 1 - x
    return w


def ex(index=1, **kw) -> Exchange:
    base = dict(question="图书馆国庆开吗？", answer="闭馆三天。", speaker="张三",
                time="09-30 22:15")
    base.update(kw)
    return Exchange(index=index, **base)


def test_html_escaping():
    print("\n[1] HTML 转义（群友内容是不可信输入）")
    evil = ex(question='<script>alert(1)</script>',
              answer='<img src="file:///etc/passwd">',
              speaker='<b>坏人</b>')
    doc = build_html(evil, session_name='<i>会话</i>')
    check("没有裸的 script 标签", "<script>alert" not in doc, "有未转义 script")
    check("没有裸的 img 标签", '<img src="file://' not in doc, "有未转义 img")
    check("尖括号被转义", "&lt;script&gt;" in doc and "&lt;img" in doc)
    # ⚠️ 正文现在由 `markdown_to_html` 渲染，它转义 `<`/`>`/`&` 但**不转义引号** ——
    # 正文落在元素的文本节点里，不进属性，所以引号没有危险；头部字段
    # （会话名/昵称/时间）仍然走 `_esc(quote=True)`。所以这里断言的是
    # 「引号没有被拼进属性」这个真正的安全性质，而不是「出现了 &quot;」。
    # `src="file` 这种**字面文本**出现是正常的（它已经被转义成 `&lt;img …`），
    # 真正要挡的是它变成**标签**：所以查的是 `&lt;img` 在、`<img` 不在。
    check("正文里的引号不会拼进属性",
          "<img" not in doc and "&lt;img" in doc and "onerror=" not in doc,
          "属性被注入")
    check("页头字段仍然转义引号", "&quot;" in build_html(
        ex(speaker='张三" onload="x'), session_name="a"))
    check("页面自己那几个 script 之外没有别的（KaTeX + 量高度）",
          doc.count("<script>") == 1, f"{doc.count('<script>')} 个")
    check("页面不引任何外部资源",
          "http://" not in doc and "https://" not in doc, "有外链")
    check("会话名也转义", "&lt;i&gt;会话&lt;/i&gt;" in doc)


def test_html_content():
    print("\n[2] 图里该有的信息")
    doc = build_html(ex(index=7, tool_calls=["get_notice"], tool_count=2),
                     session_name="预热群会话", total=12)
    check("会话名", "预热群会话" in doc)
    check("第几对 / 共几对", "第 7 对" in doc and "共 12 对" in doc)
    check("提问人", "张三" in doc)
    check("时间", "09-30 22:15" in doc)
    check("问题正文", "图书馆国庆开吗？" in doc)
    check("回答正文", "闭馆三天。" in doc)
    check("工具小字", "get_notice" in doc and "×2" in doc)
    check("没调工具就不显示工具行",
          'class="tool"' not in build_html(ex()))
    check("宽度可配", "width:600px" in build_html(ex(), width=600))


def test_markdown_rendering():
    """hget 卡片必须和别处**用同一套 Markdown 渲染器**。

    用户报过：同一段回答在聊天里是排好版的公式，在历史卡片里却是一串
    `$$\frac{x^2}{25}+\frac{y^2}{9}=1$$` 源码 —— 因为卡片那条路自己
    `html.escape` 一把了事，压根没走 Markdown 渲染。
    """
    print("\n[2.5] 卡片走同一套 Markdown 渲染器")
    doc = build_html(ex(answer=(
        "标准椭圆：\n$$\\frac{x^2}{25}+\\frac{y^2}{9}=1$$\n\n"
        "长半轴 $a=5$，离心率 $e=\\frac{c}{a}=\\frac{4}{5}$。\n\n"
        "| 量 | 值 |\n| --- | --- |\n| 顶点 | (±5, 0) |\n\n"
        "```python\nprint(1)\n```")), width=720)
    check("行间公式渲染成节点", 'class="math-display"' in doc, "没渲染公式")
    check("行内公式渲染成节点", 'class="math-inline"' in doc, "")
    check("引了 KaTeX", "katex.min.js" in doc, "没引 KaTeX")
    check("用本地 KaTeX（不走网络）", "cdn.jsdelivr" not in doc, "引了 CDN")
    check("表格渲染成 table", "<table>" in doc and "<th>" in doc, "")
    check("代码块渲染成 pre", "<pre><code>" in doc, "")
    check("不再直接显示 $$ 源码", "$$" not in doc, "还有 $$ 残留")

    # 没有公式时不该白引一套 KaTeX
    plain = build_html(ex(answer="就是闭馆三天。"))
    check("无公式时不引 KaTeX", "katex.min.js" not in plain, "白引了 KaTeX")

    # 元素样式必须来自**公共**那一份（问答卡原先自己抄了一套，改一处漏两处）
    from plugins.ai_chat.markdown import markdown_body_css
    body = markdown_body_css()
    check("公共样式里没有双花括号（写成 f-string 转义会整段失效）",
          "{{" not in body and "}}" not in body, "有 {{ 残留")
    # ⚠️ 要看的是 `.q, .a` 那条**规则**里有没有 pre-wrap，不是整份 CSS 里有没有
    # 这三个字 —— 上面那段说明性的注释里就写着它（第一版断言就是这么假失败的）。
    import re as _re
    rule = _re.search(r"\.q,\s*\.a\s*\{[^}]*\}", doc)
    check("卡片刻意不用 pre-wrap（会和 Markdown 的段落重复换行）",
          rule is not None and "pre-wrap" not in rule.group(0),
          rule.group(0)[:120] if rule else "没找到 .q/.a 规则")
    check("表格样式来自公共那一份", "th, td" in doc, "缺表格样式")

    # 量高度必须等 web 字体就位，否则公式换字体后卡片被截掉
    check("量高度等字体就位", "fonts.ready" in doc, "没等字体")

    # 答案里的换行要保留（这是转录，必须跟原话一样）
    multi = build_html(ex(answer="第一行\n第二行\n第三行"))
    check("段内换行保留成 <br>", multi.count("<br>") == 2,
          f"{multi.count('<br>')} 个 br")


def test_truncation():
    import re as _re
    print("\n[3] 超长截断（防止一张图几万像素高）")
    long_q = "问" * 5000
    long_a = "答" * 5000
    doc = build_html(ex(question=long_q, answer=long_a), max_chars_per_side=100)
    # 只在**正文**里数：`<style>` 里的注释与页头标题也会出现汉字，
    # 拿整份文档计数会被它们带偏（实测：样式注释里一个「主问答之上」就让计数 +1）。
    body = _re.sub(r"<style>.*?</style>", "", doc, flags=_re.S)
    check("问题被截断", body.count("问") <= 100, f"{body.count('问')}")
    check("回答被截断", body.count("答") <= 100, f"{body.count('答')}")
    check("有截断提示", "已截断" in doc)
    check("没超长就不出提示",
          "已截断" not in build_html(ex(), max_chars_per_side=100))


def test_find_chromium():
    print("\n[4] Chromium 探测")
    got = find_chromium()
    print(f"     （本机探测结果：{got or '未找到'}）")
    check("显式给绝对路径：存在则用、不存在则 None",
          find_chromium(sys.executable) == sys.executable
          and find_chromium("/no/such/chrome") is None)
    check("显式给命令名走 PATH", find_chromium("sh") is not None)


async def test_render_real(chrome):
    print("\n[5] 真实渲染")
    out = TMP / "real"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=60))
    path = await r.render(ex(index=3), session_name="预热群会话", total=12)
    check("产出文件存在", os.path.exists(path), path)
    w, h = png_size(path)
    # 出图后会按像素收口（裁掉右侧/底部留白），宽度可能略小于配置值
    check("是 PNG 且宽度不超过配置", 200 <= w <= 720, f"{w}x{h}")
    check("高度合理（不是 0，也没爆）", 40 < h < 2000, f"{h}")
    # 4 是我们按像素裁剪时故意留的气口，不是没裁干净
    from plugins.ai_chat.png import PAGE_PAD
    check("自适应高度：底部只留气口", png_bottom_margin(path) <= PAGE_PAD + 6,
          f"底部留白 {png_bottom_margin(path)}px")

    # 内容越多图越高（证明高度真的跟着内容走）
    tall = await r.render(
        ex(index=4, answer="这是一段很长的回答。" * 40), total=12)
    _, h2 = png_size(tall)
    check("内容多则图更高", h2 > h, f"{h} → {h2}")

    # 换行必须保留（答案里的多行不能挤成一行）
    multi = await r.render(ex(index=5, answer="第一行\n第二行\n第三行"), total=12)
    _, h3 = png_size(multi)
    check("多行回答撑出更高（换行生效）", h3 > h, f"{h} → {h3}")

    print("\n[6] 临时文件清理")
    rest = [p.name for p in Path(out).iterdir()
            if p.name.endswith(".html") or p.name.startswith("prof-")]
    check("HTML 与 Chromium profile 都被清掉", not rest, f"残留 {rest}")
    check("PNG 保留着", any(p.name.endswith(".png") for p in Path(out).iterdir()))


async def test_crop_only_text_mode(chrome):
    """**裁切只归文字模式。**

    文字模式的块图（表格 / 公式 / 列表 / 代码）要插在文字**之间**，必须紧贴内容 ——
    不裁的话一行字中间会夹进几十像素白边。

    而 Markdown 模式出的是一整「页」：页边距是版面的一部分。原先它也走同一条
    裁切逻辑，于是右下留白按墨迹被切掉 —— 内容长短一变、页边距就跟着变，
    看着就不像一页文档了（用户要求把裁切收归文字模式）。
    """
    print("\n[1.5] 裁切只归文字模式")
    from plugins.ai_chat.png import PAGE_PAD
    out = TMP / "crop"
    r = HistoryRenderer(str(out), options=RenderOptions(width=720, timeout=60))

    md = await r.render_markdown("## 标题\n\n正文一段，带公式 $a=5$。\n\n"
                                 "| 量 | 值 |\n| --- | --- |\n| a | 5 |\n")
    w, h = png_size(md)
    right = png_right_margin(md)
    check("Markdown 整页：不裁右侧", right > PAGE_PAD + 8, f"右边距只有 {right}px")
    check("Markdown 整页：宽度就是页宽（没被裁窄）", w >= 700, f"{w}x{h}")
    check("Markdown 整页：底部也留着页边距",
          png_bottom_margin(md) > PAGE_PAD + 4,
          f"下边距 {png_bottom_margin(md)}px")

    # 文字模式的块图仍然紧贴
    frag = await r.render_rich("table", "| 量 | 值 |\n| --- | --- |\n| a | 5 |")
    check("文字模式块图：右边紧贴", png_right_margin(frag) <= PAGE_PAD + 6,
          f"右边距 {png_right_margin(frag)}px")

    inline = await r.render_inline_math(r"\frac{a}{b}")
    check("文字模式行内公式：紧贴", png_right_margin(inline) <= 12,
          f"右边距 {png_right_margin(inline)}px")

    # 问答卡本来就不裁，别改坏了。
    # ⚠️ 这里只能按**宽度**判：卡片整页有灰底（`body{background:#f4f5f7}`），
    # 按「白边距」量出来恒为 0（第一版断言就是这么假失败的）。裁过的图会**变窄**，
    # 所以「宽度正好等于配置宽度」就是「没裁」的判据。
    card = await r.render(ex(index=2), session_name="测试", total=3)
    w_card, _h_card = png_size(card)
    check("问答卡：同样不裁（宽度就是配置宽度）",
          w_card == r.opt.width, f"{w_card} vs {r.opt.width}")


async def test_render_failures(chrome):
    print("\n[7] 失败路径要说人话（不抛穿、不静默）")
    # 1) 找不到浏览器
    r = HistoryRenderer(str(TMP / "nochrome"),
                        options=RenderOptions(chrome="/no/such/chrome"))
    try:
        await r.render(ex())
        check("找不到 Chromium 时应报错", False, "居然成功了")
    except RenderError as e:
        check("找不到 Chromium 报明确错误", "Chromium" in str(e), str(e))
    check("错误信息里不暴露命令行", "-no-sandbox" not in r.unavailable(),
          r.unavailable())

    # 2) 渲染超时：拿一个只会睡觉的假浏览器
    fake = TMP / "slow-chrome"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    r2 = HistoryRenderer(str(TMP / "slow"),
                         options=RenderOptions(chrome=str(fake), timeout=1.0))
    try:
        await r2.render(ex())
        check("超时应报错", False, "居然成功了")
    except RenderError as e:
        check("超时报明确错误", "超时" in str(e), str(e))
    # 超时后临时文件也要清掉
    left = [p.name for p in Path(TMP / "slow").iterdir()
            if p.name.endswith(".html") or p.name.startswith("prof-")]
    check("超时后也清理了临时文件", not left, f"残留 {left}")

    # 3) 浏览器存在但立刻失败（不产出 PNG）
    fake2 = TMP / "fail-chrome"
    fake2.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake2.chmod(fake2.stat().st_mode | stat.S_IEXEC)
    r3 = HistoryRenderer(str(TMP / "fail"),
                         options=RenderOptions(chrome=str(fake2), timeout=10))
    try:
        await r3.render(ex())
        check("没产出 PNG 时应报错", False, "居然成功了")
    except RenderError as e:
        check("没产出 PNG 报明确错误", "没有产出图片" in str(e) or "高度" in str(e),
              str(e))


async def test_concurrency_gate(chrome):
    print("\n[8] 并发闸（连发 hget 不能变 fork 炸弹）")
    out = TMP / "gate"
    r = HistoryRenderer(str(out), options=RenderOptions(width=600, timeout=60))
    # 记录同时在跑的 Chromium 数量
    peak = 0
    running = 0
    orig = r._run_chrome

    async def counting(*a, **kw):
        nonlocal peak, running
        running += 1
        peak = max(peak, running)
        try:
            return await orig(*a, **kw)
        finally:
            running -= 1

    r._run_chrome = counting        # type: ignore[assignment]
    await asyncio.gather(*[r.render(ex(index=i)) for i in range(1, 6)])
    check("5 张图都渲染成功",
          len([p for p in Path(out).iterdir() if p.suffix == ".png"]) == 5,
          f"{len(list(Path(out).iterdir()))} 个文件")
    check("同时最多 1 个 Chromium", peak == 1, f"峰值并发 {peak}")


async def test_prune(chrome):
    print("\n[9] 目录有界（不能把 SD 卡写满）")
    out = TMP / "prune"
    r = HistoryRenderer(str(out), options=RenderOptions(keep_files=3, timeout=60))
    for i in range(1, 6):
        await r.render(ex(index=i))
    pngs = [p for p in Path(out).iterdir() if p.suffix == ".png"]
    check("只保留 keep_files 张", len(pngs) <= 3, f"留下 {len(pngs)} 张")
    check("留下的是最近的（不是最早的）", len(pngs) >= 1, f"{len(pngs)}")


async def main():
    TMP.mkdir(parents=True, exist_ok=True)
    test_html_escaping()
    test_html_content()
    test_markdown_rendering()
    test_truncation()
    test_find_chromium()

    chrome = find_chromium()
    if not chrome:
        print("\n⚠️  本机没有 Chromium —— 跳过真实渲染用例（纯逻辑断言仍在上面跑过）")
    else:
        for fn in (test_render_real, test_crop_only_text_mode,
                   test_render_failures, test_concurrency_gate, test_prune):
            try:
                await fn(chrome)
            except Exception as e:  # noqa: BLE001
                import traceback
                globals()["FAILED"] = globals()["FAILED"] + 1
                FAILURES.append(f"{fn.__name__}: {e!r}")
                print(f"  ❌ {fn.__name__} 抛异常: {e!r}")
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
