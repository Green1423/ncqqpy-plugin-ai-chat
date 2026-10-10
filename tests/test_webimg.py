"""网图配图（`fetch_image`）的单元测试。

这里钉的是**安全边界与工程约束**，不是「能不能下载」这种显而易见的事：

* 魔数判定（不看后缀、不信 Content-Type）—— 否则「下载这个 .png」就能把任意
  二进制塞进群里；
* SSRF：内网/本机地址一律拒绝，而且**跳转的每一跳都要重查**（自动跳转能绕过
  首跳检查，云元数据端点就是这么被打的）；
* 大小上限、一轮张数上限、同 URL 去重、落盘目录有界；
* 只有真下载过的文件才在白名单里（模型编的路径发不出去）。

报告口径：`通过 N / 失败 M`，失败项最后列出来。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
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

from plugins.ai_chat import search as search_mod                      # noqa: E402
from plugins.ai_chat.plot import extract_markers                      # noqa: E402
from plugins.ai_chat.search import FetchBlocked, FetchTooLarge, check_url  # noqa: E402
from plugins.ai_chat.webimg import WebImageService, sniff_image       # noqa: E402

PASSED = 0
FAILED = 0
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


# ---- 假数据 ----

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
GIF = b"GIF89a" + b"\x00" * 64
WEBP = b"RIFF\x24\x00\x00\x00WEBP" + b"\x00" * 64
BMP = b"BM" + b"\x00" * 64
HTML = b"<!DOCTYPE html><html><body>not an image</body></html>"


class FakeWeb:
    """假的 `WebClient`：按 URL 回预设字节 / 抛预设异常，并记录调用。"""

    def __init__(self, mapping: dict) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int]] = []

    async def get_bytes(self, url: str, *, max_bytes: int):
        self.calls.append((url, max_bytes))
        got = self.mapping.get(url)
        if isinstance(got, Exception):
            raise got
        if got is None:
            raise RuntimeError("HTTP 404")
        return got, "image/png"


def svc(tmp: Path, **cfg) -> WebImageService:
    return WebImageService(cfg=cfg, logger=None, data_dir=str(tmp))


# ---- 用例 ----

def test_sniff() -> None:
    print("\n[1] 魔数判定（不看后缀、不信 Content-Type）")
    check("PNG", sniff_image(PNG) == "png", sniff_image(PNG))
    check("JPEG", sniff_image(JPG) == "jpg", sniff_image(JPG))
    check("GIF", sniff_image(GIF) == "gif", sniff_image(GIF))
    check("WebP（RIFF 容器）", sniff_image(WEBP) == "webp", sniff_image(WEBP))
    check("BMP", sniff_image(BMP) == "bmp", sniff_image(BMP))
    check("HTML 不算图片", sniff_image(HTML) == "", sniff_image(HTML))
    check("空内容不算图片", sniff_image(b"") == "", sniff_image(b""))
    check("只有 RIFF 头没有 WEBP 标记不算",
          sniff_image(b"RIFF\x24\x00\x00\x00AVI ") == "",
          sniff_image(b"RIFF\x24\x00\x00\x00AVI "))


def test_ssrf_gate() -> None:
    print("\n[2] SSRF 闸（图片走的是同一道）")
    for bad in ("http://127.0.0.1/x.png", "http://localhost/x.png",
                "http://10.0.0.5/x.png", "http://192.168.1.1/x.png",
                "http://169.254.169.254/latest/meta-data/",
                "file:///etc/passwd", "ftp://example.com/x.png"):
        try:
            check_url(bad)
            check(f"拒绝 {bad[:38]}", False, "居然放行了")
        except FetchBlocked:
            check(f"拒绝 {bad[:38]}", True)
    check("公网 http 放行",
          check_url("http://example.com/a.png") == "http://example.com/a.png",
          "被拒了")


async def test_fetch_ok(tmp: Path) -> None:
    print("\n[3] 下载成功：落盘 + 白名单 + 标记")
    s = svc(tmp)
    web = FakeWeb({"https://img.example.com/a.png": PNG})
    out = await s.fetch(web, "https://img.example.com/a.png", caption="一条正弦曲线")
    marks = extract_markers(out)
    check("回执里有标记", len(marks) == 1, out)
    path, title = marks[0] if marks else ("", "")
    check("标记带标题", title == "一条正弦曲线", title)
    check("文件真的落盘了", os.path.exists(path), path)
    check("落盘内容一致", Path(path).read_bytes() == PNG, "")
    check("在白名单里", path in s.rendered, f"{s.rendered}")
    check("扩展名按魔数（不是按 URL 后缀）", path.endswith(".png"), path)
    check("回执说明了「单独一条消息」", "单独一条消息" in out, out)
    check("回执提醒别贴 Markdown 图片", "Markdown" in out, out)


async def test_fetch_rejects(tmp: Path) -> None:
    print("\n[4] 各种拒绝路径都要说人话、且不留标记")
    s = svc(tmp)

    out = await s.fetch(FakeWeb({"https://x/a.png": HTML}), "https://x/a.png")
    check("不是图片 → 拒绝", "不是图片" in out, out)
    check("不是图片 → 没有标记", not extract_markers(out), out)
    check("不是图片 → 不落盘", not list((tmp / "web_img").glob("*"))
          if (tmp / "web_img").exists() else True, "")

    out = await s.fetch(FakeWeb({"https://x/a.png": FetchTooLarge(100, 500)}),
                        "https://x/a.png")
    check("太大 → 说清上限", "太大" in out, out)
    check("太大 → 没有标记", not extract_markers(out), out)

    out = await s.fetch(FakeWeb({"https://x/a.png": FetchBlocked("内网")}),
                        "https://x/a.png")
    check("被 SSRF 闸挡住 → 明说", "不能下载这个地址" in out, out)

    out = await s.fetch(FakeWeb({}), "https://x/a.png")
    check("404 → 报下载失败", "下载失败" in out, out)

    out = await s.fetch(FakeWeb({}), "")
    check("空 URL → 要 url", "请给出" in out, out)

    out = await s.fetch(None, "https://x/a.png")
    check("没有联网能力 → 明说", "不可用" in out, out)


async def test_limits(tmp: Path) -> None:
    print("\n[5] 大小上限 / 一轮张数 / 同 URL 去重")
    s = svc(tmp, max_bytes=1024, max_per_turn=2)
    web = FakeWeb({"https://x/1.png": PNG, "https://x/2.png": JPG,
                   "https://x/3.png": GIF})
    got = await s.fetch(web, "https://x/1.png")
    check("把上限传给了下载层", web.calls[0][1] == 1024, f"{web.calls}")
    check("第 1 张通过", bool(extract_markers(got)), got)
    # 同一条 URL 再来一次：不重复下载、也不占额度
    again = await s.fetch(web, "https://x/1.png")
    check("同 URL 复用（不再下载）", len(web.calls) == 1, f"{web.calls}")
    check("同 URL 仍然给标记", bool(extract_markers(again)), again)
    check("复用会说明一句", "复用" in again or "下过" in again, again)
    await s.fetch(web, "https://x/2.png")
    third = await s.fetch(web, "https://x/3.png")
    check("超过一轮上限 → 拒绝", not extract_markers(third), third)
    check("拒绝理由提到上限", "上限" in third, third)
    # 新一轮重新计数
    s.begin_turn()
    s._by_url.clear()          # 模拟「新的一轮里换一张新图」
    fresh = await s.fetch(FakeWeb({"https://x/4.png": PNG}), "https://x/4.png")
    check("新一轮重新计数", bool(extract_markers(fresh)), fresh)


async def test_prune(tmp: Path) -> None:
    print("\n[6] 落盘目录有界")
    s = svc(tmp, keep_files=3)
    web = FakeWeb({f"https://x/{i}.png": PNG for i in range(6)})
    for i in range(6):
        await s.fetch(web, f"https://x/{i}.png")
    files = sorted((tmp / "web_img").glob("*.png"))
    check("只留最近 keep_files 个", len(files) == 3, f"{[f.name for f in files]}")
    check("被删掉的也移出白名单",
          all(p in s.rendered for p in [str(f) for f in files])
          and len(s.rendered) == 3, f"{len(s.rendered)}")


async def test_get_bytes_redirects(tmp: Path) -> None:
    """`get_bytes` 自己跟跳转、**每一跳都过 SSRF 闸**，超限直接报错。

    为什么要自己跟：aiohttp 的自动跳转不过闸，一个公网地址 302 到
    `http://169.254.169.254/…` 就绕过去了。
    """
    print("\n[7] 下载层：跳转逐跳过闸 + 大小上限")
    from aiohttp import web as aiohttp_web

    async def img(request):
        return aiohttp_web.Response(body=PNG, content_type="image/png")

    async def redir(request):
        raise aiohttp_web.HTTPFound("/img.png")

    async def big(request):
        return aiohttp_web.Response(body=b"x" * 5000)

    app = aiohttp_web.Application()
    app.router.add_get("/img.png", img)
    app.router.add_get("/redir", redir)
    app.router.add_get("/big", big)
    runner = aiohttp_web.AppRunner(app)
    await runner.setup()
    site = aiohttp_web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]        # noqa: SLF001
    base = f"http://127.0.0.1:{port}"

    real_check = search_mod.check_url
    web = search_mod.WebClient(timeout=5)
    try:
        # 测试服务器就在回环地址上 —— 先把闸放开，只测「跳转/大小」逻辑
        search_mod.check_url = search_mod.clean_url
        raw, ctype = await web.get_bytes(f"{base}/img.png", max_bytes=4096)
        check("拿到原始字节", raw == PNG, f"{len(raw)} 字节")
        check("带 Content-Type", "image/png" in ctype, ctype)
        raw2, _ = await web.get_bytes(f"{base}/redir", max_bytes=4096)
        check("自己跟 302 拿到图片", raw2 == PNG, f"{len(raw2)} 字节")

        # 每一跳都要过闸：把「既不是 /img.png」的地址判为内网
        def gate(url: str) -> str:
            if "/img.png" in url:
                return search_mod.clean_url(url)
            raise FetchBlocked("拒绝访问内网或本机地址")

        search_mod.check_url = gate
        try:
            await web.get_bytes(f"{base}/redir", max_bytes=4096)
            check("跳转目标要重新过闸", False, "跳转没被拦")
        except FetchBlocked:
            check("跳转目标要重新过闸", True)

        search_mod.check_url = search_mod.clean_url
        try:
            await web.get_bytes(f"{base}/big", max_bytes=1024)
            check("超限要报错而不是截断", False, "居然成功了")
        except FetchTooLarge as e:
            check("超限要报错而不是截断", e.limit == 1024, f"{e}")
    finally:
        search_mod.check_url = real_check
        await web.close()
        await runner.cleanup()


async def main() -> int:
    test_sniff()
    test_ssrf_gate()
    base = Path(tempfile.mkdtemp(prefix="webimg-", dir=str(ROOT / "tests" / ".test-tmp")))
    await test_fetch_ok(base / "a")
    await test_fetch_rejects(base / "b")
    await test_limits(base / "c")
    await test_prune(base / "d")
    await test_get_bytes_redirects(base / "e")
    print(f"\n{'=' * 50}\n通过 {PASSED} / 失败 {FAILED}")
    if FAILURES:
        print("失败项：")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
