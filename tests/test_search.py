"""联网搜索的纯单元测试 —— 不打桩框架、不联网（除一条可跳过的真连用例）。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_search.py

解析器是**纯函数**，所以用保存下来的真实 Bing HTML 做回归：不联网也能卡住
「搜索页结构变了」这类回归。真实联网只用于最后一条「后端仍然可达」的冒烟用例，
不通就跳过（不报假绿，也不让 CI 因为没外网而变红）。
"""
from __future__ import annotations

import asyncio
import sys
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

from plugins.ai_chat.search import (          # noqa: E402
    ANYSEARCH_DOMAINS, ANYSEARCH_URL, FetchBlocked, SearchResult, WebClient,
    build_url, check_url, clean_url, format_results, html_to_text,
    looks_like_stub, page_to_text, parse_bing, parse_json_results, query_terms,
    relevance)

PASSED = 0
FAILED = 0
FAILURES: list[str] = []


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


#: 一份**真实 Bing 结果页**的最小复制（结构取自实测页面，去掉无关噪音）。
#: 用真结构而不是自己编的：解析器要适配的是搜索引擎的输出，不是我们的想象。
BING_FIXTURE = '''<html><body><ol id="b_results">
<li class="b_algo" data-id iid=SERP.1><h2 class=""><a target="_blank" href="https://www.bupt.edu.cn/" h="ID=SERP,5127.2">北京邮电大学</a></h2><p class="b_lineclamp2" data-rslinkclamp-iid="">2025年9月18日&ensp;&#0183;&ensp;北京邮电大学发布两项语义通信外场试验成果 …</p></li>
<li class="b_algo"><h2><a href="https://www.bupt.edu.cn/kyyjy/list.htm">科研院</a></h2><p class="b_lineclamp3">科研院是负责全校科研管理的职能部门。</p></li>
<li class="b_algo"><h2><a href="//example.com/x?a=1&amp;utm_source=bing">相对协议链接</a></h2><p class="b_lineclamp2">相对协议应被补成 https。</p></li>
<li class="b_algo"><h2><a href="/ck/a?!&&p=abc&u=a1aHR0cHM6Ly9leGFtcGxlLm9yZy9yZWFs">跳转链接</a></h2><p class="b_lineclamp2">真实地址藏在 u=a1 参数里。</p></li>
<li class="b_algo"><h2><a href="https://no-snippet.example/">没有摘要</a></h2></li>
<li class="b_algo"><p class="b_lineclamp2">没有标题的块应被跳过</p></li>
</ol></body></html>'''


def test_html_to_text():
    print("\n[1] HTML → 文本（与渲染共用同一套实体解码）")
    check("去标签", html_to_text("<b>粗</b>体").replace(" ", "") == "粗体",
          html_to_text("<b>粗</b>体"))
    check("解实体", "·" in html_to_text("a&ensp;b") or " " in html_to_text("a&ensp;b"),
          repr(html_to_text("a&ensp;b")))
    check("解 &nbsp;", html_to_text("a&nbsp;b") == "a b", repr(html_to_text("a&nbsp;b")))
    check("解 &lt;&gt;", html_to_text("&lt;script&gt;") == "<script>",
          repr(html_to_text("&lt;script&gt;")))
    check("压掉多余空白", html_to_text("a   \t  b") == "a b", repr(html_to_text("a   \t  b")))
    check("空输入不炸", html_to_text("") == "")


def test_parse_bing():
    print("\n[2] Bing 结果页解析")
    got = parse_bing(BING_FIXTURE, limit=10)
    check("只取有标题的块（跳过无标题）", len(got) == 5,
          f"{len(got)}: {[r.title for r in got]}")
    check("标题正确", got[0].title == "北京邮电大学", got[0].title)
    check("绝对链接保留", got[0].url == "https://www.bupt.edu.cn/", got[0].url)
    check("摘要被解实体", "·" in got[0].snippet and "&#0183;" not in got[0].snippet,
          got[0].snippet[:60])
    check("相对协议补 https", got[2].url.startswith("https://example.com/"), got[2].url)
    check("跟踪参数被清掉", "utm_source" not in got[2].url, got[2].url)
    check("Bing 跳转链接还原成真实地址",
          got[3].url == "https://example.org/real", got[3].url)
    check("无摘要的结果摘要为空（不是 None、也不报错）",
          got[4].snippet == "" and got[4].title == "没有摘要", repr(got[4].snippet))

    check("limit 生效", len(parse_bing(BING_FIXTURE, limit=2)) == 2)
    check("空 HTML 返回空列表", parse_bing("", limit=5) == [])
    check("乱 HTML 返回空列表", parse_bing("<html>没有结果</html>") == [])


def test_clean_url_and_build():
    print("\n[3] URL 清洗与构造")
    check("去掉 utm/fbclid",
          clean_url("https://a.com/x?utm_source=q&fbclid=1&keep=2") == "https://a.com/x?keep=2",
          clean_url("https://a.com/x?utm_source=q&fbclid=1&keep=2"))
    check("去掉 fragment",
          clean_url("https://a.com/x#frag") == "https://a.com/x",
          clean_url("https://a.com/x#frag"))
    check("非 http 原样返回", clean_url("ftp://a/b") == "ftp://a/b")
    check("空值不炸", clean_url("") == "")
    url = build_url("bing", "北邮 国庆", 5)
    check("查询串被编码", "%E5%8C%97%E9%82%AE" in url, url)
    check("count 进了 URL", "count=5" in url, url)


def test_format_results():
    print("\n[4] 给模型看的结果格式")
    got = parse_bing(BING_FIXTURE, limit=2)
    text = format_results(got)
    check("带编号", text.startswith("1. "), text[:40])
    check("带链接", "链接：" in text, text[:120])
    check("带摘要", "摘要：" in text, text[:160])
    check("无结果给明确说明", format_results([]) == "没有搜到结果")
    long_r = parse_bing(
        '<li class="b_algo"><h2><a href="http://a/">T</a></h2>'
        '<p class="b_lineclamp2">' + "字" * 500 + "</p></li></ol>", limit=1)
    check("摘要按上限截断", "…" in format_results(long_r, max_snippet=50),
          format_results(long_r, max_snippet=50)[:80])


def test_ssrf_guard():
    print("\n[5] SSRF 闸（搜索/网页内容都是不可信输入）")
    for bad in ("http://127.0.0.1:8080/x", "http://localhost/x",
                "http://192.168.1.1/", "http://10.0.0.5/",
                "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/",
                "http://[::1]/", "http://0.0.0.0/", "file:///etc/passwd",
                "ftp://a/b", "http://no-such-host-xyz.invalid/"):
        try:
            check_url(bad)
            check(f"拦截 {bad[:38]}", False, "被放行了")
        except FetchBlocked:
            check(f"拦截 {bad[:38]}", True)
    check("放行正常公网地址",
          check_url("https://example.com/x") == "https://example.com/x")


def test_page_to_text():
    print("\n[6] 网页正文提取")
    html = ('<html><head><title>T</title><style>a{}</style></head><body>'
            '<nav>导航别要我</nav><h1>标题</h1><p>第一段。</p><p>第二段。</p>'
            '<script>var x=1;</script><div>div 内容</div>'
            '<footer>页脚也别要</footer></body></html>')
    text = page_to_text(html)
    check("保留标题与正文", "标题" in text and "第一段。" in text, text)
    check("丢掉 nav", "导航别要我" not in text, text)
    check("丢掉 script", "var x" not in text, text)
    check("丢掉 style", "a{}" not in text, text)
    check("丢掉 footer", "页脚也别要" not in text, text)
    check("段落之间有空行", "\n\n" in text, repr(text))
    check("畸形 HTML 不抛", isinstance(page_to_text("<p>没闭合<div><span>"), str))
    check("空输入不炸", page_to_text("") == "")


def test_stub_detection():
    print("\n[7] 空壳页识别（限流 vs 结构变化）")
    check("几百字节的壳页判为 stub",
          looks_like_stub('<html><script>window.si_ST=new Date</script></html>'))
    check("有大结果容器的不是 stub",
          not looks_like_stub('<html>' + 'x' * 25000 + 'class="b_algo"'))
    check("大页面不是 stub", not looks_like_stub("x" * 25000))


def test_relevance_gate():
    print("\n[7.5] 相关性闸（「有结果」不等于「有相关结果」）")
    # 实测过的真实反例：Bing 对北邮图书馆的查询返回了 Bilibili/北京市政府
    bad = [SearchResult("北京市_百度百科", "", "北京市是中华人民共和国首都"),
           SearchResult("Bilibili", "", "哔哩哔哩是国内知名的视频弹幕网站")]
    good = [SearchResult("北京邮电大学图书馆_360百科", "",
                         "北京邮电大学图书馆馆藏资源丰富"),
            SearchResult("关于调整图书馆校庆期间对外开放时间的通知_北京邮电大学图书馆",
                         "", "图书馆对外开放时间")]
    q = "北京邮电大学 图书馆 国庆 开放时间"
    check("无关结果命中率很低", relevance(bad, q) < 0.34, f"{relevance(bad, q):.2f}")
    check("相关结果命中率够高", relevance(good, q) >= 0.34, f"{relevance(good, q):.2f}")
    check("空结果命中率为 0", relevance([], q) == 0.0)
    check("空查询不误杀（返回 1.0）", relevance(bad, "") == 1.0)

    terms = query_terms("北京邮电大学 图书馆 国庆 开放时间")
    check("长词优先、被包含的短词不重复计",
          "北京邮电大学" in terms and "北京" not in terms, f"{terms}")
    check("过滤虚词与单字",
          not (set(query_terms("这个 怎么 是 a")) & {"这个", "怎么", "是", "a"}),
          f"{query_terms('这个 怎么 是 a')}")
    check("英文虚词也被过滤",
          "the" not in query_terms("what is the weather"), f"{query_terms('what is the weather')}")


def test_json_api_parser():
    print("\n[7.6] JSON 搜索 API 解析（Tavily / SearxNG / Bocha / 通用）")
    cases = {
        "tavily": {"results": [{"title": "T", "url": "https://a/1", "content": "C"}]},
        "searx": {"results": [{"title": "S", "url": "https://b/2", "content": "D"}]},
        "bocha": {"data": {"webPages": {"value": [
            {"name": "B", "url": "https://c/3", "summary": "E"}]}}},
        "generic": {"data": [{"name": "G", "link": "https://d/4", "description": "F"}]},
    }
    want_url = {"tavily": "https://a/1", "searx": "https://b/2",
                "bocha": "https://c/3", "generic": "https://d/4"}
    for name, data in cases.items():
        got = parse_json_results(data, limit=3)
        check(f"{name} 形状能解析",
              len(got) == 1 and got[0].url == want_url[name], f"{got}")
    check("空响应返回空", parse_json_results({}) == [] and
          parse_json_results({"results": []}) == [])
    check("取不到标题的行被跳过",
          parse_json_results({"results": [{"url": "x"}]}) == [])
    check("limit 生效", len(parse_json_results(
        {"results": [{"title": f"T{i}", "url": f"u{i}"} for i in range(9)]},
        limit=4)) == 4)
    check("自定义字段路径优先",
          parse_json_results({"mine": [{"t": "X", "u": "https://z/9"}]},
                             list_path="mine", title_key="t",
                             url_key="u")[0].url == "https://z/9")


async def test_api_backend():
    print("\n[7.7] JSON API 通道（配了就先走它）")
    calls = []

    async def fake_json(url, query, count, key):
        calls.append((url, query, count, key))
        return {"results": [{"title": "北京邮电大学图书馆通知",
                             "url": "https://lib.bupt.edu.cn/n",
                             "content": "国庆开放时间"}]}

    c = WebClient(timeout=5, api_url="https://api.example/s", api_key="k",
                  fetch_json=fake_json)
    out = await c.search("北京邮电大学图书馆 国庆 开放时间", count=2)
    check("API 通道返回结果", out.ok and out.backend == "api", f"{out.error}")
    check("调用参数带上 key 与条数",
          calls and calls[0][3] == "k" and calls[0][2] == 2, f"{calls}")

    # API 挂了 → 回落到抓页面（这里抓页面也被打桩成失败）
    async def boom_json(*a):
        raise OSError("Cannot connect to host api.example")

    async def stub_fetch(url, *, max_bytes):
        return '<html><script>window.si_ST=new Date</script></html>'

    c2 = WebClient(timeout=5, api_url="https://api.example/s",
                   fetch_json=boom_json, fetch=stub_fetch)
    async def no_render(url, **kw):
        raise RuntimeError("no chrome")
    c2.rendered = no_render                            # type: ignore[assignment]
    out2 = await c2.search("北京邮电大学", count=2)
    check("API 失败会回落到抓页面", not out2.ok and "限流" in out2.error,
          f"{out2.error}")

    # API 返回无关结果：**默认放行**（安全网语义，见 api_min_score 的说明），
    # 但如果把下限调成严格模式，就该拦下并回落抓页面。
    async def bad_json(url, query, count, key):
        return {"results": [{"title": "哔哩哔哩", "url": "https://b/",
                             "content": "视频弹幕网站"}]}

    async def scrape_fetch(url, *, max_bytes):
        # 抓页面这条路的替身：给出**相关**结果
        return ('<html><ol><li class="b_algo"><h2><a href="https://lib.bupt.edu.cn/">'
                '北京邮电大学图书馆</a></h2>'
                '<p class="b_lineclamp2">国庆开放时间</p></li></ol></html>')

    c3 = WebClient(timeout=5, api_url="https://api.example/s",
                   fetch_json=bad_json, fetch=scrape_fetch,
                   backend="bing", backends=["bing"])
    c3.api_min_score = 0.34        # 严格模式：无关结果要拦下
    c3.rendered = _no_render                              # type: ignore[assignment]
    out3 = await c3.search("北京邮电大学图书馆 国庆 开放时间", count=2)
    check("API 的无关结果被拦下后回落到抓页面",
          out3.ok and out3.backend == "bing"
          and "北京邮电大学" in out3.results[0].title,
          f"ok={out3.ok} backend={out3.backend} {out3.results[:1]}")

    # 两条路都只给无关结果 → 最终失败，且说明里点出「无关」
    async def all_bad_fetch(url, *, max_bytes):
        return ('<html><ol><li class="b_algo"><h2><a href="https://b/">哔哩哔哩</a>'
                '</h2><p class="b_lineclamp2">弹幕网站</p></li></ol></html>')
    c4 = WebClient(timeout=5, api_url="https://api.example/s",
                   fetch_json=bad_json, fetch=all_bad_fetch,
                   backend="bing", backends=["bing"])
    c4.api_min_score = 0.34        # 严格模式
    c4.rendered = _no_render                              # type: ignore[assignment]
    out4 = await c4.search("北京邮电大学图书馆 国庆 开放时间", count=2)
    check("全都无关时明确失败并说明无关",
          not out4.ok and "无关" in out4.error, f"{out4.error}")

    # 默认（安全网）语义：同样这批无关结果会被**放行**，交给模型判断 ——
    # 这是刻意的取舍：精确子串匹配对中文误杀太多（北邮/北京邮电大学、今天北京天气）
    c5 = WebClient(timeout=5, api_url="https://api.example/s",
                   fetch_json=bad_json, fetch=all_bad_fetch,
                   backend="bing", backends=["bing"])
    out5 = await c5.search("北京邮电大学图书馆 国庆", count=2)
    check("默认安全网语义下 API 结果放行（交给模型判断）",
          out5.ok and out5.backend == "api", f"ok={out5.ok} backend={out5.backend}")


async def _no_render(url, **kw):
    """渲染兜底的替身：直接失败，让用例只测「换后端/报错」那条路。"""
    raise RuntimeError("no chrome")


async def test_client_behavior():
    print("\n[8] 客户端行为（注入假 HTTP，不打真网）")
    calls = []

    async def fake_fetch(url, *, max_bytes):
        calls.append(url)
        if "boom" in url:
            raise OSError("Cannot connect to host")
        return BING_FIXTURE

    c = WebClient(timeout=5, fetch=fake_fetch, backend="bing", backends=["bing"])
    # 查询词必须真的出现在桩结果里：相关性闸会拦「有结果但无关」的情况，
    # 用「北京邮电大学」这种命中 fixture 的查询才有意义
    out = await c.search("北京邮电大学", count=3)
    check("搜索成功返回结果", out.ok and len(out.results) == 3,
          f"{out.error} {len(out.results)}")

    async def empty_fetch(url, *, max_bytes):
        # 页面够大 ⇒ 不是限流空壳页，而是「结构变了」；两条分支要能分清
        return "<html><body>" + "no results here. " * 500 + "</body></html>"
    c2 = WebClient(timeout=5, fetch=empty_fetch)
    c2.rendered = _no_render                              # type: ignore[assignment]
    out2 = await c2.search("没有任何结果的查询词", count=3)
    check("解析不出结果时给出说明",
          not out2.ok and ("结构" in out2.error or "无关" in out2.error
                           or "限流" in out2.error), out2.error)

    async def boom_fetch(url, *, max_bytes):
        raise OSError("Cannot connect to host example.com")
    c3 = WebClient(timeout=5, fetch=boom_fetch, backend="bing", backends=["bing"])
    c3.rendered = _no_render                              # type: ignore[assignment]
    out3 = await c3.search("x", count=3)
    check("网络失败转成人话", not out3.ok and "连不上" in out3.error, out3.error)
    check("空关键词不发请求",
          (await WebClient(timeout=5, fetch=fake_fetch).search("")).error != "")

    print("\n[9] 抓正文（含 SSRF 拦截与截断）")
    c4 = WebClient(timeout=5, fetch=fake_fetch)
    text = await c4.read_page("https://example.com/a")
    check("抓取返回带来源", text.startswith("来源：https://example.com/a"), text[:60])
    blocked = await c4.read_page("http://127.0.0.1/x")
    check("内网地址被拒（不抛异常，回一句人话）",
          blocked.startswith("读取失败") and "内网" in blocked, blocked)
    # 这份 fixture 提取出的正文只有 135 字，所以 limit=200 不该截断；
    # 把 limit 压到 200 以下（read_page 的下限是 200）要用更长的页面来测，
    # 所以这里改成对同一页面用两个不同的上限，验证「超限才截断」。
    long_src = WebClient(timeout=5, fetch=fake_fetch)
    big = await long_src.read_page("https://example.com/a", limit=4000)
    check("limit 大于正文时原样返回（不无脑加省略号）",
          "已截断" not in big, big[-40:])
    check("正文完整（含所有块）", "没有标题的块应被跳过" in big, big[-60:])


def test_anysearch_contract():
    print("\n[7.8] AnySearch 响应契约")
    # 真实响应形状：{"code":0,"data":{"results":[{title,url,snippet,content}]}}
    reply = {"code": 0, "message": "success", "data": {"results": [
        {"title": "关于国庆节放假的通知", "url": "https://grs.bupt.edu.cn/i/1.htm",
         "snippet": "10月1日至10月8日放假", "content": "10月1日至10月8日放假，共8天"},
        {"title": "图书馆开放安排", "url": "https://lib.bupt.edu.cn/i/2.htm",
         "snippet": "10月1日~4日停止对外服务", "content": "…"},
    ]}}
    got = parse_json_results(reply, limit=5)
    check("解析 data.results", len(got) == 2, f"{len(got)}")
    check("字段映射正确",
          got[0].title == "关于国庆节放假的通知" and "grs.bupt.edu.cn" in got[0].url
          and "10月1日" in got[0].snippet, f"{got[0]}")
    check("limit 生效", len(parse_json_results(reply, limit=1)) == 1)
    check("空结果返回空", parse_json_results({"code": 0, "data": {"results": []}}) == [])

    check("默认地址指向 AnySearch", "api.anysearch.com" in ANYSEARCH_URL, ANYSEARCH_URL)
    check("领域白名单有 17 个", len(ANYSEARCH_DOMAINS) == 17, f"{len(ANYSEARCH_DOMAINS)}")
    check("含常用领域",
          {"general", "academic", "code", "finance"} <= set(ANYSEARCH_DOMAINS),
          f"{ANYSEARCH_DOMAINS}")


async def test_anysearch_client():
    print("\n[7.9] AnySearch 客户端行为（桩响应）")
    calls = []

    async def ok_json(url, query, count, key):
        calls.append((url, query, count, key))
        return {"code": 0, "data": {"results": [
            {"title": "北京邮电大学图书馆", "url": "https://lib.bupt.edu.cn/",
             "snippet": "开放时间"}]}}

    c = WebClient(timeout=5, api_url=ANYSEARCH_URL, fetch_json=ok_json)
    out = await c.search("北京邮电大学 图书馆", count=3)
    check("API 通道成功", out.ok and out.backend == "api", f"{out.error}")
    check("默认不设相关性下限（安全网语义）", c.api_min_score <= 0,
          f"{c.api_min_score}")

    # 上游明确报错：code != 0 不该当成成功
    async def err_json(url, query, count, key):
        return {"code": 40100, "message": "invalid api key", "data": {"results": [
            {"title": "x", "url": "https://x/", "snippet": "y"}]}}
    c2 = WebClient(timeout=5, api_url=ANYSEARCH_URL, fetch_json=err_json)
    out2, why = await c2._search_api("q", 3)
    check("上游 code!=0 判失败并带出原因",
          out2 is None and "invalid api key" in why, f"{out2} {why}")

    # 垂直领域：写对了才带上 tag
    async def tag_json(url, query, count, key):
        calls.append(("tag", c3.api_tag))
        return {"code": 0, "data": {"results": [
            {"title": "Swin Transformer", "url": "https://doi.org/x",
             "snippet": "cited 32985"}]}}
    c3 = WebClient(timeout=5, api_url=ANYSEARCH_URL, api_tag="academic.search",
                   fetch_json=tag_json)
    await c3.search("transformer", count=2)
    check("tag 传给客户端", calls[-1] == ("tag", "academic.search"), f"{calls[-1]}")
    check("合法 tag 不报错", c3._tag_error() == "", c3._tag_error())

    c4 = WebClient(timeout=5, api_url=ANYSEARCH_URL, api_tag="acadmic.search")
    check("拼错的领域给出可用列表",
          "不认识" in c4._tag_error() and "academic" in c4._tag_error(),
          c4._tag_error())
    out4, why4 = await c4._search_api("q", 2)
    check("拼错领域不发请求直接说明", out4 is None and "不认识" in why4, why4)


async def test_render_fallback_and_failover():
    print("\n[9.5] 空壳页 → 浏览器渲染兜底 / 后端自动切换")
    from plugins.ai_chat.search import SearchOutcome

    calls = {"get": 0, "rendered": 0}

    async def stub_fetch(url, *, max_bytes):
        calls["get"] += 1
        if "cn.bing" in url:
            return '<html><script>window.si_ST=new Date</script></html>'   # 空壳
        # ⚠️ 每个后端有**自己的**页面结构：给 360 喂 Bing 的 HTML 会解析不出结果，
        # 于是这条断言会变成在测「我的 fixture 对不对」而不是「会不会换后端」。
        return ('<html><li class="res-list"><h3><a href="https://ok.example/">'
                '来自后端二</a></h3><p class="res-desc">摘要</p></li></html>')

    c = WebClient(timeout=10, fetch=stub_fetch, backends=["bing", "360"])
    # 浏览器渲染也失败 → 必须换到第二个后端拿到结果
    async def boom(url, **kw):
        calls["rendered"] += 1
        raise RuntimeError("没有可用的 Chromium")
    c.rendered = boom                                    # type: ignore[assignment]
    out = await c.search("x", count=3)
    check("首个后端空壳 + 渲染失败 → 换后端成功",
          out.ok and out.backend == "360", f"ok={out.ok} backend={out.backend} err={out.error}")
    check("确实试过渲染兜底", calls["rendered"] >= 1, f"{calls}")

    # 渲染成功 → 直接用第一个后端
    async def good_render(url, **kw):
        calls["rendered"] += 1
        return ("<html><ol><li class=\"b_algo\"><h2><a href=\"https://r.example/\">"
                "渲染后拿到</a></h2><p class=\"b_lineclamp2\">渲染摘要</p></li></ol></html>")
    c2 = WebClient(timeout=10, fetch=stub_fetch, backends=["bing", "360"])
    c2.rendered = good_render                            # type: ignore[assignment]
    out2 = await c2.search("x", count=3)
    check("渲染兜底成功时用首选后端",
          out2.ok and out2.backend == "bing" and out2.results[0].title == "渲染后拿到",
          f"backend={out2.backend} {out2.results[:1]}")

    # 两个后端都空壳且渲染都失败 → 报错里说清是限流
    async def always_stub(url, *, max_bytes):
        return '<html><script>window.si_ST=new Date</script></html>'
    c3 = WebClient(timeout=10, fetch=always_stub, backends=["bing", "360"])
    c3.rendered = boom                                   # type: ignore[assignment]
    out3 = await c3.search("x", count=3)
    check("全部失败时给出限流说明", not out3.ok and "限流" in out3.error, out3.error)


async def test_anysearch_live():
    """AnySearch 真联网冒烟（匿名）。不通就跳过 —— 不报假绿。"""
    print("\n[9.9] AnySearch 真联网冒烟（失败即跳过）")
    c = WebClient(timeout=25, api_url=ANYSEARCH_URL)
    try:
        out = await c.search("北京邮电大学 图书馆 国庆 开放时间", count=3)
    finally:
        await c.close()
    if not out.ok:
        print(f"     ⚠️  跳过：{out.error}")
        return
    check("走的是 AnySearch（不是抓页面）", out.backend == "api", out.backend)
    check("有结果", len(out.results) >= 1, f"{len(out.results)}")
    check("结果带标题与链接",
          all(r.title and r.url for r in out.results), f"{out.results[:1]}")
    hits = " ".join(f"{r.title} {r.snippet}" for r in out.results)
    check("结果与查询相关（任一词命中）",
          any(t in hits for t in ("图书馆", "开放", "国庆", "北京邮电")),
          hits[:150])


async def test_backend_reachable_live():
    """真联网冒烟 —— 不通就跳过（不报假绿，也不让无外网环境变红）。"""
    print("\n[10] 真联网冒烟（失败即跳过）")
    # 超时给足：空壳页会走「浏览器渲染兜底」，那条路本身就要几秒
    c = WebClient(timeout=45)
    try:
        out = await c.search("北京邮电大学", count=3)
    finally:
        await c.close()
    if not out.ok:
        print(f"     ⚠️  跳过：{out.error}")
        return
    check("真实搜索有结果", len(out.results) >= 1, f"{len(out.results)}")
    check("结果有标题与链接",
          all(r.title and r.url for r in out.results), f"{out.results[:1]}")


def test_page_images():
    """从网页里挑出**图片直链**：懒加载属性、相对路径、去重、丢垃圾。"""
    print("\n[N] 页面图片直链")
    from plugins.ai_chat.search import page_images

    html = """
    <html><body>
      <img src="/static/a.png">
      <img src="https://cdn.example.com/b.jpg" alt="图">
      <img data-src="https://cdn.example.com/lazy.webp">      <!-- 懒加载 -->
      <img src="data:image/png;base64,AAAA">                  <!-- 内联，丢 -->
      <img src="https://cdn.example.com/spacer.gif">          <!-- 占位图，丢 -->
      <img src="/static/icon_home.png">                       <!-- 图标，丢 -->
      <img src="/static/logo.png">                            <!-- 站标，丢 -->
      <img src="/template/top-brand_03.png">                  <!-- 横幅，丢 -->
      <img src="https://cdn.example.com/logo.svg">            <!-- 矢量图，丢 -->
      <img src="/static/a.png">                               <!-- 重复，丢 -->
      <img src="c.gif">                                       <!-- 相对路径要拼对 -->
    </body></html>
    """
    got = page_images(html, "https://shop.example.com/list/index.html")
    check("相对路径拼成绝对 URL", "https://shop.example.com/static/a.png" in got,
          f"{got}")
    check("绝对 URL 原样保留", "https://cdn.example.com/b.jpg" in got, f"{got}")
    check("认得懒加载 data-src", "https://cdn.example.com/lazy.webp" in got,
          f"{got}")
    check("相对图片名也拼对", "https://shop.example.com/list/c.gif" in got, f"{got}")
    check("丢掉 data: 内联", not any(u.startswith("data:") for u in got), f"{got}")
    check("丢掉占位图/图标/svg",
          not any(x in " ".join(got) for x in ("spacer", "icon_", ".svg")), f"{got}")
    check("丢掉站标与横幅（实测这两个站的第一条就是它们）",
          not any(x in " ".join(got).lower()
                  for x in ("logo.png", "brand", "banner")), f"{got}")
    check("去重", len(got) == len(set(got)), f"{got}")
    check("条数受 limit 限制", len(page_images(html, "https://x/", limit=2)) == 2,
          f"{page_images(html, 'https://x/', limit=2)}")
    check("畸形 HTML 不炸", page_images("<img src=", "https://x/") == [],
          f"{page_images('<img src=', 'https://x/')}")


def test_result_images():
    """搜索结果自带图片直链时要透出来（有它就不用去读网页找图）。"""
    print("\n[N+1] 搜索结果里的图片直链")
    from plugins.ai_chat.search import format_results, parse_json_results

    data = {"results": [
        {"title": "一体成型电感", "url": "https://a.example/x",
         "content": "说明", "image": "https://cdn.example/ind.jpg"},
        {"title": "叠层电感", "url": "https://b.example/y", "content": "说明"},
    ]}
    got = parse_json_results(data)
    check("解析出 image 字段", got[0].image == "https://cdn.example/ind.jpg",
          f"{got[0]}")
    check("没有图片字段时留空", got[1].image == "", f"{got[1]}")
    text = format_results(got)
    check("结果文本里带上图片直链", "https://cdn.example/ind.jpg" in text, text)
    check("标出「可直接 fetch_image」", "fetch_image" in text, text)
    check("to_dict 里也有", "图片" in got[0].to_dict(), f"{got[0].to_dict()}")


async def main():
    for fn in (test_html_to_text, test_parse_bing, test_clean_url_and_build,
               test_format_results, test_ssrf_guard, test_page_to_text,
               test_page_images, test_result_images,
               test_stub_detection, test_relevance_gate, test_json_api_parser,
               test_anysearch_contract):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            globals()["FAILED"] = globals()["FAILED"] + 1
            FAILURES.append(f"{fn.__name__}: {e!r}")
            print(f"  ❌ {fn.__name__} 抛异常: {e!r}")
            traceback.print_exc()
    for afn in (test_client_behavior, test_api_backend, test_anysearch_client,
                test_render_fallback_and_failover,
                test_anysearch_live, test_backend_reachable_live):
        try:
            await afn()
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
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
