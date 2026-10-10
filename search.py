"""联网搜索：把「搜索引擎的 HTML」变成结构化的结果列表。

为什么自己解析 HTML 而不用 API：这台机器上 **只有 Bing 可达**（实测：
百度返回 1.5KB 反爬页、DuckDuckGo/Searx/维基全部超时/DNS 失败），而 Bing 的
公开搜索页不需要 API key。仓库里没有 bs4/lxml，所以用标准库 `html.parser`
—— 反正渲染 Markdown 也需要一个 HTML→文本的转换器，两处共用同一套实体解码。

**后端可替换**：`BACKENDS` 里每个解析器都是 `(html) -> [SearchResult]` 的纯函数，
搜索源换掉时只加一个函数 + 一条配置，不动工具层。解析器纯函数化也让它能用
**保存下来的真实 HTML** 做回归测试（见 `tests/test_search.py`），不依赖联网。
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Callable, Optional
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

#: Bing 结果块的起始标记。用它切块比 BeautifulSoup 式遍历稳 —— 只需要
#: 「块边界」这一个结构假设，块内部再用正则取 h2/p。
_BLOCK_SPLIT = re.compile(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', re.S)
_TITLE = re.compile(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", re.S)
_SNIPPET = re.compile(
    r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', re.S)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
#: Bing 有时把结果包在跳转链接里（`/ck/a?...&u=a1aHR0c...`），需要还原真实 URL
_CK_REDIRECT = re.compile(r"[?&]u=a1([A-Za-z0-9_\-]+)")


@dataclass
class SearchResult:
    """一条搜索结果。`url` 可能为空（Bing 偶尔只给标题）。

    `image`：结果自带的**图片直链**（有些搜索 API 会给）。有它就能直接
    `fetch_image`，省掉「读网页找图」那几轮 —— 实测模型会为了找图反复读页面，
    读到轮次用尽还一张没配上。
    """

    title: str
    url: str = ""
    snippet: str = ""
    image: str = ""

    def to_dict(self) -> dict:
        out = {"标题": self.title, "链接": self.url, "摘要": self.snippet}
        if self.image:
            out["图片"] = self.image
        return out


def html_to_text(raw: str) -> str:
    """HTML 片段 → 纯文本（去标签、解实体、压空白）。**共用**给渲染模块。"""
    text = _TAGS.sub(" ", str(raw or ""))
    text = unescape(text)
    text = text.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ")
    text = _WS.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _decode_bing_url(href: str) -> str:
    """还原 Bing 的跳转链接。

    Bing 有时把真实地址 base64url 编码在 `u=a1...` 里（不带 padding）。解不出来
    就原样返回 —— 宁可能点不开，也不要编一个错的地址。
    """
    href = unescape(str(href or "")).strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    m = _CK_REDIRECT.search(href)
    if not m:
        return href
    token = m.group(1)
    try:
        import base64
        pad = "=" * (-len(token) % 4)
        got = base64.urlsafe_b64decode(token + pad).decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - 解不出来不是错误，只是没有更好的链接
        return href
    return got if got.startswith("http") else href


def clean_url(url: str) -> str:
    """去掉跟踪参数，留下能直接打开的地址。"""
    url = unescape(str(url or "")).strip()
    if not url.startswith(("http://", "https://")):
        return url
    try:
        parts = urlparse(url)
    except ValueError:
        return url
    keep = [(k, v) for k, v in parse_qs(parts.query).items()
            if not k.lower().startswith(("utm_", "ref", "spm", "from"))
            and k.lower() not in ("fbclid", "gclid", "msclkid", "h", "id")]
    query = "&".join(f"{k}={quote_plus(v[0])}" for k, v in keep)
    return parts._replace(query=query, fragment="").geturl()


#: 空壳/拦截页的特征（Bing 在限流或判定为爬虫时返回几百字节的壳页）。
#: 实测：连着发请求时，同一个 URL 会从 100KB / 10 条结果掉到 ~900B / 0 条。
_STUB_HINTS = ("si_ST", "window.si_", "captcha", "验证", "unusual traffic")


def looks_like_stub(html: str) -> bool:
    """这个响应是不是搜索页的「空壳」（没结果、也没结果容器）。"""
    text = str(html or "")
    if len(text) > 20000:
        return False
    if 'class="b_algo"' in text or "b_results" in text:
        return False
    return any(hint in text for hint in _STUB_HINTS) or len(text) < 4000


def parse_bing(html: str, *, limit: int = 5) -> list[SearchResult]:
    """解析 Bing 搜索结果页。纯函数 —— 可以用保存下来的 HTML 直接测。"""
    out: list[SearchResult] = []
    for block in _BLOCK_SPLIT.findall(str(html or "")):
        m = _TITLE.search(block)
        if not m:
            continue
        url = clean_url(_decode_bing_url(m.group(1)))
        title = html_to_text(m.group(2))
        if not title:
            continue
        sm = _SNIPPET.search(block)
        snippet = html_to_text(sm.group(1)) if sm else ""
        out.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(out) >= max(1, limit):
            break
    return out


#: 360 搜索的结果块
_BLOCK_360 = re.compile(r'<li class="res-list"[^>]*>(.*?)</li>', re.S)
_TITLE_360 = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
_DESC_360 = re.compile(r'class="res-desc"[^>]*>(.*?)</p>', re.S)
_DESC_360_ALT = re.compile(r'class="res-rich[^"]*"[^>]*>(.*?)</div>', re.S)
_HREF_360 = re.compile(r'href="(https?://[^"]+)"')
_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</>", re.S | re.I)


def parse_360(html: str, *, limit: int = 5) -> list[SearchResult]:
    """解析 360 搜索（`so.com`）结果页。纯函数，可用保存的 HTML 直接测。

    360 会把真实地址包在 `so.com/link?m=...` 跳转里且**无法本地还原**，
    所以 `url` 可能是跳转链接 —— 对「给模型看标题+摘要」这个用途够用，
    真要读正文时模型拿到的是能打开的地址。"""
    out: list[SearchResult] = []
    for block in _BLOCK_360.findall(str(html or "")):
        m = _TITLE_360.search(block)
        if not m:
            continue
        title = html_to_text(_SCRIPT_STYLE.sub(" ", m.group(1)))
        if not title:
            continue
        href = _HREF_360.search(block)
        dm = _DESC_360.search(block) or _DESC_360_ALT.search(block)
        snippet = html_to_text(_SCRIPT_STYLE.sub(" ", dm.group(1))) if dm else ""
        out.append(SearchResult(title=title,
                                url=clean_url(href.group(1)) if href else "",
                                snippet=snippet))
        if len(out) >= max(1, limit):
            break
    return out


# ==================== JSON API 后端（可用 key / 自建实例） ====================
#
# 为什么要有这条路：抓公开搜索页**在有些网络里根本走不通** —— 实测这台机器上
# 百度返回「安全验证」、搜狗要验证码、知乎 403、360 给空壳、Bing 给一批无关结果。
# 所以提供一条**配置驱动**的正规通道：填一个 API（自己的 key 或自建 SearxNG），
# 就能拿到干净、稳定、可引用的结果。抓页面那条路保留为「没配 API 时的尽力而为」。

#: AnySearch 的检索领域（`tag` 用 `<领域>.<子域>`，如 `academic.search`、`code.snippet`）。
#:
#: 列在这里是为了**提前拦住拼错的 tag** 并给模型正确的选项 —— 拼错时上游会
#: 返回空结果，模型只会看到「没搜到」，很难自查。
ANYSEARCH_DOMAINS = (
    "academic", "agriculture", "business", "code", "energy", "environment",
    "film", "finance", "gaming", "general", "health", "ip", "legal",
    "resource", "security", "social_media", "travel",
)

#: AnySearch 的默认接入地址（匿名即可用，配了 key 就走认证、额度更高）
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"

#: JSON API 的默认取值路径（覆盖常见几家；可用配置覆盖）
_JSON_DEFAULTS = {
    # Tavily / Bocha / Serper(organic) / SearxNG / 通用
    "list_paths": ("results", "data.results", "data.webPages.value",
                   "webPages.value", "organic", "items", "data"),
    "title_keys": ("title", "name", "headline"),
    "url_keys": ("url", "link", "href", "sourceUrl"),
    "snippet_keys": ("content", "snippet", "description", "summary", "text"),
    # 图片直链：各家叫法不一，能认几个算几个（认不出也不影响别的字段）
    "image_keys": ("image", "image_url", "imageUrl", "img", "img_url",
                   "thumbnail", "thumbnail_url", "pic", "picture"),
}


def _dig(data, path: str):
    """按点号路径取值：`data.results` → `data['data']['results']`。"""
    node = data
    for part in str(path or "").split("."):
        if not part:
            continue
        if isinstance(node, dict):
            node = node.get(part)
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return None
    return node


def parse_json_results(data, *, limit: int = 5, list_path: str = "",
                       title_key: str = "", url_key: str = "",
                       snippet_key: str = "") -> list[SearchResult]:
    """从 JSON 响应里取结果。字段名各家不同，所以按**候选路径**逐个试。

    纯函数：可以直接拿一段 JSON 测，不需要联网。
    """
    lists = [list_path] if list_path else list(_JSON_DEFAULTS["list_paths"])
    rows = None
    for path in lists:
        got = _dig(data, path)
        if isinstance(got, list) and got:
            rows = got
            break
    if rows is None:
        return []
    titles = (title_key,) if title_key else _JSON_DEFAULTS["title_keys"]
    urls = (url_key,) if url_key else _JSON_DEFAULTS["url_keys"]
    snippets = (snippet_key,) if snippet_key else _JSON_DEFAULTS["snippet_keys"]
    images = _JSON_DEFAULTS["image_keys"]
    out: list[SearchResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = next((str(row[k]) for k in titles if row.get(k)), "")
        if not title:
            continue
        url = next((clean_url(str(row[k])) for k in urls if row.get(k)), "")
        snippet = next((str(row[k]) for k in snippets if row.get(k)), "")
        snippet = html_to_text(snippet) if "<" in snippet else snippet
        image = next((clean_url(str(row[k])) for k in images
                      if isinstance(row.get(k), str) and row.get(k)), "")
        out.append(SearchResult(title=title.strip(), url=url,
                                snippet=" ".join(snippet.split()),
                                image=image if image.startswith(("http://", "https://"))
                                else ""))
        if len(out) >= max(1, limit):
            break
    return out


#: 后端注册表：`名字 -> (搜索页 URL 模板, 解析器)`
#:
#: 两个后端是有意的：抓公开搜索页**一定会**遇到限流/改版，只挂一个的话
#: 「搜索功能」就等于「那个站点今天心情好不好」。谁先返回可用结果就用谁
#: （见 `WebClient.search`），两个都挂了才报错。
BACKENDS: dict[str, tuple[str, Callable[..., list[SearchResult]]]] = {
    "bing": ("https://cn.bing.com/search?q={query}&count={count}&setlang=zh-CN",
             parse_bing),
    "360": ("https://www.so.com/s?q={query}&pn=1", parse_360),
}

#: 默认尝试顺序。
#:
#: ⚠️ 360 放前面是**实测结论**，不是偏好：同一个查询「北京邮电大学 图书馆 国庆
#: 开放时间」，Bing 返回的是 Bilibili 与北京市政府页面（170KB、10 个结果块，
#: 看着完全正常），360 返回的是北邮图书馆的通知与简介。抓公开搜索页时
#: 「有结果」远不等于「有相关结果」，所以顺序按相关性实测结果排。
DEFAULT_BACKENDS = ("360", "bing")

DEFAULT_UA = ("Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def build_url(backend: str, query: str, count: int) -> str:
    template, _ = BACKENDS[backend]
    return template.format(query=quote_plus(str(query or "").strip()),
                           count=max(1, int(count)))


def parse(backend: str, html: str, *, limit: int = 5) -> list[SearchResult]:
    _, parser = BACKENDS[backend]
    return parser(html, limit=limit)


@dataclass
class SearchOutcome:
    """一次搜索的结果与失败原因（失败要能说给模型/用户听）。"""

    results: list[SearchResult] = field(default_factory=list)
    error: str = ""
    backend: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.results) and not self.error


#: 相关性判定里被忽略的高频虚词/泛化词（中文没有分词，只能靠这个 + 子串匹配）
_STOPWORDS = frozenset({
    "的", "了", "吗", "呢", "吧", "啊", "是", "在", "有", "和", "与", "及",
    "怎么", "如何", "什么", "为什么", "哪些", "哪个", "多少", "可以", "能否",
    "请问", "帮我", "一下", "以及", "还有", "就是", "这个", "那个",
    "the", "a", "an", "of", "to", "is", "are", "and", "or", "how", "what",
    "when", "where", "why", "which", "can", "does", "do", "for", "in", "on",
})


def query_terms(query: str) -> list[str]:
    """把查询拆成判定用的关键词。

    中文没有分词，所以：空格/标点先切一刀，再补「连续中文片段」这一个整体
    （宁可整体匹配，也不切出一堆单字 —— 单字会把「北京市」误判成命中「北邮」）。
    """
    text = str(query or "").strip().lower()
    if not text:
        return []
    words = [w for w in re.split(
        r"""[\s,，。、;；:：!！?？()（）\[\]【】"'“”]+""", text) if w]
    terms: list[str] = []
    for word in words:
        if word in _STOPWORDS or len(word) < 2:
            continue
        terms.append(word)
        # 中文片段本身就是一个整体词，不再往下切
    # 去掉被其它关键词包含的短词（"北京邮电大学" 已含 "北京" 时，后者不增加信息）
    out = []
    for term in sorted(set(terms), key=len, reverse=True):
        if not any(term in kept for kept in out):
            out.append(term)
    return out


def relevance(results: list[SearchResult], query: str) -> float:
    """结果里命中了多少比例的关键词（0.0 ~ 1.0）。

    为什么要它：搜索引擎会**返回一批与查询无关的结果**而不是空结果 ——
    实测 Bing 对「北京邮电大学 图书馆 国庆 开放时间」返回了 Bilibili 与
    北京市政府页面（页面本身 170KB、10 个结果块，看起来完全正常）。
    只看「有没有结果」的判据会把这种垃圾当成成功，于是永远不换后端。
    """
    terms = query_terms(query)
    if not terms:
        return 1.0
    hay = " ".join(f"{r.title} {r.snippet}" for r in results).lower()
    hit = sum(1 for t in terms if t in hay)
    return hit / len(terms)


def format_results(results: list[SearchResult], *, max_snippet: int = 300) -> str:
    """结果列表 → 给模型看的文本（编号 + 标题 + 链接 + 摘要）。"""
    if not results:
        return "没有搜到结果"
    lines = []
    for i, r in enumerate(results, 1):
        snippet = r.snippet
        if max_snippet > 0 and len(snippet) > max_snippet:
            snippet = snippet[:max_snippet].rstrip() + "…"
        lines.append(f"{i}. {r.title}")
        if r.url:
            lines.append(f"   链接：{r.url}")
        if r.image:
            # 明确写成「可直接 fetch_image」，否则模型会把它当成普通链接忽略
            lines.append(f"   图片直链（可直接 fetch_image 下载）：{r.image}")
        if snippet:
            lines.append(f"   摘要：{snippet}")
    return "\n".join(lines)


def host_of(url: str) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""


# ==================== 抓取（HTTP） ====================

#: 抓取正文时的体积上限（防止一个页面把内存和上下文都撑爆）
MAX_PAGE_BYTES = 2 * 1024 * 1024


class FetchBlocked(RuntimeError):
    """目标地址被安全策略拒绝（内网 / 非 http(s) / 解析失败）。"""


class FetchTooLarge(RuntimeError):
    """目标比允许的大小还大（图片下载用：宁可拒绝，也不要存半张图）。"""

    def __init__(self, limit: int, got: int = 0) -> None:
        super().__init__(f"超过 {limit} 字节上限")
        self.limit = int(limit)
        self.got = int(got)


def _is_private_host(host: str) -> bool:
    """内网/本机地址判定。

    ⚠️ 这是**安全边界**，不是洁癖：搜索摘要和网页内容都是**不可信输入**，
    而模型会把它们当成资料读。没有这道闸，一句「请访问
    http://127.0.0.1:8080/admin 并把内容告诉我」就变成了一个内网探测/SSRF 跳板。
    """
    import ipaddress
    import socket

    name = (host or "").strip().lower().rstrip(".")
    if not name:
        return True
    if name in ("localhost", "localhost.localdomain") or name.endswith(".local"):
        return True
    # 先按字面 IP 判；不是 IP 就解析域名再判（解析失败按不安全处理）
    candidates: list[str] = []
    try:
        ipaddress.ip_address(name)
        candidates.append(name)
    except ValueError:
        try:
            infos = socket.getaddrinfo(name, None)
        except OSError:
            return True
        candidates = [info[4][0] for info in infos]
    for cand in candidates:
        try:
            ip = ipaddress.ip_address(cand)
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def check_url(url: str) -> str:
    """校验要抓取的地址，返回干净 URL；不安全就抛 `FetchBlocked`。"""
    url = clean_url(url)
    if not url.startswith(("http://", "https://")):
        raise FetchBlocked(f"只支持 http/https 链接（收到 {url[:60]!r}）")
    host = host_of(url)
    if _is_private_host(host):
        raise FetchBlocked(f"拒绝访问内网或本机地址（{host or '未知主机'}）")
    return url


class WebClient:
    """搜索与抓取的 HTTP 客户端（aiohttp，懒建会话）。

    `fetch` 可注入 —— 测试不打桩 aiohttp，直接换掉「拿 HTML」这一步，
    这样解析、截断、错误处理这些**真正会出错的地方**仍然是真代码在跑。
    """

    def __init__(self, *, timeout: float = 15.0, backend: Optional[str] = None,
                 backends: Optional[list] = None,
                 api_url: str = "", api_key: str = "", api_method: str = "POST",
                 api_tag: str = "", api_fields: Optional[dict] = None,
                 user_agent: str = DEFAULT_UA, logger=None,
                 fetch: Optional[Callable] = None,
                 fetch_json: Optional[Callable] = None) -> None:
        self.timeout = float(timeout)
        self.user_agent = user_agent or DEFAULT_UA
        self.logger = logger
        self._injected_fetch = fetch
        self._injected_fetch_json = fetch_json
        # JSON API（可选的「正规通道」）：配了它就先走它
        self.api_url = str(api_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.api_method = (str(api_method or "POST").strip().upper() or "POST")
        self.api_fields = dict(api_fields or {})
        self.api_tag = str(api_tag or "").strip()
        self._session = None
        self._chrome: Optional[str] = None
        # 后端顺序：配置里写的放前面，其余默认后端按序兜底。
        # ⚠️ 「配置写了一个后端」不等于「只用它」—— 抓搜索页必然遇到限流，
        # 只认一个后端会让搜索功能随对方的风控起伏（实测被 cn.bing 限流过）。
        wanted = [str(b) for b in (backends or []) if str(b) in BACKENDS]
        # `backend=None` 表示「用默认顺序」，不再硬塞一个首选 —— 否则
        # `DEFAULT_BACKENDS` 的顺序永远被构造默认值覆盖（实测踩过：
        # 把 360 调到第一位后，`WebClient().backends` 仍是 ['bing','360']）。
        if backend and backend in BACKENDS and backend not in wanted:
            wanted.insert(0, backend)
        # ⚠️ 显式给了 `backends` 就**只**用它：再追加默认后端的话，
        # 「只指定一个后端」这种隔离（测试、排障）会失效 —— 实测表现是
        # 指定 `backends=["bing"]` 却仍然去打了 360，断言拿到另一个后端的错。
        if backends is None:
            for name in DEFAULT_BACKENDS:
                if name not in wanted:
                    wanted.append(name)
        self.backends = wanted or list(DEFAULT_BACKENDS)
        self.backend = self.backends[0]      # 首选（日志/展示用）
        #: 抓网页时的相关性下限：低于它就算「这个后端没给出有用结果」，继续换后端。
        #: 0.34 是实测折中 —— 抓来的页面经常整批跑偏（实测 Bing 对北邮图书馆的
        #: 查询返回 Bilibili 与北京市政府页面），必须卡住。
        self.min_score = 0.34
        #: 搜索 **API** 的相关性下限默认 **-1 = 不设限**，只把它当安全网。
        #:
        #: 为什么不卡：API 是**排序式**服务，返回的就是它认为最相关的结果，而
        #: 精确子串匹配对中文太苛刻 —— 实测两个都被误杀：
        #: `北邮` vs 结果里的「北京邮电大学」（同实体、无公共子串）；
        #: `今天北京天气` vs 结果里的「北京天气预报」（无整串）。
        #: 把结果交给模型判断，比用一个词法规则替它判断更准。
        #: 想恢复严格模式就把它设成 0.34（与抓网页同一口径）。
        self.api_min_score = -1.0

    async def _ensure_session(self):
        if self._injected_fetch is not None:
            return None
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout, connect=10),
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                })
        return self._session

    async def get(self, url: str, *, max_bytes: int = MAX_PAGE_BYTES) -> str:
        """取一个 URL 的文本内容。失败抛异常（由调用方转成人话）。"""
        if max_bytes <= 0:
            max_bytes = 1
        if self._injected_fetch is not None:
            return await self._injected_fetch(url, max_bytes=max_bytes)
        session = await self._ensure_session()
        async with session.get(url, allow_redirects=True,
                               max_redirects=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            raw = await resp.content.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raw = raw[:max_bytes]
            ctype = str(resp.headers.get("Content-Type") or "")
            return _decode_body(raw, ctype)

    async def get_bytes(self, url: str, *, max_bytes: int,
                        accept: str = "image/*,*/*;q=0.8") -> tuple[bytes, str]:
        """取一个 URL 的**原始字节**（下载图片用），返回 `(数据, Content-Type)`。

        与 `get()` 有两处刻意的不同，都是为「URL 来自模型、而模型读过不可信网页」
        这个前提准备的：

        * **自己跟跳转，每一跳都重查 SSRF 闸** —— `get()` 用 aiohttp 的自动跳转，
          跳转目标不过闸：一个公网地址 302 到 `http://169.254.169.254/…`
          就绕过去了（云元数据端点就是这么被打的）。这里最多跟 5 跳，每跳
          `check_url()`；
        * 超限**直接报错**而不是截断：半张图发出去比不发更糟（调用方按这个
          区分「太大」和「下载失败」）。

        测试注入 `_injected_fetch` 时返回值可以是 `bytes`（喂假图片）或 `str`
        （喂网页），`str` 按 utf-8 编回字节。
        """
        if max_bytes <= 0:
            max_bytes = 1
        if self._injected_fetch is not None:
            got = await self._injected_fetch(url, max_bytes=max_bytes)
            # 注入的东西可能是 bytes（测图片这类二进制）或 str（测网页）
            if isinstance(got, (bytes, bytearray)):
                return bytes(got), ""
            return str(got).encode("utf-8"), ""
        session = await self._ensure_session()
        target = check_url(url)
        for _hop in range(6):
            async with session.get(target, allow_redirects=False,
                                   headers={"Accept": accept}) as resp:
                if resp.status in (301, 302, 303, 307, 308):
                    loc = str(resp.headers.get("Location") or "").strip()
                    if not loc:
                        raise RuntimeError(f"HTTP {resp.status} 但没给 Location")
                    target = check_url(urljoin(target, loc))
                    continue
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                raw = await resp.content.read(max_bytes + 1)
                ctype = str(resp.headers.get("Content-Type") or "")
                if len(raw) > max_bytes:
                    raise FetchTooLarge(max_bytes, len(raw))
                return raw, ctype
        raise FetchBlocked("跳转次数过多（超过 5 跳）")

    async def rendered(self, url: str, *, wait_ms: int = 4000) -> str:
        """用无头 Chromium 打开页面并取回**渲染后的 DOM**。

        为什么需要这条兜底：搜索引擎对普通 HTTP 客户端会返回「JS 空壳页」
        （实测：aiohttp 拿到 592 字节、`b_algo` 一个都没有，而同一个 URL 用
        Chromium 渲染能拿到 10 条完整结果）。我们已经为了出图依赖 Chromium，
        所以这里复用它 —— 比再引一个搜索 API 依赖便宜。

        代价（所以只在必要时用）：一次几百 MB 的浏览器进程、几秒延迟。
        """
        import shutil
        import tempfile
        chrome = self._chrome or shutil.which("chromium") or shutil.which(
            "chromium-browser") or shutil.which("google-chrome")
        if not chrome:
            raise RuntimeError("没有可用的 Chromium，无法渲染搜索页")
        self._chrome = chrome
        profile = tempfile.mkdtemp(prefix="search-cr-")
        try:
            proc = await asyncio.create_subprocess_exec(
                chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
                "--disable-dev-shm-usage", "--no-first-run",
                "--no-default-browser-check", "--disable-extensions",
                "--disable-background-networking",
                "--user-data-dir=" + profile,
                f"--virtual-time-budget={int(wait_ms)}", "--dump-dom", url,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                out, _err = await asyncio.wait_for(
                    proc.communicate(), timeout=float(self.timeout) + 20)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
                raise RuntimeError("渲染搜索页超时")
            return (out or b"").decode("utf-8", "replace")
        finally:
            import shutil as _sh
            _sh.rmtree(profile, ignore_errors=True)

    async def search(self, query: str, *, count: int = 5) -> SearchOutcome:
        """搜一次，返回结果或**可展示的**失败原因。"""
        query = str(query or "").strip()
        if not query:
            return SearchOutcome(error="请给出搜索关键词", backend=self.backend)
        last_err = ""
        # 按后端顺序试：每个后端给自己一次重试（限流返回空壳页时，隔一下再来
        # 通常就好）。全部失败才报错 —— 这是「搜索功能」而不是「某个站点的
        # 可用性探测」，一个后端挂了不该让整个功能挂掉。
        # ⓪ 配了 JSON API 就先走它（干净、稳定、可引用；抓页面是尽力而为）
        if self.api_url:
            outcome, why = await self._search_api(query, count)
            if outcome is not None:
                return outcome
            last_err = why
            if self.logger:
                self.logger.info("[ai_chat] JSON API 未返回可用结果：%s", why)

        for name in self.backends:
            url = build_url(name, query, count)
            # ① 普通 HTTP：快（几百毫秒）。搜索引擎多数时候会给完整页面。
            html, err = await self._fetch(url, name)
            if html:
                outcome, why = self._judge(name, html, query, count)
                if outcome is not None:
                    return outcome
                last_err = why
            elif err:
                last_err = err
            # ② 无头浏览器重试：**无条件**走这一步，不看「像不像空壳页」。
            # 实测 360 对普通 HTTP 返回的是**58KB 的正常页面、但里面一个结果都没有**
            # （结果全靠 JS 填），任何基于体积/特征串的空壳判定都会漏掉它 ——
            # 于是浏览器兜底永远不触发，用户看到的就是「搜不了」。
            rendered = await self._render(url, name)
            if rendered:
                outcome, why = self._judge(name, rendered, query, count)
                if outcome is not None:
                    if self.logger:
                        self.logger.info("[ai_chat] %s 经浏览器渲染后取得结果", name)
                    return outcome
                last_err = why
            if self.logger:
                self.logger.info("[ai_chat] 搜索后端 %s 未返回可用结果，换下一个",
                                 name)
        return SearchOutcome(error=last_err, backend=self.backend)

    def _tag_error(self) -> str:
        """`api_tag` 写错时给一句可操作的说明（拼错 tag 上游只会返回空结果）。"""
        tag = self.api_tag
        if not tag:
            return ""
        head = tag.split(".", 1)[0]
        if head in ANYSEARCH_DOMAINS:
            return ""
        return (f"搜索配置里的垂直领域 {tag!r} 不认识（可用："
                f"{'、'.join(ANYSEARCH_DOMAINS[:8])} 等 {len(ANYSEARCH_DOMAINS)} 个）")

    async def _search_api(self, query: str, count: int):
        """走配置好的 JSON API。返回 `(SearchOutcome 或 None, 失败原因)`。"""
        bad_tag = self._tag_error()
        if bad_tag:
            return None, bad_tag
        if self._injected_fetch_json is not None:
            try:
                data = await self._injected_fetch_json(self.api_url, query, count,
                                                       self.api_key)
            except Exception as e:  # noqa: BLE001
                return None, f"搜索 API 请求失败：{_friendly_error(e)}"
        else:
            try:
                data = await self._request_api(query, count)
            except Exception as e:  # noqa: BLE001
                return None, f"搜索 API 请求失败：{_friendly_error(e)}"
        # 上游自带状态码：`code != 0` 是服务端明确报错，此时即便有 results
        # 也不该当成成功（信息可能不完整），直接把 message 带出去。
        if isinstance(data, dict):
            code = data.get("code")
            if code not in (None, 0, "0"):
                msg = str(data.get("message") or data.get("msg") or code)
                return None, f"搜索 API 报错：{msg[:120]}"
        results = parse_json_results(
            data, limit=count,
            list_path=str(self.api_fields.get("list_path") or ""),
            title_key=str(self.api_fields.get("title_key") or ""),
            url_key=str(self.api_fields.get("url_key") or ""),
            snippet_key=str(self.api_fields.get("snippet_key") or ""))
        if not results:
            return None, "搜索 API 没有返回结果（检查 api_url / 字段路径配置）"
        score = relevance(results, query)
        if score < self.api_min_score:
            return None, f"搜索 API 的结果与查询无关（关键词命中率 {score:.0%}）"
        return SearchOutcome(results=results, backend="api"), ""

    async def _request_api(self, query: str, count: int):
        """真正发一次 API 请求（aiohttp）。支持 POST（默认）与 GET。

        请求体同时给 `query/max_results` 与几家常见的等价字段名 —— 服务端会忽略
        不认识的键，这样同一份代码既能接 AnySearch，也能接 Tavily / SearxNG。
        """
        import aiohttp
        body = {"query": query, "count": count, "max_results": count,
                "num": count, "limit": count}
        if self.api_tag:
            body["tag"] = self.api_tag
        for key in ("zone", "language"):
            value = str(self.api_fields.get(key) or "").strip()
            if value:
                body[key] = value
        headers = {"Accept": "application/json",
                   "User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # ⚠️ 这个会话**必须**用 async with 收口：借用实例上那个长命会话的话，
        # 它的 header 与超时是给抓页面用的，且失败路径容易漏关（实测报
        # `Unclosed client session`）。
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout,
                                         headers=headers) as sess:
            if self.api_method == "GET":
                async with sess.get(self.api_url,
                                    params={"q": query, "query": query,
                                            "count": count}) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    return await resp.json(content_type=None)
            async with sess.post(self.api_url, json=body) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                return await resp.json(content_type=None)

    def _judge(self, name: str, html: str, query: str,
               count: int):
        """解析 + 相关性判定。返回 `(SearchOutcome 或 None, 失败原因)`。"""
        results = parse(name, html, limit=count)
        if not results:
            if looks_like_stub(html):
                return None, f"{name} 被限流或挡回了（返回的是空壳页）"
            return None, f"{name} 没有解析出结果（搜索页结构可能变了）"
        score = relevance(results, query)
        if score >= self.min_score:
            return SearchOutcome(results=results, backend=name), ""
        return None, (f"{name} 返回的结果与查询无关"
                      f"（关键词命中率 {score:.0%}）")

    async def _fetch(self, url: str, name: str):
        """普通 HTTP 取页面。返回 `(html, 错误说明)`，两者最多一个非空。"""
        try:
            return await self.get(url), ""
        except Exception as e:  # noqa: BLE001 - 网络失败不该炸穿一轮对话
            err = _friendly_error(e)
            if self.logger:
                self.logger.info("[ai_chat] %s 普通请求失败: %s", name, err)
            return "", err

    async def _render(self, url: str, name: str):
        """用无头浏览器取渲染后的 DOM；失败返回空串（不抛）。"""
        try:
            return await self.rendered(url)
        except Exception as e:  # noqa: BLE001 - 渲染失败就换后端
            if self.logger:
                self.logger.info("[ai_chat] 渲染 %s 失败: %s", name, e)
            return ""

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def read_page(self, url: str, *, limit: int = 4000,
                        max_images: int = 0) -> str:
        """抓一个网页、转成纯文本，返回**可直接回喂给模型**的字符串。

        `max_images > 0` 时在正文后面附上**页面里的图片直链** —— 这是「找图」
        唯一可行的路：纯文本里看不到 `<img>`，模型会为了找图反复读页面，
        读到轮次用尽还一张配不上（线上实测）。

        失败一律转成一句人话而不是抛异常（工具失败不该炸穿整轮对话）。
        """
        try:
            safe = check_url(url)
        except FetchBlocked as e:
            return f"读取失败：{e}"
        try:
            raw = await self.get(safe)
        except Exception as e:  # noqa: BLE001
            return f"读取失败：{_friendly_error(e)}"
        text = page_to_text(raw)
        images = page_images(raw, safe, limit=max_images) if max_images else []
        if not text and not images:
            return "读取成功，但页面没有可读的正文（可能是纯脚本页或需要登录）"
        limit = max(200, int(limit or 4000))
        if len(text) > limit:
            text = text[:limit].rstrip() + "\n…（正文过长已截断）"
        out = f"来源：{safe}\n\n{text}"
        if max_images:
            if images:
                listing = "\n".join(f"{i}. {u}" for i, u in enumerate(images, 1))
                out += ("\n\n——\n页面里的图片直链"
                        "（挑真正相关的调 fetch_image 下载，别全都下载）：\n"
                        + listing)
            else:
                out += "\n\n——\n（这个页面里没有找到可下载的图片直链）"
        return out


#: 正文提取时丢弃的元素（导航、脚本、样式等噪音）
_DROP_TAGS = frozenset({"script", "style", "noscript", "template", "svg",
                        "head", "nav", "footer", "form", "iframe"})
#: 块级元素 —— 收尾时补换行，避免所有文字挤成一行
_BLOCK_TAGS = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
                         "h5", "h6", "section", "article", "blockquote", "pre"})


#: 明显不是「内容图」的 URL 片段：图标、占位图、像素点、站点装饰。
#: 实测（mct8 / bcdcn 两个元件站）不过滤的话，清单第一条往往是站标或横幅 ——
#: 模型很可能就把「公司 logo」当成用户要的「电感封装图」发出去。
_JUNK_IN_URL = ("spacer", "pixel", "blank.gif", "1x1", "transparent",
                "loading", "placeholder", "/icon", "icon_", "avatar",
                "logo", "banner", "brand", "sprite", "header", "footer",
                "/nav", "/bg", "bg_", "watermark", "qrcode", "erweima")

#: 能从 `<img>` 上取地址的属性。`data-src` / `data-original` 是懒加载的常见写法
#: （电商和图片站几乎都这么写，只认 `src` 会拿到占位图）。
_IMG_ATTRS = ("src", "data-src", "data-original", "data-lazy-src", "data-echo")


def page_images(raw_html: str, base_url: str = "", *, limit: int = 8) -> list[str]:
    """HTML 里的**图片直链**（绝对 URL，去重、丢垃圾），最多 `limit` 条。

    为什么需要它：模型找不到图片直链时，会**反复读网页**找图 —— 线上实测
    （用户问「给出这几种封装的图片」）它连读了 6 轮页面，一张图没配上、
    一个字没说就撞上轮次上限。而 `fetch_page` 只回纯文本，页面里的
    `<img>` 对它完全不可见。这个函数就是补上那条信息。

    纯函数（不联网），所以能直接喂一段 HTML 做回归。
    """
    from html.parser import HTMLParser

    class _Pick(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.srcs: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag != "img":
                return
            got = {str(k).lower(): (v or "") for k, v in attrs}
            for key in _IMG_ATTRS:
                if got.get(key):
                    self.srcs.append(got[key])
                    break

        # 有些站点用自闭合写法；HTMLParser 会走 handle_startendtag
        handle_startendtag = handle_starttag

    parser = _Pick()
    try:
        parser.feed(str(raw_html or ""))
        parser.close()
    except Exception:  # noqa: BLE001 - 畸形 HTML 不该让抓取失败
        pass

    out: list[str] = []
    seen: set[str] = set()
    for src in parser.srcs:
        src = str(src or "").strip()
        if not src or src.startswith(("data:", "javascript:", "blob:", "#")):
            continue
        url = urljoin(base_url, src) if base_url else src
        if not url.startswith(("http://", "https://")):
            continue
        low = url.lower().split("?")[0]
        if low.endswith(".svg"):            # QQ 显示不了矢量图
            continue
        if any(junk in url.lower() for junk in _JUNK_IN_URL):
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max(1, int(limit)):
            break
    return out


def page_to_text(raw_html: str) -> str:
    """HTML → 可读正文（去脚本/样式/导航，保留段落换行）。

    用标准库 `html.parser`（仓库里没有 bs4/lxml）。它不是完整浏览器，
    但对「把网页读给模型听」这个用途足够：丢掉标签，保留文字与段落。
    """
    from html.parser import HTMLParser

    class _Extract(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in _DROP_TAGS:
                self.skip += 1
            elif tag in _BLOCK_TAGS:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in _DROP_TAGS and self.skip:
                self.skip -= 1
            elif tag in _BLOCK_TAGS:
                self.parts.append("\n")

        def handle_data(self, data):
            if not self.skip and data:
                self.parts.append(data)

    parser = _Extract()
    try:
        parser.feed(str(raw_html or ""))
        parser.close()
    except Exception:  # noqa: BLE001 - 畸形 HTML 不该让抓取失败
        pass
    text = "".join(parser.parts)
    lines = [_WS.sub(" ", line).strip() for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out).strip()


def _decode_body(raw: bytes, ctype: str) -> str:
    """按 Content-Type / meta 猜编码，猜不到就 utf-8 兜底（不抛）。"""
    charset = ""
    m = re.search(r"charset=([\w\-]+)", ctype or "", re.I)
    if m:
        charset = m.group(1)
    if not charset:
        head = raw[:2048].decode("ascii", "ignore")
        m = re.search(r'charset=["\']?([\w\-]+)', head, re.I)
        if m:
            charset = m.group(1)
    for enc in (charset, "utf-8", "gb18030", "latin-1"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def _friendly_error(e: Exception) -> str:
    """把网络异常转成一句人话（不把 traceback 丢给用户）。"""
    name = type(e).__name__
    text = str(e) or name
    if "Timeout" in name or "timeout" in text.lower():
        return "搜索超时（网络慢或搜索引擎不可达）"
    if "Name or service not known" in text or "DNS" in text.upper():
        return "搜索失败：域名解析不了（可能没有外网 DNS）"
    if "Cannot connect" in text or "Connect" in name:
        return f"搜索失败：连不上搜索引擎（{text[:80]}）"
    return f"搜索失败：{text[:120]}"
