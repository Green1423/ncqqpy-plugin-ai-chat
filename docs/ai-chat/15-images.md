# 十五、网图配图（`fetch_image`）

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[`/md`：把上一条 AI 回复转成图片](14-md-image.md) · 下一章：[结束符](16-terminator.md)


用户说「来几张 xx 的图」，模型能搜、能读网页，但**发不出图** —— 它在正文里写
Markdown 图片或直接贴链接：

```text
![](url)
```

QQ 里既不显示图片、用户还得自己点开 —— `fetch_image` 补的就是这一步：

```text
fetch_image(url="https://…/cat.png", caption="一只橘猫")
```

链路与函数图像**完全同构**（工具产出文件 → 在 `rendered` 白名单里记一笔 →
主流程从**工具结果**里取 `[[plot:…]]` 标记 → 图作为**单独一条消息**发出）：

| 环节 | 位置 |
|---|---|
| 下载 | `search.WebClient.get_bytes()`（复用搜索那套 HTTP 客户端） |
| 校验 + 落盘 | `webimg.WebImageService.fetch()` |
| 工具 | `tools.register_image_tool()` |
| 取图 + 发送 | `main._plots_from_tools()` / `_send_plots()`（与函数图共用） |

## 15.1 四条安全边界（都是真会被利用的路径）

URL 来自**模型**，而模型刚读过不可信的网页与搜索摘要 —— 所以这里的输入等同于
「陌生人给的地址」：

1. **只允许公网 http/https**：复用 `search.check_url()` 的 SSRF 闸，而且
   **每一跳跳转都重查**。`get()` 走 aiohttp 的自动跳转（跳转目标不过闸），
   一个公网地址 302 到 `http://169.254.169.254/…` 就绕过去了 —— `get_bytes()`
   因此自己跟跳转（最多 5 跳，每跳 `check_url()`）；
2. **必须是真图片**：按**魔数**判定（PNG/JPEG/GIF/WebP/BMP），**不看后缀、不信
   `Content-Type`** —— 否则「下载这个 .png」就能把任意二进制塞进群里。不是图片
   会明确回绝，并提示「要读网页正文请用 fetch_page」；
3. **只有本插件下载过的文件才发得出去**：`rendered` 是白名单，模型自己编一个
   `[[plot:/etc/passwd]]` 在 `_plots_from_tools()` 里就被挡掉（与 `PlotService`
   同一条规矩）；
4. **文件名按 URL 哈希**（`sha1(url)[:20].<ext>`）：远程文件名是不可信输入，
   `../../etc/x` 这类名字进不了本地路径；同时让同一条链接天然去重。

另有三条工程约束：单张 `max_bytes`（默认 8MB，**超限直接拒绝而不是截断** ——
半张图发出去比不发更糟）、一轮 `max_per_turn`（默认 3 张）、目录 `keep_files`
（默认只留最近 60 个文件，防止 SD 卡被写满）。

## 15.2 配置

```jsonc
"web_image": {
  "enabled": true,
  "max_bytes": 8388608,   // 单张上限，超过就拒绝（不发半张图）
  "max_per_turn": 4,      // 一轮最多配几张（与 reply.chunk_max_images 对齐）
  "keep_files": 60,       // 落盘目录只留最近多少个文件
  "page_images": 6        // fetch_page 顺带列出几条页面图片直链
}
```

> 依赖 `search.enabled`：下载复用搜索的 HTTP 客户端与 SSRF 闸。关掉搜索
> （或 `web_image.enabled=false`）时**工具与提示词一起消失** —— 讲一个模型调不到
> 的工具，只会让它空转一轮。

## 15.2.1 模型**怎么找到**图片直链（这条最容易做错）

工具能下载不等于用得上 —— 模型得先拿到**直链**。第一版没管这件事，线上实测
（用户问「给出这几种封装的图片」）表现是：

```text
22:31:28  web_search ×3        → 只拿到一堆网页链接
22:32:05  fetch_page ×3        → 纯文本里看不到 <img>，一无所获
22:32:36  web_search ×2        → 再来一轮
22:32:53  fetch_page + search  → 再来一轮
22:33:13  fetch_page ×2        → 知乎 403、期刊页无关
22:33:27  web_search ×2        → 第 6 轮，轮次用尽
```
用户最后只收到一句 **「（已达单轮上限 6 次模型调用，先说到这里）」**，一张图没配上、
一个字没说。两个根因，都补掉了：

1. **纯文本里没有图片** → 两条信息通道：
   * `fetch_page` 现在会在正文后面附上**页面里的图片直链**
     （`search.page_images()`：认 `src` 与懒加载的 `data-src`/`data-original`，
     相对路径用 `urljoin` 补全，丢掉 `data:`/`svg`/占位图/图标，去重、限量）；
   * 搜索 API 若自带图片字段（`image`/`thumbnail`/…）会在结果里直接写成
     「图片直链（可直接 fetch_image 下载）：…」（`SearchResult.image`）。
2. **轮次全被工具吃掉** → `_model_loop` 的**最后一轮不带工具**，并在系统提示里
   明说「工具次数已用尽，请直接用已有信息回答」。这样无论模型怎么绕，用户至少
   能拿到一句真话（哪部分没查到），而不是只看到「已达上限」。

提示词里也把路径写死了：**读两三个页面还没有直链就直接说清楚**，别再读页面。

## 15.3 提示词怎么交代

`compaction._IMAGE_GUIDANCE` 只在能力开启时追加，四条：① 要图时用**直链**（先搜，
拿到网页用 `fetch_page` 找直链）；② 图会**单独一条消息**发，正文里**不要**再写
`![]()`/`<img>`/裸链接（QQ 显示不出来），用一两句话说明这张图是什么；
③ 只配真有帮助的图，一轮最多几张；④ 失败（太大/不是图片/链接失效/内网地址）
如实说明原因，可换链接但别反复重试同一个；⑤ 别用它下网页（那是 `fetch_page`）
或文件（`.zip/.pdf/.exe` 一律被拒）。

## 15.4 测试

* `tests/test_webimg.py`（48 项）：魔数判定、SSRF 闸、跳转逐跳过闸、超限报错、
  一轮张数、同 URL 去重、目录有界、白名单；
* `tests/test_ai_chat.py::test_fetch_image_tool`：端到端 —— 工具出现在模型看到的
  清单里、调用后图片真的作为单独一条消息发出、系统提示里讲了怎么用、关掉开关
  后工具不再注册。
