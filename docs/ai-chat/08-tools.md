# 八、内置工具

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[思考强度（reasoning effort）](07-reasoning.md) · 下一章：[配置项全表](09-config.md)


| 工具 | 可用范围 | 说明 |
|------|---------|------|
| `get_current_time` | 所有 | 日期/时间/星期 |
| `get_session_info` | 所有 | 会话名、消息数、压缩次数 |
| `search_history` | 所有 | 在历史里按关键词做子串检索（带上下文） |
| `query_group_members` | 群聊 | 群成员，可按昵称/名片/QQ 过滤 |
| `query_group_info` | 群聊 | 群名、人数 |
| `mute_member` | 群聊 + 群主/管理员 | 禁言（`tools.max_mute_seconds` 封顶） |
| `list_group_bans` | 群聊 + 群主/管理员 | 当前禁言列表 |
| `web_search` | 所有（`search.enabled=true`） | 联网搜索：编号/标题/链接/摘要 |
| `fetch_page` | 同上 | 读取某个网页的正文（纯文本） |
| `plot_function` | 所有（`plot.enabled=true`） | 画一元函数图像（坐标轴 + 曲线），见(13-plot.md#十三函数图像plot_function) |
| `fetch_image` | 所有（`web_image.enabled=true` + `search.enabled=true`） | 从网上下载一张图片**配在回复里**发给用户，见(15-images.md#十五网图配图fetch_image) |

两条硬规则：

1. **可见即可用，但执行前再校验一次**：模型看到的清单按调用者权限过滤，真正执行时
   `ToolRegistry.call()` 会**再校验一遍** —— 提示词注入不该变成提权。
2. **工具失败不是对话失败**：任何异常都转成 `is_error=True` 的结果回喂给模型，
   由模型决定怎么向用户解释；直接把异常抛给用户是最差的选择。

## 8.1 与其它插件彻底解耦

**本插件不认识任何兄弟插件**：不 import 它们、不按文件路径加载它们、不读它们的
数据文件，也不为它们预留配置项。反之亦然（`bupt_news` 里没有一行提到 `aichat`）。

这里曾经有过一段耦合：内置工具会去读 `plugins/bupt_news/data/notices.jsonl` 拿校内
通知，并借 `bupt_news.weather` 的客户端报天气。那段已经**整块删除**，理由是它违反了
插件边界 ——

* 读的是别的插件的**私有数据文件**，对方改一次落盘格式就静默失效；
* 按文件路径加载对方模块，是为了绕开「不能在函数里做相对 import」才写的补丁，
  本身就是两层框架约束互相打架的产物；
* 对方的插件没装 / 加载失败 / 改过名，本插件的工具表就出现半可用状态。

要接外部数据，走**配置驱动**的路子（下一步的 `tools.custom`），那是显式的、可见的、
不改代码的。想查校内通知就直接在 QQ 里用 (../bupt-news/04-commands.md#四命令)；要让 AI 也能查，就把它配成一个
自定义 HTTP 工具。

> 老部署的 `config.json` 里可能还留着 `notices` / `weather` 两段（当年模板写进去的）。
> 插件启动时会**自动删掉**并记一条日志，不需要人工清理 —— 留着它们只会让人以为
> 这个插件还在读别人的数据。

## 8.2 自定义 HTTP 工具

不改代码就能挂工具（`tools.custom`）：

```jsonc
{
  "name": "campus_bus",
  "description": "查询校车时刻表",
  "parameters": {"type": "object",
                 "properties": {"line": {"type": "string"}},
                 "required": ["line"]},
  "method": "GET",
  "url": "https://example.edu/bus?line={line}",
  "headers": {"X-Token": "..."},
  "response_path": "data.items",
  "timeout": 15
}
```

`{参数名}` 会被替换且做百分号编码（模型给的参数不能借道注入查询串或请求头）；
只允许 http/https，强制超时。

---
