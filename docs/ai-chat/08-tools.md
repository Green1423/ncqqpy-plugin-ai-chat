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

要接外部数据，用**配置驱动**的 `tools.custom`（见 8.2）：显式、可见、不改代码。
想查校内通知，直接在 QQ 里用 (../bupt-news/04-commands.md#四命令)；要让 AI 也能查，
就把它配成一个自定义 HTTP 工具。

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

## 别的插件提供的技能（命令映射）

别的插件可以用一行把自己的**命令**挂给模型：

```python
# 在 bupt_news / byrdocs 之类插件里
self.register_skill("search", command="byrdocs.search")
```

对使用者来说它就是普通工具，但执行方式不同 —— 走的是**命令系统**：

| 环节 | 行为 |
|---|---|
| 参数 | 由那条命令的 `CommandSchema` 翻译而来（不另写） |
| 权限 / 作用域 / 服务范围 | **与手敲命令完全一致**（AI 这条通道不绕过任何检查） |
| 回复 | 命令返回什么就发什么 |
| AI | **不接话**：这轮交给插件，模型不转述、也不接收命令输出 |

所以「说人话」和「敲命令」在权限上是等价的：没有权限的人在群里让 AI 去搜，
结果与他自己敲 `/byrdocs search` 一样 —— 用不了。

**硬拦（不靠提示词）**：`_wants_interpretation()` 只看用户**自己写的正文** ——
命中「解读/总结/翻译/分析…」且没有「查/搜/详情…」时，模型若去调技能工具，
`_model_loop` 直接**拒掉这次调用**（回一条「这是解读类请求，请自己回答」的工具结果），
逼它继续把答案写出来。提示词是建议，这里才是兜底。

**边界：技能只用来取数据。** 用户已经把内容给模型（引用/粘贴）并要它解读、
总结、翻译、改写、分析时，模型应当**自己回答**，不要转交 —— 转交之后这一轮就
结束了，插件把原始数据发出去，不会有人再解释它。这条规则写在工具通则里
（`compaction._TOOL_GUIDANCE_PLAIN`），提醒也写进了插件工具的描述。

关掉/收窄口径：

```json
"skills": {"enabled": true, "allow": [], "deny": ["byrdocs.search"]}
```

`ai_chat` 每轮重新解析技能表（命令被禁用/卸载后工具立刻消失），装/卸/热重载
插件**不用重启**。详见 [技能](https://github.com/Green1423/ncqqpy/blob/main/docs/plugin-dev/17-skills.md)。

### 插件工具的「使用时机」（`prompt_hint`）

工具 schema 的 `description` 只有模型主动读工具时才看得到；时机判断（「用户说搜题
就该存题库」）更该出现在**系统提示**里。所以插件工具可以带一句
`prompt_hint`，`ai_chat` 每轮把它拼进「可用能力」：

```text
插件工具的使用时机：
- 用户让你搜题、找题，或说「存进题库」时：先调 oj_bank_save_problem 存题，再回答。
```

工具通则里「插件工具默认只用来取数据」那条会让位给这里的**具体动作** ——
否则一个写操作类工具（存题库）永远等不到调用。

### 技能描述要写成「决策规则」，不是「功能简介」

模型的误判几乎都来自描述太笼统。给模型看的描述里必须包含**三件事**：

1. **触发条件**：`【仅在用户明确要『找/搜/查』某类通知时用】`；
2. **反例**（最容易错的那几种）：『现在呢』『可以吗』『你能读到吗』这类确认句不要用、
   正文已在上下文里不要再用、用户要「解读/总结」时不要用；
3. **与兄弟工具的分界**：`recent` 不返回正文、要正文用 `notice`。

实测（`s0028`）：用户连发三次「现在呢」，模型每次都用 `bupt_news_search` 反复搜 ——
描述里没有「什么时候**不**该用我」，它就会拿最像的那个工具硬套。
