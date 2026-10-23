# 二十、为框架做的改动

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[插件包内的 import 纪律](19-imports.md) · 下一章：[与 DSH 的对应关系](21-dsh-mapping.md)


实现过程中发现若干框架层问题，都在 `napcat/` 内修掉了（不是插件 workaround）：

| 文件 | 改动 | 原因 |
|------|------|------|
| `napcat/command.py` | 新增 `CommandParser.strip_prefix()` / `skip_prefix()`；纯前缀（`/`、`//`）不再被当成命令名 | 旧实现把 `//` 解析成「名为 `/` 的命令」，用户每次发结束符都收到「未知命令: //」 |
| `napcat/command.py` | 命令名必须是合法标识符（`[a-zA-Z][\w-]*`），否则判 `illegal` | `//`、`/ /` 这类「前缀后跟分隔符」是垃圾输入，应静默丢弃 |
| `napcat/config.py` | 新增 `PluginConfig.all()` | 插件读整段配置原本只能 `get("")`，而它会返回 `None`（`get("__all__")` 也读不到），是个静默陷阱 |
| `napcat/plugin.py` | 插件生命周期改为**幂等**：`unload()` 强制解绑离场实例的监听器；`on_disable()` 只撤销「上次启用后新增的那一段」 | 热重载「新版本加载失败 → 回滚旧实例」会再调一次 `on_enable()`，而 `EventBus` **不按 hid 去重**，于是 `/events` 里同一个插件出现两对监听器、每条群消息被处理两次（实测每回滚一次多一份） |
| `napcat/command.py` | 新增 `/help` 可见性过滤器 `add_visibility_filter()` | 框架的可见性判据只有「scope + permission」；插件还有「本群是否开通服务」这类条件。没有钩子，插件只能自绘 help，于是同一份命令清单有两条渲染路径 |
| `napcat/command.py` | 命名空间下未知子命令返回「未知命令」，不再回落到父命令本体 | `/aichat help`（子命令已删）原本会执行注册子命令时自建的占位 handler，而它的 `lambda p: None` 签名与框架注入的 `event` 不匹配 → TypeError 被吞成「没反应」 |
| `napcat/config.py` | 新增 `PluginConfig.set_all()` | 补齐配置模板要写几十个键，逐键 `set()` 会每键全量重写一次 `config.json` |
| `napcat/config.py` | 新增 `ConfigManager.reload()` | 原本只在构造时读一次文件，「改了 config.json 不生效」无从解释；现在手工改文件可用 `/aichat reload --confirm=true` 显式重读 |
| `plugins/ai_chat/llm.py` | `LlmClient.reconfigure(*, reasoning_policy=...)` 走显式关键字 | 它在构造器里是从 `reasoning_effort` 派生的，不在初始字段里 —— 靠 `hasattr` 兜底的通用循环会把它静默丢掉（「改了配置但思考强度没变」） |

回归保护：`tests/test_command_parser.py` 钉住全部解析口径、`PluginConfig.all()` 与
`ConfigManager.reload()` 语义。

---
