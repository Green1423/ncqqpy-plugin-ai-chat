# 十九、插件包内的 import 纪律

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[测试](18-tests.md) · 下一章：[为框架做的改动](20-framework-changes.md)


这一节是踩坑记录，改 `plugins/ai_chat/` 之前请先读。

## 19.1 不许有「函数内相对 import」

插件是被 `spec_from_file_location("napcat_plugin_ai_chat", …)` 加载的，
包名是合成出来的。热重载时 `PluginManager.reload()` 会把
`napcat_plugin_ai_chat*` **整棵子树**从 `sys.modules` 摘掉再重新导入。
此时如果一个函数**在调用时**才去 `from .llm import err_text`，Python 会重新解析
父包 —— 拿到：

```
ModuleNotFoundError: No module named 'napcat_plugin_ai_chat'
```

而它发生在**一轮对话中间**（工具执行 / 异常处理里），用户只看到
`⚠️ AI 对话出错（E-xxxx）：ModuleNotFoundError…`，完全看不出是热重载引起的。

**规则**：包内 import 一律写在模块顶层。顶层写完，模块加载时解析一次并绑定到
模块全局，之后函数调用只做属性查找，不再碰 `sys.modules`。

回归保护：`tests/test_ai_chat.py` 的 `[30]` 用例会把包从 `sys.modules` 摘掉后
跑完一整轮（含工具失败 + 压缩），并静态扫描源码禁止函数内相对 import。

## 19.2 不许跨插件加载（历史教训）

本插件曾经有一个 `_load_sibling_module("bupt_news", "weather")`：按文件路径去加载
**兄弟插件的子模块**。它存在的原因是相对 import 在任何加载方式下都不成立 ——

* 本插件叫 `napcat_plugin_ai_chat`，`..bupt_news` 会被解析成
  `napcat_plugin_ai_chat.bupt_news`（不存在）；
* `plugins/` 目录**没有** `__init__.py`，`plugins.ai_chat` 也不是正规包。

**这个补丁连同它服务的天气工具一起删掉了**，因为它是在跟框架边界对抗：绕开 import
系统去别人的包里取代码，等于把「两个插件」变成了「一个插件的两个目录」。现在插件之间
零耦合（见 (08-tools.md#81-与其它插件彻底解耦)），这条纪律是：

> 需要别的插件的数据或能力 → 让对方向外暴露 HTTP 接口，本插件用 `tools.custom` 接；
> 不要按路径加载别人的模块，也不要读别人的数据文件。

## 19.3 相对路径配置的统一口径

`data_dir` 走 `_resolve_relative()`：

| 写法 | 基准 |
|------|------|
| 留空 / 裸名字（`data`） | 插件包目录（插件自带的数据放插件里） |
| 显式相对（`./data/ai_chat`、`../shared`） | `config.json` 所在目录（从顶层配置写路径的直觉预期） |
| 绝对路径 | 原样使用 |

> 当年 `notices.path` 与 `data_dir` 共用这套规则，是因为前者默认值写的是
> `./plugins/bupt_news/data` —— 若按「相对插件目录」解析会变成
> `plugins/ai_chat/plugins/bupt_news/data`，一个永远不存在的路径。那段配置
> 已随解耦删除，但**规则本身要保留**：从顶层 `config.json` 写相对路径，
> 预期基准就是 `config.json` 所在目录。

---
