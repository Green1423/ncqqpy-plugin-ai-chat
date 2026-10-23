# 二十一、与 DSH 的对应关系

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[为框架做的改动](20-framework-changes.md)


| 本插件 | DSH |
|--------|-----|
| `segments.py` 的 `Block` / `Message` | `dsh-llm` 的 `ContentBlock` / `Message` |
| `metadata.origin`（`user.*` / `system.compact` / `tool`） | `MessageSource` + `ContextForm` |
| `compaction.build_system_prompt()` | `dsh-system-prompt` + `dsh-persona` 分节装配 |
| `CompactionPolicy` | `dsh-compaction-basic` 的 `thresholdRatio` / `retainRatio` |
| `Compactor` + `frame_checkpoint()` | 压缩事务与 `<compacted-summary>` 检查点 |
| `tools.py` 的 `ToolRegistry` | `dsh-tools` 的工具声明与执行 |
| `/aichat *` 子命令 | `dsh-commands`（本框架另有权限/scope 体系） |

内部机制的完整考证报告见 (../../reference/dsh-internals/README.md)，按主题分四块：
消息形态与来源 (../../reference/dsh-internals/01-message-shape.md#1-message-shape-roles-blocks-and-provenance)、
会话日志与投影 (../../reference/dsh-internals/02-sessions.md#2-sessions-log-seq-surface-identity)、
系统提示装配 (../../reference/dsh-internals/03-system-prompt.md#3-system-prompt-composition)、
压缩 (../../reference/dsh-internals/04-compaction.md#4-compaction)。
