# ai_chat 插件

> 上级：[插件文档](README.md)

> 版本 1.0.0 · 依赖框架 ≥ 1.0.0
把 DSH（DeepSeek Harness）的对话语义搬进 QQ：**多会话**、**DSH 式消息包装**、
**上下文压缩**、**内置工具（function calling）**。对话不需要命令前缀，
但需要**结束符**收尾；管理动作才走命令。

## 章节

| 章节 |
|------|
| [快速开始](01-quickstart.md) |
| [服务范围（只服务绑定的群 / 私聊）](02-scope.md) |
| [会话机制](03-sessions.md) |
| [DSH 式消息包装](04-wrapping.md) |
| [上下文压缩](05-compaction.md) |
| [上游失败自动重试](06-retries.md) |
| [思考强度（reasoning effort）](07-reasoning.md) |
| [内置工具](08-tools.md) |
| [配置项全表](09-config.md) |
| [历史对话](10-history.md) |
| [联网搜索](11-web-search.md) |
| [Markdown 模式](12-markdown.md) |
| [函数图像（`plot_function`）](13-plot.md) |
| [`/md`：把上一条 AI 回复转成图片](14-md-image.md) |
| [网图配图（`fetch_image`）](15-images.md) |
| [结束符](16-terminator.md) |
| [多模态（图片 / 语音）](17-multimodal.md) |

## 开发与实现参考

内部机制（持久化、测试、import 纪律、框架改动、DSH 对应关系、Markdown 渲染细节）见 [reference/ai-chat](https://github.com/Green1423/ncqqpy/blob/main/docs/reference/ai-chat/README.md)，不属于使用手册。
