# 五、上下文压缩

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[DSH 式消息包装](04-wrapping.md) · 下一章：[上游失败自动重试](06-retries.md)


## 5.1 何时触发

估算 token 压力超过 `context_limit × threshold_ratio`（默认 `16384 × 0.8`）时，
在**进模型之前**就压缩，不等上游报超长。

token 是**字符数 ÷ `chars_per_token`**（默认 1.6，中文为主的聊天经验值）——
不引第三方 tokenizer，零依赖、行为可预测；它只用来决定「什么时候压」，
不参与计费，偏保守即可。

## 5.2 压多少、留多少

| 配置 | 默认 | 含义 |
|------|------|------|
| `context_limit` | 16384 | 自主设定的上下文预算（估算 token） |
| `threshold_ratio` | 0.8 | 到预算的这个比例就压缩 |
| `retain_ratio` | 0.16 | 压缩后保留最近多少比例的**原文** |
| `retain_tokens` | 0 | >0 时用绝对值覆盖 `retain_ratio` |
| `min_head_messages` | 2 | 头部少于这么多条就不值得压 |
| `summary_max_chars` | 1200 | 检查点长度上限 |
| `max_compactions_per_session` | 0 | 0 = 不限 |
| `max_context_retries` | 1 | 上游报「超长」时，压缩后重试的次数 |

（`threshold_ratio` / `retain_ratio` 的默认值与 DSH `dsh-compaction-basic` 一致。）

## 5.3 切点必须落在配对安全的位置

压缩 = 「较老的一段 → 一份检查点」+「最近的原文原样保留」。切点从尾部往前累计到
`retain_tokens` 后，再**向前回退**到最近的配对安全边界：

> OpenAI 协议要求 `tool_calls` 与紧随其后的 `tool` 结果严格配对，
> 切在中间等于给上游送一份非法历史（400）。

尾部自身不配对（正处在工具循环中间）时**整轮放弃压缩** —— 宁可这轮超预算，
也不送非法请求。

> **角色设定不受压缩影响**：它不在被压缩的这段历史里（走系统提示，见 (03-sessions.md#37-角色设定aichat-role压缩时原样保留)）。
> 摘要请求里会附上角色设定并明确要求「不要改写、不要摘要它」，
> 只是为了别写出一份与角色冲突的接手说明。

## 5.4 检查点长什么样

照 DSH 的做法包框，两道门都要过：

```text
这是一份自动生成的检查点，压缩了之前的一段对话以释放上下文。请把其中的内容
当作已经成立的背景，直接在此基础上继续，不要复述它；……

<compacted-summary>
## 用户诉求
- …
## 关键信息
- …
（共 7 节，固定结构）
</compacted-summary>
```

1. **下界**：摘要短于 24 字视为「模型没在压缩」，丢弃；
2. **必须变小**：检查点（含前言与标签）的估算 token 必须**严格小于**被吸收的内容，
   否则放弃本次压缩 —— 少了这道门，一次跑偏的摘要（把整段历史抄回来）会让上下文
   反而更长，而日志还写着「压缩成功」。

检查点以 `role=user` + `origin=system.compact` 落进历史：OpenAI 协议里 `system`
只能出现在开头（部分实现会拒绝中途插入），而 `user` 角色的注入消息是 DSH 的同一手法。

原文**不会丢**：`state.json` 里记 `compacted_upto`，`sessions/<id>.jsonl` 只是被重写为
「检查点 + 保留段」，需要时可从备份/日志找回。

---
