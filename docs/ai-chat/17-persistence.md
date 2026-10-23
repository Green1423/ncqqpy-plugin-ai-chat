# 十七、持久化与并发

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[结束符](16-terminator.md) · 下一章：[测试](18-tests.md)


**配置在 `config.json`，状态在数据目录** —— 分开的理由：`config.json` 是人的领域
（改一次要 review），会话状态是机器的领域（每句话都在写）。混在一起会让每句对话都
重写一次核心配置文件。

```
plugins/ai_chat/data/
├── staging.json            未提交输入的暂存区（重启不丢）
└── sessions/
    ├── state.json          会话索引 + 绑定表
    ├── s0001.jsonl         每个会话一个 append-only 历史
    └── s0001.jsonl.deleted 删除后改名（可人工找回）
```

- **暂存区落盘**：用户可能在「@我 + 长文 + 结束符」之间隔几分钟，期间
  (../../reference/watchdog/01-overview.md#一它做什么)会主动重启 —— 内存态会让整段输入消失。
- **历史 append-only + 定期裁剪**：只在超过 `history_max_messages × 1.5` 时重写一次，
  避免每加一条消息就整文件重写（SD 卡写放大）。
- **每会话一把 `asyncio.Lock`**：同一会话的对话串行化，避免两条消息同时进模型。
- **`state.json` 用 `threading.Lock`**（锁内无 `await`）：事件处理、定时任务、
  卸载可能在不同 task 里调用。

---
