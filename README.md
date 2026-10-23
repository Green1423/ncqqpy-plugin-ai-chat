# ai_chat —— 多会话 AI 对话

把 DSH 式的对话语义搬进 QQ：多会话、上下文压缩、内置工具、联网搜索、函数图像与 Markdown 出图。

这是 [ncqqpy](https://github.com/Green1423/ncqqpy) 的插件，**独立仓库**：
框架按 `plugin_dir`（默认 `./plugins`）扫描子目录加载插件，所以把它 clone 到
框架的 `plugins/` 下即可。

## 安装

```bash
cd /path/to/ncqqpy
git clone https://github.com/Green1423/ncqqpy-plugin-ai-chat plugins/ai_chat
# 重新加载（QQ 内，超级用户 · 仅私聊）
#   /reload ai_chat
```

要求：框架 [ncqqpy](https://github.com/Green1423/ncqqpy)（`napcat` 包需可导入）。

## 配置

插件被**构造**时（`on_load()` 之前）会把自己的**全部**配置项写进
`config.json` 的 `plugins.ai_chat` 段（只补缺键、不覆盖已有值、幂等），所以打开
文件就能看到所有开关和默认值。模板如下：

```json
{
  "enabled": true,
  "data_dir": "",
  "llm": {
    "base_url": "",
    "api_key": "",
    "model": "",
    "max_tokens": 4096,
    "temperature": null,
    "timeout": 270,
    "stream": true,
    "stall_timeout": 60,
    "extra_headers": {},
    "max_retries": 2,
    "retry_backoff": 1.5,
    "retry_max_wait": 8.0,
    "retry_notice": true,
    "reasoning": {
      "dialect": "effort",
      "default": "off",
      "per_model": {},
      "params": {},
      "max_tokens_floor": 0,
      "compaction": "off"
    }
  },
  "web_image": {
    "enabled": true,
    "max_bytes": 8388608,
    "max_per_turn": 4,
    "keep_files": 60,
    "page_images": 6
  },
  "binding": {
    "groups": [],
    "users": [],
    "empty_means": "deny",
    "notify_denied": false,
    "notify_cooldown": 300
  },
  "trigger": {
    "private": true,
    "group_require_at": true,
    "ignore_bot_self": true,
    "ignore_command_messages": true,
    "terminators": [
      "//"
    ],
    "markdown_suffix": "md",
    "max_staged_parts": 12,
    "max_staged_chars": 6000,
    "auto_submit_seconds": 0
  },
  "session": {
    "name_template": "{who}",
    "history_max_messages": 400,
    "max_sessions": 200
  },
  "wrapping": {
    "speaker_prefix": "auto",
    "include_timestamp": false,
    "system_prompt": "",
    "persona": ""
  },
  "compaction": {
    "enabled": true,
    "context_limit": 16384,
    "threshold_ratio": 0.8,
    "retain_ratio": 0.16,
    "retain_tokens": 0,
    "chars_per_token": 1.6,
    "summary_max_chars": 1200,
    "min_head_messages": 2,
    "max_compactions_per_session": 0,
    "max_context_retries": 1
  },
  "reply": {
    "chunk_max_chars": 1200,
    "chunk_max_images": 4,
    "quote": false,
    "inline_math": true,
    "max_rounds": 6,
    "fallback_text": "（我调用完工具但没有要补充的内容）"
  },
  "tools": {
    "enabled": true,
    "builtin": true,
    "max_mute_seconds": 2592000,
    "custom": []
  },
  "search": {
    "enabled": true,
    "backend": "360",
    "count": 5,
    "api_url": "https://api.anysearch.com/v1/search",
    "api_key": "",
    "api_method": "POST",
    "api_tag": "",
    "api_fields": {},
    "timeout": 15,
    "max_page_chars": 4000,
    "user_agent": ""
  },
  "plot": {
    "enabled": true,
    "width": 720,
    "height": 460,
    "samples": 1200,
    "grid": true,
    "show_title": false,
    "x_step": null,
    "y_step": null,
    "unit_px": null,
    "equal_aspect": null,
    "omit_y": null,
    "omit_x": null,
    "t_min": 0.0,
    "t_max": 6.283185307179586,
    "t_samples": 1200
  },
  "history": {
    "enabled": true,
    "list_limit": 10,
    "summarize": true,
    "summary_chars": 20,
    "summary_read_chars": 400,
    "max_chars_per_side": 2000,
    "render_width": 720,
    "render_timeout": 20,
    "keep_files": 40,
    "chromium": ""
  },
  "log_verbose": false
}
```

环境变量（优先级高于文件）：

| 变量 | 说明 |
|------|------|
| `（无）` | 全部配置都在 config.json → plugins.ai_chat；LLM key、搜索 key 等也在这里 |


详细文档：[`docs/ai-chat/README.md`](docs/ai-chat/README.md)。
## 测试

```bash
# 独立仓库里跑：框架需可导入（PYTHONPATH 指向框架仓库，或 pip install -e）
PYTHONPATH=/path/to/ncqqpy python -B tests/test_ai_chat.py
```

套件：`tests/test_ai_chat.py` `tests/test_history.py` `tests/test_history_command.py` `tests/test_history_render.py` `tests/test_markdown.py` `tests/test_plot.py` `tests/test_search.py` `tests/test_webimg.py`。测试会在 `tests/.test-tmp/` 下建临时目录，不碰真实数据。

## 注意

- 会话数据在 `data/`（可用 `data_dir` 改到别处）；**不要**把它提交到仓库。
- 结束符由 `trigger.terminators` 配置（默认 `//`）：分段输入的最后一条用它提交。

## License

MIT，见 [LICENSE](LICENSE)。
