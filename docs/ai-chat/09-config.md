# 九、配置项全表

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[内置工具](08-tools.md) · 下一章：[历史对话](10-history.md)


```jsonc
{
  "plugins": {
    "ai_chat": {
      "enabled": true,
      "binding": {                       // 服务范围：只服务列在这里的群/私聊（见第二节）
        "groups": [100000003],           // 留空 = 一个群都不许（不回落 bound_groups）
        "users": [10000001],             // 私聊白名单（框架不管私聊）
                                         // 群开通 = 全群可用；群里每人各有自己的会话
        "notify_denied": true,           // 回绝时是否回一句说明
        "notify_cooldown": 300           // 回绝提示冷却（秒）
      },
      // 相对路径基准：裸名字（data）随插件包；显式相对（./data/ai_chat）随 config.json
      "data_dir": "",
      "llm": {
        "base_url": "", "api_key": "", "model": "",
        "max_tokens": 4096,
        "temperature": null,             // null = 不发送该字段
        "timeout": 270,                  // 单次调用**总时长**上限（秒）
        "stream": true,                  // SSE 流式（见 6.2）
        "stall_timeout": 60,             // 流式下「多久没新数据」算卡死（秒）
        "extra_headers": {},
        "max_retries": 2,                // 失败后最多**再试**几次（0 = 不重试）
        "retry_backoff": 1.5,            // 首次退避秒数，之后翻倍
        "retry_max_wait": 8.0,           // 单次等待上限（秒）
        "retry_notice": true,            // 第一次重试时告诉用户一句
        "reasoning": {                   // 思考强度（见第七节）
          "dialect": "effort",           // effort|thinking|enable_thinking|none|raw
          "default": "off",              // off|minimal|low|medium|high|max|4k|thinking:on
          "per_model": {},               // {"模型名": "high"}
          "params": {},                  // dialect=raw 时原样并入 payload
          "max_tokens_floor": 0,         // >0：开思考时的 max_tokens 下界
          "compaction": "off"            // 压缩摘要单独一档
        }
      },
      "trigger": {
        "private": true,
        "group_require_at": true,
        "ignore_bot_self": true,
        "ignore_command_messages": true,
        "terminators": ["//"],
        "markdown_suffix": "md",       // 结束符后跟 md → Markdown 模式出图
        "max_staged_parts": 12,
        "max_staged_chars": 6000,
        "auto_submit_seconds": 0
      },
      "session": {
        // 会话只由 /aichat new 创建，不会自动建（没有 auto_create 开关）
        "name_template": "{who}",         // 可用 {who} {kind} {chat_id}
        "history_max_messages": 400,
        "max_sessions": 200
      },
      "wrapping": {
        "speaker_prefix": "auto",         // auto | always | never
        "include_timestamp": false,
        "system_prompt": "",              // 追加在默认人格之后
        "persona": ""                     // 覆盖默认人格
      },
      "compaction": {
        "enabled": true, "context_limit": 16384,
        "threshold_ratio": 0.8, "retain_ratio": 0.16, "retain_tokens": 0,
        "chars_per_token": 1.6, "summary_max_chars": 1200,
        "min_head_messages": 2, "max_compactions_per_session": 0,
        "max_context_retries": 1
      },
      "reply": {
        "chunk_max_chars": 1200, "chunk_max_images": 4,
        "quote": false, "max_rounds": 6,
        "fallback_text": "（我调用完工具但没有要补充的内容）"
      },
      "tools": {
        "enabled": true, "builtin": true,
        "max_mute_seconds": 2592000, "custom": []
      },
      "search": {
        "enabled": true, "backend": "bing", "count": 5, "timeout": 15,
        "max_page_chars": 4000, "user_agent": ""
      },
      "plot": {                          // 函数 / 方程图像（见第十三节）
        "enabled": true, "width": 720, "height": 460,
        "samples": 1200, "grid": true,
        "t_min": 0.0, "t_max": 6.283185307179586, "t_samples": 1200
      },
      "web_image": {                     // 网图配图（见第十五节）
        "enabled": true,                 // 依赖 search.enabled（下载复用它的客户端）
        "max_bytes": 8388608,            // 单张上限，超过就拒绝
        "max_per_turn": 4,               // 一轮最多配几张（与 reply.chunk_max_images 对齐）
        "keep_files": 60,                // 落盘目录只留最近多少个文件
        "page_images": 6                 // fetch_page 顺带列出几条页面图片直链
      },
      "history": {
        "enabled": true, "list_limit": 10,
        "summarize": true, "summary_chars": 20, "summary_read_chars": 400,
        "max_chars_per_side": 2000, "render_width": 720,
        "render_timeout": 20, "keep_files": 40, "chromium": ""
      },
      "log_verbose": false
    }
  }
}
```

---
