# 十八、测试

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[持久化与并发](17-persistence.md) · 下一章：[插件包内的 import 纪律](19-imports.md)


测试与诊断脚本统一放在 `tests/` 下（仓库根只留运行入口）：

```bash
cd /ncqqpy
./venv-arm/bin/python -B tests/test_ai_chat.py          # 626 项 · 本插件端到端（含画图/角色/流式/配图/公开会话/邀请码/引用展示）
./venv-arm/bin/python -B tests/test_history.py          # 139 项 · 历史归并/摘要缓存/引用链/卡片引用（纯单元）
./venv-arm/bin/python -B tests/test_history_render.py   #  39 项 · 真 Chromium 出图质量
./venv-arm/bin/python -B tests/test_history_command.py  # 119 项 · history/hget 端到端
./venv-arm/bin/python -B tests/test_markdown.py         # 207 项 · Markdown 子集 / 富文本块 / 裸 LaTeX / KaTeX 资源
./venv-arm/bin/python -B tests/test_search.py           # 118 项 · 搜索解析 / SSRF 闸 / 页面图片直链
./venv-arm/bin/python -B tests/test_webimg.py           #  48 项 · 网图配图（魔数 / SSRF 逐跳 / 上限 / 白名单）
./venv-arm/bin/python -B tests/test_plot.py             # 271 项 · 函数图像（求值安全 / 几何 / 坐标轴 / 方程 / 方格 / 省略 / 提示词分工）
./venv-arm/bin/python -B tests/test_command_parser.py   #  45 项 · 命令解析/配置/插件生命周期基线
./venv-arm/bin/python -B tests/test_bupt_news.py        # 495 项 · bupt_news 验收
./venv-arm/bin/python -B tests/test_watchdog.py         # 249 项 · 框架回归
./venv-arm/bin/python -B tests/test_command_permissions.py  # 52 项 · 命令权限
```

> 这些套件**共用 `tests/.test-tmp/`**，所以不要并行跑两个（会互相踩临时目录）。

| 目录 | 放什么 |
|------|--------|
| (../../../tests) | 测试套件 + 一次性维护脚本（`backfill_images.py`、`import_history.py`）|
| (../../../tests/diagnostics) | 诊断探针（只读，不构成长期 owner）|
| (../../../tests/backups) | 配置/日志备份（`.bak-*`、孤儿日志）|
| `tests/.test-tmp/` | 测试的临时工作目录（可随时删）|

`test_ai_chat.py` **不打桩框架**：真起 OneBot WebSocket 服务端、真跑 `NapCatClient`，
事件以原始 OneBot JSON 从 WS 推进去，走完 `parse_event → EventBus → 插件 →
ApiClient → WS` 全链路；只把「外部 LLM」换成本地脚本桩（否则测试要联网、要花钱、
还不稳定）。每个用例一个全新实例（独立 `data_dir`），互不污染。

---
