# tests/ —— ai_chat 插件验收套件

这些套件需要框架可导入（它们 `import napcat`）。两种跑法：

```bash
# ① 独立 clone 本插件仓库：把框架仓库挂到 PYTHONPATH
PYTHONPATH=/path/to/napcat-py python -B tests/test_ai_chat.py

# ② 框架仓库布局（本插件已 clone 到框架的 plugins/ai_chat/ 下）
cd /path/to/napcat-py && python -B tests/test_ai_chat.py
```

| 套件 | 覆盖 |
|------|------|
| `test_ai_chat.py` | 端到端：会话、触发、工具、错误路径 |
| `test_history.py` | 历史问答归并 + 摘要解析/缓存（纯单元，最快） |
| `test_history_command.py` | history/hget/搜索工具/Markdown/内联图文端到端 |
| `test_history_render.py` | 问答出图（真起 Chromium） |
| `test_markdown.py` | Markdown→HTML→图片（含表格/公式块） |
| `test_plot.py` | 函数图像（解析/样式/出图） |
| `test_search.py` | 联网搜索（解析/相关性/SSRF/API/兜底/页面图片直链） |
| `test_webimg.py` | 网图配图（魔数/SSRF 逐跳/上限/白名单） |

## 说明

- 临时夹具落在 `tests/.test-tmp/`（不进仓库，可随时删）；
- 需要 Chromium 的套件（出图类）在没有浏览器的机器上会跳过相关断言；
- 依赖别的插件作夹具的断言，在本插件单独 checkout 时会打印 `⏭️ 跳过`。
