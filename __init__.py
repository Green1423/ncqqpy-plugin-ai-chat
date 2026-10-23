"""ai_chat 插件包 —— 多会话 AI 对话（DSH 式消息包装 + 上下文压缩 + 内置工具）。

入口类在 `main.py`；这里只做导出与 `PLUGIN` 声明（拆分写法见
`docs/plugin-dev/14-layout.md` §14.1）。
"""
from .main import AiChatPlugin

PLUGIN = AiChatPlugin
__all__ = ["AiChatPlugin"]
