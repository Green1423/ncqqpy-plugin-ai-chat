"""ai_chat 插件包 —— 多会话 AI 对话（DSH 式消息包装 + 上下文压缩 + 内置工具）。

只在这里导出**一个** `PluginBase` 子类：`PluginManager.load()` 遍历模块属性、
取第一个 `PluginBase` 子类来实例化，导出多个会让加载结果取决于字母序。
"""
from .main import AiChatPlugin

__all__ = ["AiChatPlugin"]
