"""兼容壳：Markdown 渲染已经搬到框架（`napcat.markdown`）。

2026-10-29 起，Markdown→HTML 的实现只有**一份**，在 `napcat/markdown.py`；
这里只做转发，保证 `from plugins.ai_chat.markdown import …` 与旧测试继续可用。

⚠️ **为什么要带文件回退**：框架模块只在**进程启动**时加载，而插件是热重载的。
新加的 `napcat.markdown` 在老进程里还不存在 —— 直接 `import` 会把整个
ai_chat 打挂（插件启用失败，比渲染不对严重得多）。所以先试着正常导入，
拿不到就按**文件路径**把框架里那份加载进来：实现仍然只有一份，
只是绕过了 import 缓存。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_framework():
    """正常导入框架版；老进程里没有就按文件路径加载同一份文件。"""
    try:
        from napcat import markdown as mod      # 正常路径
        return mod
    except ImportError:
        pass
    here = Path(__file__).resolve()
    path = here.parents[2] / "napcat" / "markdown.py"
    name = "napcat_markdown_boot"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:     # 文件都没了 → 只能报错
        raise ImportError(f"找不到框架的 Markdown 渲染器：{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_framework()

#: 把框架版的全部名字转发出去（`from .markdown import X` 照常可用）。
#: ⚠️ **连私有名字也要转发**：测试与 `render.py` 会直接摸 `_math_css`、
#: `_SIZE_SCRIPT` 这类下划线常量 —— 只转公开名会让它们 ImportError（踩过）。
__all__ = [n for n in dir(_mod) if not n.startswith("__")]

globals().update({n: getattr(_mod, n) for n in __all__})
__all__ = [n for n in __all__ if not n.startswith("_")]
