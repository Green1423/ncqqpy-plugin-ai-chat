"""函数图像：表达式 → PNG（matplotlib）。

**这一版直接用 matplotlib 画**，不再自己算坐标生成 SVG。早先那套（大约两千行
手写几何 + Chromium 出图）能跑，但每加一个版面要求都要自己反解一遍窗口与画布：
「格子要正方」推一次高度、「两轴要同标度」再推一次、「省略一段」又推一次，
互相打架，最后要靠 `_square_tolerance` 这类经验常数兜着。

换成 matplotlib 之后：

| 要求 | 之前 | 现在 |
|------|------|------|
| 两轴同标度 | 反解窗口长宽比 + 画布高度 | `set_aspect("equal")` |
| 格子是正方形 | 两根轴共用一个步长 + 反解高度 | 同上 + 两根轴同一个 `MultipleLocator` |
| 隐式方程 | 手写 marching squares | `contour` |
| 多条曲线 / 图例 / 配色 | 自己维护配色表与线型 | 默认配色环 + `legend` |
| 紧贴裁剪 | 自己解码 PNG 找墨迹边界 | `bbox_inches="tight"` |
| 省略一段 | 把区间压成窄带 + 手绘波浪 | 两块面板 + 波浪断口 |

**保留下来的**是表达式解析与求值安全（`latex_to_expr` / `compile_xy` /
`parse_curve`）—— 那部分与「用什么画」无关，而且是安全边界（表达式直接来自
用户消息），不该跟着渲染一起重写。

⚠️ `MPLCONFIGDIR` 必须在 import matplotlib **之前**设好：默认路径
`~/.config/matplotlib` 在这台机器上位于**只读**的根文件系统，不设的话每次都会
退回临时目录并往 stderr 抱怨（见 `_mpl_config_dir`）。
"""

from __future__ import annotations

import ast
import asyncio
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# 顶层导入（不留函数内相对导入 —— 本仓库有静态扫描测试盯着这一条：
# 函数内的相对导入在插件被 `spec_from_file_location` 加载时会 ModuleNotFoundError，
# 而异常会被上层吞成「AI 对话出错」，非常难查）
from .tools import PLOT_MARKER_PREFIX, PLOT_MARKER_SUFFIX

# ⚠️ matplotlib / numpy **不在顶层导入**：它们合计要一秒多，而插件加载发生在
# 每次热重载与启动时；而画图只是偶尔用一次。改用 `_mpl()` 懒加载（见下）。

#: 允许出现的常量
CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

#: 允许调用的函数（白名单）
FUNCTIONS: dict[str, Callable] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "ln": math.log,
    "log10": math.log10, "log2": math.log2, "sqrt": math.sqrt,
    "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "sign": lambda v: (v > 0) - (v < 0),
    "min": min, "max": max,
    "atan2": math.atan2,
    # 注意：这里**没有** `pow`。`pow` 交给 compile_expr 里的受限版本，
    # 否则 `pow(x, 2**60)` 会绕过指数上限，算到天荒地老。
}

#: 允许的 AST 节点（**白名单**，不是黑名单）
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
    ast.Load, ast.Tuple,
)

#: 可以**被调用**的名字 = 白名单函数 + 受限的 `pow`。
#: `pow` 单独拿出来：它不在 `FUNCTIONS` 里（`FUNCTIONS` 会被原样塞进求值命名空间），
#: 真正的实现是 `compile_expr` 里那个带指数上限的版本 —— 直接放内置 `pow`
#: 就等于给 `pow(x, 2**60)` 开了一道绕过上限的后门。
_CALLABLE_NAMES = frozenset(set(FUNCTIONS) | {"pow"})

#: LaTeX 命令 → 普通写法（够覆盖常见数学记号）
#:
#: ⚠️ 长命令必须排在短命令**前面**：这里是按顺序做字符串替换，
#: `\arcsin` 若排在 `\sin` 后面，会被先替换成一个孤零零的 `\arc` + `sin`。
_LATEX_WORDS = {
    r"\cdot": "*", r"\times": "*", r"\div": "/", r"\pm": "+",
    # 反三角（LaTeX 标准写法 + 常见的 `\asin` 简写）必须排在 \sin/\cos/\tan 之前
    r"\arcsin": "asin", r"\arccos": "acos", r"\arctan": "atan",
    r"\asin": "asin", r"\acos": "acos", r"\atan": "atan",
    r"\sinh": "sinh", r"\cosh": "cosh", r"\tanh": "tanh",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan",
    r"\exp": "exp", r"\ln": "log", r"\log": "log", r"\lg": "log10",
    r"\sqrt": "sqrt", r"\abs": "abs",
    r"\pi": "pi", r"\tau": "tau", r"\infty": "inf",
    r"\left": "", r"\right": "", r"\,": "", r"\;": "", r"\!": "",
    r"\quad": " ", r"\qquad": " ", r"\ ": " ",
}

#: 只匹配到**左花括号之前**：花括号本身必须留给 `_take_group` 按配对去读。
#: （早期版本把 `{` 也吃进 match，`m.end()` 就指到了内容上，于是每次都说「不配对」。）
_FRAC_RE = re.compile(r"\\[dt]?frac\s*")
_SQRT_RE = re.compile(r"\\sqrt\s*(?:\[[^\]]*\])?\s*")
_TEXT_RE = re.compile(r"\\text\s*\{[^{}]*\}")
#: `\log_{10}` / `\lg` / `\ln` / `\log_2` 后面**紧跟**底数括号时 → 函数名
_LOG_BASE_RE = re.compile(r"\\(log|lg|ln)\s*_\s*\{?\s*(\d+|e)\s*\}?\s*(?=\()")
#: 绝对值：`|x|` → `abs(x)`（不支持嵌套 `|`，够用且不会猜错）
_ABS_RE = re.compile(r"\|([^|]+)\|")
#: `**` 的右结合会写出 `x**x**9` 这种指数塔：底数稍大就是天文数字级别的计算量，
#: 实测能把渲染进程卡死。求值前把幂运算收进安全区间。
_MAX_POW = 1e6
_MAX_POW_EXP = 64.0

#: 相邻采样点的跳变超过「纵轴量级 × 这个倍数」就判为跨过了奇点。
#: 太小会把陡峭的正常曲线切碎，太大则 `tan(x)` 的假线还在 —— 取 1.0
#: （一跳就是一整个纵轴的高度，正常连续函数在两倍采样密度下不会这样）。
_JUMP_FACTOR = 1.0

#: 纵轴「核心带」的分位区间。
#:
#: 2%/98% 是**实测选出来的**：拿 14 个典型函数（多项式、三角、倒数、指数、对数、
#: 根式、渐近线型）比过 2%/3%/5%/10% 四档 —— 2% 的数据覆盖率最高（渐近线型 95%、
#: 多项式 96%），而收紧到 10% 会把 `x^3-3x` 压成 ±488（经典形状没了）。
#: 真正解决问题的是下面的**对称化**，不是把带收紧：`e^x/\sin x` 的 2% 带本来是
#: [−690, +76]（原点被顶到 90% 处），对称化之后变成 ±383、原点回到正中。
_CORE_LOW = 0.02
_CORE_HIGH = 0.98

#: 补乘号时认得出来的名字（**最长匹配**，所以 `log10` 不会被拆成 `log`+`10`）
_KNOWN_NAMES = tuple(sorted(set(FUNCTIONS) | set(CONSTANTS) | {"x"},
                            key=len, reverse=True))

#: 函数幂写法 `\sin^2(x)`：LaTeX 习惯把指数写在函数名上，含义是 `(sin(x))^2`。
#: 直接当 `sin**2` 求值会得到 nan，必须重写成先调用再乘方。
#: 中间的 `\*?` 与 `\s*` 是为了吃掉隐式乘法扫描器插进来的乘号与空格
#: （`sin**2*(x)`）；参数那对括号由第 3 组捕获，替换时**放回括号里面**
#: —— 漏掉它就会剩下一个 `(x)` 挂在表达式尾巴上。
_FUNC_POW_RE = re.compile(
    r"\b(" + "|".join(sorted(FUNCTIONS, key=len, reverse=True))
    + r")\s*\*\*\s*\(?\s*(-?\d+(?:\.\d+)?)\s*\)?\s*\*?\(\s*([^()]*)\s*\)")
#: 裸参数形式 `\cos^2 x`。**必须在这里把参数一起吃掉**，不能只补一对括号了事：
#: `()` 是嵌套结构，事后按个数补齐会把 `sin(x` 提前闭合（`sin**2(x* (cos(x))**2)`）。
#: 两条约束：指数后不能直接跟 `(`（那是 `x**(2)` 这种正常写法，不是函数幂），
#: 且后面必须真有因子 —— 否则 `(x)**2` 会被吞掉收尾的 `*` 改成坏表达式。
_FUNC_POW_BARE_RE = re.compile(
    r"\b(" + "|".join(sorted(FUNCTIONS, key=len, reverse=True))
    + r")\s*\*\*\s*(-?\d+(?:\.\d+)?)\s*(?!\()\s*\*?\s*"
    + r"(?:\(\s*([^()]*)\s*\)|\*?\s*(\d+(?:\.\d+)?\s*\*?\s*)?([a-zA-Z]\w*))")
#: 裸科学计数法 `1e-3 x`：Python 认 `1e-3`，但后面的隐式乘法扫描器会把
#: `3` 当成普通数字并给 `3x` 补乘号，于是变成 `1e-3*x*…` 之类 —— 先固化下来。
#: 结尾的 lookahead 只挡「`e` 的指数又接着指数」这种非数学写法，不挡字母：
#: `1e3x` 要读成 `1000x`（指数结束、后面又一个因子）。
_SCI_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?[eE][-+]?\d+)(?![eE\d])")

#: 保护科学计数法片段用的哨兵（正文里不可能出现的控制字符）
_PROTECT_OPEN = "\x01"
_PROTECT_CLOSE = "\x02"


class PlotError(ValueError):
    """表达式无法解析 / 不在白名单内 / 无法绘制。消息可直接展示给用户。"""


def _take_group(s: str, start: int) -> tuple[str, int]:
    """从 `s[start] == '{'` 开始取出一整个**配对**花括号组（支持嵌套）。

    原来的正则 `\\{([^{}]*)\\}` 遇到 `\\frac{x^{2}-1}{x+1}` 这种带嵌套的分子就
    匹配不上，于是整串落到「不认识 LaTeX 命令」的错误分支里。分组必须按配对来读。
    """
    if start >= len(s) or s[start] != "{":
        raise PlotError("LaTeX 花括号不配对")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    raise PlotError("LaTeX 花括号不配对")


def _expand_groups(s: str) -> str:
    """展开 `\\frac{}{}` 与 `\\sqrt{}`，从**外层往内层**逐轮处理。"""
    for _ in range(12):
        out: list[str] = []
        i = 0
        changed = False
        while i < len(s):
            if s[i] != "\\":
                out.append(s[i])
                i += 1
                continue
            m = _FRAC_RE.match(s, i)
            if m:
                num, j = _take_group(s, m.end())
                while j < len(s) and s[j] == " ":
                    j += 1
                den, k = _take_group(s, j)
                out.append(f"(({num})/({den}))")
                i, changed = k, True
                continue
            m = _SQRT_RE.match(s, i)
            if m:
                inner, j = _take_group(s, m.end())
                out.append(f"sqrt({inner})")
                i, changed = j, True
                continue
            out.append(s[i])
            i += 1
        s = "".join(out)
        if not changed:
            return s
    return s


def _read_factor(s: str, start: int) -> str:
    """从 `start` 读**恰好一个因子**（可以带数字系数，如 `2*x`），不碰加减号。

    与 `_FUNC_POW_BARE_RE` 的「参数」保持同一套口径：两处对「参数到哪儿为止」
    的看法必须一致，否则 `\\sin 2x` 会在一边被看成 `sin(2x)`、在另一边看成 `sin(2)`。
    """
    i = start
    while i < len(s) and s[i] == " ":
        i += 1
    if i < len(s) and s[i] == "(":                 # 括号组：按配对读
        depth = 0
        for k in range(i, len(s)):
            if s[k] == "(":
                depth += 1
            elif s[k] == ")":
                depth -= 1
                if depth == 0:
                    return s[i:k + 1]
        return s[i:]
    out: list[str] = []
    num: list[str] = []
    while i < len(s) and (s[i].isdigit() or s[i] == "."):
        num.append(s[i]); i += 1
    if num:
        out.append("".join(num))
        j = i
        while j < len(s) and s[j] == " ":
            j += 1
        if j < len(s) and s[j] == "*":             # `2*x` 整体算一个因子
            j += 1
            while j < len(s) and s[j] == " ":
                j += 1
            i = j
    name = _longest_name(s, i)
    if name:
        out.append(name)
    if not out:
        return ""
    return "*".join(out) if len(out) > 1 else out[0]


def _bind_bare_args(s: str) -> str:
    """给没写括号的调用补上参数：`sin x` → `sin(x)`、`sin 2*x` → `sin(2*x)`。

    只吃**紧跟其后的一个因子**（配对括号组 / 数字 / 名字，可带数字系数），
    不吞整串 —— `sin x + 1` 必须是 `sin(x)+1` 而不是 `sin(x+1)`。把参数吃多了
    是那种「图看上去挺像那么回事、其实完全画错」的失败，比报错难发现得多。
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        if not s[i].isalpha():
            out.append(s[i]); i += 1; continue
        name = _longest_name(s, i)
        if not name or name not in FUNCTIONS:
            out.append(s[i]); i += 1; continue
        j = i + len(name)
        while j < len(s) and s[j] == " ":
            j += 1
        if j >= len(s) or s[j] in "(+-*/^)":
            out.append(s[i:j]); i = j; continue
        if s[j] == "{":                        # 还没归一的花括号组
            arg, k = _take_group(s, j)
            out.append(f"{name}({arg})"); i = k; continue
        arg = _read_factor(s, j)
        if arg:
            out.append(f"{name}({arg})")
            i = j + len(arg)
            continue
        out.append(s[i:j]); i = j
    return "".join(out)


def _protect_spans(text: str) -> str:
    """把科学计数法片段用哨兵字符包起来，并**在两侧补空格**。

    为什么要这么绕：`1e3x` 里那个 `3` 后面紧跟着 `x`，扫描器应该看成
    「`1e3` 后面又一个因子」→ 补成 `1e3*x`（= 1000x）；而 `2x` 里的 `2`
    必须补成 `2*x`（= 2x）。两者只差在「这个数字是不是指数的一部分」，
    扫描器逐字符看是分不出来的。所以先把确定的科学计数法片段**标记**出来，
    让扫描器整段跳过；两侧补空格则保证标记内外不会连成一个词。
    """
    out: list[str] = []
    last = 0
    for m in _SCI_RE.finditer(text):
        out.append(text[last:m.start()])
        out.append(f"{_PROTECT_OPEN}{m.group(1)}{_PROTECT_CLOSE}")
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def _unprotect(text: str) -> str:
    return text.replace(_PROTECT_OPEN, "").replace(_PROTECT_CLOSE, "")


def _mul_pass(text: str) -> str:
    """补一趟隐式乘号，**科学计数法片段全程受保护**。

    保护必须包在**每一次**补乘号外面：这个函数在 `latex_to_expr` 里被调用两次
    （绑参数前、绑参数后），只在其中一次加保护，另一次照样会把 `2.5e2`
    拆成 `2.5*e*2`（实测踩过）。
    """
    return _unprotect(_insert_implicit_mul(_protect_spans(text)))


def _insert_implicit_mul(s: str) -> str:
    """补隐式乘号：`2x` → `2*x`、`3sin(2x)` → `3*sin(2*x)`、`pi x` → `pi*x`。

    逐字符扫描而不是正则：正则分不清 `log10(x)` 里的 `10(`（数字与左括号相邻，
    但那是**函数名的一部分**）和 `2(x)`（确实要补乘号），也分不清 `xsin(x)`
    该拆成 `x*sin(x)` 还是当成一个名字。

    规则：在当前位置取**最长**的已知名字（函数或常量），取到就整段跳过；
    取不到而下一个字符又能开启一个因子（字母/数字/左括号），就补 `*`。
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == _PROTECT_OPEN:
            end = s.find(_PROTECT_CLOSE, i)
            end = len(s) if end < 0 else end + 1
            out.append(s[i:end])                 # 受保护片段：原样带过
            i = end
            continue
        out.append(ch)
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if not nxt:
            break
        # 中间夹空格也算相邻：`pi x` / `(x+1) (x-1)` / `2 sin(x)` 都要补乘号
        j = i + 1
        while j < len(s) and s[j] == " ":
            j += 1
        far = s[j] if j < len(s) else ""
        if ch in ")" and (far == "(" or far.isalnum()):
            out.append("*"); i += 1; continue
        if ch.isdigit() and (far == "(" or far.isalpha()):
            if s[i - 1:i].isalnum() or s[i - 1:i] == "_":
                i += 1; continue                 # `log10` / `log2` 的函数名内部
            out.append("*"); i += 1; continue
        if ch.isalpha() and (nxt == "(" or nxt.isalnum() or nxt == "."):
            name = _longest_name(s, i)
            if name:
                out.extend(s[i + 1:i + len(name)])   # 已知名字：整段带过
                i += len(name)
                # `sin(x)` 是调用，不补乘号；`pi x` / `x cos(x)` 要补。
                # 注意跳过空格再判断：`\pi x` 归一后是 `pi x`，中间有个空格，
                # 只看紧邻字符会漏掉。
                k = i
                while k < len(s) and s[k] == " ":
                    k += 1
                after = s[k] if k < len(s) else ""
                if name not in FUNCTIONS and (after == "(" or after.isalnum()):
                    out.append("*")
                continue
            if nxt == "(" or nxt.isalnum():
                out.append("*")                  # 未知名字：按隐式乘法拆开
        elif (ch.isalnum() or ch == ")" or ch == "}") and j > i + 1 and (
                far == "(" or far.isalnum()):
            # 空格分隔的并置（`x sin(x)` / `2 sin x`）：紧邻判据看不到，
            # 但空格两边都是因子时同样要补乘号。
            out.append("*")
        i += 1
    return "".join(out)


def _longest_name(s: str, start: int) -> str:
    """`s[start:]` 开头能匹配到的最长**已知**名字（函数或常量），没有则返回空串。"""
    best = ""
    for name in _KNOWN_NAMES:
        if len(name) > len(best) and s.startswith(name, start):
            best = name
    return best


def latex_to_expr(text: str) -> str:
    """把 LaTeX（或普通写法）归一成求值用的表达式。

    支持 `\\frac{a}{b}`、`\\sqrt{x}`、`\\sin` 等常见命令，以及 `y = …` / `f(x) = …`
    这类前缀（会被剥掉）。识别不出的命令**报错**而不是猜 —— 猜错会画出一条
    形状完全不对的曲线，比报错更糟。
    """
    s = str(text or "").strip()
    if not s:
        raise PlotError("表达式为空")
    # 内部多余空格先归一：`\cos^2 x`、`\cos^2(x)`、`\cos^2  x` 必须是同一条曲线。
    # 留着尾随空格的话，下面按字符扫描的那几趟（补参数 / 补乘号）会在
    # 「空格后面还是个因子」和「已经是结尾」之间摇摆，同一式子得出不同结果。
    s = " ".join(s.split())
    # 剥掉 `y =` / `f(x) =` 之类的左值
    s = re.sub(r"^\s*(?:[a-zA-Z]\w*\s*\(?\s*\w*\s*\)?\s*)?=\s*", "", s)
    # 去掉行间/行内公式定界符
    s = s.strip("$")
    s = s.replace("$$", "")
    s = _TEXT_RE.sub("", s)
    s = s.replace(r"\\", "\\")             # 用户可能贴了转义过的 LaTeX
    # 顺序很关键：`\frac` / `\sqrt` / `\log_b` 都靠**花括号**定位，
    # 必须在「花括号归一成圆括号」之前展开（`{}` 之后就没有配对信息了）。
    s = _expand_groups(s)
    s = _LOG_BASE_RE.sub(lambda m: _log_base(m.group(1), m.group(2)), s)
    s = _ABS_RE.sub(r"abs(\1)", s)
    # 花括号只是分组：`x^{2}` → `x^(2)`
    s = s.replace("{", "(").replace("}", ")")
    for cmd, rep in _LATEX_WORDS.items():
        s = s.replace(cmd, rep)
    s = s.replace("^", "**")
    s = re.sub(r"\*{3,}", "**", s)
    if re.search(r"\\[a-zA-Z]+", s):
        bad = re.search(r"\\[a-zA-Z]+", s).group(0)
        raise PlotError(f"不认识 LaTeX 命令 {bad}（支持 \\frac \\sqrt \\sin \\cos "
                        f"\\tan \\log \\ln \\exp \\pi 等）")
    # 科学计数法必须夹在「命令替换」之后、「补乘号」之前：
    # * 放到前面没用 —— `\cdot` 之类还没换成符号，后面照样会动到这些字符；
    # * 放到补乘号之后更糟，`2.5e2` 已经被拆成 `2.5*e*2`（= 13.59，一个合法但
    #   完全错误的数），先固化才能让扫描器把 `2.5e2` 看成一个整体。
    # 每一个「补乘号」的入口都要先加保护（下面两处都走 `_mul_pass`）。
    # 顺序不能换，这里踩过两个坑：
    # ① 补乘号那一趟看到「数字后面跟字母」就插 `*`。若在绑参数**之前**跑它，
    #    `cos**2 x` 里的 `2 x` 会被拆成两个因子乘起来（`((cos(x))**2)*x`，
    #    数值恰好差一个 x）；若在绑参数**之后**但只跑一次，`\sin 2x` 又会
    #    先被绑成 `sin(2)*x`（参数只吃到一个因子）。
    # ② 所以正确做法是：先把 `2x` 这类并置摊成 `2*x`（第一趟补乘号），
    #    再绑参数（这时 `2*x` 能整体作为参数），最后再补一趟收尾的乘号。
    # 科学计数法先固化，再交给带保护的补乘号趟
    s = _SCI_RE.sub(r"(\1)", s)
    s = _mul_pass(s)
    s = _FUNC_POW_RE.sub(r"((\1(\3))**\2)", s)      # `sin**2*(x)` → `((sin(x))**2)`
    s = _FUNC_POW_BARE_RE.sub(
        lambda m: (f"(({m.group(1)}({m.group(4) or ''}"
                   f"{m.group(5) or m.group(3)}))**{m.group(2)})"), s)
    s = _bind_bare_args(s)                          # `sin 2*x` → `sin(2*x)`
    s = _mul_pass(s)
    return s.strip()


def _log_base(cmd: str, base: str) -> str:
    """`\\log_{10}(x)` 这类带底数的对数 → 函数名。

    只认 2/10/e 三个底数。其它底数要按换底公式重写，而下一个记号未必是函数的
    参数（`\\log_3 x + 1` 该包住谁？），猜错就是一条错的曲线，所以直接报错。
    """
    if cmd == "ln" or base == "e":
        return "log"
    if base == "10":
        return "log10"
    if base == "2":
        return "log2"
    raise PlotError(f"只支持 log 的 2/10/e 底数（\\log_{{{base}}} 请改写成 log(x)/log({base})）")


def compile_xy(expr: str, *, variables: tuple[str, ...] = ("x",)) -> Callable[..., float]:
    """把表达式编译成可求值的函数。**白名单求值**，不是裸 eval。

    `variables` 是允许出现的自变量（`("x",)` / `("x", "y")` / `("t",)`）——
    画方程要认 `x` 和 `y`，参数方程要认 `t`，所以这里做成可配的，
    而不是把 `x` 写死在名字检查里。
    """
    tree = _validate_expr(expr, variables)
    code = compile(tree, "<plot>", "eval")
    namespace = {"__builtins__": {}, **CONSTANTS, **FUNCTIONS}

    def power(a, b, _pow=pow):
        """受限的幂运算：`x**x**9` 这种指数塔会算出天文数字，
        不仅是画不出来，光是算它就够把渲染进程拖死 —— 超出范围直接判为「这里没定义」。"""
        try:
            if abs(b) > _MAX_POW_EXP:
                return math.nan
            v = _pow(a, b)
            if isinstance(v, complex):
                # `(-8)**(1/3)`：Python 给复数。实数图上没有它，如实判为无定义，
                # 而不是自作主张取实部（那会画出一条不存在的曲线）。
                return math.nan
            return v if abs(v) <= _MAX_POW else math.nan
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return math.nan

    namespace["pow"] = power

    def f(*args: float) -> float:
        local = dict(zip(variables, args))
        try:
            value = eval(code, namespace, local)      # noqa: S307 - 白名单已校验
        except (ValueError, OverflowError, ZeroDivisionError):
            return math.nan
        except Exception:  # noqa: BLE001 - 任何求值异常都当成「这里没定义」
            return math.nan
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    return f


def _validate_expr(expr: str, variables: tuple[str, ...]):
    """白名单校验，返回 AST。**标量版与数组版共用这一份**。

    两版各写一份校验的话，「标量版挡住的写法数组版放行」这类漏洞迟早出现 ——
    而这条边界挡的是**用户消息里直接来的表达式**。
    """
    src = str(expr or "").strip()
    if not src:
        raise PlotError("表达式为空")
    allowed = set(variables)
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise PlotError(f"表达式语法不对：{e.msg}") from None
    unknown: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise PlotError(f"表达式里有不允许的写法（{type(node).__name__}）")
        if isinstance(node, ast.Constant) and not (
                isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            # 字符串/None/复数常量在数学表达式里没有意义。放行 `'a' * 3` 这种
            # 写法只会让错误从「不支持」变成「算出来是 nan」，报错更晚也更难懂。
            raise PlotError("表达式里只能出现数字")
        if isinstance(node, ast.Name):
            name = node.id
            if name in allowed or name in CONSTANTS or name in _CALLABLE_NAMES:
                continue
            # `ast.walk` 是广度优先，报出来的名字未必是**最左边**那个。
            # 用户写 `foo(x)` 时被拆成 `f*o*o*(x)`，先收集再按位置排序，
            # 才能报「不认识的名字 f」——报 `o` 会让人完全摸不着头脑。
            unknown.append((getattr(node, "col_offset", 0), name))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) \
                    or node.func.id not in _CALLABLE_NAMES:
                raise PlotError("只能调用支持的数学函数（sin/cos/exp/log/sqrt 等）")
    if unknown:
        unknown.sort()
        hint = "、".join(variables)
        raise PlotError(f"不认识的名字 {unknown[0][1]}（这里的变量只有 {hint}）")
    return tree


#: 曲线的三种形态。解析 `expression` 时自动判定，也可以由调用方指定。
KIND_EXPLICIT = "explicit"        # y = f(x)
KIND_IMPLICIT = "implicit"        # F(x, y) = 0
KIND_PARAMETRIC = "parametric"    # (x(t), y(t))


@dataclass
class CurveSpec:
    """一条待画的曲线：表达式已经被解析成「怎么求值」。"""

    kind: str = KIND_EXPLICIT
    #: explicit → f(x)；implicit → F(x, y)；parametric → (x(t), y(t)) 两个函数
    f: Optional[Callable] = None
    g: Optional[Callable] = None
    #: 原文（工具回执与图注用）
    source: str = ""
    #: 隐式方程专用：`compile_xy` 的**数组版**，给 `contour` 用（要整网格求值）。
    #: 标量版留着，是因为它还担着「这条曲线在这个窗口里有没有值」的探测。
    vec: Optional[Callable] = None


def _looks_like_param_pair(text: str) -> bool:
    """`x=cos(t), y=sin(t)` 这种逗号分隔的两半。"""
    parts = _split_top_level_commas(text)
    if len(parts) != 2:
        return False
    return all("=" in p for p in parts)


def _split_top_level_commas(text: str) -> list[str]:
    """按**顶层**逗号切分（跳过括号里的逗号，`(1, 2)` 不当分隔符）。"""
    out, depth, cur = [], 0, []
    for ch in str(text or ""):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail or out:
        out.append(tail)
    return [p for p in out if p]


def _split_equation(text: str) -> tuple[str, str]:
    """把 `左边 = 右边` 拆开（只认最外层的那个等号，`<=`/`>=` 不算）。"""
    raw = str(text or "")
    depth = 0
    for i, ch in enumerate(raw):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0:
            before = raw[i - 1] if i else ""
            after = raw[i + 1] if i + 1 < len(raw) else ""
            if before in "<>=!" or after == "=":
                continue                     # `<=` / `==` 之类，不是方程
            return raw[:i], raw[i + 1:]
    return raw, ""


def parse_curve(expr: str, *, variables: str = "") -> CurveSpec:
    """把用户/模型给的一串东西解析成「怎么画」。

    支持四种写法（自动判定，不用调用方声明）：

    | 写法 | 判定 | 画法 |
    |------|------|------|
    | `x^2-4` | 只有 x | 一元函数（按 x 采样） |
    | `y = x^2-4` | 等号一边只有 y | 同上（左值剥掉） |
    | `x^2+y^2 = 25` | 含 y 且不是显式 | 隐式方程（`F = 左边-右边 = 0`） |
    | `x=cos(t), y=sin(t)` | 两半都带 `=` 且用 t | 参数方程 |

    `variables` 可以强制指定形态（`"implicit"` / `"parametric"`），
    留空则自动判定 —— 自动判定覆盖了绝大多数写法，强制指定只是给一个
    确定的入口（模型明确说了「这是参数方程」时用）。
    """
    text = str(expr or "").strip()
    if not text:
        raise PlotError("表达式为空")
    want = str(variables or "").strip().lower()

    # ---- 参数方程 ----
    if want == KIND_PARAMETRIC or (not want and _looks_like_param_pair(text)
                                   and "t" in text):
        parts = _split_top_level_commas(text)
        pairs = dict()
        for part in parts:
            head, body = _split_equation(part)
            if not body:
                raise PlotError(f"参数方程的两半都要写成 `x=…, y=…`，看不懂：{part}")
            pairs[head.strip().lower()] = body
        if set(pairs) < {"x", "y"}:
            raise PlotError("参数方程要写成 `x=…, y=…` 两半（用变量 t）")
        fx = compile_xy(latex_to_expr(pairs["x"]), variables=("t",))
        fy = compile_xy(latex_to_expr(pairs["y"]), variables=("t",))
        return CurveSpec(kind=KIND_PARAMETRIC, f=fx, g=fy, source=text)

    # ---- 方程（含 y）----
    lhs, rhs = _split_equation(text)
    if want == KIND_IMPLICIT:
        lhs, rhs = text, ""
    # 「左边只有孤零零一个 y」= 显式函数，直接剥掉左值
    if not want and rhs and lhs.strip().lower() == "y":
        return CurveSpec(kind=KIND_EXPLICIT,
                         f=compile_xy(latex_to_expr(rhs), variables=("x",)),
                         source=text)
    body = f"({lhs}) - ({rhs})" if rhs else lhs
    normalized = latex_to_expr(body)
    if want == KIND_IMPLICIT or "y" in _bare_names(normalized):
        return CurveSpec(kind=KIND_IMPLICIT,
                         f=compile_xy(normalized, variables=("x", "y")),
                         source=text,
                         vec=compile_vec(normalized, variables=("x", "y")))
    return CurveSpec(kind=KIND_EXPLICIT,
                     f=compile_xy(normalized, variables=("x",)), source=text)


def _bare_names(expr: str) -> set[str]:
    """表达式里出现过的**裸名字**（不含函数名与常量）。用于判断「有没有 y」。"""
    try:
        tree = ast.parse(str(expr or ""), mode="eval")
    except SyntaxError:
        return set()
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    return {n for n in names
            if n not in _CALLABLE_NAMES and n not in CONSTANTS}


def compile_expr(expr: str) -> Callable[[float], float]:
    """把表达式编译成 `f(x)`（画一元函数用）。"""
    return compile_xy(expr, variables=("x",))


@dataclass
class PlotOptions:
    """绘图参数。

    关于「刻度间隔」（`x_step` / `y_step`）：留空时按 `_nice_step` 自动选一个
    1/2/5×10ⁿ 的步长。用户可以显式指定 —— **不再是固定的格子数**，
    指定 `y_step=1` 就是「每个单位一个刻度」，`x_step=0.5` 就是「每半个单位」。
    """

    width: int = 720
    height: int = 460
    samples: int = 1200
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    grid: bool = True
    #: 刻度间隔（单位 = 数据单位）。None = 自动。
    x_step: Optional[float] = None
    y_step: Optional[float] = None
    #: 单位长度：一个数据单位画多少像素。None = 按画布宽度自动铺开。
    unit_px: Optional[float] = None
    #: 坐标轴省略（截断）区间：把这区间"折叠"成一小段波浪线，两端各留一小截。
    #: `omit_y=(0, 999)` 表示「纵轴上 0 到 999 这一段省略」。None = 不省略。
    omit_y: Optional[tuple] = None
    omit_x: Optional[tuple] = None
    #: 参数方程里 t 的范围（默认一个完整周期 0..2π）与采样点数
    t_min: float = 0.0
    t_max: float = 6.283185307179586
    t_samples: int = 1200
    #: 强制指定表达式形态（explicit / implicit / parametric）；留空自动判定
    variables: str = ""
    #: 是否在图上排一行公式标题。**默认 False** —— 图里只有坐标系与曲线，
    #: 公式在聊天正文里已经有了（参考图格式如此）。
    show_title: bool = False
    #: 两轴统一标度：让「1 个单位」在 x/y 两个方向上是**同样长**的。
    #:
    #: 默认 `None` = **开**：用户要的是课本上的方格纸 —— 格子方正、两轴同标度。
    #: 这两件事是一回事：统一标度下同一个步长在两根轴上跨出的像素数相同，
    #: 格子必然是正方形（见 `uniform_layout`）。
    #: 显式给 `False` 可以关掉，回到「纵轴按数据自适应、画布固定 460」的老行为。
    #: 代价：纵轴窗口可能装不下整条曲线（`x^2` 在 ±10 上跨 100 个单位，
    #: 等比窗口只有二十几个），抛物线会从顶端出去 —— 想看全就收窄 x 范围。
    equal_aspect: Optional[bool] = None
    #: 图标题。**默认空** —— 图里只有坐标系与曲线，公式在聊天正文里已经有了。
    title: str = ""


def _omit_range(omit) -> Optional[tuple[float, float]]:
    """把 omit 参数归一成 `(lo, hi)` 或 None。

    ⚠️ 容错必须在这里，而不是只在配置入口 `_opts_from` 里归一：直接构造
    `PlotOptions(omit_y="0..999")` 是测试与内部调用的常见写法，
    少了这一层就会在 `lo, hi = omit` 处炸（字符串被拆成 3 个字符）。
    """
    if omit is None:
        return None
    if isinstance(omit, str):
        return parse_omit(omit)
    try:
        lo, hi = omit
    except (TypeError, ValueError):
        return None
    lo, hi = _coerce_float(lo), _coerce_float(hi)
    if lo is None or hi is None or hi <= lo:
        return None
    return (lo, hi)

# ============================================================================
# 采样与渲染（matplotlib）
# ============================================================================
#
# 这一段是**整套重写**过的：早先自己算坐标 → 生成 SVG → 交给 Chromium 出 PNG
# （两千行手写几何），现在直接用 matplotlib。换来的是：
#
#   * 等比例与方格：`set_aspect("equal")` 一行，不再需要自己反解窗口与画布高度；
#   * 隐式方程：`contour` 自带等值线，不再需要手写 marching squares；
#   * 多曲线 / 图例 / 配色：matplotlib 的默认行为，不再自己维护配色表；
#   * 不再依赖 Chromium（出图快一个数量级，也少一整条进程链路）。
#
# 保留下来的是**表达式解析与求值安全**（`latex_to_expr` / `compile_xy` /
# `parse_curve`）—— 那部分与「用什么画」无关，而且是安全边界，不该跟着重写。

#: 出图 DPI 与画布上限。宽度按 `PlotOptions.width`（默认 720），高度由数据比例定，
#: 但不超过 `_MAX_FIG_H` —— 超了就收窄窗口（见 `fit_window`）。
_FIG_DPI = 100
_MAX_FIG_H = 900.0
_MIN_FIG_H = 150.0

#: 默认「一个格子 = 一个数据单位」（用户明确要求「默认为 1」）。
#: 只有跨度大到这么多格以上，才换 1/2/5×10ⁿ 的更粗步长 —— 否则
#: `x∈[-100,100]` 会画成两百格、刻度数字糊成一片。
#: 注意纵轴跨度**总是**被 `plan_layout` 收到 `1.25 × 横轴跨度` 以内，
#: 所以这里的格数基本由横轴决定。
_DEFAULT_STEP_MAX_CELLS = 25

#: 隐式方程的采样网格边长（`contour` 要在网格上求值）
_IMPLICIT_GRID = 420

#: 相邻采样点的高度差超过「窗口高度的多少倍」就断开（渐近线）。
#: `tan(x)` 在极点处一步从 +∞ 跳到 −∞，不断开就会画出一条竖直的假线。
_JUMP_FACTOR = 1.0


class _Mpl:
    """matplotlib 的懒加载缓存。

    ⚠️ **`MPLCONFIGDIR` 必须在 import matplotlib 之前设好**：默认路径是
    `~/.config/matplotlib`，而这台机器的根文件系统是**只读**的，matplotlib 会退回去
    用临时目录并每次都在 stderr 上抱怨一句（缓存重建还慢）。设到插件自己的
    `data/` 下，与插件同生命周期。
    """

    mod = None
    plt = None
    locator = None


def _mpl_config_dir() -> str:
    """给 matplotlib 找一个**可写**的缓存目录。"""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.environ.get("AI_CHAT_MPL_CONFIG"),
                 os.path.join(here, "data", "mplconfig"),
                 os.path.join(os.path.dirname(here), "ai_chat", "data", "mplconfig")):
        if not cand:
            continue
        try:
            os.makedirs(cand, exist_ok=True)
        except OSError:
            continue
        if os.access(cand, os.W_OK):
            return cand
    import tempfile
    fallback = os.path.join(tempfile.gettempdir(), "ai_chat-mplconfig")
    try:
        os.makedirs(fallback, exist_ok=True)
    except OSError:
        pass
    return fallback


def _mpl():
    """懒加载 matplotlib，返回 `(matplotlib, pyplot, MultipleLocator)`。"""
    if _Mpl.mod is None:
        import os
        os.environ.setdefault("MPLCONFIGDIR", _mpl_config_dir())
        import matplotlib
        matplotlib.use("Agg")            # 无显示环境，必须在 pyplot 之前定
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MultipleLocator
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False   # 负号别画成方框
        plt.rcParams["figure.autolayout"] = False
        _Mpl.mod, _Mpl.plt, _Mpl.locator = matplotlib, plt, MultipleLocator
    return _Mpl.mod, _Mpl.plt, _Mpl.locator


#: numpy 版的白名单函数：与 `FUNCTIONS` 一一对应，但能**整数组求值**。
#: 隐式方程要在 400×400 的网格上求值（十几万次），标量循环太慢。
def _numpy_functions():
    import numpy as np

    def power(a, b):
        with np.errstate(all="ignore"):
            b = np.asarray(b, dtype=float)
            v = np.power(a, b)
        # 与标量版同一条限制：指数过大 / 结果过大一律判为「这里没定义」
        bad = (np.abs(b) > _MAX_POW_EXP) | ~np.isfinite(v) | (np.abs(v) > _MAX_POW)
        return np.where(bad, np.nan, v)

    return {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "ln": np.log,
        "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt,
        "abs": np.abs, "floor": np.floor, "ceil": np.ceil,
        "sign": np.sign, "min": np.minimum, "max": np.maximum,
        "atan2": np.arctan2, "pow": power,
    }


def compile_vec(expr: str, *, variables: tuple[str, ...] = ("x",)):
    """`compile_xy` 的**数组版**：同样的白名单校验，但用 numpy 函数求值。

    ⚠️ 校验只写一遍（`_validate_expr`），两个版本共用 —— 否则「标量版挡住的写法
    数组版放行」这类漏洞迟早出现。
    """
    import numpy as np
    tree = _validate_expr(expr, variables)
    code = compile(tree, "<plot>", "eval")
    namespace = {"__builtins__": {}, **CONSTANTS, **_numpy_functions()}

    def f(*args):
        local = dict(zip(variables, args))
        try:
            with np.errstate(all="ignore"):
                value = eval(code, namespace, local)   # noqa: S307 - 白名单已校验
        except Exception:  # noqa: BLE001 - 与标量版一致：算不出来就是「没定义」
            return np.full(np.shape(args[0]) if args else (), np.nan, dtype=float)
        try:
            out = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return np.full(np.shape(args[0]) if args else (), np.nan, dtype=float)
        return np.broadcast_to(out, np.shape(args[0])).copy() if args else out

    return f


# ==================== 窗口与刻度 ====================

def nice_step_up(value: float) -> float:
    """1/2/5×10ⁿ 里 ≥ `value` 的最小值。"""
    if not math.isfinite(value) or value <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(value))
    for mult in (1.0, 2.0, 5.0, 10.0):
        if value <= mult * mag + 1e-12:
            return mult * mag
    return 10.0 * mag


def wants_equal(o: "PlotOptions") -> bool:
    """要不要两轴同标度（+ 格子正方）。

    `equal_aspect` 是三态：`None` = **默认开**（用户要求「横纵坐标标度一定相同」）、
    `True`/`False` = 强制开/关。关掉之后纵轴按数据自适应、两根轴各用各的刻度，
    格子不再保证是正方形 —— 那是给「只想看趋势、不在乎形状」的场景留的口子。
    """
    return True if o.equal_aspect is None else bool(o.equal_aspect)


def shared_step(span_x: float, span_y: float, *,
                x_step=None, y_step=None, equal: bool = True) -> float:
    """两根轴**共用**的刻度间隔 —— 默认 **1 个单位一格**。

    为什么必须是「一个」：`set_aspect("equal")` 只保证两轴同标度；要让**格子**
    也是正方形，两根轴还得用同一个步长（同一个步长 × 同一个像素/单位 = 同一个
    像素数）。用户要求原话就是「横纵坐标标度一定相同，默认为 1」。

    用户显式给了 `x_step` / `y_step` 时按他说的来；两个都给且不同时，格子就不再
    是正方形 —— 那是用户自己的选择。
    """
    given_x = _coerce_float(x_step)
    given_y = _coerce_float(y_step)
    if given_x and given_x > 0 and given_y and given_y > 0:
        return given_x if given_x == given_y else None   # None = 各用各的
    if not equal:
        # 关掉等比例时不必强求方格，交给 matplotlib 的自动刻度（更好看）
        return given_x or given_y or None
    for given in (given_x, given_y):
        if given and given > 0:
            return given
    # 用**横轴**跨度定步长：纵轴跨度总被 `plan_layout` 收在 `1.25 × 横轴` 以内，
    # 拿两者的最大值去算只会把步长白抬一档（实测 `x∈[-100,100]` 得到 50 而不是 20）。
    span = abs(span_x)
    if 1.0 <= span <= _DEFAULT_STEP_MAX_CELLS:
        return 1.0
    return nice_step_up(span / 10.0)


def effective_omit(omit, data_lo: float, data_hi: float) -> Optional[tuple]:
    """省略区间是否**真的有用**。

    判据两条，缺一不可（都是实测踩出来的）：

    ① 数据跨在省略段两侧（`tan(x)` 配 `-10..10`）—— 用**极值**判，不用「段内有没有
       采样点」：`tan` 在极点处直接从 +1000 跳到 −1000，采样点根本不落在段内；
    ② 数据整段在省略段**一侧、且紧挨着它**（`y=sin(x)+1000` 配 `0..999`）——
       这是用户提需求时给的例子。此时波浪线只是告诉用户「下面还跳过了一段」。

    跟数据八竿子打不着的区间不算（`y=x` 配 `1000..2000`）：那只会把窗口撑到 2000。
    """
    got = _omit_range(omit)
    if got is None:
        return None
    lo, hi = got
    if data_lo < lo and data_hi > hi:
        return got
    slack = max(data_hi - data_lo, abs(data_lo) * 0.05, abs(data_hi) * 0.05, 1.0)
    if data_lo >= hi and data_lo - hi <= slack:
        return got
    if data_hi <= lo and lo - data_hi <= slack:
        return got
    return None


def plan_layout(o, x_min: float, x_max: float, x_data, ys: list[float]
                ) -> tuple[list[tuple[float, float]], set[str], float]:
    """定下「画哪几块面板、断口在哪几条边、共用步长多大」。

    返回 `(panel 列表, 断口边集合, 步长)`；面板从上到下，每个是 `(lo, hi)`。

    三种情形（`sin(x)+1000` 省略 `0..999` 是第三种，也是用户报的那个）：

    | 省略段相对窗口 | 结果 |
    |----------------|------|
    | 落在窗口**内部** | 上下两块面板，中间断开 |
    | 紧贴窗口**下方 / 上方** | 一块面板，断口画在对应的那条边上 |
    | 跟窗口无交集 | 不省略 |

    ⚠️ 紧贴时**不**把窗口撑到省略段里去。撑进去的后果是窗口一下从 2 个单位变成
    1001 个单位，等比例下要画成 500:1 的竖条；而波浪线本来只是「这里跳过了一段」
    的记号，不需要真把那一段画出来。
    """
    span_x = max(x_max - x_min, 1e-9)
    equal = wants_equal(o)
    max_ratio = _MAX_FIG_H / max(1.0, float(o.width)) if equal else float("inf")
    min_ratio = _MIN_FIG_H / max(1.0, float(o.width)) if equal else 0.0

    d_lo = min(ys) if ys else 0.0
    d_hi = max(ys) if ys else 1.0
    if not (d_hi - d_lo > 0):
        d_lo, d_hi = d_hi - 0.5, d_hi + 0.5

    band = effective_omit(o.omit_y, d_lo, d_hi)
    breaks: set[str] = set()
    if band is None:
        panels = [(d_lo, d_hi)]
    else:
        b_lo, b_hi = band
        below, above = d_lo < b_lo, d_hi > b_hi   # 数据在省略段两侧都还有点
        if below and above:
            panels = [(d_lo, b_lo), (b_hi, d_hi)]     # 省略段在中间 → 两块面板
        else:
            panels = [(d_lo, d_hi)]
            # 数据整段在省略段**上方**（`sin(x)+1000` 省略 `0..999`）→ 断口在底边：
            # 波浪线只是告诉用户「下面还跳过了一段」，那一段本来就没画进来。
            breaks.add("bottom" if above else "top")

    # 每块面板都留一点边距，别让曲线贴着框
    pads = []
    for lo, hi in panels:
        pad = max((hi - lo) * 0.06, 1e-9)
        pads.append((lo - pad, hi + pad))

    # 总高度（数据单位）→ 收进画布能承受的长宽比
    total = sum(hi - lo for lo, hi in pads)
    cap = span_x * max_ratio
    floor = span_x * min_ratio
    if total > cap and total > 0:
        k = cap / total
        pads = [(lo + (hi - lo) * (1 - k) / 2, hi - (hi - lo) * (1 - k) / 2)
                for lo, hi in pads]
        total = sum(hi - lo for lo, hi in pads)
    if total < floor and len(pads) == 1:
        lo, hi = pads[0]
        c = (lo + hi) / 2
        pads = [(c - floor / 2, c + floor / 2)]

    # 用户显式给了上下界就照用（这时只有一块面板）
    if o.y_min is not None or o.y_max is not None:
        lo = float(o.y_min) if o.y_min is not None else pads[0][0]
        hi = float(o.y_max) if o.y_max is not None else pads[0][1]
        if hi > lo:
            pads = [(lo, hi)]

    step = shared_step(span_x, sum(hi - lo for lo, hi in pads),
                       x_step=o.x_step, y_step=o.y_step, equal=equal)
    return pads, breaks, step


# ==================== 采样 ====================
#
# 三种曲线最后都归成**同一种形状**：一串 `(x 数组, y 数组)` 折线段。
# 于是渲染那边只有一条 `ax.plot` 路径 —— 显式函数、参数方程、隐式方程（等值线）
# 没有区别，配色、线宽、图例也都只有一份实现。

@dataclass
class PlotCurve:
    """一条待画的曲线：若干段（极点/无定义处断开）。"""

    label: str = ""
    segments: list = field(default_factory=list)


def _split_finite(xs, ys):
    """按「两个坐标都有限」切成若干连续段（不连续处断开）。"""
    out: list[tuple[list, list]] = []
    cur_x: list = []
    cur_y: list = []
    for a, b in zip(xs, ys):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and math.isfinite(a) and math.isfinite(b):
            cur_x.append(float(a))
            cur_y.append(float(b))
        else:
            if len(cur_x) >= 2:
                out.append((cur_x, cur_y))
            cur_x, cur_y = [], []
    if len(cur_x) >= 2:
        out.append((cur_x, cur_y))
    return out


def _break_jumps(ys: list, span_y: float) -> None:
    """把「一步跨过整个窗口」的点断开（就地改成 nan）。

    不断开的话，`tan(x)` 在极点两侧会被一条竖直的假线连起来 —— 那不是函数的一部分，
    只是采样恰好跨过了渐近线。
    """
    limit = max(span_y, 1e-9) * _JUMP_FACTOR
    for i in range(len(ys) - 1):
        a, b = ys[i], ys[i + 1]
        if isinstance(a, float) and isinstance(b, float) \
                and math.isfinite(a) and math.isfinite(b) and abs(b - a) > limit:
            ys[i + 1] = math.nan


def _samples_explicit(sp, o, x_min: float, x_max: float):
    xs = _linspace(x_min, x_max, o.samples)
    ys: list = []
    for v in xs:
        try:
            ys.append(sp.f(float(v)))
        except Exception:  # noqa: BLE001 - 与求值器一致：算不出来就是没定义
            ys.append(math.nan)
    return xs, ys


def _samples_parametric(sp, o):
    ts = _linspace(o.t_min, o.t_max, o.t_samples)
    xs, ys = [], []
    for t in ts:
        try:
            xs.append(sp.f(float(t)))
            ys.append(sp.g(float(t)))
        except Exception:  # noqa: BLE001
            xs.append(math.nan)
            ys.append(math.nan)
    return xs, ys


def _segments_implicit(sp, x_min: float, x_max: float,
                       y_lo: float, y_hi: float) -> list:
    """隐式方程 → 折线段：`contourpy` 提 0 等值线（matplotlib 的等值线引擎）。

    比手写 marching squares 省掉一整块几何代码，而且 `contourpy` 是 matplotlib 的
    硬依赖 —— 用 matplotlib 就一定在，不算新增依赖。
    """
    if not (y_hi > y_lo):
        return []
    import contourpy
    import numpy as np
    n = int(_IMPLICIT_GRID)
    gx = np.linspace(x_min, x_max, n)
    gy = np.linspace(y_lo, y_hi, n)
    X, Y = np.meshgrid(gx, gy)
    if sp.vec is None:
        return []
    Z = np.asarray(sp.vec(X, Y), dtype=float)
    # contourpy 不收 nan：把「没定义」的地方换成同号的大数，等值线自然不会穿过去
    # （`y-sqrt(x)` 在 x<0 无定义，就该在那里断开，而不是硬连一条线）
    Z = np.nan_to_num(Z, nan=1e30, posinf=1e30, neginf=-1e30)
    if not np.any(np.isfinite(Z)):
        return []
    try:
        gen = contourpy.contour_generator(x=gx, y=gy, z=Z)
        lines = gen.lines(0.0)
    except Exception:  # noqa: BLE001 - 等值线内部异常不该让整张图失败
        return []
    return [(seg[:, 0].tolist(), seg[:, 1].tolist())
            for seg in lines if len(seg) >= 2]


def _probe_span(x_min: float, x_max: float) -> float:
    """隐式方程的探查半径：按横轴量级给（圆/椭圆够用）。

    不能「先按默认窗口求一次、再拿结果扩窗口」—— 被裁过的曲线只能告诉你它被裁到
    哪儿。所以探查窗口要一次给足。
    """
    return max(abs(x_min), abs(x_max), 1.0)


def build_curves(exprs: list, o: "PlotOptions", x_min: float, x_max: float
                 ) -> tuple[list[PlotCurve], list, set, float]:
    """两趟采样：先探数据的纵向范围定窗口，再按最终窗口重采一次。

    为什么必须两趟：`y=x^2` 在 ±10 上跨 100 个单位，而等比例下窗口能覆盖多少单位
    由画布长宽比决定 —— 窗口要**看着数据**才能定，定完又得按它把渐近线断开、
    把隐式方程的网格铺对。
    """
    specs = [parse_curve(e, variables=o.variables) for e in exprs]

    # ---- 第一趟：探纵向范围 ----
    ys: list[float] = []
    for sp in specs:
        if sp.kind == KIND_EXPLICIT:
            _xs, probe = _samples_explicit(sp, o, x_min, x_max)
            ys.extend(v for v in probe
                      if isinstance(v, float) and math.isfinite(v))
        elif sp.kind == KIND_PARAMETRIC:
            _xs, probe = _samples_parametric(sp, o)
            ys.extend(v for v in probe
                      if isinstance(v, float) and math.isfinite(v))
        else:
            r = _probe_span(x_min, x_max)
            for _sx, sy in _segments_implicit(sp, x_min, x_max, -r, r):
                ys.extend(sy)
    ys = [v for v in ys if isinstance(v, float) and math.isfinite(v)]

    panels, breaks, step = plan_layout(o, x_min, x_max, None, ys)
    y_lo = min(lo for lo, _hi in panels)
    y_hi = max(hi for _lo, hi in panels)
    span_y = y_hi - y_lo

    # ---- 第二趟：按最终窗口重采 ----
    curves: list[PlotCurve] = []
    for sp in specs:
        if sp.kind == KIND_EXPLICIT:
            xs, probe = _samples_explicit(sp, o, x_min, x_max)
            _break_jumps(probe, span_y)
            curves.append(PlotCurve(sp.source,
                                    _split_finite(xs, probe)))
        elif sp.kind == KIND_PARAMETRIC:
            xs, probe = _samples_parametric(sp, o)
            curves.append(PlotCurve(sp.source, _split_finite(xs, probe)))
        else:
            curves.append(PlotCurve(
                sp.source, _segments_implicit(sp, x_min, x_max, y_lo, y_hi)))
    return curves, panels, breaks, step


# ==================== 渲染 ====================

def _default_colors(n: int) -> list[str]:
    """matplotlib 默认配色环（`axes.prop_cycle`）→ 十六进制串，不自己维护一份。

    ⚠️ `prop_cycle` 给的是 **RGBA 元组**，不是十六进制串 —— 直接拿去查颜色名会
    在 `.lower()` 上炸（实测）。统一过一道 `to_hex`。
    """
    mod, plt, _ = _mpl()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color") or []
    if not cycle:
        cycle = ["#1f77b4"]
    return [mod.colors.to_hex(cycle[i % len(cycle)]) for i in range(max(n, 1))]


#: 默认配色对应的中文名（工具回执里告诉模型，正文里才说得清哪条是哪条）
_COLOR_NAMES = {"#1f77b4": "蓝色", "#ff7f0e": "橙色", "#2ca02c": "绿色",
                "#d62728": "红色", "#9467bd": "紫色", "#8c564b": "棕色",
                "#e377c2": "粉色", "#7f7f7f": "灰色", "#bcbd22": "黄绿色",
                "#17becf": "青色"}


def color_names(n: int) -> list[str]:
    return [_COLOR_NAMES.get(c.lower(), c) for c in _default_colors(n)]


def _stroke(ax, xs, ys) -> None:
    """先描一道白线盖掉原边框、再画深色波浪 —— 出来就是「轴在这里断开」。"""
    for color, lw, z in (("white", 4.5, 9), ("#1f2329", 1.3, 10)):
        ax.plot(xs, ys, transform=ax.transAxes, color=color, lw=lw,
                clip_on=False, zorder=z, solid_capstyle="butt")


def _wavy(ax, edge: str, *, amp: float = 0.014, waves: int = 7) -> None:
    """沿**面板交界**（两块面板之间那道缝）画一条波浪线。

    ⚠️ 只该用在**两块面板之间**：那里本来就有一道缝，波浪画在缝上，
    一眼就是「坐标轴在这里断开」。只有一块面板时**不能**这么画 ——
    省略段紧贴窗口下方时，波浪会落在整条底边上，而底边正是**横轴**，
    看起来就成了「横轴在抖」（用户实际就这么问了）。那种情况走 `_wavy_spine`。
    """
    import numpy as np
    xs = np.linspace(0.0, 1.0, 360)
    base = 0.0 if edge == "bottom" else 1.0
    ys = base + amp * np.sin(2 * math.pi * waves * xs) * (1 if edge == "bottom" else -1)
    _stroke(ax, xs, ys)


def _wavy_spine(ax, edge: str, *, sides=("left", "right"),
                length_px: float = 24.0, amp_px: float = 6.0,
                waves: int = 3) -> None:
    """在**纵轴那一侧**（左边框 / 右边框）靠近某条边处画一小段波浪。

    只有一块面板、而省略段又紧贴窗口下方（或上方）时用它：那一段被跳过的区间
    **根本没画进窗口**，波浪线只是个记号，告诉用户「这条轴在这里还有内容被截掉了」。
    记号落在纵轴上才读得对 —— 省略的本来就是纵轴上的一段。

    ⚠️ 尺寸按**像素**算再折成 axes 比例：波浪的长度与摆幅要跟图的大小无关，
    否则画布一变高，同一个数字就从「一小段波浪」变成「一条细长的抖动」。
    """
    import numpy as np
    box = ax.get_window_extent()
    w = max(float(box.width), 1.0)
    h = max(float(box.height), 1.0)
    ext = min(length_px / h, 0.45)          # 沿轴方向占多长
    amp = min(amp_px / w, 0.2)              # 摆幅
    for side in sides:
        x0 = 0.0 if side == "left" else 1.0
        sgn = 1.0 if side == "left" else -1.0
        ys = (np.linspace(0.0, ext, 200) if edge == "bottom"
              else np.linspace(1.0 - ext, 1.0, 200))
        frac = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-9)
        xs = x0 + sgn * amp * np.sin(2 * math.pi * waves * frac)
        _stroke(ax, xs, ys)


def render_png(curves: list, o, *, x_min: float, x_max: float,
               panels: list, breaks: set, step, out_path: str,
               out_probe: Optional[list] = None) -> str:
    """把曲线画成 PNG，返回路径。"""
    _, plt, MultipleLocator = _mpl()

    span_x = max(x_max - x_min, 1e-9)
    total_y = sum(hi - lo for lo, hi in panels)
    equal = wants_equal(o)
    unit = _coerce_float(o.unit_px)
    if unit and unit > 0:
        # `unit_px` = 一个数据单位画多少像素。等比例下它把**坐标系的大小**定死了：
        # 宽 = 横轴跨度 × unit_px，高 = 纵轴跨度 × unit_px。
        # ⚠️ 这里必须把 subplot 边距清零，让坐标轴**就等于**整个 figure ——
        # 留边距的话坐标轴会小一圈，「一个单位 N 像素」就名不副实（实测差 7.7%）。
        # 刻度数字不会因此被切掉：`bbox_inches="tight"` 会把画布**向外扩**到装下它们。
        fig_w = min(max(1.0, span_x * unit / _FIG_DPI), 16.0)
        fig_h = min(max(0.6, total_y * unit / _FIG_DPI), 16.0)
    else:
        fig_w = max(2.0, float(o.width) / _FIG_DPI)
        fig_h = max(1.1, fig_w * total_y / span_x) if equal else 4.6

    if len(panels) == 1:
        fig, ax0 = plt.subplots(figsize=(fig_w, fig_h), dpi=_FIG_DPI)
        axes = [ax0]
    else:
        # 断轴：上下两块面板共享横轴，高度按各自的数据跨度给 ——
        # 这样 `set_aspect("equal")` 在两块里解出的「像素/单位」是**同一个**，
        # 跨断口也仍然是统一标度。
        fig, axs = plt.subplots(
            len(panels), 1, sharex=True, figsize=(fig_w, fig_h), dpi=_FIG_DPI,
            gridspec_kw={"height_ratios": [hi - lo for lo, hi in panels],
                         "hspace": 0.12})
        axes = list(axs)

    for ax, (lo, hi) in zip(axes, panels):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(lo, hi)
        # 两轴同标度（默认）；`equal_aspect=false` 时交回 matplotlib 自动缩放
        ax.set_aspect("equal" if equal else "auto", adjustable="box")
        ax.grid(True, color="#c0c0c0", linewidth=0.7, alpha=0.9)
        ax.tick_params(labelsize=9)
        if step:
            ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.yaxis.set_major_locator(MultipleLocator(step))
        else:
            # 用户给了两个**不同**的步长：照他说的来，此时格子不再是正方形
            if _coerce_float(o.x_step):
                ax.xaxis.set_major_locator(MultipleLocator(float(o.x_step)))
            if _coerce_float(o.y_step):
                ax.yaxis.set_major_locator(MultipleLocator(float(o.y_step)))

    colors = _default_colors(max(len(curves), 1))
    for idx, cur in enumerate(curves):
        color = colors[idx]
        for j, (xs, ys) in enumerate(cur.segments):
            for ax in axes:
                ax.plot(xs, ys, color=color, linewidth=1.8,
                        label=cur.label if j == 0 else None)

    if len(curves) >= 2:
        axes[0].legend(fontsize=9, loc="best", framealpha=0.9)

    if o.show_title and o.title:
        axes[0].set_title(o.title, fontsize=12)

    if unit and unit > 0:
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    if len(axes) > 1:
        # 断口：交界处两侧各画一条波浪，并把相对的那两条边框去掉
        for ax in axes[:-1]:
            _wavy(ax, "bottom")
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False,
                           labelbottom=False)
        for ax in axes[1:]:
            _wavy(ax, "top")
            ax.spines["top"].set_visible(False)
    else:
        # 只有一块面板：记号画在**纵轴**上（底边就是横轴，画上去会被当成横轴在抖）
        for edge in breaks:
            _wavy_spine(axes[0], edge)

    if out_probe is not None:
        # 调试/测试出口：把每个面板的「一个数据单位多少像素」交出去。
        # 「两轴标度相同」「格子是正方形」这类断言只能靠它 —— 从 PNG 上量网格线
        # 会被坐标取整与像素栅格带偏。
        fig.canvas.draw()
        for ax, (lo, hi) in zip(axes, panels):
            box = ax.get_window_extent()
            out_probe.append({
                "x_span": float(ax.get_xlim()[1] - ax.get_xlim()[0]),
                "y_span": float(hi - lo),
                "w_px": float(box.width), "h_px": float(box.height),
                "step": step,
            })

    fig.savefig(out_path, dpi=_FIG_DPI, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    return out_path


def build_plot_png(expr, o: "PlotOptions", *, out_dir: str, title: str = "",
                   out_probe: Optional[list] = None) -> tuple[str, list[str]]:
    """表达式 → PNG。返回 `(路径, 每条曲线的颜色名)`。

    `expr` 可以是一个表达式，也可以是一串（多条曲线画在同一张图里）。
    """
    exprs = [expr] if isinstance(expr, str) else [e for e in (expr or []) if e]
    if not exprs:
        raise PlotError("没有可画的表达式")
    o.title = title

    x_min = -10.0 if o.x_min is None else float(o.x_min)
    x_max = 10.0 if o.x_max is None else float(o.x_max)
    if not (x_max > x_min):
        raise PlotError("横轴范围不对：x_max 必须大于 x_min")

    curves, panels, breaks, step = build_curves(exprs, o, x_min, x_max)
    if not any(c.segments for c in curves):
        raise PlotError("这段范围内画不出任何东西（表达式处处无定义？）")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"plot-{int(time.time() * 1000)}.png")
    render_png(curves, o, x_min=x_min, x_max=x_max, panels=panels,
               breaks=breaks, step=step, out_path=out_path, out_probe=out_probe)
    return out_path, color_names(len(curves))


def _linspace(lo: float, hi: float, n) -> list[float]:
    n = max(2, int(n or 2))
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def _coerce_float(value) -> Optional[float]:
    """把模型给的上下界转成 float；给不出数字就当作「没给」（走自动适配）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _opts_from(cfg: dict, raw: Optional[dict]) -> PlotOptions:
    """配置默认值 + 模型给的参数 → PlotOptions。"""
    # ⚠️ `equal_aspect` 不要在这里传：它是三态（None/True/False），
    # `bool(cfg.get(...))` 会把 None 塌成 False，于是「指定刻度就统一标度」的
    # 默认行为永远不生效。它在函数末尾单独处理。
    opts = PlotOptions(
        width=int(cfg.get("width") or 720),
        height=int(cfg.get("height") or 460),
        samples=int(cfg.get("samples") or 1200),
        grid=bool(cfg.get("grid", True)),
        show_title=bool(cfg.get("show_title", False)),
    )
    for key in ("x_min", "x_max", "y_min", "y_max", "x_step", "y_step", "unit_px",
                "t_min", "t_max"):
        value = _coerce_float((raw or {}).get(key))
        if value is not None:
            setattr(opts, key, value)
    if (raw or {}).get("variables"):
        opts.variables = str((raw or {}).get("variables")).strip().lower()
    for axis in ("y", "x"):
        got = parse_omit((raw or {}).get(f"omit_{axis}"))
        if got is None:
            got = parse_omit(cfg.get(f"omit_{axis}"))
        if got is not None:
            setattr(opts, f"omit_{axis}", got)
    # 配置里的刻度间隔作为默认值（模型没显式给时生效）
    for key in ("x_step", "y_step", "unit_px"):
        if getattr(opts, key) is None:
            value = _coerce_float(cfg.get(key))
            if value is not None:
                setattr(opts, key, value)
    # `equal_aspect` 是三态：None = 用默认（**开**）、True/False = 强制开关。
    # 模型没给就看配置；配置也没写（None）就保持 None —— 不能在这一步塌成 False，
    # 否则「默认统一标度」会永远失效（实测踩过）。
    aspect = (raw or {}).get("equal_aspect")
    if aspect is None:
        aspect = cfg.get("equal_aspect")
    if aspect is not None and str(aspect) != "":
        opts.equal_aspect = _as_bool(aspect)
    return opts


#: 从「省略区间」字符串里取两端的带符号数字
_OMIT_ENDS_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
#: 两个数字之间允许出现的分隔符（`..`、`~`、`-`、`—`、逗号、空白）
_OMIT_ONLY_SEP_RE = re.compile(r"[\s.,~\-—]*")


def parse_omit(text) -> Optional[tuple[float, float]]:
    """解析「省略区间」写法：`0..999`、`0-999`、`0~999`、`0 999`、`(0, 999)`。

    为什么要支持这么多种：模型和用户都会随手写，而这一段信息只影响「轴怎么截断」，
    为格式差异去报错、让整张图画不出来，代价完全不成比例。
    """
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        if len(text) != 2:
            return None
        lo, hi = _coerce_float(text[0]), _coerce_float(text[1])
    else:
        raw = str(text).strip().strip("()[]").strip()
        if not raw:
            return None
        # 从**两端**各取一个带符号的数，中间那段就是分隔符。
        # ⚠️ 不能用「按分隔符 split」的写法：分隔符里含 `-`，会把负号一起吃掉
        # （`-10..10` 被切成 `['10','10']`，于是「省略 -10..10」静默失效 —— 实测）。
        found = _OMIT_ENDS_RE.findall(raw)
        if len(found) != 2 or not _OMIT_ONLY_SEP_RE.fullmatch(
                raw[raw.index(found[0]) + len(found[0]):raw.rindex(found[1])]):
            return None
        lo, hi = _coerce_float(found[0]), _coerce_float(found[1])
    if lo is None or hi is None:
        return None
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi) if hi > lo else None


def _as_bool(value) -> bool:
    """把模型给的开关转成 bool（它会传 "true" / True / 1 各种形态）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in ("1", "true", "yes", "on", "是")


def marker_for(path: str, title: str = "") -> str:
    """工具结果里那行占位标记（主流程负责换成图片）。"""
    safe = str(title or "").replace(PLOT_MARKER_PREFIX, "").replace(
        PLOT_MARKER_SUFFIX, "").replace("\n", " ").strip()
    return f"{PLOT_MARKER_PREFIX}{path}|{safe}{PLOT_MARKER_SUFFIX}"


_MARKER_RE = re.compile(r"\[\[plot:([^\]|]+)(?:\|([^\]]*))?\]\]")


def extract_markers(text: str) -> list[tuple[str, str]]:
    """从回复正文里取出 `[[plot:路径|标题]]` → `[(路径, 标题), …]`。"""
    return [(m.group(1), (m.group(2) or "").strip())
            for m in _MARKER_RE.finditer(text or "")]


def strip_markers(text: str) -> str:
    """把标记从正文里去掉，并压掉它留下的空行。"""
    cleaned = _MARKER_RE.sub("", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class PlotService:
    """把「表达式 → 图片」包成工具能用的形状。

    **直接用 matplotlib 出图**，不再经过 Chromium（早先那条路是「自己生成 SVG →
    丢给无头浏览器截图」，现在只有 Markdown / 公式 / 历史卡片还需要浏览器）。

    仍然保留的两条老规矩：

    * **渲染成功才留标记**：失败时返回给模型的是错误说明，主流程不会去附件路径。
      宁可告诉用户「画不出来 + 原因」，也不要发一个坏掉的图片节点；
    * `rendered` 记下真的产出过哪些文件，主流程只认这些路径 —— 模型被提示注入
      诱导、自己编一个 `[[plot:/etc/passwd]]` 是发不出图片的。
    """

    def __init__(self, renderer=None, cfg: dict = None, logger=None,
                 data_dir: str = "") -> None:
        # `renderer` 保留在签名里只是为了兼容旧调用（历史卡片那条路还在用它），
        # 函数图已经完全不碰它了。
        self.renderer = renderer
        self.cfg = cfg or {}
        self.logger = logger
        self.rendered: set[str] = set()
        #: 出图落盘目录。由主流程传进来（与 sessions/ 同级），不再去猜配置键。
        self.data_dir = str(data_dir or "")

    def _out_dir(self) -> str:
        base = self.data_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data")
        return os.path.join(base, "plots")

    async def plot(self, expression, *, opts: Optional[dict] = None,
                   title: str = "") -> str:
        """`expression` 可以是一条（字符串）或多条（列表）。

        多条时按 matplotlib 的默认配色环依次取色，并把**每条的颜色**写进工具回执 ——
        图里虽然有图例，但正文里按颜色说一句更清楚。
        """
        exprs = [expression] if isinstance(expression, str) else list(expression or [])
        try:
            for e in exprs:
                parse_curve(e)
            o = _opts_from(self.cfg, opts)
        except PlotError as e:
            return f"无法画图：{e}。请修正表达式后重试。"
        except Exception as e:  # noqa: BLE001 - 解析阶段的意外
            return f"无法画图：{e}"

        try:
            path, names = await asyncio.get_event_loop().run_in_executor(
                None, lambda: build_plot_png(exprs, o, out_dir=self._out_dir(),
                                             title=title))
        except PlotError as e:
            return f"无法画图：{e}。请修正表达式后重试。"
        except Exception as e:  # noqa: BLE001 - 出图是重活，什么都可能
            if self.logger:
                self.logger.warning("[ai_chat] 函数图像渲染失败: %s", e)
            return (f"图渲染失败（{e}）。可以先用文字描述这个函数的形状，"
                    f"并说明图片暂时画不出来。")

        self.rendered.add(path)
        kinds = {sp.kind for sp in (parse_curve(e) for e in exprs)}
        kind_label = ("方程" if kinds == {KIND_IMPLICIT}
                      else "参数曲线" if kinds == {KIND_PARAMETRIC}
                      else "函数")
        if len(exprs) == 1:
            lines = [f"已生成{kind_label}图像：{exprs[0].strip()}。"]
        else:
            pairs = [f"  {i + 1}. {e.strip()} —— {names[i]}"
                     for i, e in enumerate(exprs)]
            lines = [f"已把 {len(exprs)} 条曲线画在**同一张图**里（图内已有图例）：",
                     "\n".join(pairs)]
        lines.append("（图片会作为单独一条消息附在回复里，"
                     "正文里不用再贴表达式或描述坐标轴。）")
        cap = title or "、".join(e.strip() for e in exprs)
        return "\n".join(lines) + f"\n{marker_for(path, cap)}"
