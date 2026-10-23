"""会话历史的「问答对」视图 —— 纯数据，不碰渲染、不起进程、不读配置。

为什么要单独一层：一对问答在落盘结构里是**一串**消息（用户输入 + 若干次
「模型决定调工具」+ 若干条工具结果 + 最终回答），而用户想看的是
「我问了什么 / 它答了什么」。这个「串 → 对」的归并规则是本功能唯一的语义核心，
所以它单独一个模块、单独一组单测；渲染和命令处理都只是它的消费者。

**编号从 1 开始、按时间递增（最老 = 1）**，且 `list` 与 `get` 用**同一个**编号空间：
两个入口都调 `extract_exchanges()`，不各算一套。

编号是「当前历史里第几对」，不是永久 id —— 压缩（compaction）会把老消息换成一段
摘要，那些问答就**从编号里消失**，后面的整体前移。这是有意的：压缩后正文已经不在
历史里了，渲染它就只能靠摘要瞎编。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .segments import (BLOCK_TEXT, BLOCK_TOOL_CALL, BLOCK_TOOL_RESULT,
                       ORIGIN_COMPACT, USER_ORIGINS, Message)

#: 清单里单条预览最多取多少字符（一行放不下就截断，清单本身还会再分页）
PREVIEW_CHARS = 60

_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})")


def short_time(ts: str) -> str:
    """ISO8601 → `MM-DD HH:MM`；解析不出来就原样返回（不猜）。"""
    m = _TS_RE.match(str(ts or ""))
    if not m:
        return str(ts or "")
    _, month, day, hour, minute = m.groups()
    return f"{month}-{day} {hour}:{minute}"


def speakername(meta: dict) -> str:
    """提问人显示名：群名片 > 昵称 > QQ 号。

    与包装层（`core.wrap_user_text`）的口径一致 —— 群里用名片是因为那才是
    群友互相认识的名字。
    """
    for key in ("card", "nickname"):
        value = str((meta or {}).get(key) or "").strip()
        if value:
            return value
    uid = int((meta or {}).get("user_id") or 0)
    return f"用户{uid}" if uid else "（未知）"


@dataclass
class Exchange:
    """一对问答（`index` 从 1 开始，按时间递增）。"""

    index: int
    question: str
    answer: str
    speaker: str = ""
    user_id: int = 0
    time: str = ""                      # 已格式化成 MM-DD HH:MM
    tool_calls: list = field(default_factory=list)   # 调过的工具名（去重、保序）
    tool_count: int = 0                 # 工具调用总次数（含重复）
    seq: int = 0                        # 首条消息的 seq（诊断用）
    answer_seq: int = 0                 # 回答那条消息的 seq（卡片上标 `[#N]` 用）
    #: 这一对各条**提问消息**的 QQ 消息 id —— 「按 id 反查是哪一对」靠它
    message_ids: list = field(default_factory=list)
    #: 这一对引用了哪些 QQ 消息 id（沿引用链追溯的起点）
    quotes: list = field(default_factory=list)

    @property
    def tool_note(self) -> str:
        """「🔧 调用了 get_notice ×2」这类一行小字；没调工具就是空串。"""
        if not self.tool_calls:
            return ""
        names = "、".join(self.tool_calls)
        if self.tool_count > len(self.tool_calls):
            return f"🔧 调用了 {names} ×{self.tool_count}"
        return f"🔧 调用了 {names}"


def _text_of(msg: Message) -> str:
    """消息里的纯文本（不含工具块），两端空白已去掉。"""
    return "".join(b.text for b in msg.blocks if b.type == BLOCK_TEXT).strip()


def is_question(msg: Message) -> bool:
    """算不算「用户提问」。

    只认 `user.*` 来源：压缩摘要（`role=user` + `origin=system.compact`）是系统
    注入的接手说明，把它当成用户提问会让清单里冒出一堆「没人问过的问题」。
    """
    return msg.origin in USER_ORIGINS


def is_answer(msg: Message) -> bool:
    """算不算「模型的回答」：assistant 角色且真的有正文。"""
    return msg.role == "assistant" and bool(_text_of(msg))


def _is_toolish(msg: Message) -> bool:
    """工具调用（模型发起）或工具结果 —— 都不进正文，只计数。"""
    if msg.role == "tool":
        return True
    return any(b.type in (BLOCK_TOOL_CALL, BLOCK_TOOL_RESULT) for b in msg.blocks)


def _collect_tools(span: Iterable[Message]) -> tuple[list[str], int]:
    """统计这一对里调过哪些工具。返回 `(去重保序的工具名, 总次数)`。"""
    names: list[str] = []
    total = 0
    for msg in span:
        for block in msg.blocks:
            if block.type != BLOCK_TOOL_CALL:
                continue
            total += 1
            name = str(block.name or "").strip() or "（未命名工具）"
            if name not in names:
                names.append(name)
    return names, total


def extract_exchanges(messages: Iterable[Message]) -> list[Exchange]:
    """把会话历史归并成问答对。

    切分规则（四条例外都必须这样处理，否则清单会骗人）：

    1. **连续多条 `user.*` 算同一对**（一次输入可能分几段发），起点取这段的第一条；
    2. 到**下一段**用户输入之前都算这一对；
    3. 答案是这一对里**最后一个有正文的 assistant 消息** —— 工具循环中间那几条
       （「我先查一下」）不是答案，取第一条会让用户看到半句话；
    4. 没有 assistant 正文的对（答到一半崩了 / 刚问还没答）**直接跳过**，
       不占用编号 —— 否则 `list` 里的编号与 `get` 会错位（都跳过就一致）。
    """
    msgs = list(messages)
    # 先找每一对的起点：一段**连续**用户输入的第一条。
    # ⚠️ 不能「每条 user 消息都当起点」—— 那样一次分两段发的提问会被拆成两对，
    # 前一对没有答案而被跳过，用户只看到最后一段（实测踩过）。
    starts: list[int] = []
    for i, msg in enumerate(msgs):
        if not is_question(msg):
            continue
        if i == 0 or not is_question(msgs[i - 1]):
            starts.append(i)
    out: list[Exchange] = []
    for n, start in enumerate(starts):
        stop = starts[n + 1] if n + 1 < len(starts) else len(msgs)
        span = msgs[start:stop]
        head = span[0]

        # 把这一对里**所有** user 消息拼成问题
        parts = [t for t in (_text_of(m) for m in span if is_question(m)) if t]
        question = "\n".join(parts)
        answer = ""
        for msg in reversed(span):
            if is_answer(msg):
                answer = _text_of(msg)
                break
        if not question or not answer:
            continue                # 不完整的一对：跳过，不占编号

        names, total = _collect_tools(span)
        msg_ids: list[int] = []
        quotes: list[int] = []
        for m in span:
            if not is_question(m):
                continue
            meta = m.metadata or {}
            for raw in meta.get("message_ids") or []:
                try:
                    msg_ids.append(int(raw))
                except (TypeError, ValueError):
                    continue
            try:
                quoted = int(meta.get("reply_to") or 0)
            except (TypeError, ValueError):
                quoted = 0
            if quoted and quoted not in quotes:
                quotes.append(quoted)
        answer_seq = 0
        for m in reversed(span):
            if is_answer(m):
                answer_seq = int(m.seq or 0)
                break
        out.append(Exchange(
            index=len(out) + 1,
            question=question,
            answer=answer,
            speaker=speakername(head.metadata),
            user_id=int(head.metadata.get("user_id") or 0),
            time=short_time(head.ts),
            tool_calls=names,
            tool_count=total,
            seq=int(head.seq or 0),
            answer_seq=answer_seq,
            message_ids=msg_ids,
            quotes=quotes,
        ))
    return out


#: 引用段落的标记（`main._quoted_suffix` 拼进去的那个开头）
QUOTE_MARK = "[引用上文]"


#: 引用段里那些「结构性」的行 —— 进卡片时要去掉，只留真正的引文内容
_QUOTE_PREFACES = ("对方引用的那条消息内容如下：", "上面那条消息自己又引用了：",
                   "上面那条消息自己又引用了", "对方引用的那条消息内容如下")


def clean_quoted(quoted: str) -> str:
    """把引用段落里的**结构行**去掉，只留引文正文（给卡片里的引用框用）。

    引用段长这样（`main._quoted_suffix` 拼的）：

        [引用上文] 对方引用的那条消息内容如下：
        鲍勃：听说三楼在装修，是真的吗？

    卡片里的引用框已经有「引用上文」这个标题了，再把 `[引用上文]` 和
    「……内容如下：」原样画出来就是三重标题、纯噪声。
    """
    out: list[str] = []
    for line in str(quoted or "").splitlines():
        text = line.strip()
        if not text:
            out.append("")
            continue
        while text.startswith(QUOTE_MARK):
            text = text[len(QUOTE_MARK):].strip()
        if not text or any(text.startswith(x) for x in _QUOTE_PREFACES):
            continue
        out.append(text)
    # 压掉连续空行与首尾空行
    lines: list[str] = []
    for line in out:
        if not line and (not lines or not lines[-1]):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def split_quoted(text: str) -> tuple[str, str]:
    """把用户输入拆成 `(自己写的部分, 引用段落)`。

    引用段落**总是拼在末尾**（`main._quoted_suffix` 里 `text += suffix`），所以
    找到第一个标记就够；用户自己写的正文里出现这个标记的概率极低，真出现了
    也只是被当成引用段的一部分，不会丢内容。

    `hget` 用它来决定引用怎么展示：引用的是**一轮对话**就渲染那一轮（引用链区块），
    是**普通消息**就把内容原样贴进引用框 —— 两种情况的差别就是「有没有 id 能对上」。
    """
    body = str(text or "")
    pos = body.find(QUOTE_MARK)
    if pos < 0:
        return body.strip(), ""
    return body[:pos].strip(), body[pos:].strip()


#: 回复**开头**的 `[#N]`（模型会模仿段落编号，把它当排版标记写出来）
_LEADING_LABEL_RE = re.compile(r"^\s*\[#(\d+)\]\s*")


def strip_stray_label(text: str, valid_seqs) -> str:
    """去掉回复开头那个**不是真编号**的 `[#N]`。

    为什么需要它：段落编号进了提示之后，模型会顺手在回答开头也写一个
    `[#N]`（实测 `xcpc` 会话里出现过 `[#19] 应用其实特别多…`，而那个会话当时
    只有 9 条消息），用户看到的就是一串不明所以的符号。

    为什么只剥**开头**、且只在**编号对不上**时剥：句中写 `[#12]` 是在合法地指代
    上文；开头写一个**真实存在**的编号，也可能是「回答 [#12] 是什么意思」这种
    正当引用 —— 只有「开头 + 对不上任何消息」才确定是模仿排版。
    """
    body = str(text or "")
    m = _LEADING_LABEL_RE.match(body)
    if not m:
        return body
    try:
        seq = int(m.group(1))
    except (TypeError, ValueError):
        return body
    if seq in set(int(x) for x in (valid_seqs or ()) if x):
        return body                      # 真在指代某一段：保留
    return body[m.end():].lstrip()


#: 引用段里的段落编号（`[引用上文] 对方引用的是上文 [#12]…`）
_QUOTE_LABEL_RE = re.compile(r"\[#(\d+)\]")


def quoted_labels(question: str) -> list[int]:
    """从**引用段**里取出被引用的段落编号（`[#N]` 里的 N，按出现顺序去重）。"""
    _, quoted = split_quoted(question)
    out: list[int] = []
    for raw in _QUOTE_LABEL_RE.findall(quoted or ""):
        try:
            seq = int(raw)
        except (TypeError, ValueError):
            continue
        if seq > 0 and seq not in out:
            out.append(seq)
    return out


def trace_chain(messages: Iterable[Message], index: int, *,
                max_depth: int = 8) -> list[int]:
    """沿「引用」往回追溯，返回引用链上的**问答编号**（最早 → 最晚，含自己）。

    用途：`hget` 看到的是「一对问答」，但那一对往往是在接着上文说的
    （用户引用着上一条提问、或者模型引着上一段回答）。只看一对会缺上下文，
    所以把整条链一并渲染出来。

    两类线索都认，**都只查本地**（`get_msg` 每层要 1~5 秒，是框架节流）：

    | 引用形态 | 线索 | 例 |
    |---|---|---|
    | 标签式（上文还在上下文里） | 引用段里的 `[#N]` → 消息 `seq` | `[引用上文] 对方引用的是上文 [#6]（…）` |
    | 文本式（贴了原文） | `metadata.reply_to` → 被引消息的 QQ id | `[引用上文] 对方引用的那条消息内容如下：…` |

    ⚠️ **标签式必须认**：机器人回答的 QQ 消息 id 不在任何 `Exchange.message_ids`
    里（那里面只有用户消息的 id），所以「引用上一段回答」时按 id 查必然落空 ——
    早先正是这样：引用 `[#6]` 却渲染成一个没头没脑的引用框。
    标签里的 `seq` 能直接对到消息，正交。

    找不到（老消息没记 reply_to、或上文已被压缩吸收）就到此为止 ——
    「追溯不到」比「编一段上文」诚实。环用 `seen` 挡住；`max_depth` 兜底。
    """
    msgs = list(messages)
    exchanges = extract_exchanges(msgs)
    if not exchanges:
        return []
    by_msg_id: dict[int, int] = {}
    by_seq: dict[int, int] = {}
    for ex in exchanges:
        for mid in ex.message_ids:
            by_msg_id.setdefault(int(mid), ex.index)
        for seq in (getattr(ex, "seq", 0), getattr(ex, "answer_seq", 0)):
            if seq:
                by_seq.setdefault(int(seq), ex.index)
    by_index = {ex.index: ex for ex in exchanges}

    chain: list[int] = []
    seen: set[int] = set()
    cur = int(index)
    while cur and cur in by_index and cur not in seen and len(chain) < max_depth:
        seen.add(cur)
        chain.append(cur)
        ex = by_index[cur]
        nxt = 0
        # ① 标签式：引用段里的 `[#N]` 直接就是 seq
        for seq in quoted_labels(ex.question):
            if seq in by_seq and by_seq[seq] != ex.index:
                nxt = by_seq[seq]
                break
        # ② 文本式：reply_to 是 QQ 消息 id
        if not nxt:
            for quoted in ex.quotes:
                if int(quoted) in by_msg_id and by_msg_id[int(quoted)] != ex.index:
                    nxt = by_msg_id[int(quoted)]
                    break
        cur = nxt
    return list(reversed(chain))


def has_compaction(messages: Iterable[Message]) -> bool:
    """历史里有没有压缩摘要（用于给用户解释「老对话去哪了」）。"""
    return any(m.origin == ORIGIN_COMPACT for m in messages)


def _preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """单行预览：换行压成空格、超长截断加省略号。"""
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rstrip() + "…"


def format_exchange(ex: Exchange, *, preview: int = PREVIEW_CHARS,
                    summaries: Optional[dict] = None) -> str:
    """清单里的一条（**固定两行**，方便按条分页时不把一条劈两半）。

    只有「谁/什么时候」+「问」：**清单不展示 AI 的回答**（连预览也不给）——
    它的用途是「我之前问过什么」的目录，而回答动辄上千字、一条里还覆盖多个要点，
    塞进目录只会把清单撑爆、把编号冲散。要看某一条的完整问答走
    `/aichat hget <编号>`（出图）。

    给了 `summaries`（`{编号: 提问概括}`）就用**提问摘要**代替原文。
    """
    ask = ""
    if summaries and summaries.get(ex.index):
        ask = str(summaries[ex.index])
    if not ask:
        ask = _preview(ex.question, preview)
    return (f"{ex.index}. {ex.speaker} {ex.time}\n"
            f"   问：{ask}")


def render_list(exchanges: list[Exchange], *, header: str = "",
                chunk_max_chars: int = 1200,
                preview: int = PREVIEW_CHARS,
                summaries: Optional[dict] = None) -> list[str]:
    """渲染问答清单，按 `chunk_max_chars` 切成多条消息（**不截断任何一条**）。

    分页在**条与条之间**切，而不是对整段文本硬切：一条问答被劈成两半、
    后半截跑到下一页，是清单最难读的形态。（每条现在只有两行「谁/何时 + 问」，
    回答不进清单 —— 见 `format_exchange`。）
    """
    if not exchanges:
        return [header] if header else []
    chunks: list[str] = []
    current = header
    for ex in exchanges:
        entry = format_exchange(ex, preview=preview, summaries=summaries)
        candidate = f"{current}\n{entry}" if current else entry
        if current and len(candidate) > chunk_max_chars:
            chunks.append(current)
            current = f"{header}\n{entry}" if header else entry
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def parse_range(text: str, *, total: int) -> tuple[list[int], Optional[str]]:
    """解析编号参数，返回 `(编号列表, 错误说明)`。

    只接受**单个**编号（`3`）。区间（`3-5`）在 `hget` 里被明确拒绝 ——
    一次只取一条记录；这个函数保留区间解析是因为它是通用工具（测试与
    其它入口都用得到），但**调用方**决定要不要放行区间。
    """
    raw = str(text or "").strip()
    if not raw:
        return [], "给出要查看的编号，如 `/aichat hget 3`"
    if total <= 0:
        return [], "这个会话还没有可查看的问答"
    m = re.fullmatch(r"(\d+)\s*(?:-\s*(\d+))?", raw)
    if not m:
        return [], (f"看不懂编号「{raw}」。用 `/aichat hget 3` 给一个编号，"
                    f"当前共 {total} 对")
    first = int(m.group(1))
    last = int(m.group(2)) if m.group(2) else first
    if first > last:
        first, last = last, first
    if first < 1:
        return [], f"编号从 1 开始（当前共 {total} 对）"
    if last > total:
        return [], f"没有第 {last} 对 —— 当前共 {total} 对"
    return list(range(first, last + 1)), None


def reject_range(text: str) -> Optional[str]:
    """`hget` 只取一条：参数里出现区间写法就回一句可操作的说明。"""
    raw = str(text or "").strip()
    if re.fullmatch(r"\d+\s*-\s*\d+", raw):
        return (f"`hget` 一次只能取一条记录（收到区间「{raw}」）。"
                f"用 `/aichat hget {raw.split('-')[0].strip()}` 取其中一条，"
                f"需要别的再发一次。")
    return None


# ==================== 清单摘要（调模型写 20 字） ====================

#: 清单里「问」与「答」各自最多几个字
SUMMARY_CHARS = 20

#: 清单只概括**用户提问**，不概括 AI 的回答。
#: 理由：目录的用途是「我之前问过什么」——按提问找记录最符合人的回忆方式，
#: 而回答往往很长、且同一条回答里可能覆盖好几个问题点，压成 20 字反而失真。
SUMMARY_SYSTEM_PROMPT = (
    "你是对话历史的目录编辑。用户会给你若干条「用户提问」，你要为每一条写一句"
    f"不超过 {SUMMARY_CHARS} 个字的概括，用于在目录里一眼看出这条问的是什么。\n"
    "硬要求：\n"
    f"1. 每条概括**不超过 {SUMMARY_CHARS} 个字**（中文按字算，标点也算）；\n"
    "2. 抓**具体话题**（问的是什么事、哪个对象），不要写「询问某事」「提问了一下」"
    "这种空壳；\n"
    "3. 只概括提问本身，**不要**写答案、不要写回答；\n"
    "4. 不要复述原文开头，不要加书名号/引号，不要写序号；\n"
    "5. 输出格式**严格**为每行一条：`编号|概括`，编号与输入一致，不要多写别的行。\n"
    "示例输入：\n1. 图书馆国庆开吗？顺便说说开放时间\n"
    "示例输出：\n1|图书馆国庆开放时间"
)


def summarize_context(exchanges: list[Exchange], *,
                      read_chars: int = 400) -> str:
    """把待摘要的**提问**拼成给模型的输入（不含 AI 的回答）。

    每条只给前 `read_chars` 字（配置 `history.summary_read_chars`）：写 20 字
    概括不需要读完一整段提问，长问题往往是粘贴的代码/报错。
    """
    limit = max(50, int(read_chars))
    lines = []
    for ex in exchanges:
        q = _clip(ex.question, limit)
        lines.append(f"{ex.index}. {q}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


def _clean_summary(text: str, limit: int) -> str:
    """去掉包裹的引号/书名号、压平空白、按上限截断。"""
    got = " ".join(str(text or "").split()).strip()
    got = got.strip("「」『』\"'“”《》").strip()
    if len(got) > limit:
        got = got[:limit].rstrip()
    return got


def parse_question_summaries(text: str, *, wanted: list[int],
                             limit: int = SUMMARY_CHARS) -> tuple[dict, str]:
    """解析提问摘要：每行 `编号|概括`。

    返回 `({编号: 概括}, 错误说明)`；解析不出来的编号不进结果，由调用方逐条
    回退到截断。整批都解析不出来时给一句可展示的说明（模型跑偏要能看出来）。
    """
    out: dict[int, str] = {}
    want = set(wanted)
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().lstrip("-*•#").strip()
        if not line:
            continue
        m = re.match(r"^(\d+)\s*[|｜:：.、]\s*(.*)$", line)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in want or idx in out:
            continue
        summary = _clean_summary(m.group(2), limit)
        if not summary:
            continue
        out[idx] = summary
    if not out:
        return {}, "模型没有按「编号|概括」的格式回答"
    return out, ""


def fallback_summary(ex: Exchange, limit: int = SUMMARY_CHARS, *,
                     prefer_answer: bool = False) -> str:
    """摘要失败时的兜底：取前 `limit` 字。

    `prefer_answer=True` 时取回答（用于「答」这一半缺失的兜底）——
    两半各取各自的内容，不要拿问题冒充答案。
    """
    first = ex.answer if prefer_answer else ex.question
    second = ex.question if prefer_answer else ex.answer
    flat = " ".join(str(first or "").split())
    if not flat:
        flat = " ".join(str(second or "").split())
    return flat[:limit].rstrip() if len(flat) > limit else flat


def summary_key(ex: Exchange) -> str:
    """单条问答的内容指纹 —— 内容没变就不必重新问模型。"""
    raw = f"{ex.index}\x00{ex.question}\x00{ex.answer}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class SummaryCache:
    """清单短摘要的落盘缓存：`{会话: {内容指纹: {q, a}}}`。

    为什么必须缓存：写摘要要调模型，而 `history` 是个**翻页查看**的命令 ——
    没有缓存的话，每翻一次、每看一遍都要重新烧一次 token（而且同样的前 10 条
    提问会被反复摘要）。指纹里含内容，所以对话被压缩/追加后自然失效重算。

    落盘而不是只放内存：重启一次就全丢的话，最常用的那几条摘要会被反复重算。
    文件本身有上限（`max_entries`），超了按插入顺序淘汰最老的。
    """

    def __init__(self, path: str, *, logger=None, max_entries: int = 2000) -> None:
        self.path = path
        self.logger = logger
        self.max_entries = max(1, int(max_entries))
        self._data: dict[str, dict[str, dict]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                got = json.load(f)
            if isinstance(got, dict):
                self._data = {str(k): v for k, v in got.items()
                              if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError) as e:
            if self.logger:
                self.logger.warning("[ai_chat] 摘要缓存读取失败（忽略）: %s", e)

    def get(self, sid: str, key: str) -> Optional[str]:
        """命中返回提问概括，否则 None（空串按未命中处理，交给上层回退）。"""
        item = (self._data.get(sid) or {}).get(key)
        if isinstance(item, dict):
            got = str(item.get("q") or "")
        else:
            got = str(item or "")
        return got or None

    def put(self, sid: str, items: dict[str, str]) -> None:
        if not items:
            return
        bucket = self._data.setdefault(sid, {})
        for key, summary in items.items():
            if summary:
                bucket[key] = {"q": summary}
        self._prune()
        self._dirty = True
        self.save()

    def _prune(self) -> None:
        total = sum(len(b) for b in self._data.values())
        if total <= self.max_entries:
            return
        # 按「最老的会话先丢」淘汰（dict 保持插入顺序，够用且不引额外时间戳）
        for sid in list(self._data):
            if total <= self.max_entries:
                break
            total -= len(self._data[sid])
            del self._data[sid]

    def save(self) -> None:
        """原子落盘（半截文件会让下次启动丢掉整份缓存）。"""
        if not self._dirty:
            return
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as e:
            if self.logger:
                self.logger.warning("[ai_chat] 摘要缓存写入失败: %s", e)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
