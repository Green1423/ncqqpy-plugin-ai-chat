"""会话与历史持久化 —— 仿 DSH 的 session 记录结构。

磁盘布局（`data_dir` 默认 `plugins/ai_chat/data/`）::

    ai_chat/
      sessions/
        state.json          # 小状态：会话索引 + 绑定/附着表
        s0007.jsonl         # 每个会话一个 append-only 历史（一条消息一行）
        s0007.jsonl.deleted # 删除后改名，不立刻物理删除（可人工找回）

**为什么拆两个文件**：`state.json` 每次改绑定都要重写，而历史是 append-only。
如果混在一起，每加一条消息就要重写整个文件（SD 卡上写放大极重 ——
本仓库另一个插件就在这个坑里栽过）。

**为什么历史一行一条**：压缩（compaction）只需要「重写被压掉的那一段」，
所以 `rewrite()` 支持按 seq 保留尾部 —— 不用整文件重写。
"""
from __future__ import annotations

import json
import os

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .segments import Message

from napcat.persist import write_json

STATE_VERSION = 1

# 会话归属种类（原 `scope` 只剩「谁建的」这一层含义）
OWNER_USER = "user"      # 个人会话：所有者是某个人
OWNER_GROUP = "group"    # 群会话：所有者是某个群

# 会话状态
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"


# ==================== 聊天地址 ====================

@dataclass(frozen=True)
class ChatTarget:
    """一个「聊天地址」 —— 私聊用户 或 群聊。

    `group_id` 仅在 `kind="private"` 且是**群临时会话（小窗）**时非空：
    OneBot 把它报成 private + 来源群，回复时必须把 group_id 带回去，
    否则服务端按普通好友私聊发（非好友直接失败）。
    """

    kind: str                     # "private" | "group"
    chat_id: int                  # 私聊=user_id，群聊=group_id
    group_id: Optional[int] = None

    @property
    def is_group(self) -> bool:
        return self.kind == "group"

    @property
    def key(self) -> str:
        """稳定字符串键（落盘 / 比较用）。"""
        if self.is_group:
            return f"group:{self.chat_id}"
        if self.group_id:
            return f"private:{self.chat_id}@group:{self.group_id}"
        return f"private:{self.chat_id}"

    @property
    def conversation_key(self) -> str:
        """「这是**跟谁**的对话」—— 与 `key` 的区别只在私聊小窗。

        ⚠️ **暂存区、会话焦点、并发闸一律用这个，不要用 `key`。**
        群临时会话（小窗）的 `key` 是 `private:<uid>@group:<gid>`，于是同一个人在
        「主私聊」和「群小窗」里打字会落到**两个不同的暂存区 / 两个不同的焦点**：
        在一处压栈、在另一处发结束符，就会被判成「暂存区是空的」（用户报过）。
        人就是同一个人，问他「跟谁的对话」就够了；`group_id` 只用来决定
        **回复发回哪个窗口**（见 `_send`）。

        `key` 仍然保留：它是「往哪儿发」的地址，落盘在绑定表里当标识。
        """
        if self.is_group:
            return f"group:{self.chat_id}"
        return f"private:{self.chat_id}"

    @property
    def label(self) -> str:
        if self.is_group:
            return f"群 {self.chat_id}"
        if self.group_id:
            return f"用户 {self.chat_id}（群 {self.group_id} 小窗）"
        return f"用户 {self.chat_id}"

    @staticmethod
    def parse(key: str) -> "ChatTarget":
        """`key` 的反向解析（绑定表里只用 `private:<uid>` / `group:<gid>`）。

        ⚠️ 前缀长度必须用 `len()` 取，不能写死数字：`"private:"` 是 8 个字符，
        写 `[7:]` 会切出 `":20001"`，`int()` 直接抛 ValueError。
        """
        if key.startswith("group:"):
            return ChatTarget("group", int(key[len("group:"):]))
        body = key[len("private:"):] if key.startswith("private:") else key
        if "@group:" in body:
            uid, gid = body.split("@group:", 1)
            return ChatTarget("private", int(uid), int(gid))
        return ChatTarget("private", int(body))


@dataclass
class Session:
    """一个会话（= DSH 里的一条 session）。

    归属只有「谁建的」，不再有 scope 分类：

    * `owner_id` + `owner_kind` —— 所有者（个人或群）；
    * `guests` —— **受邀名单**：除所有者外，被允许 attach 这个会话的人。
      只有所有者能邀请（`/aichat invite`），受邀者需**自己** attach 才生效。

    会话本身不绑定聊天地址：「谁在哪儿用它」由各自的**焦点**（focus）决定，
    所以同一个会话既可以出现在群里，也可以出现在某人的私聊里。
    """

    id: str
    name: str
    owner_id: int = 0
    owner_kind: str = OWNER_USER     # user | group
    owner_name: str = ""
    guests: list = field(default_factory=list)   # 受邀者 QQ 号
    status: str = STATUS_ACTIVE
    created_at: float = 0.0
    updated_at: float = 0.0
    message_count: int = 0
    compact_count: int = 0
    # 压缩游标：seq 小于它的历史已被摘要吸收（无 seq 语义，仅用于展示）
    compacted_upto: int = 0
    last_summary: str = ""
    #: 这个会话里 AI 扮演的角色（`/aichat role` 设的）。
    #:
    #: 为什么是**独立字段**而不是塞进 `meta`：它每轮都要进系统提示，是会话的
    #: 一等属性 —— 放 `meta` 里以后没人说得清哪个键是有效的，做数据迁移时也
    #: 不敢动。放这里则读写都显式。
    #:
    #: ⚠️ 它**不参与上下文压缩**：角色走的是系统提示（每轮重新拼），而压缩只处理
    #: 消息历史 —— 两者根本不在一条路上，所以角色文本天然不会进摘要。
    #: 详见 `docs/plugins/ai-chat/03-sessions.md` §3.7「角色设定」。
    role: str = ""
    #: **公开会话**：群归属的会话打开它之后，**本群成员无需邀请**即可 attach。
    #: 只对群会话有意义 —— 私聊会话用它没有对象（私聊里只有两个人），
    #: 那种场景用邀请码（`/aichat invite <会话>`）更合适。
    public: bool = False
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "owner_id": self.owner_id, "owner_kind": self.owner_kind,
            "owner_name": self.owner_name, "guests": list(self.guests),
            "status": self.status, "created_at": self.created_at,
            "updated_at": self.updated_at, "message_count": self.message_count,
            "compact_count": self.compact_count,
            "compacted_upto": self.compacted_upto,
            "last_summary": self.last_summary, "role": self.role,
            "public": bool(self.public),
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Session":
        # 旧记录只有 scope（user/group）→ 迁移成 owner_kind
        kind = str(d.get("owner_kind") or "")
        if kind not in (OWNER_USER, OWNER_GROUP):
            kind = OWNER_GROUP if str(d.get("scope")) == "group" else OWNER_USER
        guests: list[int] = []
        for v in (d.get("guests") or []):
            try:
                guests.append(int(v))
            except (TypeError, ValueError):
                continue
        return Session(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")) or str(d.get("id", "")),
            owner_id=int(d.get("owner_id") or 0),
            owner_kind=kind,
            guests=guests,
            owner_name=str(d.get("owner_name", "")),
            status=str(d.get("status", STATUS_ACTIVE)),
            created_at=float(d.get("created_at") or 0.0),
            updated_at=float(d.get("updated_at") or 0.0),
            message_count=int(d.get("message_count") or 0),
            compact_count=int(d.get("compact_count") or 0),
            compacted_upto=int(d.get("compacted_upto") or 0),
            last_summary=str(d.get("last_summary", "")),
            # 旧记录没有这个字段 → 默认空串（等于「没设角色」），不需要迁移
            role=str(d.get("role") or ""),
            # 旧记录没有这个字段 → 默认私有（公开是显式动作）
            public=bool(d.get("public")),
            meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
        )

    @property
    def archived(self) -> bool:
        return self.status == STATUS_ARCHIVED

    @property
    def owner_label(self) -> str:
        who = self.owner_name or str(self.owner_id)
        return f"{who}" if self.owner_kind == OWNER_USER else f"群 {who}"

    def may_attach(self, user_id: int) -> bool:
        """谁能把这个会话挂到自己身上：所有者本人，或被邀请过的人。

        群归属的会话**不**等于「全群都能用」—— 那是「邀请」要解决的问题，
        所以群成员也要么是所有者、要么在受邀名单里。
        """
        uid = int(user_id)
        if self.owner_kind == OWNER_USER and uid == int(self.owner_id):
            return True
        if self.owner_kind == OWNER_GROUP:
            # 群归属：群主/管理员以群的身份使用，具体成员仍需被邀请
            pass
        return uid in set(self.guests)

    def describe(self) -> str:
        kind = "个人" if self.owner_kind == OWNER_USER else "群"
        bits = [f"`{self.id}`", self.name, f"[{kind}:{self.owner_label}]",
                f"{self.message_count} 条"]
        if self.public:
            bits.append("公开（本群免邀请）")
        if self.guests:
            bits.append(f"受邀 {len(self.guests)} 人")
        if self.compact_count:
            bits.append(f"压缩×{self.compact_count}")
        if self.archived:
            bits.append("已归档")
        return " · ".join(bits)


# ==================== 邀请码 ====================

#: 邀请码字符表：**去掉 0/O/1/I/L** 这些「抄错一个字母就废」的字符。
#: 用户是从 QQ 里手抄或转发的，可读性直接影响能不能用。
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
#: 6 位 → 约 7 亿种，配合「24 小时 + 1 次」的使用约束足够。
CODE_LEN = 6
#: 默认有效期（秒）与默认可用次数
CODE_TTL = 24 * 3600
CODE_USES = 1
#: 每个会话最多同时留几个未用完的码（防「一直生成」把 state.json 撑大）
CODE_MAX_PER_SESSION = 10
#: 全局上界（跑飞了也不会无限增长）
CODE_MAX_TOTAL = 200


def normalize_code(raw: str) -> str:
    """把人给的邀请码归一：去空白、转大写、去掉常见分隔符（`A-3K 9MZ`）。

    只做归一，**不校验** —— 认不出来时的提示由调用方给（它知道上下文）。
    """
    text = str(raw or "").strip().upper()
    for junk in (" ", "-", "－", "_", "|"):
        text = text.replace(junk, "")
    return text


def generate_code(rng=None) -> str:
    """生成一个邀请码（`CODE_LEN` 位，字符表里没有易混字符）。"""
    import secrets
    pick = getattr(rng or secrets, "choice", None) or random.choice
    return "".join(pick(CODE_ALPHABET) for _ in range(CODE_LEN))


# ==================== 会话存储 ====================

class _JsonIndex:
    """`message_id → 值` 的 JSON 索引骨架（落盘 + 上限淘汰 + 读坏忽略）。

    三个索引（发出的图片 / 发出的文本 / 发送记录）的簿记**逐字相同**，只有
    「值怎么归一化」与日志里的名字不一样。以前是三份复制 —— 改一处忘两处
    是迟早的事，而且它们的淘汰顺序、原子替换、读坏处理本来就该永远一致。

    子类只需要两个东西：`LABEL`（日志分类名）与 `_entry()`（值归一化）。
    """

    #: 日志里的分类名（「发出图片索引」/「发送索引」…）
    LABEL = "索引"

    def __init__(self, path: str, *, logger=None, max_entries: int = 400) -> None:
        self.path = path
        self.logger = logger
        self.max_entries = max(1, int(max_entries))
        self._data: dict = {}
        self._load()

    def _entry(self, value):
        """磁盘上的一个值 → 内存里的样子；返回 None 表示这条丢掉。"""
        return value

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                got = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._warn("读取失败（忽略）", e)
            return
        if not isinstance(got, dict):
            return
        data: dict = {}
        for key, value in got.items():
            clean = self._entry(value)
            if clean is not None:
                data[str(key)] = clean
        self._data = data

    def save(self) -> None:
        """落盘（原子替换）。超过 `max_entries` 时先淘汰最老的条目。"""
        # 淘汰最老的：dict 保持插入顺序，直接切掉前面多余的
        while len(self._data) > self.max_entries:
            self._data.pop(next(iter(self._data)), None)
        try:
            write_json(self.path, self._data)
        except OSError as e:
            self._warn("落盘失败", e)

    def _warn(self, what: str, exc: BaseException) -> None:
        if self.logger:
            self.logger.warning("[ai_chat] %s%s: %s", self.LABEL, what, exc)


class SentImagesIndex(_JsonIndex):
    """机器人发出去的 `message_id → [图片 url, ...]`（`message_sent` 事件喂）。

    **为什么需要**：一条长通知常带 7 张图，推送时被 `chunk.max_images` 切成几条
    （3+3+1）。用户只引用其中一条时，`get_msg` 只回**那一条**的 3 张 ——
    模型看到的就是「第 5、6、7 页」，前面几页永远进不来（用户报过）。
    这里按 id 存下每条发了哪些图，引用时把相邻几条拼起来。
    """

    LABEL = "发出图片索引"

    def _entry(self, value):
        if not isinstance(value, list):
            return None
        return [str(u) for u in value if u]

    def remember(self, message_id: int, urls: list) -> None:
        key = str(int(message_id))
        clean = [str(u) for u in (urls or []) if u]
        if not clean or self._data.get(key) == clean:
            return
        self._data[key] = clean
        self.save()

    def get(self, message_id: int) -> list:
        return list(self._data.get(str(int(message_id or 0))) or [])

    def get_block(self, message_id: int, *, max_parts: int = 6) -> list:
        """把一次推送被切成的几条里的图**按顺序拼起来**（去重保序）。"""
        mid = int(message_id or 0)
        if not mid or not self._data.get(str(mid)):
            return []
        lo = hi = mid
        while lo - 1 >= 1 and lo - mid < max_parts and self._data.get(str(lo - 1)):
            lo -= 1
        while hi + 1 <= hi + max_parts and self._data.get(str(hi + 1)):
            hi += 1
        out: list = []
        for i in range(lo, hi + 1):
            for url in self._data.get(str(i)) or []:
                if url not in out:
                    out.append(url)
        return out


class SentTextIndex(_JsonIndex):
    """机器人发出去的 `message_id → 完整文本`（**跨插件**，`message_sent` 事件喂）。

    **为什么需要**：推送常被切成多条（长通知 3 段），用户往往只引用其中一段。
    `get_msg` 只回那一段，模型拿到的就是半截话。`message_sent` 带的是那条消息的
    完整内容，按 id 存下来，引用任意一段都能还原全文。
    """

    LABEL = "发出文本索引"

    def _entry(self, value):
        return str(value) if value else None

    def remember(self, message_id: int, text: str) -> None:
        key = str(int(message_id))
        if not text or self._data.get(key) == text:
            return
        self._data[key] = str(text)
        self.save()

    def get(self, message_id: int) -> str:
        return str(self._data.get(str(int(message_id or 0))) or "")

    def get_block(self, message_id: int, *, max_parts: int = 6) -> str:
        """把**同一次推送被切成的多段**拼回来（引用其中一段也要给全文）。

        做法：以引用的那个 id 为中心，向前往后各走 `max_parts` 步，只要相邻 id
        也在索引里（说明是同一次连续发送）就连起来。索引里只存机器人自己发的，
        所以不会把别人的消息混进来；真实推送的 `message_id` 是连续的。
        """
        mid = int(message_id or 0)
        if not mid or not self._data.get(str(mid)):
            return ""
        lo = hi = mid
        while lo - 1 >= 1 and lo - mid < max_parts and self._data.get(str(lo - 1)):
            lo -= 1
        while hi + 1 <= hi + max_parts and self._data.get(str(hi + 1)):
            hi += 1
        parts = [self._data[str(i)] for i in range(lo, hi + 1) if self._data.get(str(i))]
        return "\n\n".join(parts)


class SentReplyIndex(_JsonIndex):
    """机器人**自己发出去**的消息 id → 它对应的那条会话消息 `(sid, seq)`。

    **为什么需要它**：用户会引用机器人发的**图片** —— Markdown 模式的整篇回答是
    一张图、函数图像、公式图也都是图。这些消息在 QQ 那边**没有文字**，
    `get_msg` 读回来就是「一张图片」，于是只能回一句「暂不支持多模态」。
    可那段内容本来就是我们自己生成的，原文一字不差地躺在会话历史里 ——
    这个索引就是那座桥（用户要求：引用 AI 的回答时，含图片的用原文替代）。

    只存 `(sid, seq)`、**不存原文**：原文已经在会话 jsonl 里了，存两份迟早不一致；
    而且图片回复动辄几千字，存两份会让这个索引比会话本身还大。

    落盘 + 有上限（与摘要缓存同一套路数）：重启后仍能引用旧消息，超了淘汰最老的。
    """

    LABEL = "发送索引"

    def _entry(self, value):
        if not isinstance(value, list) or len(value) != 2:
            return None
        return value

    def add(self, message_ids, sid: str, seq: int) -> None:
        """记下「这些消息 id 对应的原文在 `sid` 的第 `seq` 条」。"""
        changed = False
        for mid in message_ids or []:
            try:
                key = str(int(mid))
            except (TypeError, ValueError):
                continue
            if not key or key == "0":
                continue
            self._data[key] = [str(sid), int(seq)]
            changed = True
        if changed:
            self.save()

    def get(self, message_id) -> Optional[tuple[str, int]]:
        try:
            key = str(int(message_id))
        except (TypeError, ValueError):
            return None
        got = self._data.get(key)
        if not isinstance(got, list) or len(got) != 2:
            return None
        try:
            return str(got[0]), int(got[1])
        except (TypeError, ValueError):
            return None


class SessionStore:
    """会话索引 + 历史读写。

    并发模型：框架的事件总线是多协程的，而这里是「读改写」的小状态，
    和 `ConfigManager` 一样用 `threading.Lock`（不是 asyncio.Lock）——
    `on_disable` / 定时任务 / 事件处理可能在不同 task 里调用，且锁内没有 await。
    """

    def __init__(self, data_dir: str, *, logger=None,
                 max_messages: int = 400, max_sessions: int = 200,
                 compact_ratio: float = 0.6) -> None:
        self.data_dir = data_dir
        self.sessions_dir = os.path.join(data_dir, "sessions")
        self.state_path = os.path.join(self.sessions_dir, "state.json")
        self.logger = logger
        self.max_messages = max(20, int(max_messages))
        self.max_sessions = max(10, int(max_sessions))
        # 每次落盘时顺手裁掉多少比例的历史（0 = 不裁）
        self.compact_ratio = min(0.95, max(0.0, float(compact_ratio)))

        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        # 焦点表：`<chat key>@user:<uid>` → 会话 id。
        # 「谁在哪个聊天里当前用哪个会话」—— 同一个人在不同群可以有不同焦点，
        # 同一个群里的不同人也可以各有各的焦点。
        self._bindings: dict[str, str] = {}
        self._seq: int = 0                       # 会话 id 自增
        self._cache: dict[str, list[Message]] = {}
        self._dirty: set[str] = set()            # 需要触发裁剪的会话
        #: 邀请码：`{code: {"sid": ..., "expires": 时间戳, "uses": 剩余次数, "by": 创建者}}`
        self._codes: dict[str, dict] = {}

        os.makedirs(self.sessions_dir, exist_ok=True)
        self._load()

    # ---------------- 磁盘 ----------------

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            if self.logger:
                self.logger.warning("[ai_chat] state.json 解析失败，按空状态启动")
            return
        if not isinstance(data, dict):
            return
        raw = data.get("sessions")
        if isinstance(raw, dict):
            for sid, rec in raw.items():
                if isinstance(rec, dict):
                    rec.setdefault("id", sid)
                    s = Session.from_dict(rec)
                    if s.id:
                        self._sessions[s.id] = s
        bind = data.get("focus")
        if not isinstance(bind, dict):
            bind = data.get("bindings")          # 旧字段名
        if isinstance(bind, dict):
            fresh: dict = {}
            for k, v in bind.items():
                if isinstance(v, str) and v:
                    fresh[str(k)] = v
            # 老的群小窗焦点键并到主键（见 `_migrate_bindings`）
            self._bindings = self._migrate_bindings(fresh)
        # 自增计数要越过已存在的最大 id，否则新建会话会撞 id
        highest = 0
        for sid in self._sessions:
            if sid.startswith("s") and sid[1:].isdigit():
                highest = max(highest, int(sid[1:]))
        self._seq = max(int(data.get("seq") or 0), highest)
        raw_codes = data.get("invites")
        if isinstance(raw_codes, dict):
            now = time.time()
            for code, rec in raw_codes.items():
                if not isinstance(rec, dict):
                    continue
                sid = str(rec.get("sid") or "")
                # 会话没了 / 过期的码直接不加载（重启后不该还能用）
                if not sid or sid not in self._sessions:
                    continue
                expires = float(rec.get("expires") or 0)
                uses = int(rec.get("uses") or 0)
                if uses <= 0 or (expires and expires < now):
                    continue
                self._codes[str(code).upper()] = {
                    "sid": sid, "expires": expires, "uses": uses,
                    "by": int(rec.get("by") or 0)}

    def _save_locked(self) -> None:
        payload = {
            "version": STATE_VERSION,
            "seq": self._seq,
            "sessions": {sid: s.to_dict() for sid, s in self._sessions.items()},
            "focus": dict(self._bindings),
            # 邀请码：跟会话一起落盘（进程重启后码要还能用 —— 用户可能
            # 隔天才把码发给别人）
            "invites": {code: dict(v) for code, v in self._codes.items()},
        }
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.state_path)
        except OSError as e:            # 磁盘满 / 只读挂载：不能把插件整个带崩
            if self.logger:
                self.logger.error("[ai_chat] 写入 state.json 失败: %r", e)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _history_path(self, sid: str) -> str:
        return os.path.join(self.sessions_dir, f"{sid}.jsonl")

    # ---------------- 会话 CRUD ----------------

    def create(self, name: str, *, owner_kind: str = OWNER_USER, owner_id: int = 0,
               owner_name: str = "") -> Session:
        """新建会话。`owner_kind` 说明所有者是个人还是群。"""
        with self._lock:
            self._seq += 1
            sid = f"s{self._seq:04d}"
            now = time.time()
            s = Session(id=sid, name=name or sid,
                        owner_kind=(OWNER_GROUP if owner_kind == OWNER_GROUP
                                    else OWNER_USER),
                        owner_id=int(owner_id or 0), owner_name=owner_name,
                        created_at=now, updated_at=now)
            self._sessions[sid] = s
            self._cache[sid] = []
            self._prune_locked()
            self._save_locked()
            return s

    def get(self, sid: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(sid)

    def resolve(self, token: str) -> Optional[Session]:
        """按「id 或名字」找会话 —— 用户更愿意打名字，id 是兜底。

        名字匹配规则（大小写不敏感、先精确后前缀）：
        精确命中唯一名字 → 用它；否则唯一前缀命中 → 用它；有歧义 → None
        （调用方据此提示「有多个同名/同前缀会话，请用 id」）。
        """
        t = (token or "").strip()
        if not t:
            return None
        with self._lock:
            if t in self._sessions:
                return self._sessions[t]
            low = t.lower()
            exact = [s for s in self._sessions.values() if s.name.lower() == low]
            if len(exact) == 1:
                return exact[0]
            if exact:
                return None
            prefix = [s for s in self._sessions.values()
                      if s.name.lower().startswith(low)]
            if len(prefix) == 1:
                return prefix[0]
            return None

    def rename(self, sid: str, name: str) -> bool:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            s.name = name
            s.updated_at = time.time()
            self._save_locked()
            return True

    def touch(self, sid: str) -> None:
        """会话的 `meta` 被就地改过 → 落一次盘（`media_seq` 这类计数器靠它）。"""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return
            s.updated_at = time.time()
            self._save_locked()

    def set_role(self, sid: str, role: str) -> bool:
        """设置 / 清除会话的角色设定（空串 = 清除）。"""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            s.role = str(role or "")
            s.updated_at = time.time()
            self._save_locked()
            return True

    def list(self, *, owner_kind: Optional[str] = None,
             owner_id: Optional[int] = None, guest_id: Optional[int] = None,
             include_archived: bool = True, limit: int = 50) -> list[Session]:
        """筛选会话。`guest_id` 用于「别人邀请我参与的会话」。"""
        with self._lock:
            out = [s for s in self._sessions.values()
                   if (owner_kind is None or s.owner_kind == owner_kind)
                   and (guest_id is None or int(guest_id) in set(s.guests))
                   and (owner_id is None or s.owner_id == owner_id)
                   and (include_archived or not s.archived)]
            out.sort(key=lambda s: s.updated_at or s.created_at, reverse=True)
            return out[:max(1, limit)]

    def set_status(self, sid: str, status: str) -> bool:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            s.status = status
            s.updated_at = time.time()
            self._save_locked()
            return True

    def delete(self, sid: str, *, hard: bool = False) -> bool:
        """删除会话。默认软删（历史改名保留），`hard=True` 才真删文件。"""
        with self._lock:
            if sid not in self._sessions:
                return False
            del self._sessions[sid]
            self._cache.pop(sid, None)
            self._dirty.discard(sid)
            # 一并清掉指向它的焦点，避免留下悬空引用
            for k in [k for k, v in self._bindings.items() if v == sid]:
                del self._bindings[k]
            # 邀请码也要作废：留着就是一堆指向空会话的死码
            for c in [c for c, v in self._codes.items() if v.get("sid") == sid]:
                self._codes.pop(c, None)
            self._save_locked()
            path = self._history_path(sid)
            try:
                if os.path.exists(path):
                    if hard:
                        os.remove(path)
                    else:
                        os.replace(path, path + ".deleted")
            except OSError as e:
                if self.logger:
                    self.logger.warning("[ai_chat] 删除历史 %s 失败: %r", sid, e)
            return True

    def _prune_locked(self) -> None:
        """会话数超限：把最久未更新的**已归档**会话先挤掉（活动会话不动）。"""
        if len(self._sessions) <= self.max_sessions:
            return
        overflow = len(self._sessions) - self.max_sessions
        victims = [s for s in self._sessions.values() if s.archived]
        victims.sort(key=lambda s: s.updated_at or s.created_at)
        for s in victims[:overflow]:
            self._sessions.pop(s.id, None)
            self._cache.pop(s.id, None)
            for k in [k for k, v in self._bindings.items() if v == s.id]:
                del self._bindings[k]
            if self.logger:
                self.logger.info("[ai_chat] 会话数超限，回收已归档会话 %s(%s)",
                                 s.id, s.name)

    # ---------------- 绑定 / 附着 ----------------

    @staticmethod
    def focus_key(chat: ChatTarget, user_id: int) -> str:
        """焦点键 = `聊天地址@user:人`。

        带 `user` 这一段是关键：群里每个人的焦点各自独立，
        同一个人在不同群、以及他个人的私聊，也都是不同焦点。
        """
        # ⚠️ 用 `conversation_key` 而不是 `key`：同一个人在主私聊与群小窗里
        # 应当是**同一个焦点会话**，否则换个窗口说话就「没有会话」了。
        return f"{chat.conversation_key}@user:{int(user_id)}"

    @staticmethod
    def _migrate_bindings(bindings: dict) -> dict:
        """把老的「群小窗」焦点键并到同一个人的主键上。

        焦点键原先用 `chat.key`，群临时会话会写成
        `private:<uid>@group:<gid>@user:<uid>`；换成 `conversation_key` 之后那些键
        再也读不到 —— 不并的话，一直用小窗跟机器人说话的人会突然「没有会话」。
        主键已经有值时**保留主键**（那是用户在主私聊里选的那个）。
        """
        out: dict = {}
        for key, sid in (bindings or {}).items():
            if isinstance(key, str) and key.startswith("private:") and "@group:" in key:
                head, _, tail = key.partition("@group:")
                gid, _, rest = tail.partition("@user:")
                if rest:
                    key = f"{head}@user:{rest}"
                else:
                    key = head
            out.setdefault(key, sid)
        return out

    def focus_of(self, chat: ChatTarget, user_id: int) -> Optional[Session]:
        """这个人在这个聊天里的当前会话。"""
        with self._lock:
            sid = self._bindings.get(self.focus_key(chat, user_id))
            return self._sessions.get(sid) if sid else None

    def set_focus(self, chat: ChatTarget, user_id: int, sid: str) -> bool:
        """把会话设为「这个人在这个聊天里的焦点」。

        ⚠️ **不做授权判断** —— 谁可以挂哪个会话由插件层（`_may_attach`）决定；
        store 只管记账，免得同一套规则在两层各写一遍。
        """
        with self._lock:
            s = self._sessions.get(sid)
            if s is None or s.archived:
                return False
            self._bindings[self.focus_key(chat, user_id)] = sid
            s.updated_at = time.time()
            self._save_locked()
            return True

    def clear_focus(self, chat: ChatTarget, user_id: int) -> Optional[str]:
        """解除这个人在这个聊天的焦点，返回原会话 id。"""
        with self._lock:
            sid = self._bindings.pop(self.focus_key(chat, user_id), None)
            if sid:
                self._save_locked()
            return sid

    def clear_focus_everywhere(self, sid: str) -> list[str]:
        """把某个会话从所有焦点里摘掉（删除会话时用）。"""
        with self._lock:
            keys = [k for k, v in self._bindings.items() if v == sid]
            for k in keys:
                del self._bindings[k]
            if keys:
                self._save_locked()
            return keys

    def attachments(self, sid: str) -> list[str]:
        """这个会话当前被哪些焦点引用（`chat@user:人`）。"""
        with self._lock:
            return [k for k, v in self._bindings.items() if v == sid]

    def focus_table(self) -> dict[str, str]:
        with self._lock:
            return dict(self._bindings)

    # 兼容旧名（少量旧代码/测试仍在用）
    bindings = focus_table

    # ---------------- 受邀名单 ----------------

    def invite(self, sid: str, user_id: int) -> bool:
        """邀请某人参与会话（加入受邀名单）。已在内则返回 False。"""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            uid = int(user_id)
            if uid in set(s.guests):
                return False
            s.guests = sorted(set(s.guests) | {uid})
            s.updated_at = time.time()
            self._save_locked()
            return True

    def uninvite(self, sid: str, user_id: int) -> bool:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            uid = int(user_id)
            if uid not in set(s.guests):
                return False
            s.guests = [g for g in s.guests if int(g) != uid]
            # 顺带摘掉他基于这个会话的焦点：授权没了，焦点不该留着
            for key in [k for k, v in self._bindings.items()
                        if v == sid and k.endswith(f"@user:{uid}")]:
                del self._bindings[key]
            s.updated_at = time.time()
            self._save_locked()
            return True

    # ---------------- 公开会话 / 邀请码 ----------------

    def set_public(self, sid: str, on: bool) -> Optional[bool]:
        """把会话设成公开 / 私有。返回**改之前**的状态（会话不存在返回 None）。"""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return None
            before = bool(s.public)
            if before != bool(on):
                s.public = bool(on)
                s.updated_at = time.time()
                self._save_locked()
            return before

    def new_code(self, sid: str, *, ttl: float = CODE_TTL,
                 uses: int = CODE_USES, by: int = 0) -> tuple[str, dict]:
        """给会话生成一个邀请码，返回 `(码, 记录)`；超上限时抛 `RuntimeError`。

        `uses` 是**可用次数**（1 = 一次性）；`ttl <= 0` 表示不过期。
        """
        with self._lock:
            if sid not in self._sessions:
                raise RuntimeError("会话不存在")
            self._prune_codes_locked()
            mine = [c for c, v in self._codes.items() if v.get("sid") == sid]
            if len(mine) >= CODE_MAX_PER_SESSION:
                # 自己先腾地方：把最老的一个作废，而不是让用户「先撤销再用」
                self._codes.pop(min(mine, key=lambda c: self._codes[c]["expires"]),
                                None)
            if len(self._codes) >= CODE_MAX_TOTAL:
                self._codes.pop(next(iter(self._codes)), None)
            code = generate_code()
            while code in self._codes:          # 撞了就重摇（概率极低）
                code = generate_code()
            rec = {"sid": str(sid), "expires": (time.time() + float(ttl)
                                                if float(ttl) > 0 else 0.0),
                   "uses": max(1, int(uses)), "by": int(by)}
            self._codes[code] = rec
            self._save_locked()
            return code, dict(rec)

    def peek_code(self, raw: str) -> Optional[str]:
        """看一眼邀请码指向哪个会话（**不消耗**）。无效/过期/用尽返回 None。"""
        code = normalize_code(raw)
        if not code:
            return None
        with self._lock:
            self._prune_codes_locked()
            rec = self._codes.get(code)
            return str(rec.get("sid") or "") or None if rec else None

    def consume_code(self, raw: str, *, user_id: int = 0) -> Optional[str]:
        """核销一个邀请码，返回它指向的会话 id（无效时 None）。

        ⚠️ 只在**真的要放行**时调用：先 `peek_code` 做类型/权限判定，
        判定通过再 consume —— 否则一个用不上的会话会把码白吃掉。
        """
        code = normalize_code(raw)
        if not code:
            return None
        with self._lock:
            self._prune_codes_locked()
            rec = self._codes.get(code)
            if not rec:
                return None
            sid = str(rec.get("sid") or "")
            left = int(rec.get("uses") or 0) - 1
            if left > 0:
                rec["uses"] = left
            else:
                self._codes.pop(code, None)
            self._save_locked()
            return sid or None

    def codes_of(self, sid: str) -> list[tuple[str, dict]]:
        """某会话当前有效的邀请码（按过期时间升序）。"""
        with self._lock:
            self._prune_codes_locked()
            out = [(c, dict(v)) for c, v in self._codes.items()
                   if v.get("sid") == sid]
            out.sort(key=lambda kv: kv[1].get("expires") or 0)
            return out

    def revoke_codes(self, sid: str) -> int:
        """作废某会话的全部邀请码，返回作废了几个。"""
        with self._lock:
            gone = [c for c, v in self._codes.items() if v.get("sid") == sid]
            for c in gone:
                self._codes.pop(c, None)
            if gone:
                self._save_locked()
            return len(gone)

    def code_stats(self) -> dict:
        """诊断用：`{总数, 各会话条数}`。"""
        with self._lock:
            self._prune_codes_locked()
            per: dict[str, int] = {}
            for rec in self._codes.values():
                sid = str(rec.get("sid") or "")
                per[sid] = per.get(sid, 0) + 1
            return {"total": len(self._codes), "per_session": per}

    def _prune_codes_locked(self) -> None:
        """清掉过期的码（调用方已持锁）。"""
        now = time.time()
        dead = [c for c, v in self._codes.items()
                if int(v.get("uses") or 0) <= 0
                or (float(v.get("expires") or 0) and float(v["expires"]) < now)]
        for c in dead:
            self._codes.pop(c, None)

    def guests_of(self, sid: str) -> list[int]:
        with self._lock:
            s = self._sessions.get(sid)
            return list(s.guests) if s else []

    # ---------------- 历史 ----------------

    def history(self, sid: str) -> list[Message]:
        """会话历史（带内存缓存；首次访问从 jsonl 读）。返回的是**副本列表**。"""
        with self._lock:
            if sid in self._cache:
                return list(self._cache[sid])
        msgs = self._load_history(sid)
        with self._lock:
            self._cache[sid] = msgs
            return list(msgs)

    def _load_history(self, sid: str) -> list[Message]:
        path = self._history_path(sid)
        msgs: list[Message] = []
        if not os.path.exists(path):
            return msgs
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue        # 半行（掉电）→ 跳过，不让整段历史报废
                    if isinstance(rec, dict):
                        msgs.append(Message.from_dict(rec))
        except OSError as e:
            if self.logger:
                self.logger.error("[ai_chat] 读取历史 %s 失败: %r", sid, e)
            return []
        if self.max_messages and len(msgs) > self.max_messages:
            msgs = msgs[-self.max_messages:]
        self._backfill_seqs(sid, msgs)
        return msgs

    def _backfill_seqs(self, sid: str, msgs: list[Message]) -> None:
        """给**没有 `seq` 的老文件**补上会话内序号（就地改 `msgs`）。

        为什么必须补：`find()` 是拿 `seq` 认消息的，而历史是**尾部保留**的，
        按下标算会随裁剪漂移。早先 `Message.to_dict()` 漏写了这个字段，于是
        凡是「落盘再读回来」的消息 `seq` 全是 0 —— `SentReplyIndex` 里记的
        `(sid, seq)` 一个都对不上，表现为「引用机器人的图片永远说读不到」。
        磁盘上的老记录不重写，读的时候从后往前倒推即可：

        * 最后一条的 `seq` 就是 `session.message_count`（`append()` 的赋值口径）；
        * 见到带 `seq` 的（新写入的记录）就以它为锚点，继续往前递减。
        """
        if not msgs or all(m.seq for m in msgs):
            return
        s = self._sessions.get(sid)
        # `+1`：循环里是「先减再赋值」，所以起点要比预计的最大序号大 1
        nxt = (int(getattr(s, "message_count", 0) or 0) or len(msgs)) + 1
        for m in reversed(msgs):
            if m.seq:
                nxt = m.seq
            else:
                nxt -= 1
                m.seq = nxt

    def find(self, sid: str, seq: int) -> Optional[Message]:
        """按 `seq` 取一条消息（取不到返回 None）。

        `seq` 是会话内单调递增的，但历史会被**压缩/裁剪**掉前面的部分，
        所以不能按下标算 —— 老实扫一遍（一个会话几百条，够快）。
        """
        try:
            want = int(seq)
        except (TypeError, ValueError):
            return None
        for m in self.history(sid):
            if int(getattr(m, "seq", 0) or 0) == want:
                return m
        return None

    def append(self, sid: str, message: Message) -> bool:
        """追加一条消息：写 jsonl + 更新缓存与计数。"""
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            cached = self._cache.get(sid)
            if cached is None:
                cached = self._load_history(sid)
                self._cache[sid] = cached
            s.message_count += 1
            message.seq = s.message_count
            cached.append(message)
            if self.max_messages and len(cached) > int(self.max_messages * 1.5):
                self._dirty.add(sid)
            s.updated_at = time.time()
            path = self._history_path(sid)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            if self.logger:
                self.logger.error("[ai_chat] 追加历史 %s 失败: %r", sid, e)
            return False
        with self._lock:
            if sid in self._dirty:
                self._trim_locked(sid)
            self._save_locked()
        return True

    def append_many(self, sid: str, messages: Iterable[Message]) -> int:
        n = 0
        for m in messages:
            if self.append(sid, m):
                n += 1
        return n

    def _trim_locked(self, sid: str) -> None:
        """把内存里保留的尾部**重写**回文件（裁剪文件长度）。

        只在超过 `max_messages * 1.5` 时才做，避免每加一条消息就整文件重写。
        """
        self._dirty.discard(sid)
        cache = self._cache.get(sid) or []
        if not self.max_messages or len(cache) <= self.max_messages:
            return
        keep = cache[-self.max_messages:]
        self._cache[sid] = keep
        path = self._history_path(sid)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for m in keep:
                    f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        except OSError as e:
            if self.logger:
                self.logger.warning("[ai_chat] 裁剪历史 %s 失败: %r", sid, e)

    def replace_history(self, sid: str, messages: list[Message]) -> bool:
        """整段替换历史（压缩后调用）：覆盖 jsonl + 刷缓存。

        保留每条消息**原有的 seq**（没有 seq 的按顺序补），这样
        「被压掉的边界」在日志里仍然可追溯。
        """
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return False
            path = self._history_path(sid)
            tmp = path + ".tmp"
            used: set[int] = set()
            for m in messages:
                if not m.seq:
                    nxt = 1
                    while nxt in used:
                        nxt += 1
                    m.seq = nxt
                used.add(m.seq)
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    for m in messages:
                        f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
                os.replace(tmp, path)
            except OSError as e:
                if self.logger:
                    self.logger.error("[ai_chat] 重写历史 %s 失败: %r", sid, e)
                return False
            self._cache[sid] = list(messages)
            self._dirty.discard(sid)
        return True

    def touch_compaction(self, sid: str, *, summary: str = "",
                         absorbed: int = 0, upto_seq: int = 0) -> None:
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                return
            s.compact_count += 1
            s.last_summary = summary[:400]
            s.compacted_upto = max(s.compacted_upto, upto_seq)
            s.meta["last_absorbed"] = absorbed
            s.updated_at = time.time()
            self._save_locked()

    # ---------------- 检索（给 AI 的 search_history 工具用） ----------------

    def search(self, query: str, *, session_id: Optional[str] = None,
               owner_id: Optional[int] = None, limit: int = 8,
               context: int = 1) -> list[dict]:
        """在历史里做**子串**检索，返回带上下文的片段。

        刻意不做向量检索/分词索引：聊天记录量级（几百~几万条）下，
        线性扫描 jsonl 足够快，而且零依赖、行为可预测。
        """
        q = (query or "").strip().lower()
        if not q:
            return []
        with self._lock:
            if session_id:
                targets = [session_id] if session_id in self._sessions else []
            elif owner_id:
                targets = [s.id for s in self._sessions.values()
                           if s.owner_id == owner_id]
            else:
                targets = list(self._sessions.keys())
            targets.sort()
        hits: list[dict] = []
        for sid in targets:
            msgs = self.history(sid)
            for i, m in enumerate(msgs):
                text = m.text
                if not text or q not in text.lower():
                    continue
                lo, hi = max(0, i - context), min(len(msgs), i + context + 1)
                hits.append({
                    "session": sid,
                    "seq": m.seq,
                    "role": m.role,
                    "ts": m.ts,
                    "text": text[:400],
                    "around": [x.text[:200] for x in msgs[lo:hi] if x.text],
                })
                if len(hits) >= max(1, limit):
                    return hits
        return hits

    # ---------------- 统计 ----------------

    def stats(self) -> dict:
        with self._lock:
            total = len(self._sessions)
            by_owner: dict[str, int] = {}
            guests = 0
            messages = 0
            for s in self._sessions.values():
                by_owner[s.owner_kind] = by_owner.get(s.owner_kind, 0) + 1
                guests += len(s.guests)
                messages += s.message_count
            return {
                "sessions": total,
                "by_owner": by_owner,
                "guests": guests,
                "messages": messages,
                "focus": len(self._bindings),
                "cached": len(self._cache),
            }

    def history_tokens(self, sid: str, chars_per_token: float = 1.6) -> float:
        return sum(m.approx_tokens(chars_per_token) for m in self.history(sid))
