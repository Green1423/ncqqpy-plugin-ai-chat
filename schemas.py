"""命令参数 Schema。

约定（来自 `docs/plugin-dev.md` §4.3.1）：用法行**不手写**，全部由 Schema
经 `napcat.build_usage()` 生成，`<>` = 必填、`[]` = 可选。手写用法行会和
Schema 漂移，帮助里显示的参数也就和实际校验对不上。
"""
from __future__ import annotations

from napcat.command import (
    BoolField, ChoiceField, CommandSchema, DurationField, IntField, QQField, StrField,
    VariadicField,
)

NAME_MAX = 40


class NewSchema(CommandSchema):
    """新建会话。"""

    name = StrField(required=True, max_length=NAME_MAX,
                    description="会话名称，之后可用名字代替 id 引用")
    attach = BoolField(default=True,
                       description="创建后是否设为当前会话")
    public = BoolField(default=False,
                       description="（仅群聊）注册为**公开会话**：本群成员"
                                   "免邀请即可 attach；上下文对参与者完全共享")


class AttachSchema(CommandSchema):
    """把会话附着到当前聊天（新建会话后常用；等价于 switch）。

    **没有 `--force`**：切会话就是切，既不需要先 `detach`，也不需要确认。
    （旧版本要求「已有附着 + 另一会话」时必须 `--force=true`，否则只回一句
    「请加 --force=true」—— 用户每切一次都要先被拒一次。未知参数由框架
    按位置参数吞掉，老命令不会报错。）
    """

    session = StrField(required=True, max_length=NAME_MAX,
                       description="会话 id 或名称")


class JoinSchema(CommandSchema):
    """凭**邀请码**加入会话（码由会话所有者 `/aichat invite <会话>` 生成）。

    与 `attach` 分开是刻意的：`attach` 只认会话 id/名称，`join` 只认码 ——
    否则「一个长得像邀请码的会话名」永远说不清该按哪种解析。
    """

    code = StrField(required=True, max_length=32,
                    description="邀请码（6 位字母数字，没有 0/O/1/I/L）")


class DetachSchema(CommandSchema):
    """解除当前聊天的会话附着。"""

    session = StrField(default="", max_length=NAME_MAX,
                       description="仅当当前附着的是该会话时才解除")


class ListSchema(CommandSchema):
    """列出**当前聊天**所包含的会话。

    群聊 → 本群的群聊会话；私聊 → 你自己的私聊会话（含受邀参与的）。
    想看全部用超级用户的 `/aichat listall`。
    """

    user = IntField(default=None, min=1,
                    description="查看某个人的会话（仅超级用户）")


class ListAllSchema(CommandSchema):
    """列出全部会话（超级用户视图）。"""

    archived = BoolField(default=False, description="是否包含已归档的会话")


class HistorySchema(CommandSchema):
    """列出当前会话的历史问答（一对问答 = 一问一答）。

    只列**当前焦点会话**；编号与 `/aichat hget` 同一套，从 1 开始（最老 = 1）。
    压缩掉的老对话不在编号里 —— 正文已经不在历史里了。
    """

    limit = IntField(default=10, min=1, max=50,
                     description="列出最近几对问答")


class HGetSchema(CommandSchema):
    """把历史问答渲染成图片。

    **一次只取一条记录**：给一个编号，如 `3`。编号来自 `/aichat history`，
    从 1 开始（最老 = 1）。
    """

    n = StrField(required=True, max_length=20,
                 description="编号，如 3（一次一条）")


class TargetAddSchema(CommandSchema):
    """开通服务：给某个群/用户开放 AI 对话功能。

    `kind` 选类型、`target` 给号码 —— 与本框架其它插件同一种参数形状，好记。

    「绑定」指的是**能不能用这项功能**，与具体会话无关；
    用哪个会话由各人自己的 `/aichat new|attach` 决定。
    """

    kind = ChoiceField(["group", "user"], default="group",
                       description="目标类型：group=群 / user=QQ用户")
    target = IntField(default=None, min=0,
                      description="群号或 QQ 号；群里 add group 可省略（用当前群）")


class TargetDelSchema(CommandSchema):
    """停止服务：对某个群/用户关闭 AI 对话功能。"""

    kind = ChoiceField(["group", "user"], default="group",
                       description="目标类型：group=群 / user=QQ用户")
    target = IntField(default=None, min=0,
                      description="群号或 QQ 号；群里 del group 可省略（用当前群）")


class RenameSchema(CommandSchema):
    session = StrField(required=True, max_length=NAME_MAX, description="会话 id 或名称")
    new_name = VariadicField(description="新名称（可含空格，加引号更稳）")


class DeleteSchema(CommandSchema):
    session = StrField(required=True, max_length=NAME_MAX, description="会话 id 或名称")
    yes = BoolField(default=False, description="确认删除（必须显式给 true）")


class CompactSchema(CommandSchema):
    """手动压缩。"""

    session = StrField(default="", max_length=NAME_MAX,
                       description="压缩指定会话；留空=当前会话")
    force = BoolField(default=True, description="忽略阈值直接压缩")


#: 角色设定的长度上限。够写一段人格设定，又不至于把系统提示撑爆
#: （它每轮都随系统提示发出去，太长就是每轮白烧 token）。
ROLE_MAX = 2000


class RoleSchema(CommandSchema):
    """查看 / 设置会话里 AI 扮演的角色。

    ⚠️ 参数顺序是刻意的：`session` 在前、`text` 是变参（吃掉剩余全部位置参数）。

    ⚠️ `session` 的 `max_length` 放宽到 `ROLE_MAX`（不是 `NAME_MAX`）：位置参数
    是先由 Schema 校验再交给处理器的，而「只写角色设定」时那一格收到的其实是
    角色正文。卡在 40 字会让「设一段长角色」在**校验阶段**就失败，报的还是
    「session 长度不能超过 40」这种让人摸不着头脑的错。
    真正「哪一截是会话名」的判断在处理器的 `_split_role_args` 里做。
    """

    # 上限给到 ROLE_MAX 的两倍：位置参数由 Schema **先**校验，卡得太紧会让
    # 「角色写太长」在框架层就失败，报的是「session 长度不能超过 N」——
    # 明明用户写的是角色、字段却叫 session，这种报错没人看得懂。
    # 留出余量后，超长会由插件给出「角色设定太长（N 字，最多 2000 字）」。
    session = StrField(default="", max_length=ROLE_MAX * 2,
                       description="会话 id 或名称；留空=当前会话")
    text = VariadicField(description="角色设定（可含空格；留空=查看当前角色）")
    clear = BoolField(default=False, description="清除角色设定")


class ModelSchema(CommandSchema):
    """运行期改模型参数（不写回 config.json）。"""

    model = StrField(default="", max_length=120, description="新模型名")
    base_url = StrField(default="", max_length=300, description="新 API 地址")
    max_tokens = IntField(default=0, min=0, max=200000,
                          description="新 max_tokens")
    context_limit = IntField(default=0, min=0, max=2000000,
                             description="新的上下文预算（估算 token）")
    show = BoolField(default=False, description="只显示当前生效值")


class MaxTokensSchema(CommandSchema):
    """查看 / 设置**会话级** max_tokens（`/aichat maxtokens`）。

    `value` 用字符串而不是 IntField：既要支持数字（4096）和简写（8k），也要支持
    `off`（清除会话覆盖、回到全局默认）。解析放进处理器，与 `/aichat think` 的
    `level` 一致 —— 不在 schema 里维护第二份词汇表。
    """

    value = StrField(default="", max_length=20,
                     description="数字或 8k；off=清除会话覆盖（回到全局）")
    session = StrField(default="", max_length=64,
                       description="会话 ID 或名字（默认当前会话）")
    show = BoolField(default=False, description="只显示当前生效值（不修改）")


class TimeoutSchema(CommandSchema):
    """查看 / 设置**会话级**的单次调用时长上限（`/aichat timeout`）。

    值用字符串：支持 `600`、`600s`、`10m`、`1.5h`，以及 `off`（清除覆盖）。
    解析与范围校验在处理器里（与 `/aichat maxtokens` 同一套写法）。
    """

    value = StrField(default="", max_length=16,
                     description="秒数或 10m / 1.5h；off=清除会话覆盖（回到全局）")
    session = StrField(default="", max_length=64,
                       description="会话 ID 或名字（默认当前会话）")
    show = BoolField(default=False, description="只显示当前生效值（不修改）")


class ThinkSchema(CommandSchema):
    """查看/切换思考强度。

    `level` 故意是自由字符串：它既可能是级别（`high`），也可能是预算（`4k`）、
    也可能是开关（`thinking:off`）—— 解析放在 `reasoning.spec_from_text()`，
    这样命令层不必维护第二份词汇表。
    """

    level = StrField(default="", max_length=40,
                     description="off|minimal|low|medium|high|max，或预算 4k，或 thinking:on/off")
    global_ = BoolField(default=False,
                        description="改全局默认（默认只改当前会话）")
    show = BoolField(default=False, description="只显示当前生效值（不修改）")




class ReloadSchema(CommandSchema):
    """从磁盘重读 config.json（手工改配置后用）。"""

    confirm = BoolField(default=False, description="必须显式给 true")


class ConfigSchema(CommandSchema):
    """补齐 config.json 里的 plugins.ai_chat 段（刻意不吃参数）。

    留一个空 Schema 是有意的：这样「多打了参数」会拿到一条明确的用法报错，
    而不是静默无反应。
    """

    note = StrField(default="", max_length=40, description="备注（会被忽略）")


class InviteSchema(CommandSchema):
    """邀请参与会话：直接邀请某人，或生成**邀请码**（不带 QQ 号时）。

    邀请码用于「不方便直接给 QQ 号」的场景（私聊里把码发给对方，对方
    `/aichat attach <码>` 就进来了），而且**会过期**、可限次。
    """

    session = StrField(required=True, max_length=NAME_MAX,
                       description="会话 id 或名称（必须是你的）")
    user_id = QQField(required=False,
                      description="被邀请的 QQ 号；**不给就生成邀请码**")
    ttl = DurationField(default=0,
                        description="邀请码有效期（如 12h / 2d；最长 30 天，"
                                    "默认 24 小时）")
    uses = IntField(default=0,
                    description="邀请码可用次数（默认 1 次，最多 50 次）")
    revoke = BoolField(default=False,
                       description="作废这个会话所有未用的邀请码")
