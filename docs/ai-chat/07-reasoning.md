# 七、思考强度（reasoning effort）

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[上游失败自动重试](06-retries.md) · 下一章：[内置工具](08-tools.md)


推理模型的「想多久」是一个独立开关，且**每家端点的字段名都不一样**。本插件把它
拆成两层：**意图**（用户说想要多强）与**方言**（往 payload 塞哪个字段）。

## 7.1 配置

```jsonc
"llm": {
  "reasoning": {
    "dialect": "effort",     // effort | thinking | enable_thinking | none | raw
    "default": "off",        // 全局默认：off / minimal / low / medium / high / max / 4k / thinking:on
    "per_model": {           // 按模型名精确覆盖（换模型自动换档）
      "deepseek-v4-flash": "off",
      "o4-mini": "high"
    },
    "params": {},            // dialect=raw 时原样并入 payload
    "max_tokens_floor": 0,   // >0：开思考时把 max_tokens 抬到这个下界
    "compaction": "off"      // 压缩摘要单独一档（摘要要短，思考是浪费）
  }
}
```

> 旧的 `llm.reasoning_effort: "off"` 写法仍然兼容（相当于 `dialect=effort` 的 default）。

## 7.2 方言对照

| `dialect` | 生成的 payload 字段 | 适用 |
|-----------|--------------------|------|
| `effort`（默认） | `reasoning_effort: "<level>"` | OpenAI / DeepSeek / 通义 |
| `thinking` | `thinking: {"type": "enabled"}` | DeepSeek V3.2 风格 |
| `enable_thinking` | `enable_thinking: true` | Qwen / vLLM 部署 |
| `none` | 什么都不发 | 让端点用自己的默认值 |
| `raw` | `params` 原样并入 | 自建端点万能兜底 |

给错字段名的代价很不对称：多数端点直接 400，少数**静默忽略**（你以为关掉了思考，
其实还在烧 token 和延迟）。所以方言必须显式配置，不靠猜。

## 7.3 强度取值

| 写法 | 含义 |
|------|------|
| `off` / `inherit` / `auto` / `none` | **不干预**：不发任何思考参数，由端点决定 |
| `minimal` / `low` / `medium` / `high` / `max` | 指定级别（`on` = `medium`） |
| `4096` / `4k` / `1.5k` | **思考预算**（token） |
| `thinking:on` / `thinking:off` | 显式开关（在 `effort` 方言里会翻译成级别，不会被吞掉） |

受方言能力限制时会**降级而不是报错**，并在 `/aichat think` 里说明：

- `thinking` 方言只有开关，没有级别 → `high` / `low` 都渲染成 `enabled`；
- 预算在 `thinking` 方言下退化为「开」，在 `effort` 方言下附带 `thinking_budget`。

> ⚠️ `low` 在部分 DeepSeek 部署上会被**服务端忽略**（实测 reasoning 454 vs `none` 的 0）。
> 想真正省 token 用 `off`，别指望 `low`。

## 7.4 命令

```text
/aichat think                      查看当前生效值 + 会发的 payload
/aichat think high                 只改当前会话
/aichat think 8k                   思考预算
/aichat think off                  退回不干预
/aichat think low --global_=true   改全局默认（写入 config.json）
/aichat maxtokens                  查看本会话生效的 max_tokens（全局/覆盖/思考下界后）
/aichat maxtokens 2048             只改当前会话（8k 简写亦可）
/aichat maxtokens off              清除会话覆盖，回到全局默认
/aichat maxtokens 8192 --session=<id|名字>   改指定会话（需管理权限）
/aichat timeout                    查看本会话生效的单次调用时长上限
/aichat timeout 600                只改当前会话（10m / 1.5h 亦可）
/aichat timeout off                清除会话覆盖，回到全局 llm.timeout
/aichat model                      顺带显示思考强度
```

优先级：**会话覆盖 → `per_model[模型]` → `default`**。默认只改当前会话，因为
「这件事值不值得深想」和具体会话绑在一起更合理。

## 7.5 max_tokens 下界保护

推理 token **计入 completion**（`deepseek-flash` 实测占 ~93%）。`max_tokens` 给小了
会拿到 `finish_reason=length` 且 **content 为空串** —— 用户只看到「机器人没反应」。
所以开思考时会自动抬高下界（默认 4096；预算模式给「预算 + 1024」余量），
`/aichat think` 会显示实际用的值。
## 7.5.1 上游「自己会思考」时的空响应（实测）

有些网关/模型**不管你是否传思考参数，都会先吐一段思考链**（`reasoning_content`），
而思考 token 与正文共用 `max_tokens`。这时 `llm.max_tokens=4096` 的典型表现是：

- 难题（数学 / 竞赛 / 长推导）把配额全花在思考上 → 上游结束流时**只有 reasoning、
  没有 content、也没有 `finish_reason`**；
- 插件抛 `空响应（finish_reason=无，思考链 N 字）`，用户收到
  `⚠️ AI 对话出错（E-xxxx）：…`，会话里**只留下没被回答的提问**；
- 简单问题（「1+1」）4096 也够 → 于是表现为「只有某个会话怎么问都不回」。

实测（BUPT 网关 `myai.bupt.edu.cn/llm-gw/v1` + `deepseek-v4-flash`，同一段 math 会话历史）：

| `max_tokens` | 结果 |
|---|---|
| 4096 | 37s 后「空响应（思考链 989 字）」，无回复 |
| 8192 | 思考更久 → 撞 `llm.timeout=120` 超时 |
| **16384 + `timeout=300`** | **正常回复（409 字，assistant 记录已落库）** |

处理：

- `llm.max_tokens` 提到 **≥ 16384**，`llm.timeout` 提到 **≥ 300**（思考久时 120s 不够）；
- ⚠️ `reasoning.default=off` 只是**不发思考参数**（`build_params()` 在 off 时返回空
  dict，所有方言一致），**不等于让模型不思考**；`reasoning.max_tokens_floor` 也只在
  **我们要了思考**时才生效（`min_max_tokens()` 在 off 时直接返回 base），这条路上兜不住；
- 空响应会带错误编号（`E-xxxx`），`/aichat model` 可临时改本次运行的 `max_tokens`，
  但持久化要改 `plugins.ai_chat.llm.max_tokens` 再 `/aichat reload`；
- 单个会话想单独放宽等待时间，用 `/aichat timeout 600`（见 7.6.1），
  不需要改 `config.json`。


## 7.6 会话级 max_tokens（`/aichat maxtokens`）

`llm.max_tokens` 是**全局默认**；但「这个会话要不要长回答」其实与会话绑定更合理
（和思考强度同一个理由）。所以会话可以有自己的 `max_tokens` 覆盖：

| 命令 | 作用 |
|------|------|
| `/aichat maxtokens` | 查看：全局默认 / 本会话覆盖 / 实际生效（含思考下界） |
| `/aichat maxtokens 2048` | 只改**当前会话**（`8k`、`1.5k` 简写都行） |
| `/aichat maxtokens off` | 清除会话覆盖，回到全局默认 |
| `/aichat maxtokens 8192 --session=<id\|名字>` | 改指定会话（会话所有者 / 群主管理员 / 超管） |

- 覆盖值存在会话的 `meta.max_tokens` 里，**随会话持久化**（重启不丢）；
- 生效优先级：**会话覆盖 → `llm.max_tokens`（含 `/aichat model --max_tokens` 的
  运行期改动）**，之后仍会过一遍 7.5 的思考下界抬升；
- 合法范围 64–200000；`off`/`0` 表示清除；
- 全局默认要持久化改，仍走 `plugins.ai_chat.llm.max_tokens`（或
  `/aichat model --max_tokens=N`，仅本次运行有效）。

## 7.6.1 会话级 timeout（`/aichat timeout`）

`llm.timeout` 是**单次调用的总时长上限**（「想 + 答」加起来）。推理模型在难题上
可能想几十秒到几分钟，而这个上限是全局的；所以会话可以自己覆盖：

| 命令 | 作用 |
|------|------|
| `/aichat timeout` | 查看：全局默认 / 本会话覆盖 / 实际生效（并解释 `stall_timeout`） |
| `/aichat timeout 600` | 只改**当前会话**（`10m`、`1.5h` 也能写） |
| `/aichat timeout off` | 清除会话覆盖，回到全局 `llm.timeout` |
| `/aichat timeout 600 --session=<id\|名字>` | 改指定会话（会话所有者 / 群主管理员 / 超管） |

- 值存在会话的 `meta.timeout` 里，**随会话持久化**；合法范围 10–3600 秒；
- 生效优先级：**会话覆盖 → `llm.timeout`**，通过 `LlmClient.chat(timeout=…)`
  传到流式循环的 deadline 和 socket 超时；
- ⚠️ **和 `stall_timeout` 不是一回事**：`timeout` 是「总共最多等多久」，
  `stall_timeout`（默认 60s）是「多久没有新数据就判卡死」。思考链一直在流时，
  停滞判据永远不触发，只有 `timeout` 兜底 —— 所以难题超时应该调这里，
  而不是调 `stall_timeout`（后者调大只会让真卡死时等更久）；
- 调大的代价：上游真的挂住时要等更久才报错。建议 300–600，够覆盖慢思考，
  又不至于让用户干等十几分钟。

## 7.7 压缩摘要单独一档

摘要要求「短、密、结构化」，思考链在这里纯属浪费，所以 `reasoning.compaction`
默认 `off`。但有些推理模型关掉思考后摘要质量明显下降，所以留了单独配置的余地：

```jsonc
"reasoning": { "default": "high", "compaction": "off" }
```

## 7.8 手工改了 config.json 怎么生效

`ConfigManager` 只在启动时读一次文件。自己改自己（`/aichat think --global_=true`
之类）会同时更新内存与文件；但**手工编辑文件**后需要：

```text
/aichat reload --confirm=true       # 超级用户 · 仅私聊
```
