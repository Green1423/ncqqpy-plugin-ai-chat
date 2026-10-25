# 十七、多模态（图片 / 语音）

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 上一章：[结束符](16-terminator.md) · 下一章：[实现参考](https://github.com/Green1423/ncqqpy/blob/main/docs/reference/ai-chat/README.md)


群友发来一张图、一段语音 —— 模型能拿到吗？

**图片默认拿不到，语音默认全给。** 这条不对称是刻意的：

| 媒体 | 怎么进对话 | 标号 |
|---|---|---|
| 图片（**这条消息里发的**） | 默认就下载、就传 —— 「这张图 + 这个问题」本来就是一件事 | `[图#N]` |
| 图片（**引用别的消息**） | 引用一次就把那条消息里的图一起带上（见 17.1a） | `[图#N]` |
| 图片（**更早、没引用的**） | 引用那条消息 + `/aichat mark` | `[图#N]` |
| 语音 | **默认转写**：落地 → 转成文字 → 文字直接进正文（见 17.1） | `[语音#N]` |

> 这条边界是**实测改出来的**：早先一律要求标记，结果用户把图案配着问题发过来
> （「给这个图案写一份提示词」），模型只回一句「图没收到」。现在「同一条消息里的图」
> 默认传，历史里的图才按需标记 —— 既不用每次先敲命令，也不会把几十张历史图
> 全塞进上下文。

> 为什么语音是「转文字」而不是「传音频」：qwen3.8-27b 明确拒收音频
> （实测 400 `At most 0 audio(s) may be provided in one prompt`），塞进去整轮对话就崩。
> 转成文字既绕开这个限制，又能进历史、被压缩、被引用。

为什么图片不默认全传：多模态 token 贵，群里绝大多数图与问题无关，而「要看哪张」
只有人知道。所以把「要不要看」变成一次显式动作。

## 17.1a 历史里的图：`/aichat mark`

```text
（长按那张图 → 引用 → 发送）
/aichat mark
```

回执：

```text
🎯 已标记 [图#1]（24KB）
下一条提问会把它一起交给模型 —— 直接问「这张图里是什么」就行。
（临时标记：提交一次后失效；要再看得重新标记）
```

用于**历史里**（上一轮、几十条之前）的图 —— 同一条消息里刚发的图不需要标记，默认就传。

规矩：

* **必须先引用**那条含图的消息；没引用就发命令，会回一句「引用一条含图片的消息」；
* 引用里**只有一张**图 → 不用给序号；**多张** → `/aichat mark 2` 指定第几张；
* 引用的那条里没有图 → 明说，别让用户以为标上了；
* 标记是**临时**的：跟着下一次提交走，用完即失效（再看得重标）；
* 没标记的图在正文里就是一句提示：
  `（含一张图片，未标记：引用这条消息并发送 /aichat mark 我才能看到它）`。

编号是**会话内**自增的（`session.meta["media_seq"]`），随消息落盘 —— 重启、
压缩之后 `[图#3]` 仍指同一张图。编号只在**标记时**分配：到达就下载既费流量，
而编号对用户的意义本来就是「我要用这张」。

## 17.1 语音：默认转成文字交给模型

QQ 语音到达时：下载 →（amr/silk 用 ffmpeg 转 16k 单声道 wav）→ **转写** → 转写文本
拼进正文：

```text
[#7] 明天下午三点的会改到四点
[语音#1] 明天下午三点的会改到四点
```

引擎**由配置决定，插件不绑死一家**（`asr` 段）——引擎换实现、换机器、换云端，
插件代码都不用改：

| `engine` | 跑什么 | 典型用途 |
|---|---|---|
| `cmd` | 一条外部命令，`{file}` 替换成 wav 路径；取 stdout **最后一行非空** | 本地 sherpa-onnx / whisper.cpp / 将来的 RKNN 版 |
| `http` | OpenAI 风格 `/v1/audio/transcriptions`（读 `text` 字段） | 网关开了 whisper / qwen3-asr |
| 留空 | 按配置自动选：有 `command` → cmd，有 `url` → http | —— |

转写失败**不影响这轮对话**：标号后面写 `（转写失败：原因）`，模型会如实说没听清。

### 本机怎么跑起来（CPU 版 Qwen3-ASR int8）

```bash
# 1) 装引擎（aarch64 + Python 3.14 有现成轮子）
<仓库根>/venv-arm/bin/pip install sherpa-onnx

# 2) 下模型（Qwen3-ASR 0.6B int8，约 700MB）
cd /ncqqpy/plugins/ai_chat/data
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
tar xvf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
```

`config.json → plugins.ai_chat.asr`：

```json
{
  "enabled": true,
  "command": "sherpa-onnx-offline --qwen3-asr-conv-frontend=/ncqqpy/plugins/ai_chat/data/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx --qwen3-asr-encoder=/ncqqpy/plugins/ai_chat/data/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx --qwen3-asr-decoder=/ncqqpy/plugins/ai_chat/data/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx --qwen3-asr-tokenizer=/ncqqpy/plugins/ai_chat/data/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer {file}"
}
```

配好后先自检（**别等群里发语音才发现引擎坏了**）：

```text
/aichat asr        # 超管、私聊
🎧 语音转写：引擎 cmd：sherpa-onnx-offline 可用（静音样本转写为 ''）
```

### 为什么不走 RKNN（NPU）

2026-10-25 在本机（Rock 5B+ / RK3588）核实过：

| 条件 | 现状 |
|---|---|
| NPU 设备节点 `/dev/rknpu*` | **不存在**（`dmesg` / `lsmod` 也没有 rknpu） |
| RKNN 运行时 `librknnrt.so` | 没装 |
| `rknn-toolkit-lite2` | 不在 PyPI 上；官方只给到 cp38–cp311 的轮子，本机 Python 3.14 |
| Qwen3-ASR 权重 | **开源**（`Qwen/Qwen3-ASR-0.6B` / `1.7B`），社区也有 RK3588 转换版 |

结论：**RKNN 这条路要先装 Rockchip 的 NPU 驱动 + 运行时**（系统级改动，得重启），
之后才能谈「int8 RKNN 模型」。插件这边**已经不受影响** —— 转写引擎是可插拔的，
RKNN 版做出来后把它填进 `asr.command` 就切过去了，不用改代码。

## 17.2 能力探测（这个模型到底收不收）

不能把「支持多模态」写死在配置里：换模型、网关关掉口子，写死就会一直 400/500。
所以按 `(base_url, model)` **探测一次并缓存**（`data/capabilities.json`）：

| 探测结果 | 记成 | 缓存 |
|---|---|---|
| 200 | 支持 | 7 天 |
| 4xx | 不支持 | 7 天 |
| 5xx / 超时 | **未知**（多半是上游的锅） | 10 分钟内不重试，连续 3 次后按不支持 |

探测用的是一张 8×8 的 PNG / 0.1 秒静音 WAV，只问「收不收这种 part」。
换模型自动重探（缓存键里有 model）。

实测（2026-10-24，本机网关）：

```text
deepseek-v4-flash  → 图片 500（上游崩）、音频 500  → 视为不支持
qwen3.8-27b        → 图片 200（能看图）、音频 400「At most 0 audio(s)」→ 不支持
```

## 17.3 发给模型的形态

选中的媒体拼成 OpenAI 的 content 数组（`compaction.DshWrapping._media_parts`）：

```json
{"role": "user", "content": [
  {"type": "text", "text": "[#7] 这张图是什么\n[图#1]"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,…"}}
]}
```

* 图片先**缩到 `max_side` 以内并转 JPEG** —— 多模态接口对尺寸/体积敏感；
* 语音走 `input_audio`，只收 `wav` / `mp3`；QQ 常见的 amr/silk 由 **ffmpeg**
  转成 16kHz 单声道 wav（没装 ffmpeg 或转失败 → 标号后面补「未取到」，模型
  会如实说听不到）；
* 历史重放用的是**当轮**固化的 `media_include`，所以后面再引用这些消息不会
  「重新长出」图片 —— 上下文与当时模型看到的完全一致。

> **视频整条路径已删**（2026-10-24）：OpenAI 协议里没有标准视频 part，能收
> `video_url` 的模型凤毛麟角。视频消息照旧作为「多媒体内容」出现在正文说明里，
> 但没有标号、也不下载。

## 17.4 模型知道自己的能耐

系统提示里按探测结果加一段（`_media_capability_note`）：

```text
这个模型**不能看图片、也不能听语音**：消息里的 `[图#N]`、`[语音#N]` 只是占位标号，
你看不到里面的内容。需要图里的信息时，请让用户用文字描述，或说明你看不到 ——
不要假装看到了。
```

不支持某一模态时**不传字节**（传了只会 400/500），但标号照留 —— 模型知道
「这儿本来有张图」，也就能给出「我看不到，请你描述或者把图发成文字」这种
有用的回答，而不是编一段。

## 17.5 配置

```json
"multimodal": {
  "enabled": true,
  "image": true,
  "audio": true,
  "max_bytes": 4194304,
  "max_side": 1568,
  "keep_files": 200
}
```

| 键 | 作用 |
|---|---|
| `enabled` | 总开关：关掉后媒体一律只留文字说明（省一次探测） |
| `image` / `audio` | 这一类输入**允不允许**用（还得模型收得下才真传） |
| `max_bytes` | 单条媒体下载上限 |
| `max_side` | 图片缩到的最长边 |
| `keep_files` | 本地缓存多少条媒体（超出按时间清理） |

密钥可以放环境变量：`llm.api_key` 写 `env:AICHAT`（或 `${AICHAT}`）即从环境变量读，
缺变量时回执里说的是「环境变量 AICHAT 未设置」，不是 traceback。
