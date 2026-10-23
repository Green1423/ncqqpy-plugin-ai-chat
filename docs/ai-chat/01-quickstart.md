# 一、快速开始

> 文档索引：[docs/README.md](https://github.com/Green1423/ncqqpy/blob/main/docs/README.md) · 上级：[ai_chat 插件](README.md) · 下一章：[服务范围（只服务绑定的群 / 私聊）](02-scope.md)


**所有设置都在 `config.json` 的 `plugins.ai_chat` 段**（和 `bupt_news` / `essential`
同一套 ConfigManager 机制）。首次加载时如果这个段不存在，插件会把**完整模板**写进去，
所以打开文件就能看到全部开关和当前生效值：

```jsonc
// config.json → plugins.ai_chat（由插件自动补齐，下面是最小可用改动）
{
  "enabled": true,                                   // 模板默认就是 true；false = 整个插件不响应
  "llm": {
    "base_url": "https://myai.bupt.edu.cn/llm-gw/v1",  // 任意 OpenAI 兼容端点
    "api_key": "sk-...",
    "model": "deepseek-v4-flash",
    "reasoning": { "dialect": "effort", "default": "off" }   // 思考强度，见第七节
  },
  // 群里每个人各自有当前会话，无需配置
  "compaction": { "context_limit": 16384 }
}
```

补齐规则（三条）：

1. **只补缺键，绝不覆盖已改的值** —— 升级插件新增配置项时老部署会被自动补上，
   用户改过的一个都不动；
2. **写进去的是「有效值」**（内置默认深合并用户配置的结果），
   所以文件里看到的就是真正生效的，不存在「文件写了 A、实际跑 B」；
3. **幂等** —— 没有缺键就不写盘。

> 这三条现在是**框架对所有插件的统一契约**（插件声明 `config_template()`，
> 首次运行会自动补齐），见 (../../plugin-dev/15-config.md#1531-首次运行即初始化全部配置项config_template)。

手工改完文件后：`/aichat reload --confirm=true`（超级用户 · 仅私聊），
或直接 `/aichat config` 查看/补齐。

启用后：

1. **私聊**直接说话，**群聊**先 @机器人；
2. 长内容可以分几条发 —— 因为单条 QQ 消息有长度上限 —— 最后发一条 `//` 收尾，
   意思是「**这一大段提交上去**」；
3. 机器人把这些分段**按顺序拼成一整段**当成一次输入（不加 `(1/3)` 之类的序号：
   分段只是客户端的限制，对模型来说本来就是一段话），回答再按长度自动分段发送。

```text
用户：帮我看看       ← 暂存
用户：这周六有没有   ← 暂存
用户：讲座//         ← 提交，三条拼成一次输入
机器人：……
```

> ⚠️ 不 @机器人 的群消息**完全不进上下文**（与 DSH 的「只有明确输入才算用户消息」一致），
> 避免把整群闲聊灌进模型。

---
