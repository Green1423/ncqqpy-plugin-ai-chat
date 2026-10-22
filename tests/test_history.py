"""`history.py` 的纯单元测试 —— 不起客户端、不联网、不渲染。

跑法：
    cd /ncqqpy && ./venv-arm/bin/python -B tests/test_history.py

这里只卡**归并语义**（哪条算问、哪条算答、怎么编号、怎么分页），因为那是本功能
唯一会「悄悄骗人」的地方：编号算错，用户看到的图和清单就对不上。
渲染与真实发送在 `test_history_render.py` 里测。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP_DIR = ROOT / "tests" / ".test-tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)   # 新 clone 里没有它（.gitignore 排除）

# ---- 插件包位置：两种仓库布局都要能跑 --------------------------------------
# 框架仓库布局 = <repo>/plugins/ai_chat/（直接可用）；
# 独立插件仓库布局 = 仓库根就是包目录，而且 clone 出来的目录名未必等于插件名。
# 后者搭一个「包农场」<tmp>/pkgfarm/plugins/ai_chat -> 插件包目录，让
# `import plugins.ai_chat.*` 与 `plugin_dir=<农场>/plugins` 两种口径都成立。
PLUGIN_NAME = "ai_chat"
PLUGIN_DIR = ROOT / "plugins" / PLUGIN_NAME
PLUGIN_ROOT_DIR = ROOT / "plugins"
if not (PLUGIN_DIR / "__init__.py").exists() and (ROOT / "__init__.py").exists():
    import tempfile
    PLUGIN_ROOT_DIR = (Path(tempfile.mkdtemp(prefix=f"pkgfarm-{PLUGIN_NAME}-"))
                       / "plugins")
    PLUGIN_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if not (PLUGIN_ROOT_DIR / PLUGIN_NAME).exists():
        (PLUGIN_ROOT_DIR / PLUGIN_NAME).symlink_to(ROOT, target_is_directory=True)
    sys.path.insert(0, str(PLUGIN_ROOT_DIR.parent))
    PLUGIN_DIR = PLUGIN_ROOT_DIR / PLUGIN_NAME
PLUGIN_PKG = f"plugins.{PLUGIN_NAME}"

from plugins.ai_chat.history import (          # noqa: E402
    SUMMARY_CHARS, SummaryCache, extract_exchanges, fallback_summary,
    format_exchange, has_compaction, parse_question_summaries, parse_range,
    reject_range, render_list, short_time, speakername,
    summarize_context, summary_key, trace_chain)
from plugins.ai_chat.segments import (          # noqa: E402
    Block, Message, assistant_message, compact_message, tool_message,
    user_message)

PASSED = 0
FAILED = 0
FAILURES: list[str] = []


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  ❌ {name} — {detail}")


def _ask(text, *, user_id=20001, nickname="爱丽丝", card="", ts="2026-09-30T22:15:00+08:00"):
    # ⚠️ `ts` 必须建完再赋值：`user_message(**metadata)` 会把 ts 吞进 metadata，
    # 而 `Exchange.time` 读的是 `Message.ts`（实测踩过：拿到的是当前时刻）。
    msg = user_message(text, origin="user.group", user_id=user_id,
                       nickname=nickname, card=card)
    msg.ts = ts
    return msg


def _ans(text, ts="2026-09-30T22:15:10+08:00"):
    msg = assistant_message(text)
    msg.ts = ts
    return msg


def _with_ts(msg: Message, ts: str) -> Message:
    msg.ts = ts
    return msg


def test_short_time():
    print("\n[1] 时间格式")
    check("ISO8601 → MM-DD HH:MM",
          short_time("2026-09-30T22:15:03+08:00") == "09-30 22:15",
          short_time("2026-09-30T22:15:03+08:00"))
    check("解析不出来就原样返回（不猜）", short_time("昨天") == "昨天")
    check("空值不炸", short_time("") == "")


def test_speaker():
    print("\n[2] 提问人显示名")
    check("群名片优先", speakername({"card": "小爱", "nickname": "爱丽丝"}) == "小爱")
    check("没名片用昵称", speakername({"nickname": "爱丽丝"}) == "爱丽丝")
    check("都没有用 QQ 号", speakername({"user_id": 20001}) == "用户20001")
    check("空白名片不算", speakername({"card": "  ", "nickname": "爱丽丝"}) == "爱丽丝")


def test_basic_extraction():
    print("\n[3] 基本归并")
    msgs = [_ask("图书馆国庆开吗？"), _ans("10月1日至3日闭馆。")]
    ex = extract_exchanges(msgs)
    check("切出 1 对", len(ex) == 1, f"{len(ex)}")
    check("编号从 1 开始", ex[0].index == 1, f"{ex[0].index}")
    check("问对了", ex[0].question == "图书馆国庆开吗？", ex[0].question)
    check("答对了", ex[0].answer == "10月1日至3日闭馆。", ex[0].answer)
    check("带上提问人与时间",
          ex[0].speaker == "爱丽丝" and ex[0].time == "09-30 22:15",
          f"{ex[0].speaker} {ex[0].time}")
    check("多对时编号递增",
          [e.index for e in extract_exchanges(
              msgs + [_ask("那明天呢？"), _ans("明天开。")])] == [1, 2],
          "编号不是 1,2")


def test_answer_is_last_not_first():
    print("\n[4] 答案是最后一条，不是中间那句")
    msgs = [
        _ask("国庆图书馆开吗？"),
        assistant_message("我先查一下通知。",
                          tool_calls=[Block.tool_call("c1", "get_notice", {})]),
        tool_message("c1", '{"正文": "10月1日至3日闭馆"}', name="get_notice"),
        _ans("10月1日至3日闭馆，10月4日起恢复。"),
    ]
    ex = extract_exchanges(msgs)
    check("仍然只有 1 对", len(ex) == 1, f"{len(ex)}")
    check("答案是最终回答（不是「我先查一下」）",
          ex[0].answer == "10月1日至3日闭馆，10月4日起恢复。", ex[0].answer)
    check("工具被计数", ex[0].tool_calls == ["get_notice"] and ex[0].tool_count == 1,
          f"{ex[0].tool_calls} {ex[0].tool_count}")
    check("工具小字可读", ex[0].tool_note == "🔧 调用了 get_notice", ex[0].tool_note)


def test_tool_counting():
    print("\n[5] 工具计数与去重")
    msgs = [
        _ask("查两件事"),
        assistant_message("", tool_calls=[Block.tool_call("c1", "search_notices", {})]),
        tool_message("c1", "[]", name="search_notices"),
        assistant_message("", tool_calls=[Block.tool_call("c2", "search_notices", {})]),
        tool_message("c2", "[]", name="search_notices"),
        assistant_message("", tool_calls=[Block.tool_call("c3", "get_weather", {})]),
        tool_message("c3", "晴", name="get_weather"),
        _ans("查完了。"),
    ]
    ex = extract_exchanges(msgs)
    check("同名工具只列一次、总次数照实",
          ex[0].tool_calls == ["search_notices", "get_weather"]
          and ex[0].tool_count == 3,
          f"{ex[0].tool_calls} {ex[0].tool_count}")
    check("重复调用标注 ×N", "×3" in ex[0].tool_note, ex[0].tool_note)
    check("没调工具就是空串",
          extract_exchanges([_ask("你好"), _ans("你好")])[0].tool_note == "")


def test_incomplete_pair_skipped():
    print("\n[6] 不完整的一对不占编号（否则清单与 get 错位）")
    # 第一对只有问没有答（答到一半崩了），第二对完整
    msgs = [
        _ask("这个没人答"), assistant_message("", tool_calls=[
            Block.tool_call("c1", "get_current_time", {})]),
        _ask("这个有答"), _ans("答了"),
    ]
    ex = extract_exchanges(msgs)
    check("只剩 1 对", len(ex) == 1, f"{len(ex)}")
    check("完整的那个是 1 号（不是 2 号）", ex[0].index == 1, f"{ex[0].index}")
    check("内容对得上", ex[0].question == "这个有答", ex[0].question)
    check("末尾只有问也不产生编号",
          [e.index for e in extract_exchanges([_ask("刚问的"), _ans("答")] + [_ask("还没答")])]
          == [1], "末尾未答的被算进去了")
    check("空历史 → 空清单", extract_exchanges([]) == [])
    check("全是工具消息 → 空清单",
          extract_exchanges([tool_message("c1", "x", name="t")]) == [])


def test_compact_is_not_a_question():
    print("\n[7] 压缩摘要不算用户提问")
    msgs = [
        compact_message("## 用户诉求\n查图书馆开放时间"),
        _ask("那国庆呢？"), _ans("闭馆三天。"),
    ]
    ex = extract_exchanges(msgs)
    check("摘要没被当成一对", len(ex) == 1, f"{len(ex)}")
    check("唯一那对是真实提问", ex[0].question == "那国庆呢？", ex[0].question)
    check("能识别历史里有压缩", has_compaction(msgs), "应识别出压缩摘要")
    check("没有压缩时返回 False", not has_compaction([_ask("a"), _ans("b")]))


def test_multi_part_question_and_attribution():
    print("\n[8] 一次输入的多个 user 消息算同一对；提问人取第一条")
    msgs = [
        _ask("第一段", user_id=1, nickname="甲"),
        _with_ts(_ask("第二段", user_id=1, nickname="甲"), "2026-09-30T22:16:00+08:00"),
        _ans("合起来答"),
    ]
    ex = extract_exchanges(msgs)
    check("只算 1 对", len(ex) == 1, f"{len(ex)}")
    check("两段问题都在", ex[0].question == "第一段\n第二段", ex[0].question)
    check("提问人取第一条（不被后一条覆盖）", ex[0].speaker == "甲", ex[0].speaker)
    check("时间取第一条", ex[0].time == "09-30 22:15", ex[0].time)


def test_consecutive_user_messages_share_one_pair():
    print("\n[8.1] 分成两段发的提问算一对（不是两对）")
    # 用户连发两条、中间没有回答 —— 这是「一次输入分几段」的落盘形态
    msgs = [
        _ask("我想问个事", user_id=7, nickname="乙"),
        _with_ts(_ask("图书馆国庆开吗", user_id=7, nickname="乙"),
                 "2026-09-30T22:16:00+08:00"),
        _ans("闭馆三天。"),
    ]
    ex = extract_exchanges(msgs)
    check("只切出 1 对（不是 2 对）", len(ex) == 1, f"{len(ex)}")
    check("两段都在问题里", ex[0].question == "我想问个事\n图书馆国庆开吗",
          repr(ex[0].question))
    check("时间取第一段", ex[0].time == "09-30 22:15", ex[0].time)

    # 中间有回答之后，再连发两条 → 新的那一对
    msgs2 = msgs + [_ask("那明天呢", user_id=7, nickname="乙"),
                    _ask("后天呢", user_id=7, nickname="乙"), _ans("都开。")]
    ex2 = extract_exchanges(msgs2)
    check("回答之后再问 → 新的 1 对，共 2 对", len(ex2) == 2, f"{len(ex2)}")
    check("第二对把连发两条合并", ex2[1].question == "那明天呢\n后天呢",
          repr(ex2[1].question))


def test_render_list_paging():
    print("\n[9] 清单渲染与分页")
    check("空清单不出内容", render_list([], header="x") == ["x"])
    check("空清单且无表头 → 空列表", render_list([]) == [])
    one = extract_exchanges([_ask("问"), _ans("答")])
    chunks = render_list(one, header="表头")
    check("表头在最前面", chunks[0].startswith("表头"), chunks[0][:40])
    check("一条问答占一块", len(chunks) == 1, f"{len(chunks)}")

    # 30 对 → 必然分页；每块都不超限
    many_msgs = []
    for i in range(30):
        many_msgs += [_ask(f"第 {i} 个问题" + "内容" * 30), _ans(f"第 {i} 个回答" + "答复" * 30)]
    many = extract_exchanges(many_msgs)
    check("30 对都切出来了", len(many) == 30, f"{len(many)}")
    pages = render_list(many, header="表头", chunk_max_chars=600)
    check("确实分页了", len(pages) > 1, f"{len(pages)}")
    check("每页都在配额内", all(len(p) <= 600 for p in pages),
          f"{[len(p) for p in pages]}")
    # 关键：不能把一条问答劈成两半 —— 每页的条数加起来必须正好是全部
    joined = "\n".join(pages)
    missing = [e.index for e in many if f"{e.index}. " not in joined]
    check("所有编号都出现过", not missing, f"缺 {missing}")
    # 关键：一条记录（编号头 + 问那行）必须整条落在同一页，不能被劈开
    split_across = []
    for e in many:
        entry = format_exchange(e)
        if not any(entry in page for page in pages):
            split_across.append(e.index)
    check("每条记录（两行）整条落在同一页", not split_across, f"被劈开 {split_across}")
    # 回答**不进清单**：原始回答的开头不该在任何一页里出现
    all_pages = "\n".join(pages)
    leaked = [e.index for e in many if f"答：{e.answer[:20]}" in all_pages]
    check("清单里没有回答（连预览也没有）", not leaked, f"泄漏 {leaked}")


def test_format_exchange_is_one_record():
    print("\n[10] 单条格式化")
    ex = extract_exchanges([_ask("问题"), _ans("回答")])[0]
    text = format_exchange(ex)
    check("含编号与提问人", text.startswith("1. 爱丽丝"), text[:30])
    check("只给提问：有条目头 + 问那行", "问：问题" in text, text)
    check("回答不进清单", "回答" not in text, text)
    check("恰好两行（一条记录固定占两行）", text.count("\n") == 1, repr(text))
    multi = extract_exchanges([_ask("第一行\n第二行"), _ans("答复")])[0]
    check("问题内部的换行被压成空格（不撑破一条记录）",
          "第一行 第二行" in format_exchange(multi), repr(format_exchange(multi)))
    long_ex = extract_exchanges([_ask("问" * 200), _ans("答" * 200)])[0]
    flat = format_exchange(long_ex, preview=20)
    check("超长提问按 preview 截断加省略号", "…" in flat and flat.count("\n") == 1, flat)
    check("摘要命中时用摘要代替原文",
          format_exchange(ex, summaries={1: "模型写的概括"}).endswith("问：模型写的概括"),
          format_exchange(ex, summaries={1: "模型写的概括"}))


def test_parse_range():
    print("\n[11] 编号参数解析")
    check("单个数字", parse_range("3", total=10) == ([3], None), f"{parse_range('3', total=10)}")
    check("区间", parse_range("3-5", total=10) == ([3, 4, 5], None),
          f"{parse_range('3-5', total=10)}")
    check("顺序反了自动纠正", parse_range("5-3", total=10)[0] == [3, 4, 5],
          f"{parse_range('5-3', total=10)}")
    check("带空格也行", parse_range(" 3 - 5 ", total=10)[0] == [3, 4, 5],
          f"{parse_range(' 3 - 5 ', total=10)}")
    check("上界可以正好等于总数", parse_range("10", total=10)[0] == [10])

    codes, err = parse_range("999", total=10)
    check("越界给出总数（可操作）", not codes and "共 10 对" in err, err)
    codes, err = parse_range("0", total=10)
    check("0 被拒且说明从 1 开始", not codes and "从 1 开始" in err, err)
    codes, err = parse_range("abc", total=10)
    check("非数字被拒且给用法", not codes and "hget" in err, err)
    codes, err = parse_range("", total=10)
    check("空参数给用法", not codes and "hget" in err, err)
    codes, err = parse_range("3", total=0)
    check("没有问答时明确说明", not codes and "还没有" in err, err)
    codes, err = parse_range("1-500", total=10)
    check("区间越界也拦", not codes and "没有第 500 对" in err, err)


def test_reject_range():
    print("\n[12] hget 只取一条：区间被拦")
    check("区间被拒并给出替代", "hget 3" in (reject_range("3-5") or ""),
          f"{reject_range('3-5')}")
    check("带空格的区间也拦", reject_range(" 3 - 5 ") is not None)
    check("单个编号放行", reject_range("3") is None)
    check("空串放行（交给 parse_range 报「请给出编号」）",
          reject_range("") is None)
    check("非数字放行（交给 parse_range 报用法）", reject_range("abc") is None)


def test_parse_question_summaries():
    print("\n[13] 提问概括解析（模型可能不听话）")
    got, err = parse_question_summaries(
        "1|图书馆国庆开放时间\n2|奖学金评审截止日期", wanted=[1, 2])
    check("标准格式解析正确",
          got == {1: "图书馆国庆开放时间", 2: "奖学金评审截止日期"}, f"{got}")
    check("正常解析无错误说明", err == "", err)

    got, _ = parse_question_summaries("1｜全角竖线也认", wanted=[1])
    check("全角竖线也认", got[1] == "全角竖线也认", f"{got}")

    got, _ = parse_question_summaries("1|带项目符号", wanted=[1])
    check("行首项目符号被剥掉", 1 in got, f"{got}")

    got, _ = parse_question_summaries(f"1|{'字'*50}", wanted=[1], limit=20)
    check("超长截到 20 字", len(got[1]) == 20, f"{len(got[1])}")

    got, _ = parse_question_summaries("1|「图书馆闭馆」", wanted=[1])
    check("包裹符号被去掉", got[1] == "图书馆闭馆", f"{got}")

    got, _ = parse_question_summaries("1|要的\n99|不要的\n乱码行", wanted=[1])
    check("只要 wanted 里的编号", list(got) == [1], f"{list(got)}")

    # 模型多写了一段「答：」也照收（只取第一段）—— 多余的不会污染概括
    got, _ = parse_question_summaries("1|国庆安排|答：闭馆三天", wanted=[1])
    check("多写了内容也不炸", got[1].startswith("国庆安排"), f"{got}")

    got, err = parse_question_summaries("模型完全没按格式回答", wanted=[1, 2])
    check("整批解析失败时给出说明", got == {} and "格式" in err, f"{got} {err}")
    check("空输入不炸",
          parse_question_summaries("", wanted=[1]) == ({}, "模型没有按「编号|概括」的格式回答"))


def test_fallback_and_key():
    print("\n[14] 兜底截断与内容指纹")
    ex = extract_exchanges([_ask("图书馆国庆期间到底开不开门，我想知道具体安排"),
                            _ans("10月1日至3日闭馆，4日起恢复开放")])[0]
    # 「图书馆国庆期间到底开不开门，我想知道具体安排」共 22 字 → 取前 20 字
    check("兜底取问题前 20 字", fallback_summary(ex) == "图书馆国庆期间到底开不开门，我想知道具体",
          fallback_summary(ex))
    check("兜底长度不超限", len(fallback_summary(ex)) == SUMMARY_CHARS,
          f"{len(fallback_summary(ex))}")
    check("prefer_answer 时取回答",
          fallback_summary(ex, prefer_answer=True).startswith("10月1日至3日闭馆"),
          fallback_summary(ex, prefer_answer=True))
    check("短文本原样返回（不加省略号）",
          fallback_summary(extract_exchanges([_ask("短"), _ans("答")])[0]) == "短")

    k1, k2 = summary_key(ex), summary_key(ex)
    check("同一内容指纹稳定", k1 == k2, f"{k1} {k2}")
    other = extract_exchanges([_ask("图书馆国庆期间到底开不开门，我想知道具体安排"),
                               _ans("换了一个回答")])[0]
    check("内容变了指纹就变", summary_key(other) != k1, f"{k1} {summary_key(other)}")


def test_summarize_context():
    print("\n[15] 送给摘要模型的输入（只有提问）")
    exs = extract_exchanges([_ask("问题甲"), _ans("回答甲"),
                             _ask("问题乙"), _ans("回答乙")])
    text = summarize_context(exs, read_chars=400)
    check("带编号与提问", "1. 问题甲" in text and "2. 问题乙" in text, text)
    check("**不带**回答（清单只概括提问）",
          "回答甲" not in text and "回答乙" not in text, text)
    long_ex = extract_exchanges([_ask("问" * 1000), _ans("答" * 1000)])[0]
    got = summarize_context([long_ex], read_chars=100)
    check("超长提问按 read_chars 截断（防烧 token）",
          len(got) < 300, f"长度 {len(got)}")
    check("截断有省略号", "…" in got, got[:80])


def test_summary_cache():
    print("\n[16] 摘要缓存（翻页查看不能反复烧 token）")
    import tempfile as _tf
    d = _tf.mkdtemp(prefix="sumcache-", dir=str(ROOT / "tests" / ".test-tmp"))
    path = str(Path(d) / "s.json")
    c = SummaryCache(path)
    check("空缓存未命中", c.get("s1", "k1") is None)
    c.put("s1", {"k1": "问摘要"})
    check("写入后可读（单值）", c.get("s1", "k1") == "问摘要", f"{c.get('s1','k1')}")
    c2 = SummaryCache(path)
    check("落盘后重启仍在（不必重算）", c2.get("s1", "k1") == "问摘要",
          f"{c2.get('s1','k1')}")
    check("不同键互不干扰", c2.get("s1", "other") is None)
    check("不同会话互不干扰", c2.get("s2", "k1") is None)

    # 空摘要不算命中（落盘损坏时不要给出空概括）
    c2.put("s1", {"empty": ""})
    check("空摘要不写入也不命中", c2.get("s1", "empty") is None,
          f"{c2.get('s1','empty')}")

    # 上限淘汰
    c3 = SummaryCache(str(Path(d) / "c3.json"), max_entries=3)
    for i in range(5):
        c3.put(f"sess{i}", {f"k{i}": f"q{i}"})
    total = sum(len(b) for b in c3._data.values())
    check("条目数不超上限", total <= 3, f"{total}")
    check("保留的是最新的会话", c3.get("sess4", "k4") == "q4",
          f"{sorted(c3._data)}")


def test_exchange_quote_meta():
    """一对问答要记下「它引用了谁」与「它的 QQ 消息 id」—— 引用链靠这个追。"""
    print("\n[9.9] 问答的引用元数据")
    q1, a1 = _ask("第一个问题"), _ans("第一个回答")
    q1.metadata["message_ids"] = [7001]
    q2, a2 = _ask("第二个问题"), _ans("第二个回答")
    q2.metadata["message_ids"] = [7002, 7003]      # 一次输入分两段发
    q2.metadata["reply_to"] = 7001
    ex = extract_exchanges([q1, a1, q2, a2])
    check("记下提问消息 id", ex[1].message_ids == [7002, 7003],
          f"{ex[1].message_ids}")
    check("记下引用了谁", ex[1].quotes == [7001], f"{ex[1].quotes}")
    check("没引用就是空", ex[0].quotes == [], f"{ex[0].quotes}")
    check("回答也记了 seq 字段（默认 0）", ex[1].answer_seq == 0,
          f"{ex[1].answer_seq}")


def test_trace_chain():
    """`trace_chain`：沿引用往回追溯（最早在前），并挡住环与断链。"""
    print("\n[9.95] 沿引用追溯")
    q1, a1 = _ask("问题一"), _ans("回答一")
    q1.metadata["message_ids"] = [101]
    q2, a2 = _ask("问题二（引用一）"), _ans("回答二")
    q2.metadata["message_ids"] = [102]
    q2.metadata["reply_to"] = 101
    q3, a3 = _ask("问题三（引用二）"), _ans("回答三")
    q3.metadata["message_ids"] = [103]
    q3.metadata["reply_to"] = 102
    msgs = [q1, a1, q2, a2, q3, a3]

    check("两段的链", trace_chain(msgs, 2) == [1, 2], f"{trace_chain(msgs, 2)}")
    check("三段的链（最早在前）", trace_chain(msgs, 3) == [1, 2, 3],
          f"{trace_chain(msgs, 3)}")
    check("第一对没有上文", trace_chain(msgs, 1) == [1], f"{trace_chain(msgs, 1)}")
    check("编号不存在时为空", trace_chain(msgs, 99) == [], f"{trace_chain(msgs, 99)}")
    check("max_depth 生效", trace_chain(msgs, 3, max_depth=2) == [2, 3],
          f"{trace_chain(msgs, 3, max_depth=2)}")

    # 环：第三对引用第三对（QQ 里不该出现，出现了不能转不出来）
    q4, a4 = _ask("自引用"), _ans("回答四")
    q4.metadata["message_ids"] = [104]
    q4.metadata["reply_to"] = 104
    loop = msgs + [q4, a4]
    check("环被挡住", trace_chain(loop, 4) == [4], f"{trace_chain(loop, 4)}")

    # 引用了「不在历史里」的消息（被压缩吸收/别的会话）→ 链到此为止
    q5, a5 = _ask("引用消失的消息"), _ans("回答五")
    q5.metadata["message_ids"] = [105]
    q5.metadata["reply_to"] = 999
    got = trace_chain(msgs + [q5, a5], 4)
    check("断链只剩自己", got == [4], f"{got}")


def main():
    for fn in (test_short_time, test_speaker, test_basic_extraction,
               test_answer_is_last_not_first, test_tool_counting,
               test_incomplete_pair_skipped, test_compact_is_not_a_question,
               test_multi_part_question_and_attribution,
               test_consecutive_user_messages_share_one_pair,
               test_render_list_paging, test_format_exchange_is_one_record,
               test_parse_range, test_reject_range, test_parse_question_summaries,
               test_fallback_and_key, test_summarize_context, test_summary_cache,
               test_exchange_quote_meta, test_trace_chain):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            globals()["FAILED"] = globals()["FAILED"] + 1
            FAILURES.append(f"{fn.__name__}: {e!r}")
            print(f"  ❌ {fn.__name__} 抛异常: {e!r}")
            traceback.print_exc()
    print(f"\n{'=' * 50}\n通过 {PASSED} / 失败 {FAILED}")
    if FAILURES:
        print("失败项：")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
