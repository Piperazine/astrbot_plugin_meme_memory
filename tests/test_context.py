"""语境归属测试：只认发送者本人的消息，别人的不算。

    python tests/test_context.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context import ContextBuffer, render_for_prompt

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    T = 1_700_000_000.0
    b = ContextBuffer(per_session=20, window_seconds=180)
    s = "group:1"
    A, B = "1001", "2002"

    # A 说一句，B 插一句，A 再表情包
    b.record(s, A, "这作业谁写的啊", T)
    b.record(s, B, "哈哈哈", T + 5)
    b.record(s, A, "[图片]", T + 10)

    ctx = b.gather(s, A, T + 10)
    check("只取本人的前文", ctx["before"] == ["这作业谁写的啊"], ctx["before"])
    check("别人的话不进 before", "哈哈哈" not in ctx["before"], ctx["before"])
    check("别人的话进 neighbor（弱补充）", ctx["neighbor"] == "哈哈哈", ctx["neighbor"])

    prompt = render_for_prompt(ctx)
    check("渲染里本人语境在前", "作业谁写的" in prompt)
    check("渲染里标注了别人说的", "仅作参考" in prompt, prompt)

    # 同一人连发两条
    b2 = ContextBuffer()
    b2.record(s, A, "第一句", T)
    b2.record(s, A, "第二句", T + 5)
    b2.record(s, A, "[图片]", T + 10)
    ctx2 = b2.gather(s, A, T + 10)
    check("同一人连发取最近两条", ctx2["before"] == ["第一句", "第二句"], ctx2["before"])

    # 时间窗外的旧消息不算
    b3 = ContextBuffer(window_seconds=180)
    b3.record(s, A, "十分钟前说的话", T - 600)
    b3.record(s, A, "[图片]", T)
    ctx3 = b3.gather(s, A, T)
    check("超出时间窗的不算", ctx3["before"] == [], ctx3["before"])

    # 反馈：以"发图那一刻"为查询时刻，后面同一人的消息应落在 after
    b4 = ContextBuffer()
    b4.record(s, A, "[图片]", T)
    b4.record(s, A, "笑死", T + 20)
    ctx4 = b4.gather(s, A, T)
    check("本人后一句进 after", ctx4["after"] == ["笑死"], ctx4["after"])
    check("后一句不会混进 before", ctx4["before"] == [], ctx4["before"])

    # 引用优先
    b5 = ContextBuffer()
    ctx5 = b5.gather(s, A, T, quoted="被引用的原文")
    check("引用原文保留", ctx5["quoted"] == "被引用的原文", ctx5)
    check("渲染里引用在第一行", render_for_prompt(ctx5).startswith("[被引用的消息]"), render_for_prompt(ctx5))

    # 空输入
    ctx6 = b6 = ContextBuffer().gather("", A, T)
    check("空会话不炸", ctx6["before"] == [] and ctx6["neighbor"] == "")

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
