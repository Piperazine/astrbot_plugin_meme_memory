"""会话上下文缓冲：给"这张表情是在什么语境下发的"提供素材。

关键规则（用户明确要求）：
  - 只取**发送者本人**的消息当主证据，别人的话默认不算
  - 例外一：被引用/回复的那条消息，不管谁说的，都算最强证据
  - 例外二：紧邻的前一条（别人说的）只能当弱补充
  - 反馈信号同样按人归属：只有发图对象的下一条才算

只放内存，不落库。
"""

from __future__ import annotations

from collections import defaultdict, deque

DEFAULT_WINDOW_SECONDS = 180
DEFAULT_PER_SESSION = 20


class ContextBuffer:
    def __init__(self, per_session: int = DEFAULT_PER_SESSION,
                 window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.per_session = per_session
        self.window_seconds = window_seconds
        self._buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=per_session))

    def record(self, session: str, sender: str, text: str, ts: float,
               is_bot: bool = False) -> None:
        if not session:
            return
        text = (text or "").strip()
        if not text:
            return
        self._buf[session].append({
            "sender": str(sender or ""),
            "text": text[:200],
            "ts": ts,
            "is_bot": bool(is_bot),
        })

    def entries(self, session: str) -> list[dict]:
        return list(self._buf.get(session, ()))

    def gather(self, session: str, sender: str, ts: float,
               quoted: str = "", before: int = 2, after: int = 1) -> dict:
        """按归属优先级收集语境。

        返回 {before: [...], after: [...], neighbor: str, quoted: str}
          before   同一发送者、时间窗内、这条之前的最近几条（旧 -> 新）
          after    同一发送者、这条之后的几条（用于反馈，入库时通常为空）
          neighbor 紧邻的前一条（可能是别人），只作弱补充
          quoted   被引用的消息原文
        """
        sender = str(sender or "")
        out = {"before": [], "after": [], "neighbor": "", "quoted": quoted or ""}
        if not session:
            return out

        rows = self.entries(session)
        lo = ts - self.window_seconds

        # 严格早于"这条消息"的才算前文：否则会把消息自己当成自己的语境
        earlier = [r for r in rows if r["ts"] < ts - 1e-9]
        later = [r for r in rows if r["ts"] > ts + 1e-9]

        # 同一发送者、窗口内：取最近 before 条，保持时间顺序
        mine = [r for r in earlier if r["sender"] == sender and r["ts"] >= lo]
        out["before"] = [r["text"] for r in mine[-before:]]

        # 紧邻的前一条（任何人），用作弱补充；若已经全是我自己的就不重复
        others = [r for r in earlier if r["sender"] != sender and r["ts"] >= lo]
        if others:
            out["neighbor"] = others[-1]["text"]

        # 同一发送者的后续消息（反馈用）
        nxt = [r for r in later if r["sender"] == sender and r["ts"] <= ts + self.window_seconds]
        out["after"] = [r["text"] for r in nxt[:after]]

        return out


def render_for_prompt(ctx: dict) -> str:
    """把 gather() 结果渲染成提示词里的一段，空的就不写。"""
    lines = []
    if ctx.get("quoted"):
        lines.append(f"[被引用的消息] {ctx['quoted']}")
    for t in ctx.get("before", []):
        lines.append(f"[同一人，之前说] {t}")
    if ctx.get("neighbor"):
        lines.append(f"[紧邻的上一句，别人说的，仅作参考] {ctx['neighbor']}")
    for t in ctx.get("after", []):
        lines.append(f"[同一人，之后说] {t}")
    return "\n".join(lines)
