"""纯衰减 / 状态迁移逻辑。

这个模块刻意不 import 任何 AstrBot 的东西，也不碰数据库和文件系统，
所以可以在本地用 `python tests/run_tests.py` 直接验证。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---- 状态 ----
STATE_OBSERVING = "observing"
STATE_ACTIVE = "active"
STATE_DORMANT = "dormant"
STATE_FORGOTTEN = "forgotten"

# ---- 默认阈值（可被配置覆盖）----
DEFAULT_HALF_LIFE_DAYS = 3.0
ACTIVE_THRESHOLD = 0.35
FORGET_THRESHOLD = 0.08
FORGET_GRACE_DAYS = 3.0
MAX_STRENGTH = 2.5
REINFORCE_SEEN = 0.35
REINFORCE_USED = 0.25
OBSERVE_HOURS = 6.0
REVIVE_STRENGTH = 1.5


def daily_factor(half_life_days: float) -> float:
    """半衰期 -> 每天剩多少。

    3 天半衰期 => 0.5 ** (1/3) ≈ 0.7937
    """
    if half_life_days <= 0:
        raise ValueError("half_life_days must be > 0")
    return 0.5 ** (1.0 / half_life_days)


def decay(strength: float, elapsed_days: float, half_life_days: float) -> float:
    """按经过的天数做指数衰减。elapsed_days <= 0 时原样返回。"""
    if elapsed_days <= 0:
        return strength
    return strength * (0.5 ** (elapsed_days / half_life_days))


def reinforce(strength: float, delta: float, cap: float = MAX_STRENGTH) -> float:
    return min(cap, strength + delta)


def threshold_crossing_time(
    strength: float,
    threshold: float,
    last_seen: float,
    half_life_days: float,
) -> float:
    """解析求出"什么时候跌破阈值"的绝对时间。

    为什么需要它：如果只在"检查的这一刻"才把 below_since 记成 now，
    那么隔 20 天才检查一次的那条记录，会从"已跌破 10 天"变成"刚刚跌破"，
    宽限期永远走不完。所以要用 s0 * 2^(-d/H) = threshold 反解 d。
    """
    if strength <= threshold:
        return last_seen
    if half_life_days <= 0:
        return last_seen
    days = half_life_days * math.log2(strength / threshold)
    return last_seen + days * 86400.0


@dataclass
class StickerState:
    """衰减需要的全部状态，和数据库一行对应。"""

    strength: float
    state: str
    created_at: float          # unix 秒
    last_seen: float           # 最后一次"被人看到"的时间（只由真实事件更新）
    anchor: float = 0.0        # 衰减基准时间（维护任务推进）；0 表示沿用 last_seen
    below_since: float | None = None


def next_state(
    s: StickerState,
    now: float,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    active_threshold: float = ACTIVE_THRESHOLD,
    forget_threshold: float = FORGET_THRESHOLD,
    forget_grace_days: float = FORGET_GRACE_DAYS,
    observe_hours: float = OBSERVE_HOURS,
) -> StickerState:
    """推进一次时间（通常是"每天一次"或"每次访问前"），返回新状态。

    规则：
      1. 先按从 last_seen 到 now 的间隔衰减
      2. 未过观察期 -> observing（不许主动发）
      3. strength 高于阈值 -> active
      4. 低于 active 但高于 forget -> dormant
      5. 低于 forget 且持续超过 grace -> forgotten
    """
    anchor = s.anchor or s.last_seen
    elapsed_days = max(0.0, (now - anchor) / 86400.0)
    strength = decay(s.strength, elapsed_days, half_life_days)

    # 掉到遗忘线以下时开始计时（用解析解，而不是"检查时刻"）
    if strength >= forget_threshold:
        below_since = None
    elif s.below_since is not None:
        below_since = s.below_since
    else:
        below_since = threshold_crossing_time(
            s.strength, forget_threshold, anchor, half_life_days
        )

    # 观察期优先
    age_hours = (now - s.created_at) / 3600.0
    if age_hours < observe_hours:
        state = STATE_OBSERVING
    elif below_since is not None and (now - below_since) / 86400.0 >= forget_grace_days:
        state = STATE_FORGOTTEN
    elif strength >= active_threshold:
        state = STATE_ACTIVE
    else:
        state = STATE_DORMANT

    return StickerState(
        strength=strength,
        state=state,
        created_at=s.created_at,
        last_seen=s.last_seen,   # 关键：衰减不改变"被人看到"的时间
        anchor=now,
        below_since=below_since,
    )


def proactive_probability(strength: float, base: float = 0.25, cap: float = MAX_STRENGTH) -> float:
    """强度越高越可能被主动发出；强度为 0 时不发。"""
    if strength <= 0:
        return 0.0
    return max(0.0, min(1.0, base * (strength / cap)))


def is_eligible_to_send(state: str, strength: float, active_threshold: float = ACTIVE_THRESHOLD) -> bool:
    """只有 active 且强度达标的图才允许主动发出。"""
    return state == STATE_ACTIVE and strength >= active_threshold


def should_forget(state: str) -> bool:
    return state == STATE_FORGOTTEN


def eviction_order(rows: list[dict]) -> list[dict]:
    """容量超限时的淘汰顺序：强度最低的先走；同强度按最久没出现。

    rows 里每项至少要有 strength 和 last_seen。
    """
    return sorted(rows, key=lambda r: (r["strength"], r["last_seen"]))
