"""本地测试：不需要 AstrBot、不需要联网、不需要 QQ。

    python tests/run_tests.py

覆盖三块：衰减/状态机、表情包判定打分、存储层（去重、强化、淘汰、复活）。
"""

import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import decay
import detect
from store import StickerStore, VlmCache

PASS = 0
FAIL = 0

# 沙箱可能不允许在系统临时目录里再建子目录，统一用工作区内的临时目录
TMP_ROOT = os.path.join(HERE, "_test_tmp")


def fresh_tmp():
    import shutil
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    os.makedirs(TMP_ROOT, exist_ok=True)
    return TMP_ROOT


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- 衰减
def test_decay():
    print("\n[decay]")
    check("half_life=3 -> 每天 0.794", approx(decay.daily_factor(3), 0.7937005, 1e-6),
          decay.daily_factor(3))
    check("3 天后正好一半", approx(decay.decay(1.0, 3.0, 3.0), 0.5, 1e-9))
    check("6 天后四分之一", approx(decay.decay(1.0, 6.0, 3.0), 0.25, 1e-9))
    check("时间不前进则不变", decay.decay(0.8, 0, 3.0) == 0.8)

    now = 1_000_000.0
    # 刚收藏：观察期
    st = decay.StickerState(strength=1.0, state="observing", created_at=now, last_seen=now)
    ns = decay.next_state(st, now + 3600)
    check("1 小时后仍在观察期", ns.state == "observing", ns.state)

    # 过了观察期且强度够 -> active
    ns = decay.next_state(st, now + 7 * 3600)
    check("7 小时后 active", ns.state == "active", ns.state)

    # 十几天不碰 -> forgotten
    ns = decay.next_state(st, now + 20 * 86400)
    check("20 天后 forgotten", ns.state == "forgotten", f"{ns.state} s={ns.strength:.3f}")

    check("强化不会超过上限", decay.reinforce(2.4, 0.35) == 2.5)
    check("强度 0 不发图", decay.proactive_probability(0) == 0.0)
    check("强度高概率高", decay.proactive_probability(2.0) > decay.proactive_probability(0.5))

    rows = [{"strength": 0.9, "last_seen": 10}, {"strength": 0.1, "last_seen": 20},
            {"strength": 0.5, "last_seen": 5}]
    order = decay.eviction_order(rows)
    check("淘汰先走最弱的", order[0]["strength"] == 0.1)


# ---------------------------------------------------------------- 判定
def test_detect():
    print("\n[detect]")
    gif = detect.ImageInfo(width=240, height=240, frames=12, has_alpha=True,
                           size_bytes=300_000, fmt="GIF")
    v = detect.score(gif)
    check("动图小方图 -> accept", v.decision == "accept", f"{v.score} {v.reasons}")
    check("动图得分 >= 6", v.score >= detect.DIRECT_ACCEPT, v.score)

    shot = detect.ImageInfo(width=1920, height=1080, frames=1, has_alpha=False,
                            size_bytes=2_500_000, fmt="JPEG")
    v = detect.score(shot)
    check("截图 -> drop", v.decision == "drop", f"{v.score} {v.reasons}")

    mface = detect.ProtocolHints(is_mface=True)
    v = detect.score(detect.ImageInfo(width=100, height=100, size_bytes=50_000), mface)
    check("mface 直接收", v.decision == "accept", v.score)

    v = detect.score(detect.ImageInfo(), detect.ProtocolHints(summary="[动画表情]"))
    check("summary 命中 -> 至少 suspect", v.decision in ("accept", "suspect"), v.score)

    png_mid = detect.ImageInfo(width=400, height=400, frames=1, has_alpha=True,
                               size_bytes=100_000, fmt="PNG")
    v = detect.score(png_mid)
    check("透明方形 PNG -> accept", v.decision == "accept", f"{v.score} {v.reasons}")

    # phash
    a, b = "0f0f0f0f0f0f0f0f", "0f0f0f0f0f0f0f0e"
    check("phash 距离小", detect.hamming(a, b) <= 6, detect.hamming(a, b))
    check("空 phash 不可比", detect.hamming("", a) == 999)


# ---------------------------------------------------------------- 存储
def test_store():
    print("\n[store]")
    tmp = fresh_tmp()
    st = StickerStore(os.path.join(tmp, "t.db"), os.path.join(tmp, "blobs"))
    now = 1_700_000_000.0

    r1 = st.add(sha="aaa", phash="p1", path="x.gif", fmt="GIF", width=240,
                height=240, frames=12, size=1000, now=now)
    check("首次入库 = new", r1 == "new", r1)
    check("入库状态是观察期", st.get("aaa")["state"] == "observing")

    r2 = st.add(sha="aaa", phash="p1", path="x.gif", now=now + 10)
    check("重复发 = dup", r2 == "dup", r2)
    check("重复会强化", st.get("aaa")["seen_count"] == 1, st.get("aaa")["seen_count"])

    # 过了观察期 -> active
    st.tick(now + 8 * 3600)
    check("8 小时后 active", st.get("aaa")["state"] == "active", st.get("aaa")["state"])

    # 不碰 20 天 -> forgotten
    st.tick(now + 20 * 86400)
    check("20 天后 forgotten", st.get("aaa")["state"] == "forgotten", st.get("aaa")["state"])

    # 再发一次 -> 复活
    r3 = st.add(sha="aaa", phash="p1", path="x.gif", now=now + 21 * 86400)
    check("遗忘后再发 = revived", r3 == "revived", r3)
    check("复活后强度 1.5", approx(st.get("aaa")["strength"], decay.REVIVE_STRENGTH),
          st.get("aaa")["strength"])

    # 容量淘汰
    for i in range(10):
        st.add(sha=f"k{i}", phash=f"q{i}", path=f"{i}.png", now=now + 22 * 86400 + i)
    victims = st.enforce_capacity(5)
    live = [r for r in st.all_rows() if r["state"] != "forgotten"]
    check("容量上限生效", len(live) <= 5, len(live))
    check("有被淘汰的", len(victims) > 0, victims)

    # 视觉缓存只问一次
    cache = VlmCache(st)
    check("缓存未命中返回 None", cache.get("zzz") is None)
    cache.put("zzz", True, "摊手的猫", ["无语"])
    hit = cache.get("zzz")
    check("缓存命中", hit and hit["verdict"] == 1, hit)

    s = st.stats()
    check("统计能跑", "by_state" in s and s["total"] > 0, s)
    st.close()


def test_probe_optional():
    print("\n[probe]（依赖 Pillow，没有就跳过）")
    try:
        from PIL import Image
    except Exception:
        print("  skip 没有 Pillow")
        return
    tmp = fresh_tmp()
    p = os.path.join(tmp, "a.png")
    Image.new("RGBA", (120, 120), (0, 0, 0, 0)).save(p)
    info = detect.probe(p)
    check("读到尺寸", (info.width, info.height) == (120, 120), (info.width, info.height))
    check("读到透明通道", info.has_alpha is True)
    ph = detect.first_frame_phash(p)
    check("phash 是十六进制", len(ph) == 16 and all(c in "0123456789abcdef" for c in ph), ph)


def test_regressions():
    """回归测试：两个字段不能被维护任务搅在一起。"""
    print("\n[regressions]")
    tmp = fresh_tmp()
    st = StickerStore(os.path.join(tmp, "r.db"), os.path.join(tmp, "rb"))
    T = 1_700_000_000.0
    st.add(sha="z", phash="pz", path="z.gif", now=T)
    seen_before = st.get("z")["last_seen"]

    st.tick(now=T + 86400)
    st.tick(now=T + 2 * 86400)
    seen_after = st.get("z")["last_seen"]
    check("tick 不会改写 last_seen", seen_after == seen_before,
          f"{seen_before} -> {seen_after}")

    anchor = st.get("z")["anchor_at"]
    check("tick 推进的是 anchor_at", anchor == T + 2 * 86400, anchor)

    # 隔两天后再看到：应当从 anchor 结算一次衰减，再 +0.35（而不是从收藏时间重复衰减）
    row = st.get("z")
    s_after_ticks = row["strength"]
    st.touch_seen("z", now=T + 3 * 86400)
    expected = min(decay.MAX_STRENGTH, s_after_ticks * decay.daily_factor(3.0) + decay.REINFORCE_SEEN)
    actual = st.get("z")["strength"]
    check("强化前先结算衰减", approx(actual, expected, 1e-6), f"{actual:.6f} vs {expected:.6f}")

    # 淘汰必须按"真的最后一次被看到"排
    fresh_tmp()
    st2 = StickerStore(os.path.join(tmp, "e.db"), os.path.join(tmp, "eb"))
    for i, seen in enumerate([T, T + 10, T + 20]):
        st2.add(sha=f"e{i}", phash=f"ep{i}", path=f"e{i}.png", now=seen)
    st2.tick(now=T + 30)   # 推进 anchor，但不该改变三者的 last_seen 差异
    rows = [r for r in st2.all_rows() if r["state"] != "forgotten"]
    order = decay.eviction_order(rows)
    check("淘汰按真实 last_seen 排序", order[0]["sha"] == "e0", [r["sha"] for r in order])
    st2.close()
    st.close()


if __name__ == "__main__":
    test_decay()
    test_detect()
    test_store()
    test_regressions()
    test_probe_optional()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
