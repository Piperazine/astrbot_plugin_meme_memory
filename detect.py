"""判断"这条图片是不是表情包"。

分三级：
  L1 协议线索   —— OneBot 原始事件里的 sub_type / summary / mface
  L2 图片特征   —— 尺寸、比例、透明通道、动图、文件大小（零成本，一定可用）
  L3 视觉模型   —— 只对 L1/L2 拿不准的图问一次（唯一花钱的地方）

score() 是纯函数，不读文件，方便单测。probe() 才碰 Pillow。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 分数阈值
DIRECT_ACCEPT = 6      # >= 6 直接收
SUSPECT_MIN = 3        # 3..5 送视觉模型
# < 3 丢弃

# 单个信号加分
SCORE_SUBTYPE = 4
SCORE_SUMMARY = 4
SCORE_MFACE = 10
SCORE_ALPHA = 3
SCORE_ANIMATED = 3
SCORE_SQUARE = 2
SCORE_SMALL_FILE = 1
PENALTY_SCREENSHOT = -5
PENALTY_HUGE_FILE = -3


@dataclass
class ProtocolHints:
    """来自 OneBot 原始事件的线索，拿不到就全空。"""

    sub_type: int | None = None       # image 段的 sub_type，1 常见为表情
    summary: str = ""                 # 例如 "[动画表情]"
    is_mface: bool = False            # 商城/收藏表情段


@dataclass
class ImageInfo:
    """图片本身的客观特征。"""

    width: int = 0
    height: int = 0
    frames: int = 1                   # >1 表示动图
    has_alpha: bool = False
    size_bytes: int = 0
    fmt: str = ""                     # PNG / GIF / WEBP / JPEG


@dataclass
class Verdict:
    score: int = 0
    reasons: list[str] = field(default_factory=list)
    decision: str = "drop"            # accept | suspect | drop

    @property
    def needs_model(self) -> bool:
        return self.decision == "suspect"


def score(info: ImageInfo, hints: ProtocolHints | None = None) -> Verdict:
    """纯打分。同样的输入永远得到同样的结果，方便写测试。"""
    hints = hints or ProtocolHints()
    s = 0
    reasons: list[str] = []

    # ---- L1：协议线索 ----
    if hints.is_mface:
        s += SCORE_MFACE
        reasons.append(f"+{SCORE_MFACE} mface 商城表情")
    if hints.sub_type == 1:
        s += SCORE_SUBTYPE
        reasons.append(f"+{SCORE_SUBTYPE} sub_type=1")
    if any(k in hints.summary for k in ("表情", "动画表情", "贴图")):
        s += SCORE_SUMMARY
        reasons.append(f"+{SCORE_SUMMARY} summary={hints.summary!r}")

    # ---- L2：图片特征 ----
    if info.has_alpha:
        s += SCORE_ALPHA
        reasons.append(f"+{SCORE_ALPHA} 有透明通道")
    if info.frames > 1:
        s += SCORE_ANIMATED
        reasons.append(f"+{SCORE_ANIMATED} 动图({info.frames}帧)")

    if info.width and info.height:
        ratio = info.width / info.height
        long_side = max(info.width, info.height)
        if 0.7 <= ratio <= 1.4 and long_side <= 600:
            s += SCORE_SQUARE
            reasons.append(f"+{SCORE_SQUARE} 接近方形且不大({info.width}x{info.height})")
        # 截图/照片特征：宽、长边大
        if ratio >= 1.6 and long_side >= 1200:
            s += PENALTY_SCREENSHOT
            reasons.append(f"{PENALTY_SCREENSHOT} 像截图/照片({info.width}x{info.height})")

    limit = 5 * 1024 * 1024 if info.frames > 1 else 2 * 1024 * 1024
    if info.size_bytes <= limit:
        s += SCORE_SMALL_FILE
        reasons.append(f"+{SCORE_SMALL_FILE} 体积小({info.size_bytes // 1024}KB)")
    elif info.size_bytes > 8 * 1024 * 1024:
        s += PENALTY_HUGE_FILE
        reasons.append(f"{PENALTY_HUGE_FILE} 体积过大")

    if s >= DIRECT_ACCEPT:
        decision = "accept"
    elif s >= SUSPECT_MIN:
        decision = "suspect"
    else:
        decision = "drop"

    return Verdict(score=s, reasons=reasons, decision=decision)


def probe(path: str) -> ImageInfo:
    """读图片客观特征。Pillow 缺失时退回只用文件大小。"""
    import os

    size = os.path.getsize(path)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    info = ImageInfo(size_bytes=size, fmt=ext.upper())
    try:
        from PIL import Image  # 局部导入：没有 Pillow 也不崩
    except Exception:
        return info

    try:
        with Image.open(path) as im:
            info.width, info.height = im.size
            info.fmt = (im.format or info.fmt).upper()
            info.has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
            frames = getattr(im, "n_frames", 1)
            try:
                info.frames = int(frames) if frames else 1
            except Exception:
                info.frames = 1
    except Exception:
        pass
    return info


def first_frame_phash(path: str, size: int = 8) -> str:
    """取首帧做感知哈希（动图用首帧，避免重编码导致字节不同）。

    返回 16 位十六进制字符串；Pillow 不可用时返回空串。
    """
    try:
        from PIL import Image
    except Exception:
        return ""

    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            px = list(im.getdata())
    except Exception:
        return ""

    if not px:
        return ""
    avg = sum(px) / len(px)
    bits = "".join("1" if p >= avg else "0" for p in px)
    return f"{int(bits, 2):0{(size * size) // 4}x}"


def hamming(a: str, b: str) -> int:
    """两个 phash 的汉明距离；任一为空时返回一个大数表示不可比。"""
    if not a or not b or len(a) != len(b):
        return 999
    return bin(int(a, 16) ^ int(b, 16)).count("1")
