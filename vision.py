"""让视觉模型"看懂"这张图：画面描述 + 图上文字 + 语境标签。

动图只给首帧是看不清的（猫在动 / 猫在躺 分不出来），所以 GIF 抽 3 帧
拼成一张横图再送模型。
"""

from __future__ import annotations

import json
import os
import tempfile

PROMPT = """这是聊天里的一张表情包/图片，请结合给出的语境判断它在表达什么。

{context_block}
请只输出一个 JSON 对象，不要任何解释：
{{"is_sticker": true, "desc": "画面里是什么，一句话", "ocr": "图上的文字，没有就空字符串", "ctx_tags": ["3到5个标签，描述这张图在此语境下表达的情绪或意图，如 嘲讽/无奈/自嘲/认同/敷衍/安慰/惊喜"], "confidence": 0.0}}

要求：
- is_sticker：这张图是不是聊天用的表情包/梗图（不是照片、截图、文档、UI）
- desc 只描述画面，不要推测用途
- ctx_tags 结合语境给出，如果语境为空就按画面常见的用法给
- confidence 用 0~1 表示你对 ctx_tags 的把握
"""


def _pil():
    from PIL import Image  # 局部导入：Pillow 缺失时上层会降级
    return Image


def frames_for(path: str, max_frames: int = 3) -> tuple[str, bool]:
    """返回 (送给模型的图片路径, 是否做了拼图)。

    动图：等距抽 max_frames 帧，横向拼成一张；静态图原样返回。
    """
    try:
        Image = _pil()
    except Exception:
        return path, False

    try:
        with Image.open(path) as im:
            frames = getattr(im, "n_frames", 1) or 1
            if frames <= 1:
                return path, False

            n = max(1, min(max_frames, frames))
            idx = [round(i * (frames - 1) / max(1, n - 1)) for i in range(n)] if n > 1 else [0]
            tiles = []
            for i in idx:
                im.seek(int(i))
                tiles.append(im.convert("RGB").copy())

            w = sum(t.width for t in tiles)
            h = max(t.height for t in tiles)
            canvas = Image.new("RGB", (w, h), (0, 0, 0))
            x = 0
            for t in tiles:
                canvas.paste(t, (x, 0))
                x += t.width

            out = os.path.join(tempfile.gettempdir(), f"sticker_frames_{abs(hash(path)) % (10**8)}.png")
            canvas.save(out, format="PNG")
            return out, True
    except Exception:
        return path, False


COMPOSITE_NOTE = """注意：这张图是**同一段动图**的 3 个瞬间按时间顺序横向拼接而成，用来告诉你它在"怎么动"。
请把它理解为一段连续的动画，不要把画面描述成"三格拼图/三联画/多个分镜"。"""


def build_prompt(context_block: str, composite: bool = False) -> str:
    hook = (COMPOSITE_NOTE + "\n\n") if composite else ""
    return PROMPT.format(
        context_block=hook + (context_block.strip() or "（没有可用语境）")
    )


def parse_result(text: str) -> dict:
    """宽容解析：从模型输出里抠出 JSON，字段缺失用默认值补齐。"""
    out = {"is_sticker": True, "desc": "", "ocr": "", "ctx_tags": [], "confidence": 0.0}
    if not text:
        return out
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return out
    try:
        raw = json.loads(text[start:end + 1])
    except Exception:
        return out
    if not isinstance(raw, dict):
        return out

    out["is_sticker"] = bool(raw.get("is_sticker", True))
    out["desc"] = str(raw.get("desc") or "").strip()[:300]
    out["ocr"] = str(raw.get("ocr") or "").strip()[:300]
    tags = raw.get("ctx_tags") or []
    if isinstance(tags, str):
        tags = [tags]
    out["ctx_tags"] = [str(t).strip()[:20] for t in tags if str(t).strip()][:5]
    try:
        out["confidence"] = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
    except Exception:
        out["confidence"] = 0.0
    return out


def merge_tags(old: dict, new: list[str]) -> dict:
    """标签带计数累加，这样同一张图在不同语境出现多次时会慢慢偏重。"""
    merged = dict(old or {})
    for t in new or []:
        merged[t] = int(merged.get(t, 0)) + 1
    return merged
