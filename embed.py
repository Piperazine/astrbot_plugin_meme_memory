"""向量工具：打包、余弦相似度、检索打分。

纯粹的数学部分，不碰 AstrBot、不碰网络，方便本地测试。
"""

from __future__ import annotations

import array
import math


def pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def unpack(blob: bytes | None) -> list[float]:
    if not blob:
        return []
    a = array.array("f")
    try:
        a.frombytes(blob)
    except Exception:
        return []
    return list(a)


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度；长度不一致或含 0 向量时返回 0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def sticker_text(desc: str, tags: list[str], ocr: str = "") -> str:
    """把一张图的语义拼成一段用来嵌入的文本。

    描述为主，标签补充，OCR 放最后 —— 顺序影响不大，但保持稳定便于排查。
    """
    parts = []
    if desc:
        parts.append(desc.strip())
    if tags:
        parts.append("、".join(t for t in tags if t))
    if ocr:
        parts.append(f"图上写着：{ocr.strip()}")
    return "。".join(p for p in parts if p).strip()


def rank(query_vec: list[float], rows: list[dict], *,
         top_k: int = 5, min_similarity: float = 0.35,
         strength_weight: float = 0.3, max_strength: float = 2.5) -> list[dict]:
    """给候选图打分并排序。

    rows 里每项需要：vec（bytes）、strength。
    返回的每项附加 sim 与 score，且已按 score 降序、截断 top_k。
    相似度低于 min_similarity 的直接丢掉（宁可不发，也不发错的）。
    """
    scored = []
    for r in rows:
        vec = unpack(r.get("vec"))
        if not vec:
            continue
        sim = cosine(query_vec, vec)
        if sim < min_similarity:
            continue
        strength = float(r.get("strength") or 0.0)
        norm = max(0.0, min(1.0, strength / max_strength)) if max_strength else 0.0
        score = sim * (1.0 - strength_weight) + norm * strength_weight
        item = dict(r)
        item["sim"] = round(sim, 4)
        item["score"] = round(score, 4)
        scored.append(item)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
