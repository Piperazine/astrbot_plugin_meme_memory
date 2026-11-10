"""梗百科 -> 表情记忆 的桥（读端）。

梗百科把命中记录写到 plugin_data/_shared/memewiki_hits.json；
这里只负责读。读不到、过期、格式坏了，一律当"没有"，退回纯语义检索。
"""

from __future__ import annotations

import json
import os
import time

SHARED_DIR = "_shared"
SHARED_FILE = "memewiki_hits.json"


def shared_path(plugin_data_dir: str) -> str:
    return os.path.join(plugin_data_dir, SHARED_DIR, SHARED_FILE)


def read(plugin_data_dir: str, session: str, max_age: int = 120,
         now: float | None = None) -> dict | None:
    """返回 {ts, titles, tags}；不可用返回 None。"""
    if not session:
        return None
    path = shared_path(plugin_data_dir)
    if not os.path.exists(path):
        return None
    now = time.time() if now is None else now
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    hit = (data.get("sessions") or {}).get(session)
    if not isinstance(hit, dict):
        return None
    try:
        age = now - float(hit.get("ts") or 0)
    except Exception:
        return None
    if age < 0 or age > max_age:
        return None

    tags = [str(t) for t in (hit.get("tags") or []) if str(t).strip()]
    titles = [str(t) for t in (hit.get("titles") or []) if str(t).strip()]
    if not tags and not titles:
        return None
    return {"ts": hit.get("ts"), "titles": titles, "tags": tags}


def query_suffix(hit: dict | None, limit: int = 4) -> str:
    """把桥上来的梗词拼成查询后缀，让嵌入更贴近"这个梗的气质"。"""
    if not hit:
        return ""
    parts = list(hit.get("titles") or [])[:1] + list(hit.get("tags") or [])[:limit]
    return " ".join(p for p in parts if p).strip()


def tag_overlap(sticker_tags: dict | list, meme_tags: list[str]) -> float:
    """表情标签与梗标签的重合度 0~1，用于加权。"""
    if not sticker_tags or not meme_tags:
        return 0.0
    if isinstance(sticker_tags, dict):
        keys = set(sticker_tags.keys())
    else:
        keys = set(str(t) for t in sticker_tags)
    wanted = set(meme_tags)
    if not keys or not wanted:
        return 0.0
    return len(keys & wanted) / float(len(wanted))
