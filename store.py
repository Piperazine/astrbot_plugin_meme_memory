"""SQLite 存储层：表情包元数据 + 墓碑 + 视觉判定缓存。

两个时间字段必须分开：
  last_seen  最后一次"真的被人看到"（只由事件更新）
  anchor_at  衰减基准（维护任务推进到这里）

混成一个字段的后果：维护任务每小时都会把 last_seen 推到"现在"，
于是"最久没出现"的排序失效、衰减基准也跟着漂移。

不 import AstrBot，本地可直接跑测试。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time

try:  # 作为包加载（AstrBot 实际方式）时走相对导入
    from .decay import (
        ACTIVE_THRESHOLD,
    DEFAULT_HALF_LIFE_DAYS,
    FORGET_GRACE_DAYS,
    FORGET_THRESHOLD,
    OBSERVE_HOURS,
    REINFORCE_SEEN,
    REINFORCE_USED,
    REVIVE_STRENGTH,
    STATE_ACTIVE,
    STATE_FORGOTTEN,
    STATE_DORMANT,
    STATE_OBSERVING,
    StickerState,
        decay as _decay,
        eviction_order,
        next_state,
        reinforce,
    )
except ImportError:  # 平铺加载（测试/单独运行）时兜底
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from decay import (
        ACTIVE_THRESHOLD,
        DEFAULT_HALF_LIFE_DAYS,
        FORGET_GRACE_DAYS,
        FORGET_THRESHOLD,
        OBSERVE_HOURS,
        REINFORCE_SEEN,
        REINFORCE_USED,
        REVIVE_STRENGTH,
        STATE_ACTIVE,
        STATE_FORGOTTEN,
        STATE_DORMANT,
        STATE_OBSERVING,
        StickerState,
        decay as _decay,
        eviction_order,
        next_state,
        reinforce,
    )

SCHEMA = """
CREATE TABLE IF NOT EXISTS stickers (
    sha          TEXT PRIMARY KEY,
    phash        TEXT,
    path         TEXT NOT NULL,
    fmt          TEXT,
    w            INTEGER DEFAULT 0,
    h            INTEGER DEFAULT 0,
    frames       INTEGER DEFAULT 1,
    size         INTEGER DEFAULT 0,
    desc         TEXT DEFAULT '',
    tags         TEXT DEFAULT '[]',
    ocr_text     TEXT DEFAULT '',
    ctx_tags     TEXT DEFAULT '{}',
    confidence   REAL DEFAULT 0,
    analyzed     INTEGER DEFAULT 0,
    vec          BLOB,
    strength     REAL DEFAULT 1.0,
    state        TEXT DEFAULT 'observing',
    created_at   REAL,
    last_seen    REAL,
    anchor_at    REAL,
    last_used    REAL DEFAULT 0,
    seen_count   INTEGER DEFAULT 0,
    used_count   INTEGER DEFAULT 0,
    below_since  REAL,
    sender       TEXT DEFAULT '',
    session      TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_state ON stickers(state);
CREATE INDEX IF NOT EXISTS idx_phash ON stickers(phash);

CREATE TABLE IF NOT EXISTS vlm_cache (
    phash    TEXT PRIMARY KEY,
    verdict  INTEGER NOT NULL,
    desc     TEXT DEFAULT '',
    tags     TEXT DEFAULT '[]',
    ts       REAL
);
"""


class StickerStore:
    def __init__(self, db_path: str, blob_dir: str):
        self.db_path = db_path
        self.blob_dir = blob_dir
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(blob_dir, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(stickers)")]
        if "anchor_at" not in cols:
            self.conn.execute("ALTER TABLE stickers ADD COLUMN anchor_at REAL")
            self.conn.execute("UPDATE stickers SET anchor_at = last_seen WHERE anchor_at IS NULL")
        for name, ddl in (
            ("ocr_text", "ALTER TABLE stickers ADD COLUMN ocr_text TEXT DEFAULT ''"),
            ("ctx_tags", "ALTER TABLE stickers ADD COLUMN ctx_tags TEXT DEFAULT '{}'"),
            ("confidence", "ALTER TABLE stickers ADD COLUMN confidence REAL DEFAULT 0"),
            ("analyzed", "ALTER TABLE stickers ADD COLUMN analyzed INTEGER DEFAULT 0"),
            ("vec", "ALTER TABLE stickers ADD COLUMN vec BLOB"),
        ):
            if name not in cols:
                self.conn.execute(ddl)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ---------- 读 ----------
    def _row(self, sha: str):
        cur = self.conn.execute("SELECT * FROM stickers WHERE sha=?", (sha,))
        return cur.fetchone()

    def get(self, sha: str):
        r = self._row(sha)
        return dict(r) if r else None

    def find_by_phash(self, phash: str, max_distance_fn=None, max_distance: int = 6):
        if not phash:
            return None
        if max_distance_fn is None:
            r = self.conn.execute("SELECT * FROM stickers WHERE phash=?", (phash,)).fetchone()
            return dict(r) if r else None
        for r in self.conn.execute("SELECT * FROM stickers WHERE phash != ''").fetchall():
            if max_distance_fn(phash, r["phash"]) <= max_distance:
                return dict(r)
        return None

    # ---------- 写 ----------
    def add(
        self,
        *,
        sha: str,
        phash: str,
        path: str,
        fmt: str = "",
        width: int = 0,
        height: int = 0,
        frames: int = 1,
        size: int = 0,
        sender: str = "",
        session: str = "",
        now: float | None = None,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> str:
        """返回 'new' / 'revived' / 'dup'。"""
        now = time.time() if now is None else now
        row = self._row(sha)

        if row and row["state"] != STATE_FORGOTTEN:
            self.touch_seen(sha, now=now, half_life_days=half_life_days)
            return "dup"

        if row and row["state"] == STATE_FORGOTTEN:
            self.conn.execute(
                "UPDATE stickers SET state=?, strength=?, last_seen=?, anchor_at=?, "
                "below_since=NULL, path=?, phash=? WHERE sha=?",
                (STATE_DORMANT, REVIVE_STRENGTH, now, now, path, phash, sha),
            )
            self.conn.commit()
            return "revived"

        self.conn.execute(
            """INSERT INTO stickers
               (sha, phash, path, fmt, w, h, frames, size, strength, state,
                created_at, last_seen, anchor_at, sender, session)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (sha, phash, path, fmt, width, height, frames, size,
             1.0, STATE_OBSERVING, now, now, now, sender, session),
        )
        self.conn.commit()
        return "new"

    def set_analysis(self, sha: str, desc: str, tags: list[str] | None = None,
                     ocr: str = "", ctx_tags: dict | None = None,
                     confidence: float = 0.0, merge: bool = True):
        """写入分析结果。ctx_tags 是带计数的 dict，merge=True 时累加。"""
        old = self._row(sha)
        merged = dict(ctx_tags or {})
        if merge and old is not None:
            try:
                prev = json.loads(old["ctx_tags"] or "{}")
                if isinstance(prev, dict):
                    for k, v in prev.items():
                        merged[k] = int(merged.get(k, 0)) + int(v)
            except Exception:
                pass
        self.conn.execute(
            "UPDATE stickers SET desc=?, tags=?, ocr_text=?, ctx_tags=?, confidence=?, analyzed=1 WHERE sha=?",
            (
                desc or "",
                json.dumps(tags or [], ensure_ascii=False),
                ocr or "",
                json.dumps(merged, ensure_ascii=False),
                float(confidence or 0.0),
                sha,
            ),
        )
        self.conn.commit()

    def set_vec(self, sha: str, blob: bytes):
        self.conn.execute("UPDATE stickers SET vec=? WHERE sha=?", (blob, sha))
        self.conn.commit()

    def rows_for_retrieval(self) -> list[dict]:
        """参与检索的候选：活跃 + 沉睡（沉睡也能被搜到，只是不那么优先）。"""
        cur = self.conn.execute(
            "SELECT * FROM stickers WHERE state IN (?, ?) AND vec IS NOT NULL",
            (STATE_ACTIVE, STATE_DORMANT),
        )
        return [dict(r) for r in cur.fetchall()]

    def rows_missing_vec(self) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM stickers WHERE vec IS NULL AND state != ?", (STATE_FORGOTTEN,)
        )
        return [dict(r) for r in cur.fetchall()]

    def top_ctx_tags(self, sha: str, limit: int = 5) -> list[str]:
        row = self._row(sha)
        if not row:
            return []
        try:
            d = json.loads(row["ctx_tags"] or "{}")
        except Exception:
            return []
        if not isinstance(d, dict):
            return []
        return [k for k, _ in sorted(d.items(), key=lambda kv: -int(kv[1]))[:limit]]

    def _strength_at(self, row, now: float, half_life_days: float) -> float:
        """把强度结算到 now：基准是 anchor_at。"""
        anchor = row["anchor_at"] or row["last_seen"]
        elapsed = max(0.0, (now - anchor) / 86400.0)
        return _decay(row["strength"], elapsed, half_life_days)

    def touch_seen(self, sha: str, now: float | None = None,
                   half_life_days: float = DEFAULT_HALF_LIFE_DAYS):
        """再次看到：先结算衰减，再强化，最后把基准推到 now。"""
        now = time.time() if now is None else now
        row = self._row(sha)
        if not row:
            return
        base = self._strength_at(row, now, half_life_days)
        self.conn.execute(
            "UPDATE stickers SET last_seen=?, anchor_at=?, seen_count=seen_count+1, "
            "strength=?, below_since=NULL WHERE sha=?",
            (now, now, reinforce(base, REINFORCE_SEEN), sha),
        )
        self.conn.commit()

    def touch_used(self, sha: str, now: float | None = None,
                   half_life_days: float = DEFAULT_HALF_LIFE_DAYS):
        now = time.time() if now is None else now
        row = self._row(sha)
        if not row:
            return
        base = self._strength_at(row, now, half_life_days)
        self.conn.execute(
            "UPDATE stickers SET last_used=?, anchor_at=?, used_count=used_count+1, "
            "strength=? WHERE sha=?",
            (now, now, reinforce(base, REINFORCE_USED), sha),
        )
        self.conn.commit()

    # ---------- 时间推进 ----------
    def tick(
        self,
        now: float | None = None,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        active_threshold: float = ACTIVE_THRESHOLD,
        forget_threshold: float = FORGET_THRESHOLD,
        forget_grace_days: float = FORGET_GRACE_DAYS,
        observe_hours: float = OBSERVE_HOURS,
    ) -> dict:
        now = time.time() if now is None else now
        changed = {"decayed": 0, "forgotten": 0, "revived_to_active": 0}

        rows = self.conn.execute(
            "SELECT * FROM stickers WHERE state != ?", (STATE_FORGOTTEN,)
        ).fetchall()

        for r in rows:
            st = StickerState(
                strength=r["strength"],
                state=r["state"],
                created_at=r["created_at"],
                last_seen=r["last_seen"],
                anchor=r["anchor_at"] or r["last_seen"],
                below_since=r["below_since"],
            )
            new = next_state(
                st, now,
                half_life_days=half_life_days,
                active_threshold=active_threshold,
                forget_threshold=forget_threshold,
                forget_grace_days=forget_grace_days,
                observe_hours=observe_hours,
            )
            if abs(new.strength - r["strength"]) > 1e-9 or new.state != r["state"]:
                changed["decayed"] += 1
            if new.state == STATE_FORGOTTEN and r["state"] != STATE_FORGOTTEN:
                changed["forgotten"] += 1
            if new.state == STATE_ACTIVE and r["state"] != STATE_ACTIVE:
                changed["revived_to_active"] += 1
            # 注意：只推进 anchor_at，不动 last_seen
            self.conn.execute(
                "UPDATE stickers SET strength=?, state=?, anchor_at=?, below_since=? WHERE sha=?",
                (new.strength, new.state, new.anchor, new.below_since, r["sha"]),
            )

        self.conn.commit()
        return changed

    # ---------- 检索 ----------
    def candidates(self, *, min_strength: float = ACTIVE_THRESHOLD, limit: int = 50) -> list[dict]:
        cur = self.conn.execute(
            "SELECT * FROM stickers WHERE state=? AND strength>=? ORDER BY strength DESC LIMIT ?",
            (STATE_ACTIVE, min_strength, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def all_rows(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM stickers").fetchall()]

    # ---------- 容量 ----------
    def enforce_capacity(self, max_n: int) -> list[str]:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM stickers WHERE state != ?", (STATE_FORGOTTEN,)
        ).fetchone()[0]
        overflow = total - max_n
        if overflow <= 0:
            return []

        rows = [dict(r) for r in self.conn.execute(
            "SELECT sha, strength, last_seen FROM stickers WHERE state != ?", (STATE_FORGOTTEN,)
        ).fetchall()]
        victims = eviction_order(rows)[:overflow]

        ids = []
        for v in victims:
            self.conn.execute(
                "UPDATE stickers SET state=?, strength=0, below_since=? WHERE sha=?",
                (STATE_FORGOTTEN, time.time(), v["sha"]),
            )
            ids.append(v["sha"])
        self.conn.commit()
        return ids

    # ---------- 清理 ----------
    def purge_files(self, retention_days: int = 30, now: float | None = None) -> int:
        now = time.time() if now is None else now
        cutoff = now - retention_days * 86400
        rows = self.conn.execute(
            "SELECT sha, path FROM stickers WHERE state=? AND below_since IS NOT NULL AND below_since < ?",
            (STATE_FORGOTTEN, cutoff),
        ).fetchall()
        removed = 0
        for r in rows:
            try:
                if r["path"] and os.path.exists(r["path"]):
                    os.remove(r["path"])
                    removed += 1
            except Exception:
                pass
        return removed

    # ---------- 统计 ----------
    def stats(self) -> dict:
        out = {"total": 0, "by_state": {}, "avg_strength": 0.0, "total_used": 0}
        rows = self.conn.execute(
            "SELECT state, COUNT(*) c, SUM(used_count) u FROM stickers GROUP BY state"
        ).fetchall()
        for r in rows:
            out["by_state"][r["state"]] = r["c"]
            out["total"] += r["c"]
            out["total_used"] += r["u"] or 0
        if out["total"]:
            avg = self.conn.execute("SELECT AVG(strength) a FROM stickers").fetchone()
            out["avg_strength"] = round(avg["a"] or 0.0, 3)
        return out


class VlmCache:
    def __init__(self, store: StickerStore):
        self.s = store

    def get(self, phash: str):
        if not phash:
            return None
        r = self.s.conn.execute("SELECT * FROM vlm_cache WHERE phash=?", (phash,)).fetchone()
        return dict(r) if r else None

    def put(self, phash: str, verdict: bool, desc: str = "", tags: list[str] | None = None):
        if not phash:
            return
        self.s.conn.execute(
            "INSERT OR REPLACE INTO vlm_cache (phash, verdict, desc, tags, ts) VALUES (?,?,?,?,?)",
            (phash, 1 if verdict else 0, desc, json.dumps(tags or [], ensure_ascii=False), time.time()),
        )
        self.s.conn.commit()
