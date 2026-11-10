"""表情记忆插件：收集、识别、遗忘、按语境配图。

设计要点：
  - 只收"判定为表情包"的图（L1 协议线索 / L2 图片特征 / L3 视觉模型）
  - 每张图有 strength，按半衰期自然衰减；久不用的遗忘
  - 遗忘不是删文件：先进墓碑，30 天后才物理删
  - 主动配图受强度、冷却、每日额度三重限制

本文件只做 AstrBot 胶水；判定/衰减/存储分别在 detect.py / decay.py / store.py。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register

try:
    from . import bridge, decay, detect, embed, vision
    from .context import ContextBuffer, render_for_prompt
    from .store import StickerStore, VlmCache
except ImportError:  # 平铺加载时
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import bridge
    import decay
    import detect
    import embed
    import vision
    from context import ContextBuffer, render_for_prompt
    from store import StickerStore, VlmCache

PLUGIN_NAME = "astrbot_plugin_meme_memory"

DEFAULTS = {
    "enabled": True,
    "collect_from_all": True,
    "vlm_enabled": True,
    "vlm_provider_id": "",
    "half_life_days": 3.0,
    "max_stickers": 100,
    "observe_hours": 6.0,
    "active_threshold": 0.35,
    "forget_threshold": 0.08,
    "reinforce_seen": 0.35,
    "reinforce_used": 0.25,
    "proactive_enabled": True,
    "proactive_probability_base": 0.25,
    "proactive_daily_limit": 5,
    "same_sticker_cooldown_hours": 24,
    "session_cooldown_minutes": 10,
    "daily_collect_limit": 120,
    "purge_retention_days": 30,
    "analysis_enabled": True,
    "analysis_daily_limit": 60,
    "analysis_max_frames": 3,
    "context_window_seconds": 180,
    "context_before": 2,
    "reanalyze_on_seen": True,
    "retrieval_enabled": True,
    "retrieval_min_similarity": 0.35,
    "retrieval_top_k": 5,
    "strength_weight": 0.3,
    "embedding_provider_id": "siliconflow/bge-m3",
    "bridge_enabled": True,
    "bridge_max_age_seconds": 120,
    "meme_tag_boost": 0.15,
}


@register(PLUGIN_NAME, "Piperazine", "给表情包加上记忆：会收藏、会淡忘、会按语境配图", "1.0.0")
class StickerMemory(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.context = context
        self.cfg = {**DEFAULTS, **(config or {})}
        self.data_dir = self._resolve_data_dir()
        self.blob_dir = os.path.join(self.data_dir, "stickers")
        os.makedirs(self.blob_dir, exist_ok=True)

        self.store = StickerStore(os.path.join(self.data_dir, "stickers.db"), self.blob_dir)
        self.vlm = VlmCache(self.store)
        here = Path(__file__).resolve()
        self.shared_root = str(here.parents[2] / "plugin_data") if len(here.parents) > 2 \
            else os.path.dirname(self.data_dir)
        self.ctxbuf = ContextBuffer(
            per_session=20,
            window_seconds=int(self.cfg.get("context_window_seconds", 180)),
        )
        self._analysis_calls = {"date": "", "n": 0}

        self._task: asyncio.Task | None = None
        self._daily = {"date": "", "collected": 0, "sent": 0}
        self._last_sent_in_session: dict[str, float] = {}
        self._recent_sent: dict[str, float] = {}

    # ------------------------------------------------------------------ 初始化
    def _resolve_data_dir(self) -> str:
        # 从插件目录反推 <instance>/data/plugin_data/sticker_memory
        here = Path(__file__).resolve()
        for anchor, name in ((here.parents[2], "plugin_data"), (here.parents[1], "plugin_data")):
            cand = anchor / name / "sticker_memory"
            try:
                os.makedirs(cand, exist_ok=True)
                return str(cand)
            except Exception:
                continue
        p = here.parent / "data"
        os.makedirs(p, exist_ok=True)
        return str(p)

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _roll_daily(self):
        d = self._today()
        if self._daily["date"] != d:
            self._daily = {"date": d, "collected": 0, "sent": 0}

    # ------------------------------------------------------------------ 消息收集
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        if not self.cfg.get("enabled", True):
            return
        try:
            await self._collect(event)
        except Exception as e:
            logger.debug(f"[sticker] 收集失败: {e}")

    async def _collect(self, event: AstrMessageEvent):
        self._roll_daily()
        if self._daily["collected"] >= int(self.cfg["daily_collect_limit"]):
            return

        sender = str(event.get_sender_id() or "")
        session = str(event.unified_msg_origin or "")
        now = time.time()
        ctx = self.ctxbuf.gather(
            session, sender, now,
            quoted=self._quoted_text(event),
            before=int(self.cfg.get("context_before", 2)),
        )
        self.ctxbuf.record(session, sender, self._message_text(event), now,
                           is_bot=(sender == str(getattr(self, "_self_id", "") or "")))

        hints_by_index = self._protocol_hints(event)
        components = list(event.get_messages() or [])
        img_idx = 0
        for comp in components:
            if not isinstance(comp, Image):
                continue
            hints = hints_by_index.get(img_idx)
            img_idx += 1
            if hints is not None and hints.is_mface:
                pass  # mface 视作表情，继续走流程
            try:
                await self._ingest_image(event, comp, hints, ctx)
            except Exception as e:
                logger.debug(f"[sticker] 入库失败: {e}")

    def _protocol_hints(self, event: AstrMessageEvent) -> dict[int, detect.ProtocolHints]:
        """从 OneBot 原始事件里尽量捞 sub_type / summary。捞不到就返回空。"""
        out: dict[int, detect.ProtocolHints] = {}
        raw = getattr(getattr(event, "message_obj", None), "raw_message", None)
        if not isinstance(raw, dict):
            return out
        msg = raw.get("message")
        if not isinstance(msg, list):
            return out
        idx = 0
        for seg in msg:
            if not isinstance(seg, dict):
                continue
            t = seg.get("type")
            if t == "image":
                data = seg.get("data") or {}
                st = data.get("sub_type")
                try:
                    st = int(st) if st is not None else None
                except Exception:
                    st = None
                out[idx] = detect.ProtocolHints(
                    sub_type=st,
                    summary=str(data.get("summary") or ""),
                )
                idx += 1
            elif t == "mface":
                out[idx] = detect.ProtocolHints(is_mface=True)
                idx += 1
        return out

    async def _ingest_image(self, event: AstrMessageEvent, comp: Image, hints, ctx: dict | None = None):
        # 先把图落到本地（NapCat 的 URL 有时效，必须立刻取）
        try:
            path = await comp.convert_to_file_path()
        except Exception as e:
            logger.debug(f"[sticker] 取图失败: {e}")
            return
        if not path or not os.path.exists(path):
            return

        info = detect.probe(path)
        verdict = detect.score(info, hints)
        ctx_block = render_for_prompt(ctx or {})

        inline: dict = {}
        if verdict.needs_model:
            if not self.cfg.get("analysis_enabled", True) and not self.cfg.get("vlm_enabled", True):
                return
            ok, inline = await self._analyze(path, ctx_block, need_verdict=True)
            if not ok:
                return
        elif verdict.decision != "accept":
            return

        sha = self._sha256(path)
        phash = detect.first_frame_phash(path)

        # 同一张图（内容哈希或近似哈希）不重复入库
        if self.store.get(sha):
            pass
        else:
            near = self.store.find_by_phash(phash, max_distance_fn=detect.hamming)
            if near is not None:
                self.store.touch_seen(near["sha"], half_life_days=float(self.cfg["half_life_days"]))
                if self.cfg.get("reanalyze_on_seen", True):
                    self._schedule_analysis(near["sha"], near["path"], ctx_block)
                return

        saved = self._save_blob(path, sha, info.fmt)
        status = self.store.add(
            sha=sha, phash=phash, path=saved, fmt=info.fmt,
            width=info.width, height=info.height, frames=info.frames,
            size=info.size_bytes,
            sender=str(event.get_sender_id() or ""),
            session=str(event.unified_msg_origin or ""),
            half_life_days=float(self.cfg["half_life_days"]),
        )
        if inline:
            await self._apply_analysis(sha, inline)
        elif self.cfg.get("analysis_enabled", True):
            self._schedule_analysis(sha, saved, ctx_block)
        if status == "new":
            self._daily["collected"] += 1
            logger.info(f"[sticker] 新收藏 {sha[:8]} ({info.fmt} {info.width}x{info.height}) {verdict.reasons}")


    def _save_blob(self, src: str, sha: str, fmt: str) -> str:
        ext = (fmt or "bin").lower()
        dst = os.path.join(self.blob_dir, f"{sha}.{ext}")
        if not os.path.exists(dst):
            try:
                with open(src, "rb") as f, open(dst, "wb") as o:
                    o.write(f.read())
            except Exception:
                return src
        return dst

    @staticmethod
    def _sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------ 语义分析
    def _provider(self):
        """取视觉模型 Provider：优先配置指定，其次当前默认。"""
        try:
            pid = str(self.cfg.get("vlm_provider_id") or "").strip()
            if pid and hasattr(self.context, "get_provider_by_id"):
                p = self.context.get_provider_by_id(pid)
                if p is not None:
                    return p
            if hasattr(self.context, "get_using_provider"):
                return self.context.get_using_provider()
        except Exception as e:
            logger.debug(f"[sticker] 取 Provider 失败: {e}")
        return None

    def _embedding_provider(self):
        """取嵌入 Provider（bge-m3）。取不到就返回 None，检索自动降级。"""
        try:
            pid = str(self.cfg.get("embedding_provider_id") or "").strip()
            if pid and hasattr(self.context, "get_provider_by_id"):
                p = self.context.get_provider_by_id(pid)
                if p is not None:
                    return p
        except Exception as e:
            logger.debug(f"[sticker] 取嵌入 Provider 失败: {e}")
        return None

    async def _embed(self, text: str):
        if not text or not str(text).strip():
            return None
        provider = self._embedding_provider()
        if provider is None:
            return None
        try:
            vec = await provider.get_embedding(str(text)[:1000])
            return list(vec) if vec else None
        except Exception as e:
            logger.debug(f"[sticker] 嵌入失败: {e}")
            return None

    async def _backfill_vectors(self):
        """给还没向量的老图补向量（免费，不吃视觉额度）。"""
        await asyncio.sleep(25)
        try:
            rows = self.store.rows_missing_vec()
            if not rows:
                return
            done = 0
            for r in rows:
                tags = self.store.top_ctx_tags(r["sha"])
                text = embed.sticker_text(r.get("desc", ""), tags, r.get("ocr_text", ""))
                if not text:
                    continue
                vec = await self._embed(text)
                if vec:
                    self.store.set_vec(r["sha"], embed.pack(vec))
                    done += 1
            if done:
                logger.info(f"[sticker] 补向量 {done} 张")
        except Exception as e:
            logger.debug(f"[sticker] 补向量失败: {e}")

    def _roll_analysis(self) -> bool:
        """每日分析调用上限，防止刷爆视觉模型额度。"""
        d = self._today()
        if self._analysis_calls["date"] != d:
            self._analysis_calls = {"date": d, "n": 0}
        limit = int(self.cfg.get("analysis_daily_limit", 60))
        if self._analysis_calls["n"] >= limit:
            return False
        self._analysis_calls["n"] += 1
        return True

    async def _analyze(self, path: str, ctx_block: str, need_verdict: bool = False):
        """一次调用同时拿到：是不是表情包 / 画面描述 / 图上文字 / 语境标签。"""
        if not self._roll_analysis():
            return (not need_verdict), {}
        provider = self._provider()
        if provider is None:
            logger.debug("[sticker] 没有可用 Provider，跳过语义分析")
            return (not need_verdict), {}

        frames_path, composite = vision.frames_for(path, int(self.cfg.get("analysis_max_frames", 3)))
        try:
            if ctx_block:
                logger.debug(f"[sticker] 语境: {ctx_block[:200]}")
            resp = await provider.text_chat(
                prompt=vision.build_prompt(ctx_block, composite=composite),
                image_urls=[str(frames_path)],
            )
            text = getattr(resp, "completion_text", "") or str(resp)
            data = vision.parse_result(text)
            if need_verdict and not data.get("is_sticker", True):
                return False, data
            return True, data
        except Exception as e:
            logger.debug(f"[sticker] 语义分析失败: {e}")
            return (not need_verdict), {}
        finally:
            if composite:
                try:
                    os.remove(frames_path)
                except Exception:
                    pass

    async def _apply_analysis(self, sha: str, data: dict):
        tags = list(data.get("ctx_tags") or [])
        self.store.set_analysis(
            sha,
            desc=data.get("desc", ""),
            tags=tags,
            ocr=data.get("ocr", ""),
            ctx_tags={t: 1 for t in tags},
            confidence=float(data.get("confidence", 0) or 0),
        )
        # 立刻把语义写成向量，供检索用
        text = embed.sticker_text(data.get("desc", ""), tags, data.get("ocr", ""))
        vec = await self._embed(text)
        if vec:
            self.store.set_vec(sha, embed.pack(vec))

    def _schedule_analysis(self, sha: str, path: str, ctx_block: str):
        try:
            asyncio.create_task(self._analyze_later(sha, path, ctx_block))
        except Exception as e:
            logger.debug(f"[sticker] 排队分析失败: {e}")

    async def _analyze_later(self, sha: str, path: str, ctx_block: str):
        try:
            ok, data = await self._analyze(path, ctx_block, need_verdict=False)
            if ok and data:
                await self._apply_analysis(sha, data)
                logger.info(
                    f"[sticker] 语义 {sha[:8]} desc={(data.get('desc') or '')[:40]!r} "
                    f"tags={data.get('ctx_tags')} conf={data.get('confidence')}"
                )
        except Exception as e:
            logger.debug(f"[sticker] 后台分析失败: {e}")

    @staticmethod
    def _message_text(event) -> str:
        parts = []
        try:
            for c in (event.get_messages() or []):
                t = getattr(c, "text", None)
                if t:
                    parts.append(str(t))
                elif isinstance(c, Image):
                    parts.append("[图片]")
        except Exception:
            pass
        return " ".join(parts).strip()

    @staticmethod
    def _quoted_text(event) -> str:
        try:
            for c in (event.get_messages() or []):
                if type(c).__name__ == "Reply":
                    return str(getattr(c, "message_str", "") or getattr(c, "text", "") or "")[:200]
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None

    # ------------------------------------------------------------------ 主动配图
    @filter.on_decorating_result()
    async def decorate(self, event: AstrMessageEvent):
        if not self.cfg.get("enabled", True) or not self.cfg.get("proactive_enabled", True):
            return
        try:
            self._roll_daily()
            if self._daily["sent"] >= int(self.cfg["proactive_daily_limit"]):
                return

            session = str(event.unified_msg_origin or "")
            last = self._last_sent_in_session.get(session, 0)
            if time.time() - last < int(self.cfg["session_cooldown_minutes"]) * 60:
                return

            result = event.get_result()
            if result is None:
                return
            text = " ".join(
                getattr(c, "text", "") for c in (result.chain or []) if getattr(c, "text", None)
            )
            if not text:
                return

            picked = await self._pick_for(text, session)
            if not picked:
                return

            if random.random() > decay.proactive_probability(
                picked["strength"], float(self.cfg["proactive_probability_base"])
            ):
                return

            result.chain.append(Image(file=picked["path"]))
            self.store.touch_used(picked["sha"], half_life_days=float(self.cfg["half_life_days"]))
            self._last_sent_in_session[session] = time.time()
            self._recent_sent[picked["sha"]] = time.time()
            self._daily["sent"] += 1
            logger.info(f"[sticker] 配图 {picked['sha'][:8]} strength={picked['strength']:.2f}")
        except Exception as e:
            logger.debug(f"[sticker] 配图失败: {e}")

    def _meme_hit(self, session: str) -> dict | None:
        """读梗百科发布的命中（它没装或过期都返回 None）。"""
        if not self.cfg.get("bridge_enabled", True):
            return None
        try:
            return bridge.read(self.shared_root, session,
                               max_age=int(self.cfg.get("bridge_max_age_seconds", 120)))
        except Exception:
            return None

    async def _pick_for(self, text: str, session: str) -> dict | None:
        """优先按语义挑；检索不可用或没有合格候选时，退化成按强度随机。"""
        if self.cfg.get("retrieval_enabled", True):
            picked = await self._pick_by_semantics(text, session)
            if picked is not None:
                return picked
            # 语义检索没结果（含"相似度不够"）→ 宁可不发
            if self.cfg.get("retrieval_strict", True):
                return None
        return self._pick_sticker(text)

    async def _pick_by_semantics(self, text: str, session: str) -> dict | None:
        reply_head = (text or "").strip()[:60]
        user_text = ""
        try:
            hist = [e["text"] for e in self.ctxbuf.entries(session) if not e.get("is_bot")]
            if hist:
                user_text = hist[-1]
        except Exception:
            pass
        # 以"对方说的话"为主（它决定话题），回复只取开头当语气参考。
        # 之前把整条回复拼进去，长回复会把话题信号稀释掉。
        meme_hit = self._meme_hit(session)
        meme_suffix = bridge.query_suffix(meme_hit) if meme_hit else ""
        query = " ".join(x for x in (user_text, reply_head, meme_suffix) if x).strip()

        vec = await self._embed(query)
        if not vec:
            return None

        rows = self.store.rows_for_retrieval()
        if not rows:
            return None

        ranked = embed.rank(
            vec, rows,
            top_k=int(self.cfg.get("retrieval_top_k", 5)),
            min_similarity=float(self.cfg.get("retrieval_min_similarity", 0.35)),
            strength_weight=float(self.cfg.get("strength_weight", 0.3)),
        )
        # 梗百科命中时，按标签重合度给候选加权：梗对上了，就用同样情绪的图
        if ranked and meme_hit:
            boost = float(self.cfg.get("meme_tag_boost", 0.15))
            meme_tags = list(meme_hit.get("tags") or [])
            for r in ranked:
                try:
                    st = json.loads(r.get("ctx_tags") or "{}")
                except Exception:
                    st = {}
                r["score"] = round(r["score"] + boost * bridge.tag_overlap(st, meme_tags), 4)
            ranked.sort(key=lambda x: x["score"], reverse=True)

        if not ranked:
            logger.debug(f"[sticker] 没有足够相关的图（阈值 {self.cfg.get('retrieval_min_similarity')}）")
            return None

        # 选图：默认取第一名。只有前几名分数几乎并列（差 <= 0.05）时才随机，
        # 否则"按相似度加权随机"会在多个合格候选里抽错，这正是上一轮挑歪的原因。
        pool = ranked[:3]
        best = pool[0]
        tied = [r for r in pool if (best["score"] - r["score"]) <= 0.05]
        pick = random.choice(tied) if len(tied) > 1 else best
        logger.info(
            f"[sticker] 语义命中 {pick['sha'][:8]} sim={pick['sim']} score={pick['score']} "
            f"tags={self.store.top_ctx_tags(pick['sha'], 3)} query={query[:40]!r}"
            + (f" meme={(meme_hit.get('titles') or [''])[:1]}" if meme_hit else "")
        )
        return pick

    def _pick_sticker(self, context_text: str) -> dict | None:
        cands = self.store.candidates(min_strength=float(self.cfg["active_threshold"]))
        if not cands:
            return None
        cooldown = int(self.cfg["same_sticker_cooldown_hours"]) * 3600
        now = time.time()
        fresh = [
            c for c in cands
            if now - self._recent_sent.get(c["sha"], 0) > cooldown
            and now - (c["last_used"] or 0) > cooldown
        ]
        if not fresh:
            return None
        # 有标签就先按标签粗筛，筛不出就用全部
        tagged = [c for c in fresh if c.get("tags") and c["tags"] != "[]"]
        pool = tagged or fresh
        pool.sort(key=lambda c: c["strength"], reverse=True)
        return random.choice(pool[:10])

    # ------------------------------------------------------------------ 定时维护
    @filter.on_astrbot_loaded()
    async def start_maintenance(self):
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._maintenance_loop())
        logger.info("[sticker] 维护任务已启动")
        # 补齐历史数据：库里还没描述的老图，启动时补一次（限 10 张，避免一上来刷额度）
        asyncio.create_task(self._backfill_analysis())
        asyncio.create_task(self._backfill_vectors())

    @filter.on_plugin_unloaded()
    async def stop_maintenance(self, metadata=None):
        if self._task and not self._task.done():
            self._task.cancel()
        try:
            self.store.close()
        except Exception:
            pass

    async def _backfill_analysis(self, limit: int = 10):
        await asyncio.sleep(20)  # 等启动彻底完成，别和初始化抢资源
        try:
            rows = [r for r in self.store.all_rows()
                    if not r.get("analyzed") and r.get("state") != "forgotten"]
            if not rows:
                return
            logger.info(f"[sticker] 待补描述 {len(rows)} 张，先处理 {min(limit, len(rows))} 张")
            for r in rows[:limit]:
                ok, data = await self._analyze(r["path"], "")
                if ok and data:
                    self._apply_analysis(r["sha"], data)
                    logger.info(
                        f"[sticker] 补描述 {r['sha'][:8]} desc={(data.get('desc') or '')[:40]!r} "
                        f"tags={data.get('ctx_tags')}"
                    )
        except Exception as e:
            logger.debug(f"[sticker] 补描述失败: {e}")

    async def _maintenance_loop(self):
        while True:
            try:
                await asyncio.sleep(3600)
                changes = self.store.tick(
                    half_life_days=float(self.cfg["half_life_days"]),
                    active_threshold=float(self.cfg["active_threshold"]),
                    forget_threshold=float(self.cfg["forget_threshold"]),
                    observe_hours=float(self.cfg["observe_hours"]),
                )
                victims = self.store.enforce_capacity(int(self.cfg["max_stickers"]))
                purged = self.store.purge_files(int(self.cfg["purge_retention_days"]))
                if changes["forgotten"] or victims or purged:
                    logger.info(
                        f"[sticker] 维护: 遗忘 {changes['forgotten']}, 淘汰 {len(victims)}, 清理文件 {purged}, "
                        f"统计 {self.store.stats()}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[sticker] 维护任务异常: {e}")

    # ------------------------------------------------------------------ 指令
    @filter.command("表情库")
    async def cmd_status(self, event: AstrMessageEvent):
        s = self.store.stats()
        by = s.get("by_state", {})
        yield event.plain_result(
            f"表情库：共 {s['total']} 张｜活跃 {by.get('active', 0)}｜观察 {by.get('observing', 0)}"
            f"｜沉睡 {by.get('dormant', 0)}｜遗忘 {by.get('forgotten', 0)}\n"
            f"平均强度 {s['avg_strength']}｜累计使用 {s['total_used']} 次｜今日收 {self._daily['collected']} 发 {self._daily['sent']}"
        )

    @filter.command("表情库清理")
    async def cmd_tick(self, event: AstrMessageEvent):
        try:
            is_adm = event.is_admin()
        except Exception:
            is_adm = False
        if not is_adm and str(event.get_sender_id()) not in self._admins():
            yield event.plain_result("只有管理员能手动推进衰减")
            return
        c = self.store.tick(
            half_life_days=float(self.cfg["half_life_days"]),
            active_threshold=float(self.cfg["active_threshold"]),
            forget_threshold=float(self.cfg["forget_threshold"]),
            observe_hours=float(self.cfg["observe_hours"]),
        )
        yield event.plain_result(f"已推进：{c}")

    def _admins(self) -> list[str]:
        try:
            return [str(x) for x in (self.context.get_config().get("admins_id") or [])]
        except Exception:
            return []
