"""联动测试：梗百科命中 -> 表情插件挑图加权。

覆盖：
  - 发布 / 读取 / 过期 / 缺文件
  - 查询后缀与标签重合度
  - 有梗信号时，同样情绪的图会被优先选中
"""

import asyncio
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, ROOT)   # 让 astrbot_plugin_memewiki 能作为包导入
sys.path.insert(0, os.path.join(ROOT, "astrbot_plugin_memewiki", "tests"))

from test_import import DummyContext, build_stub  # noqa: E402

build_stub()

import bridge  # noqa: E402  表情插件侧的读端
from astrbot_plugin_memewiki import publish  # noqa: E402 梗百科侧的写端
import embed  # noqa: E402
import main as pm  # noqa: E402

PASS = 0
FAIL = 0
TMP = os.path.join(HERE, "_test_tmp_bridge")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def test_roundtrip():
    print("\n[发布与读取]")
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    s = "napcat:GroupMessage:123"
    T = 1_700_000_000.0

    check("没有文件时读到 None", bridge.read(TMP, s, now=T) is None)

    ok = publish.publish(TMP, s, ["康神开播了"], ["主播梗", "直播梗"], now=T)
    check("发布成功", ok is True)

    hit = bridge.read(TMP, s, max_age=120, now=T + 10)
    check("能读回命中", hit and hit["titles"] == ["康神开播了"], hit)
    check("标签保留", hit and "主播梗" in hit["tags"], hit)

    check("过期后读到 None", bridge.read(TMP, s, max_age=120, now=T + 999) is None)
    check("别的会话读不到", bridge.read(TMP, "other:session", now=T + 10) is None)

    # 文件坏了也不能抛异常
    path = bridge.shared_path(TMP)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{坏的 json")
    check("坏文件返回 None", bridge.read(TMP, s, now=T + 10) is None)

    publish.publish(TMP, s, ["芭比Q"], ["网络用语"], now=T)
    check("坏文件后还能恢复写入", bridge.read(TMP, s, now=T) is not None)


def test_helpers():
    print("\n[辅助函数]")
    hit = {"titles": ["康神开播了"], "tags": ["主播梗", "嘲讽", "直播梗"]}
    suf = bridge.query_suffix(hit, limit=2)
    check("查询后缀含标题", "康神开播了" in suf, suf)
    check("后缀不会太长", len(suf.split()) <= 3, suf)
    check("None 命中返回空串", bridge.query_suffix(None) == "")

    check("完全重合 = 1", bridge.tag_overlap({"嘲讽": 1}, ["嘲讽"]) == 1.0)
    check("部分重合", abs(bridge.tag_overlap({"嘲讽": 1, "无语": 1}, ["嘲讽", "路过"]) - 0.5) < 1e-9)
    check("不重合 = 0", bridge.tag_overlap({"温和": 1}, ["嘲讽"]) == 0.0)
    check("空输入 = 0", bridge.tag_overlap({}, ["嘲讽"]) == 0.0)


def make_plugin(tmp):
    cfg = dict(pm.DEFAULTS)
    cfg["retrieval_min_similarity"] = 0.3
    cfg["meme_tag_boost"] = 0.15
    pm.StickerMemory._resolve_data_dir = lambda self: tmp
    inst = pm.StickerMemory(DummyContext(), cfg)
    try:
        inst.store.close()
    except Exception:
        pass
    from store import StickerStore, VlmCache
    inst.blob_dir = os.path.join(tmp, "blobs")
    os.makedirs(inst.blob_dir, exist_ok=True)
    inst.store = StickerStore(os.path.join(tmp, "s.db"), inst.blob_dir)
    inst.vlm = VlmCache(inst.store)
    inst.shared_root = tmp
    return inst


def test_boost():
    print("\n[梗信号加权]")
    tmp = os.path.join(TMP, "w1")
    os.makedirs(tmp, exist_ok=True)
    inst = make_plugin(tmp)

    # A：嘲讽（标签对得上梗），相似度略低
    inst.store.add(sha="mock", phash="p1", path="a.gif")
    inst.store.set_analysis("mock", "阴阳怪气的图", ["嘲讽"], ctx_tags={"嘲讽": 1})
    inst.store.set_vec("mock", embed.pack([0.86, 0.51]))
    # B：温和（标签对不上），相似度略高
    inst.store.add(sha="warm", phash="p2", path="b.gif")
    inst.store.set_analysis("warm", "温柔抱抱的图", ["温和"], ctx_tags={"温和": 1})
    inst.store.set_vec("warm", embed.pack([1.0, 0.0]))
    inst.store.conn.execute("update stickers set state='active', strength=2.0")
    inst.store.conn.commit()

    async def fake_embed(_text):
        return [1.0, 0.0]

    inst._embed = fake_embed
    session = "napcat:GroupMessage:9"

    # 没有梗信号：选相似度更高的 warm
    inst._meme_hit = lambda s: None
    got = asyncio.run(inst._pick_by_semantics("普通一句话", session))
    check("无梗信号时按相似度选", got and got["sha"] == "warm", got and got["sha"])

    # 有梗信号（嘲讽）：mock 被加权超过 warm
    inst._meme_hit = lambda s: {"titles": ["某人破防了"], "tags": ["嘲讽"]}
    got2 = asyncio.run(inst._pick_by_semantics("普通一句话", session))
    check("有梗信号时选同情绪的图", got2 and got2["sha"] == "mock",
          got2 and (got2["sha"], got2["score"]))
    inst.store.close()


def test_memewiki_publishes():
    print("\n[梗百科侧：注入时发布]")
    from stub_astrbot import DummyContext as WikiCtx, build_stub as wiki_stub
    wiki_stub()

    from astrbot_plugin_memewiki import main as wiki_main
    wiki_main.MemeWikiPlugin._resolve_data_dir = lambda self: os.path.join(TMP, "wikidata")

    # 用合成词库
    wdir = os.path.join(TMP, "wikidata")
    os.makedirs(wdir, exist_ok=True)
    import sqlite3
    con = sqlite3.connect(os.path.join(wdir, "memes.db"))
    con.execute("""CREATE TABLE memes(slug TEXT PRIMARY KEY, title TEXT, aliases TEXT,
                   tags TEXT, summary TEXT, keywords TEXT, updated_at TEXT)""")
    con.execute("INSERT INTO memes VALUES (?,?,?,?,?,?,?)",
                ("babiq", "芭比Q", "[]", '["网络用语","自嘲"]', "完蛋了的意思。", "[]", ""))
    con.commit(); con.close()

    inst = wiki_main.MemeWikiPlugin(WikiCtx(), dict(wiki_main.DEFAULTS))
    inst.shared_root = TMP

    class Req:
        prompt = "这局芭比Q了"
        system_prompt = "原始提示"

    ev = type("E", (), {"unified_msg_origin": "napcat:GroupMessage:7"})()
    req = Req()
    asyncio.run(inst.inject_memes(ev, req))
    check("注入成功", "芭比Q" in req.system_prompt)

    hit = bridge.read(TMP, "napcat:GroupMessage:7", now=None)
    # 注意：publish 用的是真实时间，这里用 now=None 读取
    check("命中已发布给表情插件", hit is not None and "芭比Q" in hit["titles"], hit)
    inst.wiki.close()


if __name__ == "__main__":
    test_roundtrip()
    test_helpers()
    test_boost()
    test_memewiki_publishes()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
