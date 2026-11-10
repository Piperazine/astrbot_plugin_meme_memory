"""向量检索测试：余弦、排序、阈值门控、以及"真的挑了语义最近的那张"。

    python tests/test_retrieval.py
"""

import asyncio
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from test_import import DummyContext, build_stub  # noqa: E402

build_stub()

import embed  # noqa: E402
import main as pm  # noqa: E402

PASS = 0
FAIL = 0
TMP = os.path.join(HERE, "_test_tmp_retrieval")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def test_math():
    print("\n[数学]")
    check("完全相同 = 1", abs(embed.cosine([1, 0], [1, 0]) - 1.0) < 1e-9)
    check("正交 = 0", abs(embed.cosine([1, 0], [0, 1])) < 1e-9)
    check("长度不一致 = 0", embed.cosine([1, 0], [1, 0, 0]) == 0.0)
    check("零向量 = 0", embed.cosine([0, 0], [1, 0]) == 0.0)

    v = [0.1, 0.2, 0.3]
    check("打包解包无损", all(abs(a - b) < 1e-6 for a, b in zip(embed.unpack(embed.pack(v)), v)),
          embed.unpack(embed.pack(v)))
    check("空 blob 返回空", embed.unpack(None) == [])

    text = embed.sticker_text("一只猫摊手", ["无奈", "自嘲"], "算了")
    check("拼接文本含描述", "一只猫摊手" in text)
    check("拼接文本含标签", "无奈" in text)
    check("拼接文本含 OCR", "图上写着" in text)


def test_rank():
    print("\n[排序与门控]")
    rows = [
        {"sha": "a", "vec": embed.pack([1.0, 0.0]), "strength": 2.5},
        {"sha": "b", "vec": embed.pack([0.9, 0.1]), "strength": 0.5},
        {"sha": "c", "vec": embed.pack([0.0, 1.0]), "strength": 2.5},
    ]
    out = embed.rank([1.0, 0.0], rows, top_k=5, min_similarity=0.35, strength_weight=0.3)
    check("低相似度被剔除", all(r["sha"] != "c" for r in out), [r["sha"] for r in out])
    check("最相似的排第一", out and out[0]["sha"] in ("a", "b"), [r["sha"] for r in out])
    check("带上了 sim 字段", out and "sim" in out[0])

    # 查询与两个方向都只有约 0.707 的相似度，阈值设 0.9 就应该全被剔除
    none = embed.rank([0.7071, 0.7071], rows, top_k=5, min_similarity=0.9, strength_weight=0.3)
    check("阈值高到没有候选 -> 空", none == [], none)

    check("top_k 生效", len(embed.rank([1.0, 0.0], rows, top_k=1, min_similarity=0.0)) == 1)

    # 强度加权：相似度接近时，强度高的胜出
    ok = embed.rank([1.0, 0.0], rows, top_k=2, min_similarity=0.0, strength_weight=0.5)
    check("强度参与打分", abs(ok[0]["score"] - (ok[0]["sim"] * 0.5 + (ok[0]["strength"] / 2.5) * 0.5)) < 1e-3,
          ok[0])


def make_plugin(tmp):
    cfg = dict(pm.DEFAULTS)
    cfg["retrieval_min_similarity"] = 0.5
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
    return inst


def test_pick():
    print("\n[挑图]")
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    inst = make_plugin(TMP)

    # 两张图：一张"笑哭"，一张"无语"
    inst.store.add(sha="laugh", phash="p1", path="a.gif")
    inst.store.add(sha="speechless", phash="p2", path="b.gif")
    inst.store.set_vec("laugh", embed.pack([1.0, 0.0]))
    inst.store.set_vec("speechless", embed.pack([0.0, 1.0]))
    inst.store.conn.execute("update stickers set state='active', strength=2.0")
    inst.store.conn.commit()

    async def fake_embed(text):
        if "笑" in text:
            return [1.0, 0.0]
        if "无语" in text:
            return [0.0, 1.0]
        return [0.7071, 0.7071]   # 与两者都只是中等相似

    inst._embed = fake_embed

    got = asyncio.run(inst._pick_by_semantics("哈哈哈笑死我了", "s1"))
    check("按语义挑到笑哭那张", got and got["sha"] == "laugh", got and got["sha"])

    # 连跑 10 次都应该是同一张（优势明显时不许随机）
    picks = {asyncio.run(inst._pick_by_semantics("笑死", "s1"))["sha"] for _ in range(10)}
    check("优势明显时稳定选第一", picks == {"laugh"}, picks)

    got2 = asyncio.run(inst._pick_by_semantics("我无语了", "s1"))
    check("按语义挑到无语那张", got2 and got2["sha"] == "speechless", got2 and got2["sha"])

    inst.store.close()


def test_strict_gate():
    print("\n[阈值门控：不相关就不发]")
    tmp = os.path.join(TMP, "w2")
    os.makedirs(tmp, exist_ok=True)
    inst = make_plugin(tmp)
    inst.store.add(sha="only", phash="p9", path="c.gif")
    inst.store.set_vec("only", embed.pack([1.0, 0.0]))
    inst.store.conn.execute("update stickers set state='active', strength=2.0")
    inst.store.conn.commit()

    async def unrelated(_text):
        return [0.0, 1.0]   # 与库里唯一的图正交

    inst._embed = unrelated
    got = asyncio.run(inst._pick_for("今天天气不错", "s2"))
    check("不相关时返回 None", got is None, got)
    inst.store.close()


if __name__ == "__main__":
    test_math()
    test_rank()
    test_pick()
    test_strict_gate()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
