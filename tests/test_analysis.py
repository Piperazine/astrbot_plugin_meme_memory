"""语义分析测试：解析、动图抽帧、以及"入库后真的写进了描述和标签"。

用假 Provider，不联网。
"""

import asyncio
import json
import time
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from test_import import DummyContext, build_stub  # noqa: E402

build_stub()

import vision  # noqa: E402
import main as pm  # noqa: E402
from test_ingest import FakeEvent, FakeImage, raw_segments  # noqa: E402

PASS = 0
FAIL = 0
TMP = os.path.join(HERE, "_test_tmp_analysis")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class FakeResp:
    def __init__(self, text):
        self.completion_text = text


class FakeProvider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def text_chat(self, prompt=None, image_urls=None):
        self.calls.append({"prompt": prompt, "images": image_urls})
        return FakeResp(json.dumps(self.payload, ensure_ascii=False))


def test_parse():
    print("\n[parse]")
    ok = vision.parse_result('{"is_sticker": true, "desc": "猫摊手", "ocr": "", "ctx_tags": ["无奈"], "confidence": 0.8}')
    check("正常 JSON", ok["desc"] == "猫摊手" and ok["ctx_tags"] == ["无奈"], ok)

    wrapped = vision.parse_result('好的，结果如下：\n{"is_sticker": false, "desc": "截图", "ctx_tags": []}\n以上')
    check("能从废话里抠 JSON", wrapped["desc"] == "截图" and wrapped["is_sticker"] is False, wrapped)

    bad = vision.parse_result("完全不是 JSON")
    check("坏输出不炸", bad["desc"] == "" and bad["is_sticker"] is True, bad)

    tags_str = vision.parse_result('{"desc":"x","ctx_tags":"单个标签"}')
    check("标签给成字符串也能收", tags_str["ctx_tags"] == ["单个标签"], tags_str)

    many = vision.parse_result(json.dumps({"ctx_tags": [f"t{i}" for i in range(10)]}, ensure_ascii=False))
    check("标签最多 5 个", len(many["ctx_tags"]) == 5, many["ctx_tags"])

    merged = vision.merge_tags({"嘲讽": 2}, ["嘲讽", "无语"])
    check("标签计数累加", merged == {"嘲讽": 3, "无语": 1}, merged)

    p_comp = vision.build_prompt("语境", composite=True)
    check("拼图会额外说明是动图", "同一段动图" in p_comp, p_comp[:80])
    p_plain = vision.build_prompt("语境", composite=False)
    check("静态图不加说明", "同一段动图" not in p_plain)


def test_frames():
    print("\n[frames]")
    from PIL import Image as PILImage
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)

    gif = os.path.join(TMP, "a.gif")
    frames = [PILImage.new("RGB", (240, 320), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

    path, composite = vision.frames_for(gif, max_frames=3)
    check("动图被拼图", composite is True, composite)
    if composite:
        with PILImage.open(path) as im:
            check("拼图宽度 = 3 帧", im.width == 720, im.size)
            check("拼图高度 = 单帧", im.height == 320, im.size)
        os.remove(path)

    png = os.path.join(TMP, "b.png")
    PILImage.new("RGB", (100, 100)).save(png)
    path2, composite2 = vision.frames_for(png)
    check("静态图不拼", composite2 is False and path2 == png, (path2, composite2))


def make_plugin(tmp):
    cfg = dict(pm.DEFAULTS)
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


def test_ingest_with_provider():
    print("\n[ingest + 视觉模型]")
    from PIL import Image as PILImage
    tmp = os.path.join(TMP, "w1")
    os.makedirs(tmp, exist_ok=True)

    gif = os.path.join(tmp, "s.gif")
    frames = [PILImage.new("RGB", (240, 240), c) for c in ((1, 2, 3), (4, 5, 6))]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)

    inst = make_plugin(tmp)
    prov = FakeProvider({
        "is_sticker": True,
        "desc": "一只猫摊开双手",
        "ocr": "算了",
        "ctx_tags": ["无奈", "自嘲"],
        "confidence": 0.86,
    })
    inst._provider = lambda: prov

    async def run():
        ev = FakeEvent([FakeImage(gif)], raw_segments(1))
        # 先塞一句本人语境
        inst.ctxbuf.record("test:GroupMessage:123456", "861823912", "我摆烂了", time.time())
        await inst._collect(ev)
        await asyncio.sleep(0.3)   # 等后台分析任务

    asyncio.run(run())

    rows = inst.store.all_rows()
    check("图已入库", len(rows) == 1, len(rows))
    if rows:
        r = rows[0]
        check("描述已写入", r["desc"] == "一只猫摊开双手", r["desc"])
        check("OCR 已写入", r["ocr_text"] == "算了", r["ocr_text"])
        tags = json.loads(r["ctx_tags"] or "{}")
        check("语境标签已写入", tags.get("无奈") == 1 and tags.get("自嘲") == 1, tags)
        check("置信度已写入", abs(r["confidence"] - 0.86) < 1e-6, r["confidence"])
        check("标记为已分析", r["analyzed"] == 1, r["analyzed"])

    check("确实调用了模型", len(prov.calls) >= 1, len(prov.calls))
    if prov.calls:
        p = prov.calls[0]["prompt"]
        check("提示词里带了本人语境", "我摆烂了" in p, p[:120])
    inst.store.close()


if __name__ == "__main__":
    test_parse()
    test_frames()
    test_ingest_with_provider()
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    sys.exit(1 if FAIL else 0)
