"""端到端（离线）测试：模拟一条 QQ 消息，走完"判定 -> 入库"。

用假 AstrBot 包 + 假事件对象，不需要真机器人。
验证三件事：
  1. 动图表情包 -> 收，且记录帧数
  2. 普通截图   -> 不收
  3. 透明方形图 -> 收

    python tests/test_ingest.py
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

import main as pm  # noqa: E402

PASS = 0
FAIL = 0
TMP = os.path.join(HERE, "_test_tmp_ingest")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def make_assets():
    from PIL import Image as PILImage

    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)

    # 1) 动图 GIF（3 帧）
    frames = [PILImage.new("RGB", (240, 240), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    gif = os.path.join(TMP, "anim.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=120, loop=0)

    # 2) 普通截图（16:9 大图）
    shot = os.path.join(TMP, "shot.jpg")
    PILImage.new("RGB", (1920, 1080), (128, 128, 128)).save(shot, quality=70)

    # 3) 透明方形 PNG
    png = os.path.join(TMP, "sticker.png")
    PILImage.new("RGBA", (400, 400), (0, 0, 0, 0)).save(png)

    return gif, shot, png


class FakeImage(pm.Image):
    def __init__(self, path):
        super().__init__(file=path)
        self._p = path

    async def convert_to_file_path(self):
        return self._p


class FakeEvent:
    def __init__(self, comps, raw):
        self._comps = comps
        self.message_obj = type("MO", (), {"raw_message": raw})()
        self.unified_msg_origin = "test:GroupMessage:123456"

    def get_messages(self):
        return self._comps

    def get_sender_id(self):
        return "861823912"


def raw_segments(n):
    return {"message": [{"type": "image", "data": {"sub_type": 1, "summary": "[动画表情]"}} for _ in range(n)]}


def make_plugin(tmp):
    cfg = dict(pm.DEFAULTS)
    cfg["vlm_enabled"] = False       # 不联网
    pm.StickerMemory._resolve_data_dir = lambda self: tmp
    inst = pm.StickerMemory(DummyContext(), cfg)
    # 强制把数据目录指到测试临时目录
    inst.close_store = None
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


def collect(inst, path):
    ev = FakeEvent([FakeImage(path)], raw_segments(1))
    asyncio.run(inst._collect(ev))


def main():
    gif, shot, png = make_assets()

    print("[ingest] 动图 GIF")
    inst = make_plugin(os.path.join(TMP, "w1"))
    collect(inst, gif)
    rows = inst.store.all_rows()
    check("动图被收入", len(rows) == 1, len(rows))
    if rows:
        check("记录帧数 > 1", rows[0]["frames"] > 1, rows[0]["frames"])
        check("初始状态 = observing", rows[0]["state"] == "observing", rows[0]["state"])
        check("文件名是内容哈希", len(rows[0]["sha"]) == 64, rows[0]["sha"][:12])
    inst.store.close()

    print("\n[ingest] 1920x1080 截图")
    inst = make_plugin(os.path.join(TMP, "w2"))
    collect(inst, shot)
    check("截图被拒收", len(inst.store.all_rows()) == 0, inst.store.all_rows())
    inst.store.close()

    print("\n[ingest] 透明方形 PNG")
    inst = make_plugin(os.path.join(TMP, "w3"))
    collect(inst, png)
    check("透明方形图被收", len(inst.store.all_rows()) == 1, len(inst.store.all_rows()))
    inst.store.close()

    print("\n[ingest] 同一张图发两次")
    inst = make_plugin(os.path.join(TMP, "w4"))
    collect(inst, gif)
    collect(inst, gif)
    rows = inst.store.all_rows()
    check("不会重复入库", len(rows) == 1, len(rows))
    if rows:
        check("第二次算强化", rows[0]["seen_count"] >= 1, rows[0]["seen_count"])
    inst.store.close()

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
