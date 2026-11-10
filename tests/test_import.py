"""外壳冒烟测试：用假的 astrbot 包把 main.py 导进来。

目的不是测业务逻辑（那在 run_tests.py），而是抓那种"只有在真实
AstrBot 里才会炸"的低级错误：装饰器用法、类定义、初始化路径、
以及能不能真的建出数据库。

    python tests/test_import.py
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def build_stub():
    # ---- astrbot.api ----
    import logging
    api = _mod("astrbot.api", logger=logging.getLogger("stub"))

    # ---- astrbot.api.event ----
    class AstrMessageEvent:  # noqa: D401
        pass

    class EventMessageType:
        ALL = "all"

    def _identity_decorator(*a, **kw):
        def wrap(fn):
            return fn
        return wrap

    filt = types.SimpleNamespace(
        event_message_type=_identity_decorator,
        command=_identity_decorator,
        on_decorating_result=_identity_decorator,
        on_astrbot_loaded=_identity_decorator,
        on_plugin_unloaded=_identity_decorator,
        EventMessageType=EventMessageType,
    )
    ev = _mod("astrbot.api.event", filter=filt, AstrMessageEvent=AstrMessageEvent)
    api.event = ev

    # ---- astrbot.api.message_components ----
    class Image:
        def __init__(self, file="", **kw):
            self.file = file

        async def convert_to_file_path(self):
            return self.file

    mc = _mod("astrbot.api.message_components", Image=Image)
    api.message_components = mc

    # ---- astrbot.api.star ----
    class Star:
        def __init__(self, context=None):
            self.context = context

    class Context:
        pass

    def register(*a, **kw):
        def wrap(cls):
            return cls
        return wrap

    star = _mod("astrbot.api.star", Context=Context, Star=Star, register=register)
    api.star = star

    _mod("astrbot", api=api)


class DummyContext:
    def __init__(self):
        self.cfg = {"admins_id": ["1"]}

    def get_using_provider(self):
        return None

    def get_provider_by_id(self, _):
        return None

    def get_config(self):
        return self.cfg


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    build_stub()

    print("[package mode] 模拟 AstrBot：把插件当包导入")
    pkg = os.path.basename(HERE)
    parent = os.path.dirname(HERE)
    sys.path.insert(0, parent)
    try:
        import importlib
        mod = importlib.import_module(f"{pkg}.main")
        check("包模式可导入（与 AstrBot 一致）", hasattr(mod, "StickerMemory"))
        # 包模式下兄弟模块也必须能解析
        check("包模式下 store 可导入", importlib.import_module(f"{pkg}.store") is not None)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("包模式可导入（与 AstrBot 一致）", False, repr(e))

    print("\n[flat mode] 平铺导入")
    try:
        import main as plugin_main
        check("main.py 可导入", True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("main.py 可导入", False, repr(e))
        print(f"\n==== {PASS} passed, {FAIL} failed ====")
        return 1

    check("插件类存在", hasattr(plugin_main, "StickerMemory"))
    check("配置 schema 可解析", _check_schema())

    print("\n[init]")
    try:
        import shutil
        tmp = os.path.join(HERE, "_test_tmp_import")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
        # 测试里接管数据目录，避免写到工作区外
        plugin_main.StickerMemory._resolve_data_dir = lambda self: tmp
        # 让 _resolve_data_dir 落到临时目录：把 main.py 所在目录临时指过去不方便，
        # 直接构造实例并覆盖 data_dir 相关目录即可
        inst = plugin_main.StickerMemory(DummyContext(), dict(plugin_main.DEFAULTS))
        check("能构造实例", inst is not None)
        check("数据库文件已创建", os.path.exists(inst.store.db_path), inst.store.db_path)
        check("sticker 目录已创建", os.path.isdir(inst.blob_dir), inst.blob_dir)
        s = inst.store.stats()
        check("空库统计正常", s["total"] == 0, s)
        inst.store.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("能构造实例", False, repr(e))

    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


def _check_schema():
    import json
    try:
        with open(os.path.join(HERE, "_conf_schema.json"), encoding="utf-8") as f:
            d = json.load(f)
        return isinstance(d, dict) and "enabled" in d
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
