# 表情记忆插件 · 安装与回滚

## 一、它做什么

| 能力 | 说明 |
|---|---|
| 收集 | 收到消息里的图片，判定是"表情包"才收；按内容哈希 + 感知哈希去重 |
| 记忆 | 每张图一个 `strength`，按半衰期（默认 3 天）自然衰减 |
| 加固 | 再次看到 +0.35，被使用 +0.25，上限 2.5 |
| 遗忘 | 低于 0.08 并持续 3 天 → 遗忘；文件保留 30 天，再物理删除 |
| 复活 | 已遗忘的图再出现，直接回到 1.5，跳过观察期 |
| 容量 | 超过 100 张时淘汰最弱、最久没出现的 |
| 配图 | 回复生成后，按强度概率、同图 24h 冷却、同会话 10 分钟冷却、每日 5 张上限，决定是否追加一张 |

## 二、依赖

- AstrBot >= 4.24.0
- Python 已有 Pillow（AstrBot 自带 13.x）
- 无额外 pip 包（sqlite3、hashlib、Pillow 都是现成的）
- 可选：视觉模型（用于 L3 判定），关掉也能跑

## 三、安装（两种方式）

### 方式 A：面板上传 zip

1. AstrBot 面板 → 插件 → 从文件安装 → 选 `astrbot_plugin_meme_memory.zip`
2. 启用插件
3. 重载插件（或重启 AstrBot）

### 方式 B：手工放目录

```bash
# 在服务器上
cd /opt/astrbot/instance/data/plugins/
# 解压后目录名必须是 astrbot_plugin_meme_memory
unzip astrbot_plugin_meme_memory.zip
# 然后重载插件或重启 AstrBot
```

## 四、首次验证（按顺序，每步都能定位问题）

1. **插件是否加载**：面板插件列表里能看到「表情记忆」。
2. **能不能收**：在私聊发一张表情包，然后 `/表情库`。
   - 期望：观察 `1`
   - 同时 `journalctl -u astrbot -n 50 | grep sticker` 应有一行 `新收藏`
3. **判定准不准**：
   - 发一张普通截图 → 不应被收（`/表情库` 数字不变）
   - 发一张 GIF 动图 → 应被收
4. **L1 是否可用**（关键未知项）：日志里如果 `reasons` 出现 `sub_type=1` 或 `summary=`，说明
   OneBot 原始字段能读到，判定会准很多；如果从来没有，说明只有 L2/L3 在工作，属正常降级。
5. **配图是否生效**：等观察期（默认 6 小时）过后，正常聊天几天，观察她是否偶尔多一张图。

## 五、回滚

### 最小回滚（保留数据，随时可再开）

面板里**禁用**插件即可。不占资源、不发消息、数据都在。

### 完全回滚（删干净）

```bash
# 1. 先停掉插件（面板禁用，或直接删目录后重载）
rm -rf /opt/astrbot/instance/data/plugins/astrbot_plugin_meme_memory

# 2. 删掉它存的数据（图 + 数据库）
rm -rf /opt/astrbot/instance/data/plugin_data/sticker_memory

# 3. 重载插件 或 systemctl restart astrbot
```

> 数据目录是 `data/plugin_data/sticker_memory/`，里面有 `stickers.db` 和 `stickers/`。
> 只删插件目录、不删数据目录时，重新装回来自动接上原有记忆。

### 回滚前想先备份

```bash
tar czf sticker-memory-backup-$(date +%Y%m%d).tgz \
  -C /opt/astrbot/instance/data/plugin_data sticker_memory
```

## 六、参数怎么调

| 想要的效果 | 调哪个 |
|---|---|
| 忘得更快/更慢 | `half_life_days`（2 = 更快，7 = 更慢） |
| 库里多留点图 | `max_stickers` 调大 |
| 她别太爱发图 | `proactive_probability_base` 调小 / `proactive_daily_limit` 调小 |
| 刚收的图别马上发 | `observe_hours` 调大 |
| 只收主人的图 | `collect_from_all` = false |
| 完全不想花视觉模型的钱 | `vlm_enabled` = false（只靠图片特征判断，会漏一些） |

## 七、本地开发与测试

不需要 AstrBot、不需要 QQ、不需要联网：

```bash
cd astrbot_plugin_meme_memory
python tests/run_tests.py     # 衰减 / 判定 / 存储，32 项
python tests/test_import.py   # 用假 AstrBot 包冒烟，7 项
```

## 八、已知限制（诚实清单）

1. **L1 协议线索未实测**。`sub_type` / `summary` 能不能从 `raw_message` 读到，取决于 NapCat
   的字段透传。读不到就靠 L2 图片特征 + L3 视觉模型，准确率会降。
2. **空 `id_whitelist` 时可能收到自己发的图**——本插件不区分"谁发的"，如需只收主人的要开
   `collect_from_all=false` 并在代码里补 sender 判断。
3. **`mface`（商城表情）** 目前只当作"是表情"，取图仍走 `convert_to_file_path()`，能不能
   成功取决于 URL 是否可下载。
4. **视觉模型调用签名**按主流 `provider.text_chat(prompt=..., image_urls=[...])` 写的，
   失败会静默降级成"不收这张"。若发现该收的没收，先把 `vlm_enabled` 打开并看 debug 日志。
5. **配图是"追加在回复后"**，不是替换回复；如果她本来就发了图片，可能叠加。
