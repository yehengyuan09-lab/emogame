# 02 — VLM 视觉语言模型管线

## 模型选型

### 三级管线设计

| 层级 | 模型 | 部署方式 | 用途 | 延迟 | 成本 |
|------|------|----------|------|------|------|
| L1 快速提取 | qwen2.5vl:3b | 本地 GPU (T4, 16GB) | 批量皮肤分类、色彩提取、构图分析 | ~0.5s/img | 免费 |
| L2 精细分析 | llama3.2-vision:11b | 本地 GPU (A10, 24GB) | 审美维度评分、特效层级判别、缺陷检测 | ~2s/img | 免费 |
| L3 深度理解 | GPT5.4-mini | API 调用 | 设计语言解读、文化符号识别、竞品对比描述 | ~3s/img | ~$0.00015/img |

### 模型能力对比

#### qwen2.5vl:3b（L1 — 批量快速分类）

- **优势**：开源可本地部署，Qwen2.5-VL 系列视觉语言模型，3B 参数可在 T4 上运行，Ollama 原生支持
- **适用任务**：颜色直方图、构图密度、场景分类、UI 标签检测
- **局限**：对高分辨率细节（1920×882 壁纸）理解有限

#### llama3.2-vision:11b（L2 — 精细审美分析）

- **优势**：Meta Llama 3.2 视觉模型，11B 参数，支持高分辨率图像输入，英文视觉理解能力强，Ollama 原生支持
- **适用任务**：面部细节、服饰纹理、特效粒子层数、UI 标识（限定标签、价格标签）
- **局限**：需要 A10 24GB VRAM，不能批量高并发，中文理解需 prompt 引导

#### GPT5.4-mini（L3 — 语义理解）

- **优势**：强大的语义推理和文化理解能力，通过 AutoDL API 调用（OpenAI 兼容接口）
- **适用任务**：设计风格命名（"水墨国风"、"赛博朋克"）、文化符号锚定（"山海经"、"敦煌"）、受众画像
- **局限**：API 延迟、按量付费

## 处理管线

```
  Skin Image (1920×882)
        │
        ├─[1]─► 图像预处理
        │       │  - 格式标准化 (JPEG → RGB Tensor)
        │       │  - 多尺度裁剪 (full / center / face region)
        │       │  - 色彩直方图提取 (HSV 量化 → 5 主色)
        │       │  - Canny 边缘密度检测 (特效复杂度代理)
        │       │  - 拉普拉斯方差 (清晰度/模糊检测)
        │       │
        │       ▼
        ├─[2]─► qwen2.5vl:3b: 快速分类 (~0.5s)
        │       │  - 皮肤稀有度分类 (勇者/史诗/传说/无双/荣耀典藏)
        │       │  - 主色调识别 (5 dominant colors + hex)
        │       │  - 场景类型 (战场/主城/异界/抽象/自然)
        │       │  - 人物占比估算 (0.0-1.0)
        │       │  - 特效密集度 (low/mid/high/extreme)
        │       │
        │       ▼
        ├─[3]─► llama3.2-vision:11b: 精细分析 (~2s)
        │       │  - 面部与服饰细节评价 (1-10)
        │       │  - 特效质量判别 (粒子数量、光影层次、动态暗示)
        │       │  - 构图评分 (三分法、视觉引导线、留白比例)
        │       │  - UI 标识识别 (限定标签、首周折扣、抽奖专属)
        │       │  - 与同系列皮肤的一致性判断
        │       │
        │       ▼
        ├─[4]─► GPT5.4-mini: 语义理解 (~3s)
        │       │  - 设计风格解读 (eg. "水墨国风"、"赛博朋克")
        │       │  - 文化符号锚定 (eg. "山海经异兽"、"敦煌飞天")
        │       │  - 受众群体画像 (核心向/泛用户/女性向/收藏党)
        │       │  - 与同类皮肤差异化描述
        │       │
        │       ▼
        └─[5]─► 输出: VLMFeatureVector
```

## 结构化输出 Schema

### L1 输出
```json
{
  "rarity_tier": "史诗",
  "dominant_colors": ["#C42828", "#F5E6C8", "#2B1E10", "#8B0000", "#FFD700"],
  "scene_type": "战场",
  "character_ratio": 0.65,
  "effect_density": "high",
  "confidence": 0.92
}
```

### L2 输出
```json
{
  "model_detail": 8.5,
  "effect_quality": 9.0,
  "color_scheme": 7.5,
  "composition": 8.0,
  "uniqueness": 6.5,
  "costume_design": 8.5,
  "background_quality": 7.0,
  "ui_elements": {
    "has_limited_tag": false,
    "has_discount_tag": true,
    "has_gacha_tag": false,
    "special_border": true,
    "tier_label": "传说"
  }
}
```

### L3 输出
```json
{
  "design_style": "水墨国风",
  "cultural_references": ["山海经", "穷奇"],
  "target_audience": ["核心玩家", "收藏党", "国风爱好者"],
  "similar_skins": [
    {"name": "李白-诗剑行", "similarity_reason": "同为水墨风格"},
    {"name": "上官婉儿-万象笔", "similarity_reason": "书法元素"}
  ],
  "differentiation": "相较于同类水墨皮肤，本皮肤加入了更多金色粒子特效，提升了传说品质感"
}
```

## Prompt 工程

### L1 快速分类 Prompt
```
Analyze this game skin image and output ONLY valid JSON (no markdown, no explanation):

{
  "rarity_tier": "勇者|史诗|传说|无双|荣耀典藏",
  "dominant_colors": ["#HEX1", "#HEX2", "#HEX3", "#HEX4", "#HEX5"],
  "scene_type": "战场|主城|异界|抽象|自然",
  "character_ratio": 0.0-1.0,
  "effect_density": "low|mid|high|extreme"
}
```

### L2 精细分析 System Prompt
```
你是游戏皮肤视觉评估专家。请对下方皮肤图片从以下 8 个维度打分 (1-10)：

- model_detail: 模型精细度（面数、材质贴图、光影反射质量）
- effect_quality: 特效质量（粒子效果密度、动态光效、色彩过渡自然度）
- color_scheme: 配色方案（色调和谐度、辨识度、主题一致性）
- composition: 构图（视觉焦点明确度、动感、前景/背景层次感）
- uniqueness: 独创性（与同英雄其他皮肤的差异程度）
- costume_design: 服饰设计（纹理细节、材质表现、风格统一性）
- background_quality: 背景质量（与主体融合度、氛围营造、细节丰富度）
- ui_elements: 识别所有可见的 UI 标记（限定、折扣、抽奖专属标签等）

输出严格 JSON 格式，不要包含任何额外文本。
```

## Ollama 本地烟测

仓库提供了一个轻量测试脚本，用同一张皮肤图调用 Ollama `/api/chat` 的图片输入能力，适合先验证本地模型、显存和 prompt 输出格式。

```bash
# 拉取模型
ollama pull qwen2.5vl:3b        # L1 快速分类
ollama pull llama3.2-vision:11b # L2 精细审美分析

# 测试 L1 模型
python3 vlm/ollama_vlm_test.py \
  --models qwen2.5vl:3b \
  --image hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg \
  --prompt l1

# 测试 L2 模型
python3 vlm/ollama_vlm_test.py \
  --models llama3.2-vision:11b \
  --image hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg \
  --prompt l2
```

脚本会跳过未安装模型并给出 `ollama pull <model>` 提示；已安装模型会输出模型名、耗时和原始 JSON 响应。

## 批量推理优化

```python
# 策略 1: 分级缓存
# L1/L2 结果写入特征存储，设置 TTL（如 30 天），仅图像变更时重新推理
CACHE_TTL_DAYS = 30

# 策略 2: 动态批处理
# qwen2.5vl:3b batch_size=8 on T4 (16GB VRAM)
# 900 张图片: 900/8 × 0.5s ≈ 56 秒完成全量 L1

# 策略 3: 增量更新
# 监控 hero-skin-image submodule 的 commit hash
# 仅对新皮肤（对比上次 commit 的增量）触发 VLM 推理

# 策略 4: 降级策略
# GPU 不可用时: L1+L2 降级为 AutoDL GPT5.4-mini API 模式
# L1 分类任务可用传统 CV（颜色直方图 + 边缘检测）替代
```

## GPU 需求

| 环境 | GPU | VRAM | 可运行模型 |
|------|-----|------|-----------|
| 最低 | CPU only | — | 传统 CV 替代 L1，API 替代 L2+L3 |
| 开发 | T4 | 16GB | qwen2.5vl:3b (batch=8) |
| 生产 | A10 | 24GB | qwen2.5vl:3b + llama3.2-vision:11b 并行 |

## 下一步

- → [03 — 数据采集系统](03-data-crawling.md)
- → [04 — 特征工程系统](04-feature-engineering.md)
