"""Prompt templates for the three VLM pipeline tiers.

L1 / L2 prompts are extracted from ``ollama_vlm_test.py``.
L3 prompts are new for GPT-4o-mini semantic analysis.
"""

from __future__ import annotations

# ── L1: Fast classification (Qwen2.5VL-3B via Ollama) ──

L1_PROMPT = """\
Analyze this game skin image and output ONLY valid JSON (no markdown, no explanation):

{
  "rarity_tier": "勇者|史诗|传说|无双|荣耀典藏",
  "dominant_colors": ["#HEX1", "#HEX2", "#HEX3", "#HEX4", "#HEX5"],
  "scene_type": "战场|主城|异界|抽象|自然",
  "character_ratio": 0.0-1.0,
  "effect_density": "low|mid|high|extreme",
  "confidence": 0.0-1.0
}
"""

# ── L2: Fine aesthetic analysis (Qwen2.5VL-3B via Ollama) ──

L2_PROMPT = """\
你是游戏皮肤视觉评估专家。请对下方皮肤图片从以下 8 个维度打分 (1-10)：

- model_detail: 模型精细度（面数、材质贴图、光影反射质量）
- effect_quality: 特效质量（粒子效果密度、动态光效、色彩过渡自然度）
- color_scheme: 配色方案（色调和谐度、辨识度、主题一致性）
- composition: 构图（视觉焦点明确度、动感、前景/背景层次感）
- uniqueness: 独创性（与同英雄其他皮肤的差异程度）
- costume_design: 服饰设计（纹理细节、材质表现、风格统一性）
- background_quality: 背景质量（与主体融合度、氛围营造、细节丰富度）
- ui_elements: 识别所有可见的 UI 标记，输出格式：
  {"has_limited_tag": bool, "has_discount_tag": bool, "has_gacha_tag": bool, "special_border": bool, "tier_label": "传说"}

输出严格 JSON 格式，不要包含任何额外文本。
"""

# ── L3: Semantic understanding (AutoDL GPT-5.4-mini API) ──

L3_SYSTEM_PROMPT = """\
你是游戏美术与文化分析专家。根据皮肤图片的视觉特征，进行语义层面的深度解读。
输出严格 JSON 格式，不要包含任何额外文本。
"""

L3_USER_PROMPT_TEMPLATE = """\
这是一张王者荣耀皮肤壁纸图片，已知以下视觉分析结果：
- 稀有度: {rarity_tier}
- 主色调: {dominant_colors}
- 场景类型: {scene_type}
- 特效密集度: {effect_density}
- 模型精细度: {model_detail}/10
- 配色方案: {color_scheme}/10

请从以下角度分析并输出 JSON：
{{
  "design_style": "设计风格命名，如 水墨国风/赛博朋克/机甲科技/古风仙侠/现代潮流等",
  "cultural_references": ["文化符号列表，如 山海经/敦煌/西游记 等"],
  "target_audience": ["目标受众，如 核心玩家/收藏党/国风爱好者/女性玩家 等"],
  "similar_skins": [
    {{"name": "英雄名-皮肤名", "similarity_reason": "相似原因"}}
  ],
  "differentiation": "与同类皮肤的核心差异描述"
}}
"""
