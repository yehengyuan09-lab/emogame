"""VLM Pipeline — 3-tier vision-language-model analysis for game skins.

Tiers
-----
* **L1** — Qwen2.5VL-3B via Ollama: fast classification (rarity, colours, scene)
* **L2** — Qwen2.5VL-3B via Ollama: fine aesthetic scoring (8 dimensions, 1-10)
* **L3** — AutoDL GPT-5.4-mini API: semantic / cultural understanding

Public API
----------
``VlmPipeline`` is the main entry point::

    from vlm import VlmPipeline
    pipeline = VlmPipeline()
    result = await pipeline.analyze("wallpaper.jpg")

``VLMFeatureVector`` is the flat Pydantic model returned by the pipeline.
"""

from vlm.config import VLMSettings, get_settings
from vlm.pipeline import ExecutionMode, VlmPipeline
from vlm.schemas import (
    L1Output,
    L2Output,
    L3Output,
    UIElements,
    VLMFeatureVector,
)

__all__ = [
    "ExecutionMode",
    "VlmPipeline",
    "VLMFeatureVector",
    "L1Output",
    "L2Output",
    "L3Output",
    "UIElements",
    "get_settings",
    "VLMSettings",
]
