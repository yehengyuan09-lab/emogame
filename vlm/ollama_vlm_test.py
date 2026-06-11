"""Smoke-test local Ollama vision models on a skin image.

Example:
    python3 vlm/ollama_vlm_test.py \
        --models qwen2.5vl:3b,internvl2:4b \
        --image hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_MODELS = "internvl2:4b,qwen2-vl:7b,qwen2.5vl:3b"
DEFAULT_IMAGE = "hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg"
DEFAULT_HOST = "http://127.0.0.1:11434"

PROMPTS = {
    "l1": """Analyze this game skin image and output ONLY valid JSON (no markdown, no explanation):

{
  "rarity_tier": "勇者|史诗|传说|无双|荣耀典藏",
  "dominant_colors": ["#HEX1", "#HEX2", "#HEX3", "#HEX4", "#HEX5"],
  "scene_type": "战场|主城|异界|抽象|自然",
  "character_ratio": 0.0-1.0,
  "effect_density": "low|mid|high|extreme",
  "confidence": 0.0-1.0
}
""",
    "l2": """你是游戏皮肤视觉评估专家。请对下方皮肤图片从以下 8 个维度打分 (1-10)：

- model_detail: 模型精细度（面数、材质贴图、光影反射质量）
- effect_quality: 特效质量（粒子效果密度、动态光效、色彩过渡自然度）
- color_scheme: 配色方案（色调和谐度、辨识度、主题一致性）
- composition: 构图（视觉焦点明确度、动感、前景/背景层次感）
- uniqueness: 独创性（与同英雄其他皮肤的差异程度）
- costume_design: 服饰设计（纹理细节、材质表现、风格统一性）
- background_quality: 背景质量（与主体融合度、氛围营造、细节丰富度）
- ui_elements: 识别所有可见的 UI 标记（限定、折扣、抽奖专属标签等）

输出严格 JSON 格式，不要包含任何额外文本。
""",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test InternVL/Qwen-VL style Ollama models with one skin image."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama host URL")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help="Comma-separated Ollama model names to test",
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image path")
    parser.add_argument(
        "--prompt",
        choices=sorted(PROMPTS),
        default="l2",
        help="Prompt/schema to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-model HTTP timeout in seconds",
    )
    return parser.parse_args()


def post_json(host: str, endpoint: str, payload: dict[str, Any], timeout: int) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_installed_models(host: str, timeout: int) -> set[str]:
    request = urllib.request.Request(f"{host.rstrip('/')}/api/tags")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {model["name"] for model in payload.get("models", [])}


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_model(
    host: str,
    model: str,
    image_b64: str,
    prompt: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = post_json(
        host,
        "/api/chat",
        {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "options": {
                "temperature": 0,
                "num_predict": 700,
            },
        },
        timeout,
    )
    content = response.get("message", {}).get("content", "").strip()
    result = {
        "model": model,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "response": content,
    }
    parsed = parse_jsonish(content)
    if parsed is not None:
        result["parsed_json"] = parsed
    return result


def parse_jsonish(content: str) -> Any | None:
    """Parse plain JSON or a JSON object wrapped in markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    models = [name.strip() for name in args.models.split(",") if name.strip()]
    try:
        installed = get_installed_models(args.host, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Cannot reach Ollama at {args.host}: {exc}", file=sys.stderr)
        return 2

    image_b64 = encode_image(image_path)
    results: list[dict[str, Any]] = []
    for model in models:
        if model not in installed:
            results.append(
                {
                    "model": model,
                    "status": "missing",
                    "hint": f"Run: ollama pull {model}",
                }
            )
            continue

        try:
            result = run_model(
                args.host,
                model,
                image_b64,
                PROMPTS[args.prompt],
                args.timeout,
            )
            result["status"] = "ok"
            results.append(result)
        except (urllib.error.URLError, TimeoutError) as exc:
            results.append({"model": model, "status": "error", "error": str(exc)})

    print(
        json.dumps(
            {
                "image": str(image_path),
                "prompt": args.prompt,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if any(item.get("status") == "ok" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
