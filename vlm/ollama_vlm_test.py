"""Smoke-test local Ollama vision models on a skin image.

Refactored to use the shared ``OllamaClient``, prompts, and utilities from
the ``vlm`` package instead of defining them inline.

Example:
    python3 vlm/ollama_vlm_test.py \\
        --models qwen2.5vl:3b \\
        --image hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlm.ollama_client import OllamaClient
from vlm.prompts import L1_PROMPT, L2_PROMPT
from vlm.utils import encode_image

DEFAULT_MODELS = "qwen2.5vl:3b"
DEFAULT_IMAGE = "hero-skin-image/3phone-bigskin-images/李白-3-千年之狐.jpg"

PROMPTS = {
    "l1": L1_PROMPT,
    "l2": L2_PROMPT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test InternVL/Qwen-VL style Ollama models with one skin image."
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL (overrides VLMSettings.ollama_host)",
    )
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


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    models = [name.strip() for name in args.models.split(",") if name.strip()]

    client = OllamaClient()
    # Override host for the smoke test if non-default
    if args.host != "http://127.0.0.1:11434":
        client.host = args.host.rstrip("/")

    try:
        installed = client.list_models_sync()
    except Exception as exc:
        print(f"Cannot reach Ollama at {client.host}: {exc}", file=sys.stderr)
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
            result = client.chat_sync(
                model,
                image_b64,
                PROMPTS[args.prompt],
                timeout=args.timeout,
            )
            # Rename 'content' key to 'response' for backward compat
            result["response"] = result.pop("content", "")
            result["status"] = "ok"
            results.append(result)
        except Exception as exc:
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
