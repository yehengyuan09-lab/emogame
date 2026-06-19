"""Shared utility functions for the VLM pipeline.

Extracted from ``ollama_vlm_test.py`` so every module can reuse them.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


# ── Image encoding ──

def encode_image(path: Path) -> str:
    """Base64-encode an image file for the Ollama API."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def encode_image_bytes(data: bytes) -> str:
    """Base64-encode raw image bytes."""
    return base64.b64encode(data).decode("ascii")


# ── JSON parsing ──

def parse_jsonish(content: str) -> Any | None:
    """Parse JSON from a VLM response that may contain extra text.

    Handles three common failure modes:
    1. Plain JSON (happy path)
    2. JSON wrapped in markdown fences (`` ```json ... ``` ``)
    3. Truncated / wrapped JSON — finds the outermost ``{…}``

    Returns ``None`` when no valid JSON can be extracted.
    """
    text = content.strip()

    # Strip markdown code fences
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
        # Try to find the outermost { … } block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


# ── Hashing (for caching) ──

def image_path_hash(image_path: Path) -> str:
    """SHA-256 of the *resolved absolute path* string.

    Use as the primary cache key — fast and stable.
    """
    return hashlib.sha256(str(image_path.resolve()).encode()).hexdigest()


def image_content_hash(image_path: Path) -> str:
    """SHA-256 of the image *file content*.

    Use for change detection — if the file at the same path is replaced
    with different content the hash changes and the cache is invalidated.
    """
    return hashlib.sha256(image_path.read_bytes()).hexdigest()
