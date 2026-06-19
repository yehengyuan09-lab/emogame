"""Shared Ollama HTTP client for L1 and L2 tiers.

Refactored from ``ollama_vlm_test.py``.  Both L1 (InternVL2-4B) and L2
(Qwen2-VL-7B) use the same Ollama ``/api/chat`` endpoint — only the model
name and prompt differ.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from vlm.config import get_settings
from vlm.utils import parse_jsonish


class OllamaClient:
    """HTTP client for Ollama's chat API with image support."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.host = self.settings.ollama_host.rstrip("/")

    # ── model discovery ──

    @staticmethod
    def _client_kwargs() -> dict[str, Any]:
        """Return kwargs for httpx clients — disable proxy for localhost.

        ``trust_env=False`` prevents httpx from picking up ``all_proxy`` /
        ``HTTP_PROXY`` environment variables that would route localhost
        traffic through an external proxy.
        """
        return {"trust_env": False}

    async def list_models(self) -> set[str]:
        """Return the set of installed Ollama model names."""
        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            resp = await client.get(f"{self.host}/api/tags", timeout=30)
            resp.raise_for_status()
            data = resp.json()
        return {m["name"] for m in data.get("models", [])}

    def list_models_sync(self) -> set[str]:
        """Synchronous wrapper for ``list_models``."""
        resp = httpx.get(
            f"{self.host}/api/tags",
            timeout=30,
            **self._client_kwargs(),
        )
        resp.raise_for_status()
        data = resp.json()
        return {m["name"] for m in data.get("models", [])}

    # ── chat ──

    async def chat(
        self,
        model: str,
        image_b64: str,
        prompt: str,
        timeout: int | None = None,
        temperature: float = 0.0,
        num_predict: int = 700,
    ) -> dict[str, Any]:
        """Send a chat request to Ollama with one base64-encoded image.

        Returns a dict with keys ``model``, ``elapsed_seconds``, ``content``,
        and (when the response is parseable JSON) ``parsed_json``.
        """
        timeout = timeout or self.settings.timeout_seconds
        started = time.perf_counter()

        payload = {
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
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        async with httpx.AsyncClient(**self._client_kwargs()) as client:
            resp = await client.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        result: dict[str, Any] = {
            "model": model,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "content": content,
        }

        parsed = parse_jsonish(content)
        if parsed is not None:
            result["parsed_json"] = parsed

        return result

    def chat_sync(
        self,
        model: str,
        image_b64: str,
        prompt: str,
        timeout: int | None = None,
        temperature: float = 0.0,
        num_predict: int = 700,
    ) -> dict[str, Any]:
        """Synchronous wrapper for ``chat`` (used by the smoke-test script)."""
        timeout = timeout or self.settings.timeout_seconds
        started = time.perf_counter()

        payload = {
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
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }

        resp = httpx.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=timeout,
            **self._client_kwargs(),
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "").strip()
        result: dict[str, Any] = {
            "model": model,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "content": content,
        }

        parsed = parse_jsonish(content)
        if parsed is not None:
            result["parsed_json"] = parsed

        return result
