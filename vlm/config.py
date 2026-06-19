"""Centralized configuration for the VLM pipeline.

Reads from environment variables and ``.env`` file via pydantic-settings.
All model names, hosts, API keys, cache settings, and timeouts live here.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class VLMSettings(BaseSettings):
    """VLM pipeline configuration — env vars or .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Ollama ──
    ollama_host: str = "http://127.0.0.1:11434"
    l1_model: str = "qwen2.5vl:3b"
    l2_model: str = "qwen2.5vl:3b"
    l1_fallback_model: str = ""

    # ── AutoDL (L3) — OpenAI-compatible vision API ──
    autodl_token: str = ""
    autodl_base_url: str = "https://www.autodl.art/api/v1"
    l3_model: str = "gpt-5.4-mini"

    # ── Cache ──
    cache_ttl_days: int = 30
    cache_dir: str = "data/vlm_cache"

    # ── Timeouts (seconds) ──
    timeout_seconds: int = 180
    l1_timeout: int = 120
    l2_timeout: int = 180
    l3_timeout: int = 120

    # ── Degradation ──
    use_degradation: bool = True

    # ── Image preprocessing ──
    target_width: int = 1920
    target_height: int = 882
    enable_multi_scale_crops: bool = True

    @property
    def target_resolution(self) -> tuple[int, int]:
        return (self.target_width, self.target_height)


_settings: VLMSettings | None = None


def get_settings() -> VLMSettings:
    """Return the module-level settings singleton (lazy-loaded)."""
    global _settings
    if _settings is None:
        _settings = VLMSettings()
    return _settings
