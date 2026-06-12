"""Configuration package: typed, validated settings loaded from ``.env``."""

from app.config.settings import EmbeddingBackend, Settings, get_settings

__all__ = ["EmbeddingBackend", "Settings", "get_settings"]
