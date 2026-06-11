"""Factory for QC provider implementations."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .anthropic_provider import AnthropicQCProvider
from .base import BaseQCProvider
from .openai_provider import OpenAIQCProvider


def build_qc_provider(
    provider_name: str,
    clients: Optional[Dict[str, Any]] = None,
) -> BaseQCProvider:
    """Build a concrete provider for the requested AI backend."""
    clients = clients or {}
    normalized = provider_name.lower()
    if normalized == "openai":
        return OpenAIQCProvider(client=clients.get("openai"))
    if normalized == "anthropic":
        return AnthropicQCProvider(client=clients.get("anthropic"))
    raise ValueError(
        f"Unsupported QC ai_provider '{provider_name}'. "
        "Expected one of ['openai', 'anthropic']."
    )
