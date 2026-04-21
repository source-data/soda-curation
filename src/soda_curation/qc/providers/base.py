"""Provider contract for QC model calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class QCProviderRequest:
    """Normalized request passed to provider implementations."""

    model: str
    messages: List[Dict[str, Any]]
    prompt_config: Dict[str, Any]
    response_type: Optional[Type[Any]]
    operation: str
    context: Dict[str, Any] = field(default_factory=dict)
    agentic_enabled: bool = False
    model_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCProviderResponse:
    """Normalized provider response consumed by ModelAPI."""

    content: str
    parsed: Optional[Any]
    model: str
    usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseQCProvider(ABC):
    """Abstract provider interface for QC checks."""

    provider_name: str = "unknown"
    supports_agentic: bool = False

    @abstractmethod
    def generate(self, request: QCProviderRequest) -> QCProviderResponse:
        """Generate a response for a QC request."""
