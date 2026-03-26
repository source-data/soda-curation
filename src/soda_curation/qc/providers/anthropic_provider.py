"""Anthropic implementation of the QC provider contract."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import anthropic

from ...pipeline.anthropic_utils import call_anthropic
from .base import BaseQCProvider, QCProviderRequest, QCProviderResponse

logger = logging.getLogger(__name__)


class AnthropicQCProvider(BaseQCProvider):
    """QC provider implementation backed by Anthropic Claude APIs."""

    provider_name = "anthropic"
    supports_agentic = True

    def __init__(self, client: Optional[anthropic.Anthropic] = None):
        self._init_error: Optional[Exception] = None
        if client is not None:
            self.client = client
            return
        try:
            self.client = anthropic.Anthropic()
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self.client = None
            self._init_error = exc

    def generate(self, request: QCProviderRequest) -> QCProviderResponse:
        if self.client is None:
            raise RuntimeError(
                "Anthropic client is unavailable. "
                f"Initialization failed with: {self._init_error}"
            )
        agentic_requested = bool(request.agentic_enabled or request.model_config)
        response = call_anthropic(
            client=self.client,
            model=request.model,
            messages=request.messages,
            response_format=request.response_type,
            temperature=request.prompt_config.get("temperature", 0.1),
            max_tokens=request.prompt_config.get("max_tokens", 4096),
            operation=request.operation,
            request_metadata=request.context,
            model_config=request.model_config if agentic_requested else None,
        )
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        content = message.content or ""
        if (
            parsed is None
            and isinstance(content, str)
            and request.response_type is not None
        ):
            try:
                parsed = request.response_type(**json.loads(content))
            except Exception:
                parsed = None

        usage = {
            "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0)),
            "completion_tokens": int(getattr(response.usage, "completion_tokens", 0)),
            "total_tokens": int(getattr(response.usage, "total_tokens", 0)),
        }
        return QCProviderResponse(
            content=content,
            parsed=parsed,
            model=getattr(response, "model", request.model),
            usage=usage,
            metadata={
                "api_mode": "messages",
                "agentic_requested": agentic_requested,
                "tool_config_present": bool(request.model_config.get("tools")),
            },
        )
