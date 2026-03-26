"""Gemini implementation of the QC provider contract."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import BaseQCProvider, QCProviderRequest, QCProviderResponse

logger = logging.getLogger(__name__)

try:
    from google import genai
except Exception:  # pragma: no cover - import error handled at runtime
    genai = None


class GeminiQCProvider(BaseQCProvider):
    """QC provider implementation backed by Google Gemini API."""

    provider_name = "gemini"
    supports_agentic = False

    def __init__(self, client: Optional[Any] = None):
        self._init_error: Optional[Exception] = None
        if client is not None:
            self.client = client
            return
        if genai is None:
            raise ImportError(
                "google-genai is not installed. Install with `poetry add google-genai`."
            )
        try:
            self.client = genai.Client()
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self.client = None
            self._init_error = exc

    def generate(self, request: QCProviderRequest) -> QCProviderResponse:
        if self.client is None:
            raise RuntimeError(
                "Gemini client is unavailable. "
                f"Initialization failed with: {self._init_error}"
            )
        if request.agentic_enabled or request.model_config:
            logger.warning(
                "agentic_not_supported: Agentic mode requested for Gemini QC provider; "
                "continuing with non-agentic call",
                extra={
                    "operation": request.operation,
                    "provider": self.provider_name,
                    "model": request.model,
                    "reason": "agentic_not_supported",
                },
            )

        generation_config = _build_generation_config(request)
        contents = _to_gemini_contents(request.messages)
        response = self.client.models.generate_content(
            model=request.model,
            contents=contents,
            config=generation_config,
        )

        content = _extract_gemini_text(response)
        parsed = None
        if request.response_type is not None and content:
            parsed = request.response_type(**json.loads(content))

        usage_metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage_metadata, "prompt_token_count", 0))
        completion_tokens = int(getattr(usage_metadata, "candidates_token_count", 0))
        total_tokens = int(
            getattr(
                usage_metadata, "total_token_count", prompt_tokens + completion_tokens
            )
        )

        return QCProviderResponse(
            content=content,
            parsed=parsed,
            model=getattr(response, "model_version", request.model),
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata={"api_mode": "generate_content"},
        )


def _build_generation_config(request: QCProviderRequest) -> Dict[str, Any]:
    """Build Gemini generation config from normalized request."""
    config: Dict[str, Any] = {
        "temperature": request.prompt_config.get("temperature", 0.1),
        "top_p": request.prompt_config.get("top_p", 1.0),
        "max_output_tokens": request.prompt_config.get("max_tokens", 2048),
    }
    system_chunks = [
        m.get("content", "")
        for m in request.messages
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    ]
    if system_chunks:
        config["system_instruction"] = "\n".join(system_chunks)

    if request.response_type is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = request.response_type.model_json_schema()
    return config


def _to_gemini_contents(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style messages to Gemini content format."""
    converted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            continue
        content = message.get("content")
        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    data_url = item.get("image_url", {}).get("url", "")
                    inline_data = _parse_data_url(data_url)
                    if inline_data:
                        parts.append({"inline_data": inline_data})
        if not parts:
            continue
        converted.append(
            {"role": "model" if role == "assistant" else "user", "parts": parts}
        )
    return converted


def _parse_data_url(data_url: str) -> Optional[Dict[str, str]]:
    """Parse data URL into Gemini inline_data dict."""
    if not data_url.startswith("data:") or "," not in data_url:
        return None
    header, payload = data_url.split(",", 1)
    mime_type = "image/png"
    try:
        mime_type = header.split(":")[1].split(";")[0]
    except Exception:
        pass
    return {"mime_type": mime_type, "data": payload}


def _extract_gemini_text(response: Any) -> str:
    """Extract text from Gemini response object."""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    chunks: List[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "\n".join(chunks).strip()
