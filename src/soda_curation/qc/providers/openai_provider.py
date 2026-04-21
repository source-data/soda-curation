"""OpenAI implementation of the QC provider contract."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import openai

from ...pipeline.openai_utils import call_openai_with_fallback, validate_model_config
from .base import BaseQCProvider, QCProviderRequest, QCProviderResponse

logger = logging.getLogger(__name__)

# Compatibility shim for environments/tests using older OpenAI SDKs.
if not hasattr(openai, "OpenAI"):
    if hasattr(openai, "Client"):
        openai.OpenAI = openai.Client
    else:

        class _OpenAIClientMissing:  # pragma: no cover - defensive fallback
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "OpenAI client class is unavailable in this environment"
                )

        openai.OpenAI = _OpenAIClientMissing

_OPENAI_AGENTIC_CONFIG_KEYS = {
    "tools",
    "tool_choice",
    "max_tool_calls",
    "include",
    "reasoning",
    "max_output_tokens",
}


class OpenAIQCProvider(BaseQCProvider):
    """QC provider implementation backed by OpenAI APIs."""

    provider_name = "openai"
    supports_agentic = True

    def __init__(self, client: Optional[openai.OpenAI] = None):
        self._init_error: Optional[Exception] = None
        if client is not None:
            self.client = client
            return
        try:
            self.client = openai.OpenAI()
        except Exception as exc:  # pragma: no cover - depends on runtime env
            self.client = None
            self._init_error = exc

    def generate(self, request: QCProviderRequest) -> QCProviderResponse:
        if self.client is None:
            raise RuntimeError(
                "OpenAI client is unavailable. "
                f"Initialization failed with: {self._init_error}"
            )
        agentic_requested = request.agentic_enabled or bool(request.model_config)
        has_agentic_config = any(
            key in request.model_config for key in _OPENAI_AGENTIC_CONFIG_KEYS
        )
        if agentic_requested and (has_agentic_config or request.agentic_enabled):
            return self._generate_agentic(request)
        return self._generate_standard(request)

    def _generate_standard(self, request: QCProviderRequest) -> QCProviderResponse:
        validate_model_config(request.model, request.prompt_config)
        response = call_openai_with_fallback(
            client=self.client,
            model=request.model,
            messages=request.messages,
            response_format=request.response_type,
            temperature=request.prompt_config.get("temperature", 0.1),
            top_p=request.prompt_config.get("top_p", 1.0),
            frequency_penalty=request.prompt_config.get("frequency_penalty", 0.0),
            presence_penalty=request.prompt_config.get("presence_penalty", 0.0),
            max_tokens=request.prompt_config.get("max_tokens", 2048),
            json_mode=request.prompt_config.get("json_mode", True),
            fallback_model=request.prompt_config.get("fallback_model", "gpt-5"),
            operation=request.operation,
            request_metadata=request.context,
        )
        message = response.choices[0].message
        usage = _normalize_usage(getattr(response, "usage", None))
        return QCProviderResponse(
            content=message.content or "",
            parsed=getattr(message, "parsed", None),
            model=getattr(response, "model", request.model),
            usage=usage,
            metadata={"api_mode": "chat.completions"},
        )

    def _generate_agentic(self, request: QCProviderRequest) -> QCProviderResponse:
        logger.info(
            "Running QC check in OpenAI agentic mode",
            extra={
                "operation": request.operation,
                "provider": self.provider_name,
                "model": request.model,
                "model_config_keys": sorted(list(request.model_config.keys())),
            },
        )
        system_prompt, model_input = _to_openai_responses_input(request.messages)
        call_kwargs: Dict[str, Any] = {
            "model": request.model,
            "input": model_input,
        }
        if system_prompt:
            call_kwargs["instructions"] = system_prompt
        if request.context:
            call_kwargs["metadata"] = _sanitize_metadata_for_model(request.context)
        if request.response_type is not None:
            schema = request.response_type.model_json_schema()
            call_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "qc_output",
                    "schema": schema,
                }
            }

        for key in _OPENAI_AGENTIC_CONFIG_KEYS:
            if key in request.model_config:
                call_kwargs[key] = request.model_config[key]

        response = self.client.responses.create(**call_kwargs)
        raw_text = _extract_output_text_from_response(response)
        parsed = None
        if request.response_type is not None and raw_text:
            parsed = request.response_type(**json.loads(raw_text))

        usage = _normalize_usage(getattr(response, "usage", None))
        return QCProviderResponse(
            content=raw_text,
            parsed=parsed,
            model=getattr(response, "model", request.model),
            usage=usage,
            metadata={"api_mode": "responses", "agentic": True},
        )


def _normalize_usage(raw_usage: Any) -> Dict[str, int]:
    """Normalize provider usage objects to prompt/completion/total token fields."""
    if raw_usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = getattr(raw_usage, "prompt_tokens", None)
    if prompt_tokens is None:
        prompt_tokens = getattr(raw_usage, "input_tokens", 0) or 0

    completion_tokens = getattr(raw_usage, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(raw_usage, "output_tokens", 0) or 0

    total_tokens = getattr(raw_usage, "total_tokens", None)
    if total_tokens is None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
    }


def _to_openai_responses_input(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Convert chat-completions style messages to Responses API input payload."""
    system_chunks: List[str] = []
    converted: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                system_chunks.append(content)
            continue

        parts: List[Dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append({"type": "input_text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url:
                        parts.append({"type": "input_image", "image_url": image_url})
        if not parts:
            continue
        converted.append({"role": role, "content": parts})

    return "\n".join(system_chunks), converted


def _sanitize_metadata_for_model(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Convert metadata values to strings accepted by OpenAI Responses API."""
    sanitized: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, sort_keys=True, default=str)
    return sanitized


def _extract_output_text_from_response(response: Any) -> str:
    """Extract assistant text from an OpenAI Responses API object."""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text:
        return output_text

    collected: List[str] = []
    for item in getattr(response, "output", []) or []:
        for chunk in getattr(item, "content", []) or []:
            chunk_type = getattr(chunk, "type", None)
            if chunk_type in {"output_text", "text"}:
                text = getattr(chunk, "text", "")
                if text:
                    collected.append(text)
    return "\n".join(collected).strip()
