"""Anthropic Claude API utility functions."""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import anthropic

from .ai_observability import summarize_messages

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Valid Anthropic Claude models
VALID_ANTHROPIC_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
}
ANTHROPIC_MAX_RETRIES = 3


class AnthropicUsage:
    """Usage statistics wrapper with both Anthropic and OpenAI-compatible attribute names."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        # OpenAI-compatible aliases used by update_token_usage
        self.prompt_tokens = input_tokens
        self.completion_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class AnthropicMessage:
    """Message wrapper compatible with OpenAI's response.choices[0].message."""

    def __init__(self, content: str, parsed: Any = None):
        self.content = content
        self.parsed = parsed


class AnthropicChoice:
    """Choice wrapper compatible with OpenAI's response.choices[0]."""

    def __init__(self, message: AnthropicMessage):
        self.message = message


class AnthropicResponseWrapper:
    """
    Response wrapper that mimics OpenAI's response format so that the same
    update_token_usage() and response-parsing logic works for both providers.
    """

    def __init__(self, content: str, parsed: Any, usage: AnthropicUsage, model: str):
        self.choices = [
            AnthropicChoice(AnthropicMessage(content=content, parsed=parsed))
        ]
        self.usage = usage
        self.model = model


def _is_retryable_anthropic_error(error: Exception) -> bool:
    """Return True for likely transient Anthropic failures."""
    class_name = error.__class__.__name__
    retryable_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "OverloadedError",
    }
    if class_name in retryable_names:
        return True

    message = str(error).lower()
    transient_indicators = [
        "timed out",
        "timeout",
        "rate limit",
        "overloaded",
        "temporarily unavailable",
        "connection reset",
        "503",
        "504",
        "502",
    ]
    return any(indicator in message for indicator in transient_indicators)


def is_retryable_anthropic_error(error: Exception) -> bool:
    """Public wrapper used by callers to detect transient Anthropic errors."""
    return _is_retryable_anthropic_error(error)


def _classify_anthropic_error(error: Exception) -> Dict[str, str]:
    """Classify Anthropic failures for warning/error semantics."""
    if _is_retryable_anthropic_error(error):
        return {"severity": "recoverable", "reason": "transient_api_error"}

    class_name = error.__class__.__name__
    critical_map = {
        "BadRequestError": "invalid_request",
        "AuthenticationError": "authentication_error",
        "PermissionDeniedError": "permission_denied",
        "NotFoundError": "not_found",
        "UnprocessableEntityError": "unprocessable_request",
    }
    if class_name in critical_map:
        return {"severity": "critical", "reason": critical_map[class_name]}
    return {"severity": "critical", "reason": "unexpected_error"}


def _create_with_retry(
    client: anthropic.Anthropic,
    params: Dict[str, Any],
    model: str,
    operation: str,
) -> Any:
    """Retry Anthropic calls when failures look transient."""
    for attempt in range(1, ANTHROPIC_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.warning(
                    "Retrying Anthropic call",
                    extra={
                        "operation": operation,
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": ANTHROPIC_MAX_RETRIES,
                    },
                )
            return client.messages.create(**params)
        except Exception as error:
            classification = _classify_anthropic_error(error)
            retryable = (
                _is_retryable_anthropic_error(error) and attempt < ANTHROPIC_MAX_RETRIES
            )
            if retryable:
                wait_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Recoverable Anthropic error; retrying",
                    extra={
                        "operation": operation,
                        "model": model,
                        "reason": classification["reason"],
                        "attempt": attempt,
                        "retry_in_s": wait_seconds,
                    },
                )
                time.sleep(wait_seconds)
                continue
            log_method = (
                logger.error
                if classification["severity"] == "critical"
                else logger.warning
            )
            log_method(
                "Anthropic call failed",
                extra={
                    "operation": operation,
                    "model": model,
                    "severity": classification["severity"],
                    "reason": classification["reason"],
                    "error": str(error),
                },
            )
            raise


def _convert_messages(messages: List[Dict[str, Any]]) -> tuple:
    """
    Convert OpenAI-format messages to Anthropic format.

    Returns:
        Tuple of (system_prompt: str, anthropic_messages: list)
    """
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_prompt = content if isinstance(content, str) else str(content)

        elif role in ("user", "assistant"):
            if isinstance(content, list):
                # Multimodal content (text + images)
                anthropic_content = []
                for item in content:
                    if item["type"] == "text":
                        anthropic_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            # data:image/png;base64,<data>
                            header, base64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                }
                            )
                anthropic_messages.append({"role": role, "content": anthropic_content})
            else:
                anthropic_messages.append({"role": role, "content": content})

    return system_prompt, anthropic_messages


def call_anthropic(
    client: anthropic.Anthropic,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Type[T]] = None,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    operation: str = "unspecified_operation",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> AnthropicResponseWrapper:
    """
    Call Anthropic Claude API, optionally enforcing structured output via tool use.

    Args:
        client: Anthropic client instance.
        model: Claude model name (e.g. "claude-sonnet-4-6").
        messages: OpenAI-format messages (converted internally to Anthropic format).
        response_format: Optional Pydantic model class; when provided, tool use is
            used to enforce a structured JSON response that is parsed into this type.
        temperature: Sampling temperature (0–1).
        max_tokens: Maximum tokens in the response.
        operation: Operation name for structured logs.
        request_metadata: Additional metadata for structured logs.

    Returns:
        AnthropicResponseWrapper compatible with OpenAI response format.
    """
    system_prompt, anthropic_messages = _convert_messages(messages)

    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
        "temperature": temperature,
    }
    if system_prompt:
        params["system"] = system_prompt

    logger.info(
        "Anthropic request prepared",
        extra={
            "operation": operation,
            "model": model,
            "message_summary": summarize_messages(messages),
            "request_metadata": request_metadata or {},
        },
    )

    if response_format is not None:
        # Enforce structured output via tool use
        schema = response_format.model_json_schema()
        tool_name = "structured_output"
        params["tools"] = [
            {
                "name": tool_name,
                "description": f"Return structured data as {response_format.__name__}",
                "input_schema": schema,
            }
        ]
        params["tool_choice"] = {"type": "tool", "name": tool_name}

        logger.info(
            f"Calling Anthropic API with structured output ({response_format.__name__}) "
            f"using model: {model}"
        )
        response = _create_with_retry(client, params, model, operation)

        usage = AnthropicUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        parsed = None
        raw_text = ""
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                try:
                    parsed = response_format(**block.input)
                except Exception as e:
                    logger.error(
                        f"Failed to instantiate {response_format.__name__} from tool input: {e}"
                    )
            elif block.type == "text":
                raw_text = block.text

        if parsed is None and raw_text:
            logger.warning(
                "Tool use result not found; attempting JSON parse from text response"
            )
            try:
                parsed = response_format(**json.loads(raw_text))
            except Exception as e:
                logger.error(
                    f"Failed to parse text response as {response_format.__name__}: {e}"
                )

        return AnthropicResponseWrapper(
            content=raw_text,
            parsed=parsed,
            usage=usage,
            model=response.model,
        )

    else:
        # Plain text response (JSON expected in prompt instructions)
        logger.info(f"Calling Anthropic API with model: {model}")
        response = _create_with_retry(client, params, model, operation)

        usage = AnthropicUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return AnthropicResponseWrapper(
            content=content,
            parsed=None,
            usage=usage,
            model=response.model,
        )


def validate_anthropic_model(model: str) -> None:
    """Raise ValueError if the model is not a recognised Claude model."""
    if model not in VALID_ANTHROPIC_MODELS:
        raise ValueError(
            f"Invalid Anthropic model: '{model}'. "
            f"Must be one of {sorted(VALID_ANTHROPIC_MODELS)}"
        )
