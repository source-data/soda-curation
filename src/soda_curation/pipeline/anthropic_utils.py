"""Anthropic Claude API utility functions."""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

import anthropic

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Valid Anthropic Claude models
VALID_ANTHROPIC_MODELS = {
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
}


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
        response = client.messages.create(**params)

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
        response = client.messages.create(**params)

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
