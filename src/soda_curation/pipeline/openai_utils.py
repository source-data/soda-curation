"""OpenAI utility functions with GPT-5 fallback support."""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import openai
from openai import OpenAIError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# GPT-5 model identifier
GPT5_MODEL = "gpt-5"

# Models that don't support additional parameters (temperature, top_p, etc.)
MODELS_WITHOUT_PARAMETERS = {GPT5_MODEL}


def is_context_length_error(error: Exception) -> bool:
    """
    Check if the error is related to context length limits.

    Args:
        error: The exception to check

    Returns:
        True if the error is related to context length limits
    """
    error_message = str(error).lower()
    context_indicators = [
        "context length",
        "maximum context length",
        "token limit",
        "too long",
        "context window",
        "maximum tokens",
        "input too long",
        "length limit",
        "length was reached",
    ]
    return any(indicator in error_message for indicator in context_indicators)


def prepare_model_params(
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Union[Type[T], Dict[str, Any]]] = None,
    temperature: float = 0.1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Prepare model parameters, excluding unsupported parameters for certain models.

    Args:
        model: The model name
        messages: List of messages for the API call
        response_format: Response format (Pydantic model or dict)
        temperature: Temperature parameter
        top_p: Top-p parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter
        max_tokens: Maximum tokens parameter
        json_mode: Whether to use JSON mode

    Returns:
        Dictionary of parameters for the API call
    """
    params = {
        "model": model,
        "messages": messages,
    }

    # Add response format
    if response_format:
        params["response_format"] = response_format
    elif json_mode:
        params["response_format"] = {"type": "json_object"}

    # Only add parameters that are supported by the model
    if model not in MODELS_WITHOUT_PARAMETERS:
        params.update(
            {
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
            }
        )
    else:
        logger.info(
            f"Model {model} does not support additional parameters, using basic configuration"
        )

    return params


def call_openai_with_fallback(
    client: openai.OpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Union[Type[T], Dict[str, Any]]] = None,
    temperature: float = 0.1,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = True,
    fallback_model: str = GPT5_MODEL,
) -> Any:
    """
    Call OpenAI API with automatic fallback to GPT-5 on context length errors.

    Args:
        client: OpenAI client instance
        model: Primary model to use
        messages: List of messages for the API call
        response_format: Response format (Pydantic model or dict)
        temperature: Temperature parameter
        top_p: Top-p parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter
        max_tokens: Maximum tokens parameter
        json_mode: Whether to use JSON mode
        fallback_model: Model to use as fallback (default: gpt-5)

    Returns:
        Response from OpenAI API

    Raises:
        OpenAIError: If both primary and fallback models fail
    """
    # Prepare parameters for the primary model
    params = prepare_model_params(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )

    try:
        logger.info(f"Attempting API call with model: {model}")
        response = client.beta.chat.completions.parse(**params)
        logger.info(f"Successfully completed API call with model: {model}")
        return response

    except OpenAIError as e:
        # Check if this is a context length error
        if is_context_length_error(e):
            logger.warning(f"Context length error with model {model}: {str(e)}")
            logger.info(f"Attempting fallback to model: {fallback_model}")

            # Prepare parameters for the fallback model
            fallback_params = prepare_model_params(
                model=fallback_model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )

            try:
                response = client.beta.chat.completions.parse(**fallback_params)
                logger.info(
                    f"Successfully completed API call with fallback model: {fallback_model}"
                )
                return response

            except OpenAIError as fallback_error:
                logger.error(
                    f"Fallback model {fallback_model} also failed: {str(fallback_error)}"
                )
                raise fallback_error
        else:
            # Re-raise non-context-length errors
            logger.error(f"Non-context-length error with model {model}: {str(e)}")
            raise e

    except Exception as e:
        logger.error(f"Unexpected error with model {model}: {str(e)}")
        raise e


def validate_model_config(model: str, config: Dict[str, Any]) -> None:
    """
    Validate model configuration, checking for GPT-5 specific requirements.

    Args:
        model: The model name
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid for the model
    """
    if model == GPT5_MODEL:
        # GPT-5 doesn't support additional parameters
        unsupported_params = []
        for param in [
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
        ]:
            if param in config and config[param] is not None:
                unsupported_params.append(param)

        if unsupported_params:
            logger.warning(
                f"Model {GPT5_MODEL} does not support parameters: {unsupported_params}. "
                "These will be ignored during API calls."
            )
    else:
        # Standard validation for other models
        if not 0 <= config.get("temperature", 0.1) <= 2:
            raise ValueError(
                f"Temperature must be between 0 and 2, value: `{config.get('temperature', 0.1)}`"
            )
        if not 0 <= config.get("top_p", 1.0) <= 1:
            raise ValueError(
                f"Top_p must be between 0 and 1, value: `{config.get('top_p', 1.0)}`"
            )
        if "frequency_penalty" in config and not -2 <= config["frequency_penalty"] <= 2:
            raise ValueError(
                f"Frequency penalty must be between -2 and 2, value: `{config.get('frequency_penalty', 0.)}`"
            )
        if "presence_penalty" in config and not -2 <= config["presence_penalty"] <= 2:
            raise ValueError(
                f"Presence penalty must be between -2 and 2, value: `{config.get('presence_penalty', 0.)}`"
            )
