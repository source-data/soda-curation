"""OpenAI utility functions with GPT-5 fallback support."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import openai
from openai import OpenAIError

from .ai_observability import summarize_messages

try:
    import tiktoken
except ImportError:
    tiktoken = None

logger = logging.getLogger(__name__)

T = TypeVar("T")
_TIKTOKEN_FALLBACK_WARNED_MODELS: set[str] = set()

# GPT-5 model identifier
GPT5_MODEL = "gpt-5"

# Model name prefixes that don't support additional parameters (temperature, top_p, etc.)
MODELS_WITHOUT_PARAMETERS_PREFIXES = {"gpt-5"}
# Backward-compatible symbol expected by existing tests/importers.
MODELS_WITHOUT_PARAMETERS = {GPT5_MODEL}

# Model-specific token limits (input tokens)
# Using conservative limits to account for response tokens and safety margin
MODEL_TOKEN_LIMITS = {
    "gpt-4o": 120000,  # Actual: 128k, using 120k for safety
    "gpt-4o-mini": 120000,  # Actual: 128k, using 120k for safety
    "gpt-4o-2024-08-06": 120000,
    "gpt-4o-mini-2024-07-18": 120000,
    "gpt-5": 270000,  # Actual: 272k, using 270k for safety (error showed 272k limit)
}

# Default token limit for unknown models
DEFAULT_TOKEN_LIMIT = 120000
OPENAI_MAX_RETRIES = 3


def _fallback_encoding_for_model(model: str) -> str:
    """Return best-effort encoding fallback when tiktoken has no model mapping."""
    modern_o200k_prefixes = (
        "gpt-5",
        "gpt-4o",
        "gpt-4.1",
        "gpt-4.5",
        "o1",
        "o3",
        "o4-mini",
    )
    if any(
        model == prefix or model.startswith(prefix + "-")
        for prefix in modern_o200k_prefixes
    ):
        return "o200k_base"
    return "cl100k_base"


def _is_model_without_parameters(model: str) -> bool:
    """Return True if model belongs to a family with restricted parameters."""
    return any(
        model == prefix
        or model.startswith(prefix + "-")
        or model.startswith(prefix + "/")
        for prefix in MODELS_WITHOUT_PARAMETERS_PREFIXES
    )


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
        "maximum context",
        "token limit",
        "too long",
        "context window",
        "context_window_exceeded",
        "maximum tokens",
        "input too long",
        "length limit",
        "context_length_exceeded",
    ]
    return any(indicator in error_message for indicator in context_indicators)


def extract_context_length_error_details(error: Exception) -> Dict[str, int]:
    """Extract token counts from provider error messages when available."""
    message = str(error)
    details: Dict[str, int] = {}

    patterns = {
        "api_max_context_tokens": r"maximum context length is\s*([0-9,]+)\s*tokens",
        "api_requested_tokens": r"(?:resulted in|you requested)\s*([0-9,]+)\s*tokens",
        "api_prompt_tokens": r"([0-9,]+)\s*in your prompt",
        "api_completion_tokens": r"([0-9,]+)\s*for completion",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            details[key] = int(match.group(1).replace(",", ""))
        except ValueError:
            continue

    return details


def is_output_length_error(error: Exception) -> bool:
    """
    Check if the model stopped due to output token length limits.

    This is different from input context overflow and should not trigger a
    model-context fallback.
    """
    error_message = str(error).lower()
    output_length_indicators = [
        "lengthfinishreasonerror",
        "finish_reason",
        "length limit was reached",
        "response was truncated",
    ]
    return any(indicator in error_message for indicator in output_length_indicators)


def is_safety_block_error(error: Exception) -> bool:
    """
    Check if the error is a content safety block (e.g. gpt-5 refusing
    sensitive biological content with HTTP 400 / invalid_prompt).

    Args:
        error: The exception to check

    Returns:
        True if the error is a safety/content-policy refusal
    """
    error_message = str(error).lower()
    safety_indicators = [
        "invalid_prompt",
        "safety reasons",
        "limited access to this content",
        "content_policy_violation",
        "content policy",
    ]
    return any(indicator in error_message for indicator in safety_indicators)


def is_retryable_openai_error(error: Exception) -> bool:
    """Return True when the OpenAI error is likely transient."""
    retryable_error_names = {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }
    error_name = error.__class__.__name__
    if error_name in retryable_error_names:
        return True

    message = str(error).lower()
    transient_indicators = [
        "timed out",
        "timeout",
        "rate limit",
        "temporarily unavailable",
        "connection reset",
        "503",
        "504",
        "502",
    ]
    return any(indicator in message for indicator in transient_indicators)


def classify_openai_error(error: Exception) -> Dict[str, str]:
    """Classify OpenAI errors for consistent warning/error behavior."""
    if is_context_length_error(error):
        return {"severity": "recoverable", "reason": "context_length"}
    if is_output_length_error(error):
        return {"severity": "recoverable", "reason": "output_length_limit"}
    if is_safety_block_error(error):
        return {"severity": "recoverable", "reason": "safety_block"}
    if is_retryable_openai_error(error):
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


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string for a given model.

    Args:
        text: The text to count tokens for
        model: The model name to use for tokenization

    Returns:
        Number of tokens in the text
    """
    if tiktoken is None or not hasattr(tiktoken, "encoding_for_model"):
        # Fallback: rough estimation (1 token ≈ 4 characters for English text)
        logger.warning(
            "tiktoken unavailable or incompatible, using rough token estimation. "
            "Install tiktoken for accurate token counting: pip install tiktoken"
        )
        return len(text) // 4

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        if not hasattr(tiktoken, "get_encoding"):
            logger.warning(
                "tiktoken get_encoding unavailable, using rough token estimation"
            )
            return len(text) // 4
        fallback_encoding = _fallback_encoding_for_model(model)
        if model not in _TIKTOKEN_FALLBACK_WARNED_MODELS:
            logger.warning(
                f"Model {model} not found in tiktoken, using {fallback_encoding} encoding"
            )
            _TIKTOKEN_FALLBACK_WARNED_MODELS.add(model)
        encoding = tiktoken.get_encoding(fallback_encoding)

    return len(encoding.encode(text))


def count_messages_tokens(messages: List[Dict[str, Any]], model: str = "gpt-4o") -> int:
    """
    Count the total number of tokens in a list of messages.

    This accounts for the message structure overhead as per OpenAI's token counting.

    Args:
        messages: List of message dictionaries
        model: The model name to use for tokenization

    Returns:
        Total number of tokens including message overhead
    """
    num_tokens = 0

    # Different models have different overhead per message
    tokens_per_message = (
        3  # Every message follows <|start|>{role/name}\n{content}<|end|>\n
    )
    tokens_per_name = 1  # If there's a name field

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += count_tokens(value, model)
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # Every reply is primed with <|start|>assistant<|message|>

    return num_tokens


def get_token_limit(model: str) -> int:
    """
    Get the token limit for a given model.

    Args:
        model: The model name

    Returns:
        Token limit for the model
    """
    return MODEL_TOKEN_LIMITS.get(model, DEFAULT_TOKEN_LIMIT)


def chunk_file_list(file_list_str: str, chunk_size: int, model: str) -> List[str]:
    """
    Split a file list string into chunks that fit within the token limit.

    Args:
        file_list_str: The file list as a newline-separated string
        chunk_size: Maximum number of tokens per chunk
        model: The model name for token counting

    Returns:
        List of file list chunks
    """
    files = file_list_str.split("\n")
    if not files:
        return [file_list_str]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for file in files:
        file_tokens = count_tokens(file + "\n", model)

        # If a single file exceeds the chunk size, we still need to include it
        if file_tokens > chunk_size:
            logger.warning(
                f"Single file path exceeds chunk size: {file[:100]}... ({file_tokens} tokens)"
            )
            # Save current chunk if it has content
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            # Add the large file as its own chunk
            chunks.append(file)
            continue

        # Check if adding this file would exceed the limit
        if current_tokens + file_tokens > chunk_size:
            # Save current chunk and start a new one
            chunks.append("\n".join(current_chunk))
            current_chunk = [file]
            current_tokens = file_tokens
        else:
            # Add to current chunk
            current_chunk.append(file)
            current_tokens += file_tokens

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def create_chunked_messages(
    messages: List[Dict[str, Any]], model: str, token_limit: int
) -> List[List[Dict[str, Any]]]:
    """
    Create multiple message lists by chunking the content when it exceeds token limits.

    This function assumes the last user message contains a file list that can be chunked.
    All other messages are preserved in each chunk.

    Args:
        messages: Original list of messages
        model: The model name for token counting
        token_limit: Maximum token limit

    Returns:
        List of message lists, each within the token limit
    """
    # Find the last user message (which typically contains the file list)
    user_message_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            user_message_idx = i
            break

    if user_message_idx is None:
        logger.warning("No user message found to chunk")
        return [messages]

    # Get the user message content
    user_content = messages[user_message_idx]["content"]

    # Try to find the file list section
    # Look for patterns like "file_list:", "Files:", etc.
    file_list_markers = [
        "\nFile list:\n",
        "\nfile_list:\n",
        "\nFiles:\n",
        "\nfiles:\n",
        "\n\nFile list:\n",
        "\n\nfile_list:\n",
    ]

    file_list_start = -1
    for marker in file_list_markers:
        idx = user_content.find(marker)
        if idx != -1:
            file_list_start = idx + len(marker)
            break

    if file_list_start == -1:
        # If no marker found, assume the entire user message can be chunked by lines
        logger.warning(
            "No file list marker found, will try to chunk entire user message"
        )
        # Split by double newline to separate sections
        parts = user_content.split("\n\n")
        if len(parts) > 1:
            # Assume the last section is the file list
            prefix = "\n\n".join(parts[:-1]) + "\n\n"
            file_list = parts[-1]
        else:
            # No clear structure, try to split by lines
            lines = user_content.split("\n")
            if len(lines) > 10:
                # Keep first few lines as context, chunk the rest
                prefix = "\n".join(lines[:5]) + "\n"
                file_list = "\n".join(lines[5:])
            else:
                # Too small to chunk meaningfully
                logger.error("User message too small to chunk effectively")
                return [messages]
    else:
        # Extract prefix and file list
        prefix = user_content[:file_list_start]
        file_list = user_content[file_list_start:]

    # Count tokens for fixed parts
    fixed_messages = messages[:user_message_idx] + messages[user_message_idx + 1 :]
    fixed_tokens = count_messages_tokens(fixed_messages, model)
    prefix_tokens = count_tokens(prefix, model)

    # Calculate available tokens for file list
    # Reserve some tokens for response (already accounted for in limits, but be extra safe)
    available_tokens = (
        token_limit - fixed_tokens - prefix_tokens - 100
    )  # 100 token buffer

    if available_tokens <= 0:
        logger.error(
            f"Fixed message content ({fixed_tokens + prefix_tokens} tokens) "
            f"exceeds token limit ({token_limit}). Cannot chunk effectively."
        )
        return [messages]

    # Chunk the file list
    file_list_chunks = chunk_file_list(file_list, available_tokens, model)

    logger.info(
        f"Chunking large request into {len(file_list_chunks)} chunks "
        f"(token limit: {token_limit}, available for files: {available_tokens})"
    )

    # Create message lists for each chunk
    chunked_message_lists = []
    for i, chunk in enumerate(file_list_chunks):
        # Reconstruct user message with this chunk
        chunked_user_content = prefix + chunk

        # Add chunk information to help the model understand this is part of a larger set
        if len(file_list_chunks) > 1:
            chunk_info = f"\n\n[Note: This is chunk {i+1} of {len(file_list_chunks)} total chunks. Please analyze only the files in this chunk.]"
            chunked_user_content += chunk_info

        # Create new message list
        new_messages = messages[:user_message_idx] + [
            {"role": "user", "content": chunked_user_content}
        ]

        chunked_message_lists.append(new_messages)

    return chunked_message_lists


def merge_pydantic_responses(
    responses: List[Any], response_format: Optional[Type[T]]
) -> Any:
    """
    Merge multiple Pydantic model responses into a single response.

    This function handles merging of common response formats used in the pipeline.

    Args:
        responses: List of response objects to merge
        response_format: The Pydantic model class used for responses

    Returns:
        Merged response object
    """
    if not responses:
        return None

    if len(responses) == 1:
        return responses[0]

    # Get the parsed content from each response
    parsed_objects = []
    for response in responses:
        if hasattr(response.choices[0].message, "parsed"):
            parsed_objects.append(response.choices[0].message.parsed)
        else:
            # Fallback for non-structured responses
            try:
                content = json.loads(response.choices[0].message.content)
                parsed_objects.append(content)
            except json.JSONDecodeError:
                logger.error("Failed to parse response content as JSON")
                parsed_objects.append(None)

    # Check if we're dealing with a Pydantic model that has assigned_files and not_assigned_files
    # This handles AsignedFilesList and similar models
    if response_format and parsed_objects:
        # Check if first parsed object has the structure we expect
        first_obj = next((obj for obj in parsed_objects if obj is not None), None)
        if (
            first_obj
            and hasattr(first_obj, "assigned_files")
            and hasattr(first_obj, "not_assigned_files")
        ):
            # Merge assigned files lists
            all_assigned_files = []
            all_not_assigned_files = []

            for obj in parsed_objects:
                if obj is not None:
                    if hasattr(obj, "assigned_files"):
                        all_assigned_files.extend(obj.assigned_files)
                    if hasattr(obj, "not_assigned_files"):
                        all_not_assigned_files.extend(obj.not_assigned_files)

            # Create merged response using the response format class
            merged = response_format(
                assigned_files=all_assigned_files,
                not_assigned_files=all_not_assigned_files,
            )

            # Update the first response object with merged data
            first_response = responses[0]
            first_response.choices[0].message.parsed = merged

            # Update token usage to sum all responses
            if hasattr(first_response, "usage"):
                total_prompt_tokens = sum(
                    r.usage.prompt_tokens for r in responses if hasattr(r, "usage")
                )
                total_completion_tokens = sum(
                    r.usage.completion_tokens for r in responses if hasattr(r, "usage")
                )
                total_tokens = sum(
                    r.usage.total_tokens for r in responses if hasattr(r, "usage")
                )

                first_response.usage.prompt_tokens = total_prompt_tokens
                first_response.usage.completion_tokens = total_completion_tokens
                first_response.usage.total_tokens = total_tokens

            return first_response

    # Generic merging for other types
    # Attempt to merge lists or combine dictionaries
    logger.warning(
        f"Generic merging for response format: {response_format}. "
        "Results may need custom handling."
    )

    # If all parsed objects are lists, concatenate them
    if all(isinstance(obj, list) for obj in parsed_objects if obj is not None):
        merged_list = []
        for obj in parsed_objects:
            if obj is not None:
                merged_list.extend(obj)

        # Update first response with merged list
        first_response = responses[0]
        first_response.choices[0].message.content = json.dumps(merged_list)
        return first_response

    # If all are dicts with similar keys, try to merge
    if all(isinstance(obj, dict) for obj in parsed_objects if obj is not None):
        merged_dict = {}
        for obj in parsed_objects:
            if obj is not None:
                for key, value in obj.items():
                    if key not in merged_dict:
                        merged_dict[key] = value
                    elif isinstance(value, list) and isinstance(merged_dict[key], list):
                        merged_dict[key].extend(value)
                    elif isinstance(value, dict) and isinstance(merged_dict[key], dict):
                        merged_dict[key].update(value)
                    else:
                        # Can't merge, keep first value
                        pass

        # Update first response with merged dict
        first_response = responses[0]
        first_response.choices[0].message.content = json.dumps(merged_dict)
        return first_response

    # Fallback: return first response
    logger.warning(
        "Could not determine how to merge responses, returning first response"
    )
    return responses[0]


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
    if not _is_model_without_parameters(model):
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


def _parse_with_retry(
    client: openai.OpenAI,
    params: Dict[str, Any],
    model: str,
    operation: str,
) -> Any:
    """Parse chat completion with retries for transient errors."""
    for attempt in range(1, OPENAI_MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.warning(
                    "Retrying OpenAI call",
                    extra={
                        "operation": operation,
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": OPENAI_MAX_RETRIES,
                    },
                )
            return client.beta.chat.completions.parse(**params)
        except OpenAIError as error:
            classification = classify_openai_error(error)
            retryable = (
                is_retryable_openai_error(error) and attempt < OPENAI_MAX_RETRIES
            )
            if retryable:
                wait_seconds = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Recoverable OpenAI error; retrying",
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
            raise


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
    enable_chunking: bool = True,
    operation: str = "unspecified_operation",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Call OpenAI API with automatic fallback to GPT-5 on context length errors
    and automatic chunking for oversized requests.

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
        enable_chunking: Whether to enable automatic chunking for large requests
        operation: Operation name for structured logs
        request_metadata: Additional metadata for structured logs

    Returns:
        Response from OpenAI API (or merged responses if chunked)

    Raises:
        OpenAIError: If both primary and fallback models fail
    """
    logger.info(
        "OpenAI request prepared",
        extra={
            "operation": operation,
            "model": model,
            "fallback_model": fallback_model,
            "message_summary": summarize_messages(messages),
            "request_metadata": request_metadata or {},
        },
    )

    # Check if messages exceed token limit and need chunking
    token_limit = get_token_limit(model)
    current_tokens = count_messages_tokens(messages, model)

    if enable_chunking and current_tokens > token_limit:
        logger.warning(
            "Fallback strategy activated: chunking",
            extra={
                "operation": operation,
                "model": model,
                "reason": "context_length_precheck",
                "context_length_source": "local_estimate",
                "current_tokens": current_tokens,
                "token_limit": token_limit,
            },
        )
        return _call_openai_with_chunking(
            client=client,
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            json_mode=json_mode,
            fallback_model=fallback_model,
            operation=operation,
            request_metadata=request_metadata,
        )

    # Standard call without chunking
    return _call_openai_single(
        client=client,
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        json_mode=json_mode,
        fallback_model=fallback_model,
        operation=operation,
        request_metadata=request_metadata,
    )


def _call_openai_single(
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
    operation: str = "unspecified_operation",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Make a single API call with fallback support.

    This is the original call_openai_with_fallback logic, extracted for reuse.
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
        logger.info(
            "Attempting OpenAI API call",
            extra={
                "operation": operation,
                "model": model,
                "request_metadata": request_metadata or {},
            },
        )
        response = _parse_with_retry(client, params, model, operation)
        logger.info(
            "OpenAI API call succeeded",
            extra={
                "operation": operation,
                "model": model,
            },
        )
        return response

    except OpenAIError as e:
        classification = classify_openai_error(e)
        # Check if this is a context length error
        if is_context_length_error(e):
            local_token_count = count_messages_tokens(messages, model)
            local_token_limit = get_token_limit(model)
            api_context_details = extract_context_length_error_details(e)
            logger.warning(
                "Fallback strategy activated: context length",
                extra={
                    "operation": operation,
                    "from_model": model,
                    "to_model": fallback_model,
                    "severity": classification["severity"],
                    "reason": classification["reason"],
                    "context_length_source": "api_error",
                    "local_estimated_tokens": local_token_count,
                    "local_model_token_limit": local_token_limit,
                    **api_context_details,
                },
            )

            # If we haven't tried the fallback model yet, try it
            if model != fallback_model:
                logger.info(
                    "Attempting model fallback",
                    extra={
                        "operation": operation,
                        "from_model": model,
                        "to_model": fallback_model,
                        "reason": "context_length",
                    },
                )

                # Check if fallback model would still exceed limits
                fallback_token_limit = get_token_limit(fallback_model)
                current_tokens = count_messages_tokens(messages, fallback_model)

                if current_tokens > fallback_token_limit:
                    logger.warning(
                        "Fallback model also exceeds token limit; chunking",
                        extra={
                            "operation": operation,
                            "model": fallback_model,
                            "reason": "context_length",
                            "context_length_source": "local_estimate",
                            "current_tokens": current_tokens,
                            "token_limit": fallback_token_limit,
                        },
                    )
                    return _call_openai_with_chunking(
                        client=client,
                        model=fallback_model,
                        messages=messages,
                        response_format=response_format,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                        fallback_model=fallback_model,  # No further fallback
                        operation=operation,
                        request_metadata=request_metadata,
                    )

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
                    response = _parse_with_retry(
                        client, fallback_params, fallback_model, operation
                    )
                    logger.info(
                        "Fallback model call succeeded",
                        extra={
                            "operation": operation,
                            "model": fallback_model,
                            "reason": "context_length",
                        },
                    )
                    return response

                except OpenAIError as fallback_error:
                    fallback_classification = classify_openai_error(fallback_error)
                    # If fallback also has context length error, try chunking
                    if is_context_length_error(fallback_error):
                        fallback_api_context_details = (
                            extract_context_length_error_details(fallback_error)
                        )
                        logger.warning(
                            "Fallback model context-length error; chunking as last resort",
                            extra={
                                "operation": operation,
                                "model": fallback_model,
                                "reason": "context_length",
                                "context_length_source": "api_error",
                                "local_estimated_tokens": count_messages_tokens(
                                    messages, fallback_model
                                ),
                                "local_model_token_limit": get_token_limit(
                                    fallback_model
                                ),
                                **fallback_api_context_details,
                            },
                        )
                        return _call_openai_with_chunking(
                            client=client,
                            model=fallback_model,
                            messages=messages,
                            response_format=response_format,
                            temperature=temperature,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            max_tokens=max_tokens,
                            json_mode=json_mode,
                            fallback_model=fallback_model,
                            operation=operation,
                            request_metadata=request_metadata,
                        )
                    logger.error(
                        "Fallback model failed",
                        extra={
                            "operation": operation,
                            "model": fallback_model,
                            "severity": fallback_classification["severity"],
                            "reason": fallback_classification["reason"],
                            "error": str(fallback_error),
                        },
                    )
                    raise fallback_error
            else:
                # Already using fallback model, can't fallback further - try chunking
                logger.warning(
                    "Context length with fallback model; chunking as last resort",
                    extra={
                        "operation": operation,
                        "model": fallback_model,
                        "reason": "context_length",
                        "context_length_source": "api_error",
                        "local_estimated_tokens": count_messages_tokens(
                            messages, fallback_model
                        ),
                        "local_model_token_limit": get_token_limit(fallback_model),
                    },
                )
                return _call_openai_with_chunking(
                    client=client,
                    model=fallback_model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    fallback_model=fallback_model,
                    operation=operation,
                    request_metadata=request_metadata,
                )
        elif is_safety_block_error(e) and model != fallback_model:
            # Safety/content-policy block — retry with fallback model (e.g. gpt-4o)
            logger.warning(
                "Fallback strategy activated: safety block",
                extra={
                    "operation": operation,
                    "from_model": model,
                    "to_model": fallback_model,
                    "reason": "safety_block",
                },
            )
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
            response = _parse_with_retry(
                client, fallback_params, fallback_model, operation
            )
            logger.info(
                "Fallback model call succeeded",
                extra={
                    "operation": operation,
                    "model": fallback_model,
                    "reason": "safety_block",
                },
            )
            return response
        elif is_output_length_error(e):
            # Output truncation/parse length issues are not input-context overflows.
            # Retry once with a higher completion budget when supported.
            if not _is_model_without_parameters(model):
                retry_max_tokens = min(max(max_tokens * 2, 1024), 8192)
                if retry_max_tokens > max_tokens:
                    logger.warning(
                        "Output length limit reached; retrying with higher max_tokens",
                        extra={
                            "operation": operation,
                            "model": model,
                            "severity": "recoverable",
                            "reason": "output_length_limit",
                            "max_tokens_before": max_tokens,
                            "max_tokens_after": retry_max_tokens,
                        },
                    )
                    retry_params = prepare_model_params(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        temperature=temperature,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        max_tokens=retry_max_tokens,
                        json_mode=json_mode,
                    )
                    try:
                        response = _parse_with_retry(
                            client, retry_params, model, operation
                        )
                        logger.info(
                            "OpenAI call succeeded after max_tokens increase",
                            extra={
                                "operation": operation,
                                "model": model,
                                "reason": "output_length_limit",
                                "max_tokens_after": retry_max_tokens,
                            },
                        )
                        return response
                    except OpenAIError as retry_error:
                        retry_classification = classify_openai_error(retry_error)
                        logger.error(
                            "OpenAI retry after output-length limit failed",
                            extra={
                                "operation": operation,
                                "model": model,
                                "severity": retry_classification["severity"],
                                "reason": retry_classification["reason"],
                                "error": str(retry_error),
                            },
                        )
                        raise retry_error
            logger.error(
                "OpenAI call hit output-length limit and could not be recovered",
                extra={
                    "operation": operation,
                    "model": model,
                    "severity": classification["severity"],
                    "reason": classification["reason"],
                    "error": str(e),
                },
            )
            raise e
        else:
            # Re-raise other errors
            logger.error(
                "OpenAI call failed",
                extra={
                    "operation": operation,
                    "model": model,
                    "severity": classification["severity"],
                    "reason": classification["reason"],
                    "error": str(e),
                },
            )
            raise e

    except Exception as e:
        logger.error(
            "Unexpected OpenAI call failure",
            extra={
                "operation": operation,
                "model": model,
                "error": str(e),
            },
        )
        raise e


def _call_openai_with_chunking(
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
    operation: str = "unspecified_operation",
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Make multiple API calls by chunking large messages and merge the responses.

    Args:
        Same as call_openai_with_fallback

    Returns:
        Merged response from all chunks
    """
    logger.info(
        "Chunking request payload",
        extra={
            "operation": operation,
            "model": model,
        },
    )

    # Get token limit for the model
    token_limit = get_token_limit(model)

    # Create chunked message lists
    chunked_message_lists = create_chunked_messages(messages, model, token_limit)

    if len(chunked_message_lists) <= 1:
        # Chunking didn't help, try with the single message
        logger.warning(
            "Chunking did not split messages. Attempting single call anyway."
        )
        return _call_openai_single(
            client=client,
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            json_mode=json_mode,
            fallback_model=fallback_model,
            operation=operation,
            request_metadata=request_metadata,
        )

    logger.info(f"Processing {len(chunked_message_lists)} chunks...")

    # Process each chunk
    responses = []
    for i, chunk_messages in enumerate(chunked_message_lists):
        logger.info(f"Processing chunk {i+1}/{len(chunked_message_lists)}")

        # Verify this chunk is within limits
        chunk_tokens = count_messages_tokens(chunk_messages, model)
        logger.info(f"Chunk {i+1} token count: {chunk_tokens}/{token_limit}")

        try:
            response = _call_openai_single(
                client=client,
                model=model,
                messages=chunk_messages,
                response_format=response_format,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                max_tokens=max_tokens,
                json_mode=json_mode,
                fallback_model=fallback_model,
                operation=operation,
                request_metadata=request_metadata,
            )
            responses.append(response)
        except Exception as e:
            logger.error(f"Failed to process chunk {i+1}: {str(e)}")
            raise

    logger.info(
        f"Successfully processed all {len(responses)} chunks, merging responses..."
    )

    # Merge responses
    merged_response = merge_pydantic_responses(responses, response_format)

    logger.info("Successfully merged all chunk responses")
    return merged_response


def validate_model_config(model: str, config: Dict[str, Any]) -> None:
    """
    Validate model configuration, checking for GPT-5 specific requirements.

    Args:
        model: The model name
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid for the model
    """
    if _is_model_without_parameters(model):
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
