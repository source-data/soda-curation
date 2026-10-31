"""Token usage and cost tracking utilities."""

from typing import Any

from ..pipeline.manuscript_structure.manuscript_structure import TokenUsage

# USD per 1M tokens. Optional "cached_input_tokens" / "cache_write_tokens" rates
# apply to the cached / cache-write subsets of prompt tokens; models without them
# are priced at the input rate for all prompt tokens.
pricing = {
    # OpenAI models (USD per 1M tokens)
    "gpt-4o": {"input_tokens": 5.00, "output_tokens": 10.00},
    "gpt-4o-mini": {"input_tokens": 0.15, "output_tokens": 0.60},
    "gpt-6-astra": {"input_tokens": 10.00, "output_tokens": 50.00},
    "gpt-6-luna": {
        "input_tokens": 0.10,
        "cached_input_tokens": 0.01,
        "cache_write_tokens": 0.125,
        "output_tokens": 0.50,
    },
    # Before "gpt-5": prefix matching takes the first key that fits.
    "gpt-5.6-sol": {
        "input_tokens": 4.00,
        "cached_input_tokens": 0.40,
        "cache_write_tokens": 5.00,
        "output_tokens": 20.00,
    },
    "gpt-5": {"input_tokens": 5.00, "output_tokens": 15.00},
    # Anthropic Claude models (USD per 1M tokens)
    "claude-sonnet-5-5": {"input_tokens": 2.00, "output_tokens": 10.00},
    "claude-opus-4-6": {"input_tokens": 15.00, "output_tokens": 75.00},
    "claude-sonnet-4-6": {"input_tokens": 3.00, "output_tokens": 15.00},
    "claude-haiku-4-5": {"input_tokens": 0.25, "output_tokens": 1.25},
}


def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Calculate cost based on token usage and model pricing.

    ``cached_tokens`` and ``cache_write_tokens`` are subsets of ``prompt_tokens``.
    """

    # Retrieve pricing — exact match first, then prefix match for versioned IDs
    # e.g. "gpt-5-2025-04-14" matches "gpt-5"
    model_pricing = pricing.get(model)
    if model_pricing is None:
        for key in pricing:
            if model.startswith(key):
                model_pricing = pricing[key]
                break
    if model_pricing is None:
        return 0.0

    input_rate = model_pricing["input_tokens"]
    cached_rate = model_pricing.get("cached_input_tokens", input_rate)
    write_rate = model_pricing.get("cache_write_tokens", input_rate)
    uncached_tokens = prompt_tokens - cached_tokens - cache_write_tokens

    return (
        uncached_tokens * input_rate
        + cached_tokens * cached_rate
        + cache_write_tokens * write_rate
        + completion_tokens * model_pricing["output_tokens"]
    ) / 1_000_000


def update_token_usage(
    token_usage: TokenUsage, response: Any, model: str
) -> TokenUsage:
    """Update TokenUsage object with data from API response."""
    # Handle response object with usage attribute
    if hasattr(response, "usage"):
        # Accumulate token counts
        token_usage.prompt_tokens += response.usage.prompt_tokens
        token_usage.completion_tokens += response.usage.completion_tokens
        token_usage.total_tokens += response.usage.total_tokens
        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        completion_details = getattr(response.usage, "completion_tokens_details", None)
        cached = getattr(prompt_details, "cached_tokens", 0) or 0
        cache_write = getattr(prompt_details, "cache_write_tokens", 0) or 0
        token_usage.cached_tokens += cached
        token_usage.cache_write_tokens += cache_write
        token_usage.reasoning_tokens += (
            getattr(completion_details, "reasoning_tokens", 0) or 0
        )

        # Calculate and accumulate cost for this call
        call_cost = calculate_cost(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            cached_tokens=cached,
            cache_write_tokens=cache_write,
        )
        token_usage.cost += call_cost
    elif isinstance(response, dict) and "usage" in response:
        # Handle dictionary response format
        token_usage.prompt_tokens += response["usage"]["prompt_tokens"]
        token_usage.completion_tokens += response["usage"]["completion_tokens"]
        token_usage.total_tokens += response["usage"]["total_tokens"]

        # Calculate and accumulate cost for this call
        call_cost = calculate_cost(
            model,
            response["usage"]["prompt_tokens"],
            response["usage"]["completion_tokens"],
        )
        token_usage.cost += call_cost
    else:
        # Try to convert to dict if it's a Pydantic model
        try:
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
                if "usage" in response_dict:
                    token_usage.prompt_tokens += response_dict["usage"]["prompt_tokens"]
                    token_usage.completion_tokens += response_dict["usage"][
                        "completion_tokens"
                    ]
                    token_usage.total_tokens += response_dict["usage"]["total_tokens"]

                    # Calculate and accumulate cost for this call
                    call_cost = calculate_cost(
                        model,
                        response_dict["usage"]["prompt_tokens"],
                        response_dict["usage"]["completion_tokens"],
                    )
                    token_usage.cost += call_cost
        except Exception:
            # If all else fails, skip token tracking
            pass
    return token_usage
