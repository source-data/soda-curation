"""Token usage and cost tracking utilities."""

from typing import Any

from ..pipeline.manuscript_structure.manuscript_structure import TokenUsage

pricing = {
    # OpenAI models (USD per 1M tokens)
    "gpt-4o": {"input_tokens": 5.00, "output_tokens": 10.00},
    "gpt-4o-mini": {"input_tokens": 0.15, "output_tokens": 0.60},
    "gpt-5": {"input_tokens": 5.00, "output_tokens": 15.00},
    # Anthropic Claude models (USD per 1M tokens)
    "claude-opus-4-6": {"input_tokens": 15.00, "output_tokens": 75.00},
    "claude-sonnet-4-6": {"input_tokens": 3.00, "output_tokens": 15.00},
    "claude-haiku-4-5": {"input_tokens": 0.25, "output_tokens": 1.25},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on token usage and model pricing."""
    # Convert tokens to millions
    input_tokens_million = prompt_tokens / 1_000_000
    output_tokens_million = completion_tokens / 1_000_000

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

    input_cost = model_pricing["input_tokens"]
    output_cost = model_pricing["output_tokens"]

    # Calculate total cost
    total_cost = (input_tokens_million * input_cost) + (
        output_tokens_million * output_cost
    )
    return total_cost


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

        # Calculate and accumulate cost for this call
        call_cost = calculate_cost(
            model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
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
