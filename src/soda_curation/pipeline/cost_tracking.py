"""Token usage and cost tracking utilities."""

from typing import Any

from ..pipeline.manuscript_structure.manuscript_structure import TokenUsage

pricing = {
    "gpt-4o": {"input_tokens": 5.00, "output_tokens": 10.00},
    "gpt-4o-mini": {"input_tokens": 0.15, "output_tokens": 0.60},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost based on token usage and model pricing."""
    # Convert tokens to millions
    input_tokens_million = prompt_tokens / 1_000_000
    output_tokens_million = completion_tokens / 1_000_000

    # Retrieve pricing
    input_cost = pricing[model]["input_tokens"]
    output_cost = pricing[model]["output_tokens"]

    # Calculate total cost
    total_cost = (input_tokens_million * input_cost) + (
        output_tokens_million * output_cost
    )
    return total_cost


def update_token_usage(
    token_usage: TokenUsage, response: Any, model: str
) -> TokenUsage:
    """Update TokenUsage object with data from API response."""
    if not isinstance(response, dict):
        response = response.model_dump()
    if "usage" in response:
        # Accumulate token counts
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
    return token_usage
