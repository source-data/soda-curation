"""Tests for model pricing in cost_tracking."""

from unittest.mock import MagicMock

import pytest

from src.soda_curation.pipeline.cost_tracking import calculate_cost, update_token_usage
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    TokenUsage,
)


def test_luna_priced():
    assert calculate_cost("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)


def test_cached_and_cache_write_tokens_use_their_own_rates():
    # 1M prompt tokens: 600k cached reads, 300k cache writes, 100k uncached.
    cost = calculate_cost(
        "gpt-6-luna",
        1_000_000,
        0,
        cached_tokens=600_000,
        cache_write_tokens=300_000,
    )
    assert cost == pytest.approx(0.6 * 0.01 + 0.3 * 0.125 + 0.1 * 0.10)


def test_models_without_cache_rates_price_all_input_at_input_rate():
    cost = calculate_cost(
        "gpt-4o", 1_000_000, 0, cached_tokens=500_000, cache_write_tokens=500_000
    )
    assert cost == pytest.approx(5.00)


def test_update_token_usage_prices_cache_tokens():
    response = MagicMock(
        usage=MagicMock(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            total_tokens=1_000_000,
            prompt_tokens_details=MagicMock(
                cached_tokens=1_000_000, cache_write_tokens=0
            ),
            completion_tokens_details=MagicMock(reasoning_tokens=0),
        )
    )
    usage = update_token_usage(TokenUsage(), response, "gpt-5.6-sol")
    assert usage.cost == pytest.approx(0.40)
