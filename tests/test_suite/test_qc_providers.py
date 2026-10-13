"""Unit tests for QC provider implementations and factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.qc.providers.base import QCProviderRequest
from src.soda_curation.qc.providers.factory import build_qc_provider
from src.soda_curation.qc.providers.openai_provider import OpenAIQCProvider


def test_build_qc_provider_supported():
    clients = {"openai": MagicMock(), "anthropic": MagicMock()}
    assert build_qc_provider("openai", clients=clients).provider_name == "openai"
    assert build_qc_provider("anthropic", clients=clients).provider_name == "anthropic"


def test_build_qc_provider_invalid():
    with pytest.raises(ValueError):
        build_qc_provider("unknown-provider")


def test_build_qc_provider_gemini_no_longer_supported():
    with pytest.raises(ValueError):
        build_qc_provider("gemini")


@patch("src.soda_curation.qc.providers.anthropic_provider.call_anthropic")
def test_anthropic_provider_passes_agentic_model_config(mock_call_anthropic):
    from src.soda_curation.qc.providers.anthropic_provider import AnthropicQCProvider

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", parsed=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model="claude-sonnet-4-6",
    )
    mock_call_anthropic.return_value = mock_response

    provider = AnthropicQCProvider(client=MagicMock())
    request = QCProviderRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "test"}],
        prompt_config={},
        response_type=None,
        operation="qc.test",
        context={},
        agentic_enabled=True,
        model_config={
            "tools": [{"type": "web_search_20250305", "max_uses": 2}],
            "tool_choice": "auto",
        },
    )

    response = provider.generate(request)

    assert response.metadata["api_mode"] == "messages"
    assert response.metadata["agentic_requested"] is True
    assert response.metadata["tool_config_present"] is True
    kwargs = mock_call_anthropic.call_args.kwargs
    assert kwargs["model_config"]["tool_choice"] == "auto"


@patch("src.soda_curation.qc.providers.anthropic_provider.call_anthropic")
def test_anthropic_provider_uses_model_config_without_agentic_flag(mock_call_anthropic):
    from src.soda_curation.qc.providers.anthropic_provider import AnthropicQCProvider

    mock_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}", parsed=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model="claude-sonnet-4-6",
    )
    mock_call_anthropic.return_value = mock_response

    provider = AnthropicQCProvider(client=MagicMock())
    request = QCProviderRequest(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "test"}],
        prompt_config={},
        response_type=None,
        operation="qc.test",
        context={},
        agentic_enabled=False,
        model_config={"tools": [{"type": "web_search_20250305"}]},
    )

    _ = provider.generate(request)
    kwargs = mock_call_anthropic.call_args.kwargs
    assert kwargs["model_config"]["tools"][0]["type"] == "web_search_20250305"


def test_openai_provider_agentic_uses_responses_api():
    mock_client = MagicMock()
    mock_response = SimpleNamespace(
        output_text='{"ok": true}',
        output=[],
        usage=SimpleNamespace(input_tokens=2, output_tokens=1, total_tokens=3),
        model="gpt-5",
    )
    mock_client.responses.create.return_value = mock_response

    provider = OpenAIQCProvider(client=mock_client)
    request = QCProviderRequest(
        model="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
        prompt_config={},
        response_type=None,
        operation="qc.test",
        context={"test_name": "agentic_check"},
        agentic_enabled=True,
        model_config={"tools": [{"type": "web_search_preview"}], "tool_choice": "auto"},
    )

    response = provider.generate(request)
    assert response.content == '{"ok": true}'
    assert response.metadata["api_mode"] == "responses"
    kwargs = mock_client.responses.create.call_args.kwargs
    assert "tools" in kwargs
    assert kwargs["tool_choice"] == "auto"


def test_call_anthropic_sonnet_5_5_omits_sampling_and_forced_tool_choice():
    from pydantic import BaseModel

    from src.soda_curation.pipeline.anthropic_utils import (
        DEFAULT_ANTHROPIC_MODEL,
        call_anthropic,
    )

    class Answer(BaseModel):
        value: str

    client = MagicMock()
    stream = client.messages.stream.return_value.__enter__.return_value
    stream.get_final_message.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use", name="structured_output", input={"value": "ok"}
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model=DEFAULT_ANTHROPIC_MODEL,
        stop_reason="tool_use",
    )

    response = call_anthropic(
        client=client,
        model=DEFAULT_ANTHROPIC_MODEL,
        messages=[{"role": "user", "content": "test"}],
        response_format=Answer,
        temperature=0.1,
    )

    params = client.messages.stream.call_args.kwargs
    client.messages.create.assert_not_called()
    assert params["max_tokens"] == 64000
    assert "temperature" not in params
    assert "extra_body" not in params
    assert params["tool_choice"] == {"type": "auto"}
    assert "structured_output" in params["system"]
    assert response.choices[0].message.parsed == Answer(value="ok")


def test_production_models_have_their_own_price():
    from src.soda_curation.pipeline.cost_tracking import calculate_cost

    assert calculate_cost("gpt-5.6-sol", 1_000_000, 1_000_000) == 24.0
    assert calculate_cost("gpt-6-astra", 1_000_000, 1_000_000) == 60.0
    assert calculate_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == 12.0


def test_call_anthropic_older_model_sends_temperature_in_body():
    from src.soda_curation.pipeline.anthropic_utils import call_anthropic

    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="claude-sonnet-4-6",
        stop_reason="end_turn",
    )

    call_anthropic(
        client=client,
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.2,
    )

    params = client.messages.create.call_args.kwargs
    assert "temperature" not in params
    assert params["extra_body"] == {"temperature": 0.2}


def test_call_anthropic_raises_when_structured_output_is_cut_off():
    from pydantic import BaseModel

    from src.soda_curation.pipeline.anthropic_utils import call_anthropic

    class Answer(BaseModel):
        value: str

    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="structured_output", input={})],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2048),
        model="claude-sonnet-4-6",
        stop_reason="max_tokens",
    )

    with pytest.raises(ValueError, match="hit max_tokens"):
        call_anthropic(
            client=client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "test"}],
            response_format=Answer,
        )
