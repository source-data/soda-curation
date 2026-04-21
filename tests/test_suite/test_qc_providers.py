"""Unit tests for QC provider implementations and factory."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.qc.providers.base import QCProviderRequest
from src.soda_curation.qc.providers.factory import build_qc_provider
from src.soda_curation.qc.providers.openai_provider import OpenAIQCProvider


def test_build_qc_provider_supported():
    clients = {"openai": MagicMock(), "anthropic": MagicMock(), "gemini": MagicMock()}
    assert build_qc_provider("openai", clients=clients).provider_name == "openai"
    assert build_qc_provider("anthropic", clients=clients).provider_name == "anthropic"
    assert build_qc_provider("gemini", clients=clients).provider_name == "gemini"


def test_build_qc_provider_invalid():
    with pytest.raises(ValueError):
        build_qc_provider("unknown-provider")


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


def test_gemini_provider_warns_when_agentic_requested(caplog):
    from src.soda_curation.qc.providers.gemini_provider import GeminiQCProvider

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = SimpleNamespace(
        text='{"ok": true}',
        usage_metadata=SimpleNamespace(
            prompt_token_count=5,
            candidates_token_count=3,
            total_token_count=8,
        ),
        model_version="gemini-2.5-flash",
        candidates=[],
    )
    provider = GeminiQCProvider(client=mock_client)
    request = QCProviderRequest(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        prompt_config={},
        response_type=None,
        operation="qc.test",
        context={},
        agentic_enabled=True,
        model_config={"tools": [{"name": "lookup"}]},
    )

    with caplog.at_level("WARNING"):
        response = provider.generate(request)

    assert response.model == "gemini-2.5-flash"
    assert "agentic_not_supported" in caplog.text
