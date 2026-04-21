"""Tests for the provider-agnostic ModelAPI."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.soda_curation.qc.model_api import ModelAPI
from src.soda_curation.qc.providers.base import QCProviderResponse


class TestModelAPI:
    @pytest.fixture
    def model_config(self):
        return {"ai_provider": "openai", "default": {"openai": {"model": "gpt-4o"}}}

    @patch("src.soda_curation.qc.model_api.build_qc_provider")
    def test_generate_response(self, mock_build_provider, model_config):
        mock_provider = MagicMock()
        mock_provider.generate.return_value = QCProviderResponse(
            content='{"test": "value"}',
            parsed=None,
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_build_provider.return_value = mock_provider

        api = ModelAPI(model_config)
        response = api.generate_response(
            encoded_image="base64encodedimage",
            caption="Test caption",
            prompt_config={
                "prompts": {
                    "system": "System prompt",
                    "user": "User prompt with $figure_caption",
                },
                "model": "gpt-4o",
            },
        )

        assert response == {"test": "value"}
        mock_provider.generate.assert_called_once()
        assert api.token_usage.total_tokens == 15

    @patch("src.soda_curation.qc.model_api.build_qc_provider")
    def test_generate_response_prefers_structured_parsed(
        self, mock_build_provider, model_config
    ):
        mock_provider = MagicMock()
        parsed = MagicMock()
        parsed.model_dump.return_value = {"outputs": [{"panel_label": "A"}]}
        mock_provider.generate.return_value = QCProviderResponse(
            content="",
            parsed=parsed,
            model="gpt-4o",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        mock_build_provider.return_value = mock_provider

        api = ModelAPI(model_config)
        response = api.generate_response(
            encoded_image="base64encodedimage",
            caption="Test caption",
            prompt_config={"prompts": {"system": "s", "user": "$figure_caption"}},
            response_type=MagicMock(),
        )

        assert json.loads(response)["outputs"][0]["panel_label"] == "A"

    @patch("src.soda_curation.qc.model_api.build_qc_provider")
    def test_generate_response_requires_valid_payload(
        self, mock_build_provider, model_config
    ):
        mock_build_provider.return_value = MagicMock()
        api = ModelAPI(model_config)
        with pytest.raises(ValueError):
            api.generate_response(
                prompt_config={"prompts": {"system": "", "user": ""}},
            )
