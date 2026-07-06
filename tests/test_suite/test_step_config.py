"""Tests for unified pipeline step configuration resolution."""

import unittest

from src.soda_curation.pipeline.step_config import (
    infer_provider_from_model,
    resolve_pipeline_provider,
    resolve_step_config,
)


class TestInferProviderFromModel(unittest.TestCase):
    def test_openai_gpt(self):
        self.assertEqual(infer_provider_from_model("gpt-5.4-mini"), "openai")
        self.assertEqual(infer_provider_from_model("gpt-4o"), "openai")

    def test_openai_o_series(self):
        self.assertEqual(infer_provider_from_model("o1-preview"), "openai")
        self.assertEqual(infer_provider_from_model("o3-mini"), "openai")

    def test_anthropic_claude(self):
        self.assertEqual(infer_provider_from_model("claude-sonnet-4-6"), "anthropic")

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            infer_provider_from_model("llama-3")


class TestResolveStepConfig(unittest.TestCase):
    def test_flat_shape(self):
        step = {
            "model": "gpt-5.4-mini",
            "prompts": {"system": "sys", "user": "usr"},
        }
        resolved = resolve_step_config(step)
        self.assertEqual(resolved["model"], "gpt-5.4-mini")
        self.assertEqual(resolved["provider"], "openai")
        self.assertEqual(resolved["prompts"]["system"], "sys")

    def test_legacy_nested_openai(self):
        step = {
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.1,
                "prompts": {"system": "sys", "user": "usr"},
            }
        }
        resolved = resolve_step_config(step)
        self.assertEqual(resolved["model"], "gpt-4o")
        self.assertEqual(resolved["provider"], "openai")


class TestResolvePipelineProvider(unittest.TestCase):
    def test_single_provider_across_steps(self):
        pipeline = {
            "extract_sections": {
                "model": "gpt-5.4-mini",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_caption_title": {
                "model": "gpt-4o",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_panel_sequence": {
                "model": "gpt-4o",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_data_sources": {
                "model": "gpt-4o-mini",
                "prompts": {"system": "s", "user": "u"},
            },
            "match_caption_panel": {
                "model": "gpt-4o",
                "prompts": {"system": "s", "user": "u"},
            },
            "assign_panel_source": {
                "model": "gpt-4o",
                "prompts": {"system": "s", "user": "u"},
            },
        }
        self.assertEqual(resolve_pipeline_provider(pipeline), "openai")

    def test_mixed_providers_raises(self):
        pipeline = {
            "extract_sections": {
                "model": "gpt-4o",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_caption_title": {
                "model": "claude-sonnet-4-6",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_panel_sequence": {
                "model": "claude-sonnet-4-6",
                "prompts": {"system": "s", "user": "u"},
            },
            "extract_data_sources": {
                "model": "claude-sonnet-4-6",
                "prompts": {"system": "s", "user": "u"},
            },
            "match_caption_panel": {
                "model": "claude-sonnet-4-6",
                "prompts": {"system": "s", "user": "u"},
            },
            "assign_panel_source": {
                "model": "claude-sonnet-4-6",
                "prompts": {"system": "s", "user": "u"},
            },
        }
        with self.assertRaises(ValueError):
            resolve_pipeline_provider(pipeline)
