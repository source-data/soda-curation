"""Resolve unified pipeline step configuration (model + shared prompts)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

SUPPORTED_PROVIDERS = ("openai", "anthropic")

# Steps that call an LLM in the main curation pipeline.
AI_PIPELINE_STEPS = (
    "extract_sections",
    "extract_caption_title",
    "extract_panel_sequence",
    "extract_data_sources",
    "match_caption_panel",
    "assign_panel_source",
)


def infer_provider_from_model(model: str) -> str:
    """
    Infer API provider from a model identifier.

    - ``gpt*``, ``o1*``, ``o3*``, ``o4*`` → OpenAI
    - ``claude*`` → Anthropic
    """
    if not model or not str(model).strip():
        raise ValueError("Model name is required to infer AI provider.")

    name = str(model).strip().lower()
    openai_prefixes = ("gpt", "o1", "o3", "o4")
    if any(
        name == prefix or name.startswith(f"{prefix}-") for prefix in openai_prefixes
    ):
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    raise ValueError(
        f"Cannot infer AI provider from model '{model}'. "
        "Use an OpenAI (gpt/o-series) or Anthropic (claude) model name."
    )


def is_ai_step(step_config: Dict[str, Any]) -> bool:
    """Return True when a pipeline step configures an LLM (flat or legacy nested)."""
    if not isinstance(step_config, dict):
        return False
    if "model" in step_config and "prompts" in step_config:
        return True
    return any(provider in step_config for provider in SUPPORTED_PROVIDERS)


def _legacy_provider_block(step_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the single legacy nested provider block, if present."""
    found = [step_config[p] for p in SUPPORTED_PROVIDERS if p in step_config]
    if not found:
        return None
    if len(found) > 1:
        raise ValueError(
            "Step defines both 'openai' and 'anthropic' blocks. "
            "Use a single flat step with one 'model' and shared 'prompts'."
        )
    block = found[0]
    if not isinstance(block, dict):
        raise ValueError("Legacy provider block must be a mapping.")
    return block


def resolve_step_config(step_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a pipeline step to ``{model, prompts, provider}``.

    Supports the flat shape::

        model: gpt-5.4-mini
        prompts: {system: ..., user: ...}

    and the legacy nested shape (backward compatible)::

        openai:
          model: ...
          prompts: ...
    """
    if not isinstance(step_config, dict):
        raise ValueError("Step configuration must be a mapping.")

    if "model" in step_config:
        model = step_config.get("model")
        prompts = step_config.get("prompts")
        if not model:
            raise ValueError("Step is missing required 'model'.")
        if not isinstance(prompts, dict):
            raise ValueError("Step is missing required 'prompts' mapping.")
        provider = infer_provider_from_model(str(model))
        return {"model": str(model), "prompts": prompts, "provider": provider}

    legacy = _legacy_provider_block(step_config)
    if legacy is None:
        raise ValueError("Step has no 'model'/'prompts' and no legacy provider block.")

    model = legacy.get("model")
    prompts = legacy.get("prompts")
    if not model:
        raise ValueError("Legacy provider block is missing 'model'.")
    if not isinstance(prompts, dict):
        raise ValueError("Legacy provider block is missing 'prompts' mapping.")

    provider = next(p for p in SUPPORTED_PROVIDERS if p in step_config)
    return {"model": str(model), "prompts": prompts, "provider": provider}


def get_step_prompts(step_config: Dict[str, Any]) -> Dict[str, str]:
    """Return the ``system`` and ``user`` prompt templates for a step."""
    resolved = resolve_step_config(step_config)
    prompts = resolved["prompts"]
    system = prompts.get("system", "")
    user = prompts.get("user", "")
    if not system or not user:
        raise ValueError("Step prompts must include non-empty 'system' and 'user'.")
    return {"system": system, "user": user}


def resolve_pipeline_provider(
    pipeline_config: Dict[str, Any],
    steps: Iterable[str] = AI_PIPELINE_STEPS,
) -> str:
    """
    Infer the single runtime provider for all AI steps from their model names.

    Raises when steps disagree (e.g. one gpt model and one claude model).
    """
    providers: set[str] = set()
    for step_name in steps:
        step_config = pipeline_config.get(step_name)
        if not isinstance(step_config, dict):
            raise ValueError(f"Missing pipeline configuration for step '{step_name}'.")
        resolved = resolve_step_config(step_config)
        providers.add(resolved["provider"])

    if len(providers) != 1:
        raise ValueError(
            "All AI pipeline steps must use models from the same provider. "
            f"Found providers: {sorted(providers)}."
        )
    return providers.pop()
