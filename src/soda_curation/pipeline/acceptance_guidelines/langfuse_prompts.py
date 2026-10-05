"""Fetch acceptance-guideline text from the Langfuse project.

The Langfuse project is ``AIP-guidelines``. Its API keys are separate
from the QC project's ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``.
Each guideline is a text prompt. The step reads the ``production`` label only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_NAME = "AIP-guidelines"
PROMPT_LABEL = "production"
COMMON_PROMPT_NAME = "common"
PUBLIC_KEY_ENV = "LANGFUSE_ACCEPTANCE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_ACCEPTANCE_SECRET_KEY"
HOST_ENV = "LANGFUSE_ACCEPTANCE_HOST"

_client = None
_client_failed = False
_client_failure_reason = ""
_prompt_cache: Dict[str, "FetchedPrompt"] = {}


@dataclass(frozen=True)
class FetchedPrompt:
    """A production text prompt returned by Langfuse."""

    name: str
    label: str
    version: str
    text: str


def reset_client() -> None:
    """Drop the cached client and prompts. Tests use this between cases."""
    global _client, _client_failed, _client_failure_reason
    _client = None
    _client_failed = False
    _client_failure_reason = ""
    _prompt_cache.clear()


def _host() -> Optional[str]:
    host = (
        os.environ.get(HOST_ENV, "").strip()
        or os.environ.get("LANGFUSE_HOST", "").strip()
        or os.environ.get("LANGFUSE_BASE_URL", "").strip()
    )
    return host or None


def _get_client():
    """Lazily create the Langfuse client for the AIP-guidelines project."""
    global _client, _client_failed, _client_failure_reason
    if _client is not None or _client_failed:
        return _client

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    public_key = os.environ.get(PUBLIC_KEY_ENV, "").strip()
    secret_key = os.environ.get(SECRET_KEY_ENV, "").strip()
    missing = [
        name
        for name, value in (
            (PUBLIC_KEY_ENV, public_key),
            (SECRET_KEY_ENV, secret_key),
        )
        if not value
    ]
    if missing:
        _client_failed = True
        _client_failure_reason = (
            f"Missing Langfuse environment variables for project '{PROJECT_NAME}': "
            + ", ".join(missing)
        )
        logger.warning(_client_failure_reason)
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=_host(),
        )
    except Exception as exc:
        _client_failed = True
        _client_failure_reason = str(exc)
        logger.warning("Langfuse client init failed for acceptance guidelines: %s", exc)
    return _client


def get_production_prompt(name: str) -> FetchedPrompt:
    """Return the production text prompt ``name`` from the acceptance project."""
    cached = _prompt_cache.get(name)
    if cached is not None:
        return cached

    client = _get_client()
    if client is None:
        raise RuntimeError(f"{_client_failure_reason}. Cannot fetch prompt '{name}'.")

    logger.debug(
        "Fetching Langfuse prompt %s label=%s project=%s",
        name,
        PROMPT_LABEL,
        PROJECT_NAME,
    )
    try:
        prompt = client.get_prompt(name, label=PROMPT_LABEL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch Langfuse prompt '{name}' with label '{PROMPT_LABEL}' "
            f"from project '{PROJECT_NAME}': {exc}"
        ) from exc

    text = getattr(prompt, "prompt", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"Langfuse prompt '{name}' (label '{PROMPT_LABEL}') must be a "
            f"non-empty text prompt in project '{PROJECT_NAME}'."
        )

    fetched = FetchedPrompt(
        name=getattr(prompt, "name", None) or name,
        label=PROMPT_LABEL,
        version=str(getattr(prompt, "version", "") or ""),
        text=text,
    )
    _prompt_cache[name] = fetched
    return fetched
