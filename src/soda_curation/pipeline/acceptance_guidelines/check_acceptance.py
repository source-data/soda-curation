"""Check a manuscript against journal acceptance guidelines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import anthropic
import openai

from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from ..openai_utils import call_openai
from ..step_config import resolve_step_config
from .langfuse_prompts import (
    COMMON_PROMPT_NAME,
    PROMPT_LABEL,
    FetchedPrompt,
    get_production_prompt,
)

logger = logging.getLogger(__name__)

STEP_NAME = "check_acceptance_guidelines"
UNKNOWN_JOURNAL_GUIDELINES = (
    "No journal-specific guidelines prompt matched this manuscript. "
    "Apply only the common guidelines."
)


@dataclass(frozen=True)
class Journal:
    """A journal this pipeline can match to a Langfuse prompt."""

    key: str
    title: str
    prefixes: Tuple[str, ...]


# Longest manuscript-id prefix is matched first (EMBOR before any shorter EMBO* token).
# ``key`` is the Langfuse prompt name in the AIP-guidelines project.
JOURNALS: Tuple[Journal, ...] = (
    Journal("embo_reports", "EMBO Reports", ("EMBOR",)),
    Journal("the_embo_journal", "The EMBO Journal", ("EMBOJ",)),
    Journal("embo_molecular_medicine", "EMBO Molecular Medicine", ("EMM",)),
    Journal("molecular_systems_biology", "Molecular Systems Biology", ("MSB",)),
    Journal("life_science_alliance", "Life Science Alliance", ("LSA",)),
)


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


def resolve_journal(
    journal_title: str, manuscript_id: str
) -> Tuple[Optional[Journal], str]:
    """
    Match a journal from the XML ``journal-title``, then from the manuscript id prefix.

    Returns ``(journal, matched_from)`` where ``matched_from`` is
    ``journal_title``, ``manuscript_id``, or ``unmatched``.
    """
    title = _normalize(journal_title)
    for journal in JOURNALS:
        names = {_normalize(journal.title)}
        if journal.title.casefold().startswith("the "):
            names.add(_normalize(journal.title[4:]))
        if title and title in names:
            return journal, "journal_title"

    prefix = (manuscript_id or "").split("-", 1)[0].upper()
    by_prefix = sorted(
        JOURNALS,
        key=lambda journal: max(len(item) for item in journal.prefixes),
        reverse=True,
    )
    for journal in by_prefix:
        if prefix in journal.prefixes:
            return journal, "manuscript_id"
    return None, "unmatched"


def load_guidelines(
    journal: Optional[Journal],
) -> Tuple[FetchedPrompt, Optional[FetchedPrompt]]:
    """Fetch the shared prompt and, when known, the matching journal prompt."""
    common = get_production_prompt(COMMON_PROMPT_NAME)
    if journal is None:
        return common, None
    return common, get_production_prompt(journal.key)


def _complete(provider: str, model: str, messages: list) -> object:
    if provider == "anthropic":
        validate_anthropic_model(model)
        response = call_anthropic(
            client=anthropic.Anthropic(),
            model=model,
            messages=messages,
            max_tokens=8192,
            operation="main.check_acceptance_guidelines",
        )
        return response

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return call_openai(
        client=openai.OpenAI(api_key=api_key),
        model=model,
        messages=messages,
        json_mode=False,
        max_tokens=8192,
        operation="main.check_acceptance_guidelines",
    )


def check_acceptance_guidelines(
    config: Dict,
    prompt_handler,
    zip_structure: ZipStructure,
    manuscript_text: str,
) -> ZipStructure:
    """
    Ask the configured model whether the manuscript follows the journal guidelines.

    Stores the Markdown report on ``zip_structure.acceptance_guidelines``
    so it is part of the main pipeline JSON.
    Only the common prompt and the matched journal prompt are sent.
    Both are the Langfuse ``production`` label.
    """
    resolved = resolve_step_config(config["pipeline"][STEP_NAME])
    journal, matched_from = resolve_journal(
        getattr(zip_structure, "journal_title", "") or "",
        zip_structure.manuscript_id,
    )
    common_prompt, journal_prompt = load_guidelines(journal)
    common_guidelines = common_prompt.text
    journal_guidelines = (
        journal_prompt.text
        if journal_prompt is not None
        else UNKNOWN_JOURNAL_GUIDELINES
    )
    journal_title = (
        journal.title
        if journal is not None
        else (getattr(zip_structure, "journal_title", "") or "Unknown journal")
    )
    journal_key = journal.key if journal is not None else "unknown"

    prompts = prompt_handler.get_prompt(
        step=STEP_NAME,
        variables={
            "journal_title": journal_title,
            "journal_key": journal_key,
            "common_guidelines": common_guidelines,
            "journal_guidelines": journal_guidelines,
            "manuscript_text": manuscript_text or "",
        },
    )
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]},
    ]
    response = _complete(resolved["provider"], resolved["model"], messages)
    update_token_usage(
        zip_structure.cost.check_acceptance_guidelines, response, resolved["model"]
    )

    report = (response.choices[0].message.content or "").strip()
    if not report:
        raise ValueError("Acceptance guidelines model returned an empty report")

    zip_structure.journal_title = journal_title
    zip_structure.acceptance_guidelines = {
        "journal_key": journal_key,
        "journal_title": journal_title,
        "matched_from": matched_from,
        "prompt_label": PROMPT_LABEL,
        "common_prompt_version": common_prompt.version,
        "journal_prompt_version": (
            journal_prompt.version if journal_prompt is not None else ""
        ),
        "report": report,
    }
    logger.info(
        "Acceptance guidelines report stored on the pipeline JSON",
        extra={
            "operation": "main.check_acceptance_guidelines",
            "journal_key": journal_key,
            "matched_from": matched_from,
            "common_prompt_version": common_prompt.version,
            "journal_prompt_version": (
                journal_prompt.version if journal_prompt is not None else ""
            ),
        },
    )
    return zip_structure
