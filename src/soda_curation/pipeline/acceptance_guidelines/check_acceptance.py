"""Check a manuscript against journal acceptance guidelines."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import anthropic
import openai

from ..anthropic_utils import call_anthropic, validate_anthropic_model
from ..cost_tracking import update_token_usage
from ..manuscript_structure.manuscript_structure import ZipStructure
from ..openai_utils import call_openai
from ..step_config import resolve_step_config

logger = logging.getLogger(__name__)

STEP_NAME = "check_acceptance_guidelines"
GUIDELINES_DIR = Path(__file__).parent / "guidelines"
UNKNOWN_JOURNAL_GUIDELINES = (
    "PLACEHOLDER: No journal-specific guidelines file matched this manuscript. "
    "Apply only the common guidelines."
)


@dataclass(frozen=True)
class Journal:
    """A journal this pipeline can match to a guidelines file."""

    key: str
    title: str
    prefixes: Tuple[str, ...]
    guidelines_file: str


# Longest manuscript-id prefix is matched first (EMBOR before any shorter EMBO* token).
JOURNALS: Tuple[Journal, ...] = (
    Journal("embo_reports", "EMBO Reports", ("EMBOR",), "embo_reports.md"),
    Journal("the_embo_journal", "The EMBO Journal", ("EMBOJ",), "the_embo_journal.md"),
    Journal(
        "embo_molecular_medicine",
        "EMBO Molecular Medicine",
        ("EMM",),
        "embo_molecular_medicine.md",
    ),
    Journal(
        "molecular_systems_biology",
        "Molecular Systems Biology",
        ("MSB",),
        "molecular_systems_biology.md",
    ),
    Journal(
        "life_science_alliance",
        "Life Science Alliance",
        ("LSA",),
        "life_science_alliance.md",
    ),
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


def load_guidelines(journal: Optional[Journal]) -> Tuple[str, str]:
    """Load the shared guidelines and, when known, the matching journal file."""
    common_path = GUIDELINES_DIR / "common.md"
    common = common_path.read_text(encoding="utf-8")
    if journal is None:
        return common, UNKNOWN_JOURNAL_GUIDELINES
    journal_path = GUIDELINES_DIR / journal.guidelines_file
    return common, journal_path.read_text(encoding="utf-8")


def acceptance_report_path(output_path: Optional[str], manuscript_id: str) -> Path:
    """Markdown report path beside the pipeline JSON, or under ``data/output``."""
    if output_path:
        out = Path(output_path)
        return out.with_name(f"{out.stem}_acceptance_guidelines.md")
    manuscript_id = manuscript_id or "manuscript"
    return Path("data/output") / f"{manuscript_id}_acceptance_guidelines.md"


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
    output_path: Optional[str] = None,
) -> ZipStructure:
    """
    Ask the configured model whether the manuscript follows the journal guidelines.

    Writes a Markdown file and stores its path on
    ``zip_structure.acceptance_guidelines``.
    Only the common guidelines and the matched journal file are sent.
    """
    resolved = resolve_step_config(config["pipeline"][STEP_NAME])
    journal, matched_from = resolve_journal(
        getattr(zip_structure, "journal_title", "") or "",
        zip_structure.manuscript_id,
    )
    common_guidelines, journal_guidelines = load_guidelines(journal)
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

    report_path = acceptance_report_path(output_path, zip_structure.manuscript_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")

    zip_structure.journal_title = journal_title
    zip_structure.acceptance_guidelines = {
        "journal_key": journal_key,
        "journal_title": journal_title,
        "matched_from": matched_from,
        "report_path": str(report_path),
    }
    logger.info(
        "Acceptance guidelines report written",
        extra={
            "operation": "main.check_acceptance_guidelines",
            "journal_key": journal_key,
            "matched_from": matched_from,
            "report_path": str(report_path),
        },
    )
    return zip_structure
