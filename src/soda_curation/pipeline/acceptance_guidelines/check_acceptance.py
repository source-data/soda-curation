"""Check a manuscript against journal acceptance guidelines."""

from __future__ import annotations

import base64
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
from .langfuse_prompts import (
    GUIDELINES_PROMPT_NAME,
    PROMPT_LABEL,
    get_production_prompt,
)

logger = logging.getLogger(__name__)

STEP_NAME = "check_acceptance_guidelines"
# call_anthropic streams large requests; a longer report raises instead of truncating.
ANTHROPIC_MAX_OUTPUT_TOKENS = 64000


@dataclass(frozen=True)
class Journal:
    """A journal this pipeline can match to a Langfuse prompt."""

    key: str
    title: str
    prefixes: Tuple[str, ...]


# Longest manuscript-id prefix is matched first (EMBOR before any shorter EMBO* token).
# ``key`` identifies the journal in the prompt and the JSON output.
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


def _existing(path: Path) -> Optional[Path]:
    return path if path.is_file() else None


def resolve_manuscript_xml(extract_dir: Path, zip_structure: ZipStructure) -> Path:
    """Return the extracted manuscript XML.

    The archive XML is stored beside the manuscript directory or inside it.
    """
    root = Path(extract_dir)
    manuscript_id = zip_structure.manuscript_id or ""
    named = (zip_structure.xml or "").strip()
    candidates = []
    if named:
        candidates.extend([root / named, root.parent / named])
    if manuscript_id:
        candidates.extend(
            [
                root / f"{manuscript_id}.xml",
                root.parent / f"{manuscript_id}.xml",
            ]
        )
    for path in candidates:
        found = _existing(path)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"Manuscript XML not found for {manuscript_id or 'unknown manuscript'} "
        f"under {root}"
    )


def resolve_merged_pdf(extract_dir: Path, zip_structure: ZipStructure) -> Path:
    """Return the merged PDF from the archive ``pdf`` folder."""
    root = Path(extract_dir)
    manuscript_id = zip_structure.manuscript_id or ""
    pdf_dir = root / "pdf"
    named = pdf_dir / f"{manuscript_id}.pdf" if manuscript_id else None
    if named is not None:
        found = _existing(named)
        if found is not None:
            return found

    relative = (zip_structure.pdf or "").strip()
    if relative:
        found = _existing(root / relative)
        if found is not None:
            return found

    if pdf_dir.is_dir():
        pdfs = sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
        if len(pdfs) == 1:
            return pdfs[0]
    raise FileNotFoundError(
        f"Merged PDF not found in {pdf_dir} for "
        f"{manuscript_id or 'unknown manuscript'}"
    )


def _pdf_file_part(pdf_path: Path) -> dict:
    encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    return {
        "type": "file",
        "file": {
            "filename": pdf_path.name,
            "file_data": f"data:application/pdf;base64,{encoded}",
        },
    }


def _complete(provider: str, model: str, messages: list) -> object:
    if provider == "anthropic":
        validate_anthropic_model(model)
        response = call_anthropic(
            client=anthropic.Anthropic(),
            model=model,
            messages=messages,
            max_tokens=ANTHROPIC_MAX_OUTPUT_TOKENS,
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
        enable_chunking=False,
        operation="main.check_acceptance_guidelines",
    )


def check_acceptance_guidelines(
    config: Dict,
    prompt_handler,
    zip_structure: ZipStructure,
    extract_dir: Path,
) -> ZipStructure:
    """
    Ask the configured model whether the manuscript follows the journal guidelines.

    The manuscript input is the extracted XML plus the merged PDF from ``pdf/``.
    Stores the Markdown report on ``zip_structure.acceptance_guidelines``
    so it is part of the main pipeline JSON.
    The guidelines are the Langfuse ``aip_guidelines`` prompt, ``production`` label.
    It covers every journal; the matched journal is named in the user prompt.
    """
    resolved = resolve_step_config(config["pipeline"][STEP_NAME])
    journal, matched_from = resolve_journal(
        getattr(zip_structure, "journal_title", "") or "",
        zip_structure.manuscript_id,
    )
    guidelines = get_production_prompt(GUIDELINES_PROMPT_NAME)
    journal_title = (
        journal.title
        if journal is not None
        else (getattr(zip_structure, "journal_title", "") or "Unknown journal")
    )
    journal_key = journal.key if journal is not None else "unknown"

    xml_path = resolve_manuscript_xml(extract_dir, zip_structure)
    pdf_path = resolve_merged_pdf(extract_dir, zip_structure)
    xml_text = xml_path.read_text(encoding="utf-8")
    prompts = prompt_handler.get_prompt(
        step=STEP_NAME,
        variables={
            "journal_title": journal_title,
            "journal_key": journal_key,
            "guidelines": guidelines.text,
        },
    )
    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompts["user"]},
                {
                    "type": "text",
                    "text": f"Manuscript XML ({xml_path.name}):\n{xml_text}",
                },
                _pdf_file_part(pdf_path),
            ],
        },
    ]
    response = _complete(resolved["provider"], resolved["model"], messages)
    update_token_usage(
        zip_structure.cost.check_acceptance_guidelines, response, resolved["model"]
    )

    if response.choices[0].finish_reason == "length":
        raise ValueError(
            "Acceptance guidelines report was cut off at the output token limit"
        )
    report = (response.choices[0].message.content or "").strip()
    if not report:
        raise ValueError("Acceptance guidelines model returned an empty report")

    zip_structure.journal_title = journal_title
    zip_structure.acceptance_guidelines = {
        "journal_key": journal_key,
        "journal_title": journal_title,
        "matched_from": matched_from,
        "prompt_name": guidelines.name,
        "prompt_label": PROMPT_LABEL,
        "prompt_version": guidelines.version,
        "source_xml": xml_path.name,
        "source_pdf": str(pdf_path.relative_to(extract_dir)),
        "report": report,
    }
    logger.info(
        "Acceptance guidelines report stored on the pipeline JSON",
        extra={
            "operation": "main.check_acceptance_guidelines",
            "journal_key": journal_key,
            "matched_from": matched_from,
            "prompt_version": guidelines.version,
        },
    )
    return zip_structure
