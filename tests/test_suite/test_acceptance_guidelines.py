"""Tests for journal matching and the acceptance-guidelines pipeline step."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lxml import etree

from src.soda_curation.pipeline.acceptance_guidelines.check_acceptance import (
    acceptance_report_path,
    check_acceptance_guidelines,
    load_guidelines,
    resolve_journal,
)
from src.soda_curation.pipeline.acceptance_guidelines.langfuse_prompts import (
    FetchedPrompt,
    get_production_prompt,
    reset_client,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    ProcessingCost,
    ZipStructure,
)
from src.soda_curation.pipeline.prompt_handler import PromptHandler

SAMPLE_XML = Path(__file__).parent / "test_data" / "EMBOJ-DUMMY-ZIP.xml"


def _pipeline_config() -> dict:
    return {
        "check_acceptance_guidelines": {
            "model": "gpt-5.4-mini",
            "prompts": {
                "system": "Review the manuscript. Return Markdown.",
                "user": (
                    "Journal: $journal_title\n"
                    "Key: $journal_key\n"
                    "Common:\n$common_guidelines\n"
                    "Journal-specific:\n$journal_guidelines\n"
                    "Manuscript:\n$manuscript_text\n"
                ),
            },
        }
    }


def test_resolve_journal_from_title():
    journal, matched_from = resolve_journal("EMBO Reports", "ignored")
    assert journal is not None
    assert journal.key == "embo_reports"
    assert matched_from == "journal_title"

    journal, matched_from = resolve_journal("embo journal", "EMBOR-1")
    assert journal.key == "the_embo_journal"
    assert matched_from == "journal_title"


def test_resolve_journal_from_manuscript_id_when_title_missing():
    cases = {
        "EMBOR-2025-62929V1-T": "embo_reports",
        "EMBOJ-DUMMY-ZIP": "the_embo_journal",
        "EMM-2023-18636": "embo_molecular_medicine",
        "MSB-2024-1": "molecular_systems_biology",
        "LSA-2025-03296-TR": "life_science_alliance",
    }
    for manuscript_id, key in cases.items():
        journal, matched_from = resolve_journal("", manuscript_id)
        assert journal is not None
        assert journal.key == key
        assert matched_from == "manuscript_id"


def test_unknown_journal():
    journal, matched_from = resolve_journal("Some Other Journal", "OTHER-1")
    assert journal is None
    assert matched_from == "unmatched"


def test_sample_xml_journal_title():
    root = etree.parse(str(SAMPLE_XML)).getroot()
    nodes = root.xpath("//journal-title")
    title = nodes[0].text.strip()
    journal, matched_from = resolve_journal(title, "EMBOJ-DUMMY-ZIP")
    assert title == "The EMBO Journal"
    assert journal.key == "the_embo_journal"
    assert matched_from == "journal_title"


def _prompt(name: str, text: str, version: str = "1") -> FetchedPrompt:
    return FetchedPrompt(name=name, label="production", version=version, text=text)


def test_load_guidelines_fetches_common_and_the_matched_journal():
    journal, _ = resolve_journal("EMBO Reports", "")
    fetched = {
        "common": _prompt(
            "common",
            "shared embo press rules\n"
            "Life Science Alliance rules are not in this file",
            "2",
        ),
        "embo_reports": _prompt(
            "embo_reports",
            "EMBO Reports\nThe Paper Explained\nno additional checklist items",
        ),
    }
    with patch(
        "src.soda_curation.pipeline.acceptance_guidelines.check_acceptance"
        ".get_production_prompt",
        side_effect=lambda name: fetched[name],
    ) as get_prompt:
        common, specific = load_guidelines(journal)

    assert [item.args[0] for item in get_prompt.call_args_list] == [
        "common",
        "embo_reports",
    ]
    assert specific is not None
    assert "shared embo press rules" in common.text.lower()
    assert "The Paper Explained" in specific.text
    assert "Molecular Systems Biology" not in specific.text
    assert common.version == "2"


def test_check_acceptance_guidelines_writes_markdown(tmp_path):
    structure = ZipStructure(
        manuscript_id="EMBOR-2025-62929V1-T",
        journal_title="EMBO Reports",
        cost=ProcessingCost(),
    )
    output_json = tmp_path / "EMBOR-2025-62929V1-T.json"
    response = MagicMock()
    response.choices[0].message.content = "# Report\n\nOverall: unclear."
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15

    prompts = {
        "common": _prompt("common", "shared embo press rules", "3"),
        "embo_reports": _prompt(
            "embo_reports",
            "no additional checklist items",
            "5",
        ),
    }
    with (
        patch(
            "src.soda_curation.pipeline.acceptance_guidelines.check_acceptance"
            "._complete",
            return_value=response,
        ) as complete,
        patch(
            "src.soda_curation.pipeline.acceptance_guidelines.check_acceptance"
            ".get_production_prompt",
            side_effect=lambda name: prompts[name],
        ),
    ):
        result = check_acceptance_guidelines(
            config={"pipeline": _pipeline_config()},
            prompt_handler=PromptHandler(_pipeline_config()),
            zip_structure=structure,
            manuscript_text="<p>Manuscript body</p>",
            output_path=str(output_json),
        )

    messages = complete.call_args.args[2]
    user_prompt = messages[1]["content"]
    assert "EMBO Reports" in user_prompt
    assert "no additional checklist items" in user_prompt
    assert "Please upload the graphical abstract" not in user_prompt
    assert "<p>Manuscript body</p>" in user_prompt

    report_path = acceptance_report_path(str(output_json), structure.manuscript_id)
    assert report_path.read_text(encoding="utf-8").startswith("# Report")
    assert result.acceptance_guidelines["journal_key"] == "embo_reports"
    assert result.acceptance_guidelines["matched_from"] == "journal_title"
    assert result.acceptance_guidelines["report_path"] == str(report_path)
    assert result.acceptance_guidelines["prompt_label"] == "production"
    assert result.acceptance_guidelines["common_prompt_version"] == "3"
    assert result.acceptance_guidelines["journal_prompt_version"] == "5"
    assert result.cost.check_acceptance_guidelines.total_tokens == 15


def test_get_production_prompt_requires_acceptance_project_keys(monkeypatch):
    reset_client()
    monkeypatch.delenv("LANGFUSE_ACCEPTANCE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_ACCEPTANCE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "qc-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "qc-secret")
    try:
        with pytest.raises(RuntimeError, match="AIP-guidelines"):
            get_production_prompt("common")
    finally:
        reset_client()


def test_get_production_prompt_reads_production_label(monkeypatch):
    reset_client()
    monkeypatch.setenv("LANGFUSE_ACCEPTANCE_PUBLIC_KEY", "pk-acceptance")
    monkeypatch.setenv("LANGFUSE_ACCEPTANCE_SECRET_KEY", "sk-acceptance")
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    prompt = MagicMock()
    prompt.prompt = "shared rules"
    prompt.version = 4
    prompt.name = "common"
    client = MagicMock()
    client.get_prompt.return_value = prompt

    try:
        with patch("langfuse.Langfuse", return_value=client) as langfuse_cls:
            fetched = get_production_prompt("common")
            again = get_production_prompt("common")

        assert langfuse_cls.call_args.kwargs["public_key"] == "pk-acceptance"
        assert langfuse_cls.call_args.kwargs["secret_key"] == "sk-acceptance"
        assert langfuse_cls.call_args.kwargs["host"] == "https://cloud.langfuse.com"
        client.get_prompt.assert_called_once_with("common", label="production")
        assert fetched.text == "shared rules"
        assert fetched.version == "4"
        assert again is fetched
    finally:
        reset_client()
