"""Tests for journal matching and the acceptance-guidelines pipeline step."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lxml import etree

from src.soda_curation.pipeline.acceptance_guidelines.check_acceptance import (
    check_acceptance_guidelines,
    load_guidelines,
    resolve_journal,
    resolve_manuscript_xml,
    resolve_merged_pdf,
)
from src.soda_curation.pipeline.acceptance_guidelines.langfuse_prompts import (
    FetchedPrompt,
    get_production_prompt,
    reset_client,
)
from src.soda_curation.pipeline.anthropic_utils import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicResponseWrapper,
    AnthropicUsage,
    validate_anthropic_model,
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
            "model": "gpt-6-astra",
            "prompts": {
                "system": "Review the manuscript. Return Markdown.",
                "user": (
                    "Journal: $journal_title\n"
                    "Key: $journal_key\n"
                    "Common:\n$common_guidelines\n"
                    "Journal-specific:\n$journal_guidelines\n"
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


def test_resolve_merged_pdf_uses_the_pdf_folder(tmp_path):
    manuscript_id = "EMBOR-2025-62929V1-T"
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    merged = pdf_dir / f"{manuscript_id}.pdf"
    merged.write_bytes(b"%PDF-1.4")
    (pdf_dir / "other.pdf").write_bytes(b"%PDF-other")
    structure = ZipStructure(manuscript_id=manuscript_id, pdf="pdf/other.pdf")

    assert resolve_merged_pdf(tmp_path, structure) == merged


def test_resolve_manuscript_xml_finds_sibling_of_the_extract_dir(tmp_path):
    manuscript_id = "EMBOR-2025-62929V1-T"
    extract_dir = tmp_path / manuscript_id
    extract_dir.mkdir()
    xml_path = tmp_path / f"{manuscript_id}.xml"
    xml_path.write_text(
        "<article><journal-title>EMBO Reports</journal-title></article>",
        encoding="utf-8",
    )
    structure = ZipStructure(manuscript_id=manuscript_id, xml=f"{manuscript_id}.xml")

    assert resolve_manuscript_xml(extract_dir, structure) == xml_path


def test_check_acceptance_guidelines_stores_report_on_the_json(tmp_path):
    manuscript_id = "EMBOR-2025-62929V1-T"
    xml_path = tmp_path / f"{manuscript_id}.xml"
    xml_path.write_text("<article><title>Example</title></article>", encoding="utf-8")
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    (pdf_dir / f"{manuscript_id}.pdf").write_bytes(b"%PDF-1.4 merged")
    structure = ZipStructure(
        manuscript_id=manuscript_id,
        journal_title="EMBO Reports",
        xml=xml_path.name,
        cost=ProcessingCost(),
    )
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
            extract_dir=tmp_path,
        )

    messages = complete.call_args.args[2]
    user_content = messages[1]["content"]
    user_text = user_content[0]["text"]
    assert "EMBO Reports" in user_text
    assert "no additional checklist items" in user_text
    assert "Please upload the graphical abstract" not in user_text
    assert "<article><title>Example</title></article>" in user_content[1]["text"]
    assert user_content[2]["type"] == "file"
    assert user_content[2]["file"]["filename"] == f"{manuscript_id}.pdf"
    assert user_content[2]["file"]["file_data"].startswith(
        "data:application/pdf;base64,"
    )

    assert result.acceptance_guidelines["journal_key"] == "embo_reports"
    assert result.acceptance_guidelines["matched_from"] == "journal_title"
    assert result.acceptance_guidelines["report"] == "# Report\n\nOverall: unclear."
    assert result.acceptance_guidelines["prompt_label"] == "production"
    assert result.acceptance_guidelines["common_prompt_version"] == "3"
    assert result.acceptance_guidelines["journal_prompt_version"] == "5"
    assert result.acceptance_guidelines["source_xml"] == f"{manuscript_id}.xml"
    assert result.acceptance_guidelines["source_pdf"] == f"pdf/{manuscript_id}.pdf"
    assert result.cost.check_acceptance_guidelines.total_tokens == 15


def test_check_acceptance_guidelines_rejects_a_truncated_report(tmp_path):
    manuscript_id = "EMBOR-2025-62929V1-T"
    (tmp_path / f"{manuscript_id}.xml").write_text("<article/>", encoding="utf-8")
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf" / f"{manuscript_id}.pdf").write_bytes(b"%PDF-1.4 merged")
    structure = ZipStructure(
        manuscript_id=manuscript_id,
        journal_title="EMBO Reports",
        cost=ProcessingCost(),
    )
    response = MagicMock()
    response.choices[0].finish_reason = "length"
    response.choices[0].message.content = "# Report\n\n1. Figures: pass\n2."
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15

    with (
        patch(
            "src.soda_curation.pipeline.acceptance_guidelines.check_acceptance"
            "._complete",
            return_value=response,
        ),
        patch(
            "src.soda_curation.pipeline.acceptance_guidelines.check_acceptance"
            ".get_production_prompt",
            side_effect=lambda name: _prompt(name, "rules", "1"),
        ),
        pytest.raises(ValueError, match="cut off"),
    ):
        check_acceptance_guidelines(
            config={"pipeline": _pipeline_config()},
            prompt_handler=PromptHandler(_pipeline_config()),
            zip_structure=structure,
            extract_dir=tmp_path,
        )
    assert structure.acceptance_guidelines == {}


def test_anthropic_max_tokens_stop_reads_as_openai_length():
    wrapper = AnthropicResponseWrapper(
        content="partial",
        parsed=None,
        usage=AnthropicUsage(1, 1),
        model=DEFAULT_ANTHROPIC_MODEL,
        stop_reason="max_tokens",
    )
    assert wrapper.choices[0].finish_reason == "length"
    validate_anthropic_model(DEFAULT_ANTHROPIC_MODEL)


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
