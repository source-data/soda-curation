"""Tests for the provider-independent caption verification guardrail."""

from __future__ import annotations

from src.soda_curation._main_utils import (
    UNVERIFIED_CAPTION_PLACEHOLDER,
    audit_caption_hallucination_scores,
    caption_hallucination_score,
    caption_partial_ratio,
    caption_present_in_manuscript,
    caption_similarity_ratio,
    dedupe_consecutive_paragraphs,
    finalize_figure_output,
    repair_empty_panel_markers,
    sort_panels_by_label,
    verify_captions_against_manuscript,
)
from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)


def _make_panel(label: str = "A") -> Panel:
    return Panel(panel_label=label, panel_caption=f"panel {label}")


def _make_figure(
    label: str,
    caption: str,
    *,
    caption_title: str = "",
    panels: list[Panel] | None = None,
) -> Figure:
    return Figure(
        figure_label=label,
        img_files=[f"graphic/{label}.tif"],
        sd_files=[],
        panels=list(panels) if panels else [_make_panel()],
        figure_caption=caption,
        caption_title=caption_title,
    )


def _make_zip_structure(figures: list[Figure]) -> ZipStructure:
    return ZipStructure(
        manuscript_id="TEST-001",
        xml="TEST-001.xml",
        docx="Doc/manuscript.docx",
        figures=figures,
    )


def test_caption_partial_ratio_high_when_present():
    caption = "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells."
    manuscript = (
        "Methods\nWe performed Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells. "
        "Additional details are described below.\n"
    )
    assert caption_partial_ratio(caption, manuscript) >= 90.0


def test_caption_partial_ratio_low_when_absent():
    caption = "Unrelated invented caption about XYZ123 transcription factor."
    manuscript = "Methods\nWe performed unrelated experiments on mitochondria.\n"
    assert caption_partial_ratio(caption, manuscript) < 90.0


def test_caption_partial_ratio_zero_for_empty_inputs():
    assert caption_partial_ratio("", "any text") == 0.0
    assert caption_partial_ratio("caption", "") == 0.0
    assert caption_partial_ratio("", "") == 0.0


def test_caption_similarity_ratio_penalizes_hallucinated_prefix_suffix():
    """Use ratio against the best manuscript span, not partial_ratio directly."""
    real_caption = "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells."
    manuscript = (
        "Methods. We performed "
        + real_caption
        + " Additional details are described below. " * 100
    )
    hallucinated = (
        "Invented introductory sentence. "
        + real_caption
        + " Invented closing sentence."
    )

    assert caption_similarity_ratio(real_caption, manuscript) >= 99.0
    assert caption_similarity_ratio(hallucinated, manuscript) < 90.0


def test_caption_hallucination_score_bounds():
    """Score is always in [0, 1] and inverse to caption similarity."""
    manuscript = "Methods. Result of experiment described in detail."
    perfect = caption_hallucination_score(
        "Result of experiment described in detail.", manuscript
    )
    absent = caption_hallucination_score(
        "Totally unrelated XYZ123 nonsense.", manuscript
    )

    assert 0.0 <= perfect <= 0.1
    assert 0.0 <= absent <= 1.0
    assert absent > perfect


def test_caption_present_in_manuscript_back_compat():
    """The legacy helper still works on top of caption_partial_ratio."""
    caption = "Methods of analysis are described below."
    manuscript = "Methods of analysis are described below in detail."
    ok, score = caption_present_in_manuscript(caption, manuscript, threshold=90.0)
    assert ok is True
    assert score >= 90.0


def test_verify_keeps_valid_caption_and_sets_low_score():
    """When the caption is in the manuscript, content is kept and score is low."""
    manuscript = (
        "Figure 1. Western blot analysis of FLAG-AID-Fzo1. "
        "Solidity of the mitochondrial network after Fzo1 depletion."
    )
    figure = _make_figure(
        "Figure 1",
        caption="Western blot analysis of FLAG-AID-Fzo1.",
        caption_title="Figure 1",
        panels=[_make_panel("A"), _make_panel("B")],
    )
    # Start from a non-zero score to prove the guardrail overwrites it.
    figure.hallucination_score = 0.7
    zs = _make_zip_structure([figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 0
    kept = zs.figures[0]
    assert kept.figure_caption == "Western blot analysis of FLAG-AID-Fzo1."
    assert kept.caption_title == "Figure 1"
    assert [p.panel_label for p in kept.panels] == ["A", "B"]
    # Always set by the guardrail (not zero by chance from the dataclass default)
    assert 0.0 <= kept.hallucination_score <= 0.1


def test_verify_marks_hallucinated_caption_and_clears_dependents():
    manuscript = (
        "Figure 1 caption is here.\nFigure 2 describes something concrete.\n"
        "No mention of Figure 8 at all in the manuscript body."
    )
    original_caption = (
        "Quantification of XYZ123 fluorescence intensity in transfected "
        "cells normalized to control conditions across three replicates."
    )
    bad_figure = _make_figure(
        "Figure 8",
        caption=original_caption,
        caption_title="Figure 8 (made up)",
        panels=[_make_panel("A"), _make_panel("B"), _make_panel("C")],
    )
    zs = _make_zip_structure([bad_figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 1
    figure = zs.figures[0]
    assert figure.figure_caption == original_caption
    assert figure.caption_title == "Figure 8 (made up)"
    assert figure.panels == []
    expected_score = caption_hallucination_score(original_caption, manuscript)
    assert figure.hallucination_score == expected_score
    assert figure.hallucination_score > 0.0  # non-zero proves the EMBOR bug is gone
    # Figure metadata that does not come from the AI must be preserved.
    assert figure.figure_label == "Figure 8"
    assert figure.img_files == ["graphic/Figure 8.tif"]


def test_verify_mixed_only_alters_hallucinated_one():
    manuscript = (
        "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells under "
        "control conditions. We then examined further outputs."
    )
    good_figure = _make_figure(
        "Figure 1",
        caption="Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells.",
        caption_title="Western blots",
        panels=[_make_panel("A")],
    )
    bad_original_caption = (
        "Quantification of XYZ123 fluorescence intensity across three "
        "independent biological replicates."
    )
    bad_figure = _make_figure(
        "Figure 2",
        caption=bad_original_caption,
        caption_title="Quantification",
        panels=[_make_panel("A"), _make_panel("B")],
    )
    zs = _make_zip_structure([good_figure, bad_figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 1

    kept, sanitized = zs.figures
    assert kept.figure_caption == (
        "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells."
    )
    assert kept.caption_title == "Western blots"
    assert [p.panel_label for p in kept.panels] == ["A"]
    assert 0.0 <= kept.hallucination_score <= 0.1

    assert sanitized.figure_caption == bad_original_caption
    assert sanitized.caption_title == "Quantification"
    assert sanitized.panels == []
    assert sanitized.hallucination_score == caption_hallucination_score(
        bad_original_caption, manuscript
    )
    assert sanitized.hallucination_score > 0.0


def test_verify_empty_caption_scored_as_hallucinated():
    """An empty caption stays empty and gets score 1.0."""
    manuscript = "Figure 1 actually exists in the body of the manuscript."
    figure = _make_figure("Figure 1", caption="", caption_title="", panels=[])
    zs = _make_zip_structure([figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 1
    assert zs.figures[0].figure_caption == ""
    assert zs.figures[0].caption_title == ""
    assert zs.figures[0].hallucination_score == 1.0
    assert zs.figures[0].caption_verified is False


def test_verify_clears_ev_caption_returned_for_main_figure():
    """Figure EV8 must never be accepted as the caption for main Figure 8."""
    manuscript = (
        "Extended View Figure Legends\n\n"
        "Figure EV8: Sensitivity analysis of model parameters. "
        "A) Histogram with lognormal and gamma distribution fits."
    )
    ev_caption = (
        "Figure EV8: Sensitivity analysis of model parameters. "
        "A) Histogram with lognormal and gamma distribution fits."
    )
    figure = _make_figure(
        "Figure 8",
        caption=ev_caption,
        caption_title="Sensitivity analysis of model parameters.",
        panels=[_make_panel("A")],
    )
    zs = _make_zip_structure([figure])

    unverified = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert unverified == 1
    assert zs.figures[0].figure_caption == ""
    assert zs.figures[0].caption_title == ""
    assert zs.figures[0].panels == []
    assert zs.figures[0].hallucination_score == 1.0
    assert zs.figures[0].caption_verified is False


def test_finalize_figure_output_writes_formula_and_strips_conflicting_panels():
    """Final JSON step: rapidfuzz score + no conflicting_panels when caption fails."""
    manuscript = (
        "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells under "
        "control conditions. Additional methods follow."
    )
    good = _make_figure(
        "Figure 1",
        caption="Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells.",
    )
    good.hallucination_score = 0.99  # stale value that must be overwritten
    bad = _make_figure(
        "Figure 8",
        caption="Figure 8 not present in text",
    )
    bad.hallucination_score = 0.0  # stale
    bad._conflicting_panels = [
        {
            "panel_label": "A",
            "detection_idx": 1,
            "confidence": 0.34,
            "bbox": [0.0, 0.0, 1.0, 0.5],
        }
    ]
    zs = _make_zip_structure([good, bad])

    finalize_figure_output(zs, manuscript, threshold=90.0)

    assert zs.figures[0].hallucination_score == caption_hallucination_score(
        good.figure_caption, manuscript
    )
    assert zs.figures[0].hallucination_score <= 0.1
    assert getattr(zs.figures[0], "_conflicting_panels", []) == []
    assert zs.figures[1].hallucination_score == caption_hallucination_score(
        "Figure 8 not present in text", manuscript
    )
    assert zs.figures[1].hallucination_score > 0.0
    assert zs.figures[1].figure_caption == "Figure 8 not present in text"
    assert getattr(zs.figures[1], "_conflicting_panels", []) == []
    encoded = __import__("json").dumps(
        zs,
        cls=__import__(
            "src.soda_curation.pipeline.manuscript_structure.manuscript_structure",
            fromlist=["CustomJSONEncoder"],
        ).CustomJSONEncoder,
        ensure_ascii=False,
    )
    assert "conflicting_panels" not in encoded


def test_finalize_clears_ev_caption_returned_for_main_figure():
    """Finalization catches EV/main mismatches even if earlier verify was skipped."""
    manuscript = (
        "Extended View Figure Legends\n\n"
        "Figure EV8: Sensitivity analysis of model parameters. "
        "A) Histogram with lognormal and gamma distribution fits."
    )
    figure = _make_figure(
        "Figure 8",
        caption=(
            "Figure EV8: Sensitivity analysis of model parameters. "
            "A) Histogram with lognormal and gamma distribution fits."
        ),
        caption_title="Sensitivity analysis of model parameters.",
        panels=[_make_panel("A")],
    )
    figure.hallucination_score = 0.0
    zs = _make_zip_structure([figure])

    unverified = finalize_figure_output(zs, manuscript, threshold=90.0)

    assert unverified == 1
    assert zs.figures[0].figure_caption == ""
    assert zs.figures[0].caption_title == ""
    assert len(zs.figures[0].panels) == 1
    assert zs.figures[0].panels[0].panel_label == "A"
    assert zs.figures[0].panels[0].panel_caption == ""
    assert zs.figures[0].hallucination_score == 1.0
    assert zs.figures[0].caption_verified is False


def test_finalize_keeps_empty_caption_empty():
    """Missing captions remain empty; only the score carries the warning signal."""
    manuscript = (
        "Title here. Figure 8 was mentioned in passing. "
        "Methods. Results. Many other paragraphs follow."
    )
    figure = _make_figure(
        "Figure 8",
        caption="",
        panels=[],
    )
    figure.hallucination_score = 0.0
    zs = _make_zip_structure([figure])

    finalize_figure_output(zs, manuscript, threshold=90.0)

    assert zs.figures[0].figure_caption == ""
    assert zs.figures[0].hallucination_score == 1.0
    assert getattr(zs.figures[0], "_conflicting_panels", []) == []
    assert len(zs.figures[0].panels) == 1
    assert zs.figures[0].panels[0].panel_label == "A"
    assert zs.figures[0].panels[0].panel_caption == ""


def test_json_output_omits_caption_verified_and_panel_hallucination_score():
    """Output JSON must not add caption_verified; panel scores stay internal."""
    import json as _json

    from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
        CustomJSONEncoder,
    )

    figure = _make_figure(
        "Figure 1",
        caption="Some caption.",
        panels=[
            Panel(panel_label="A", panel_caption="A. text", panel_bbox=[0, 0, 1, 1])
        ],
    )
    figure.caption_verified = False
    figure.hallucination_score = 0.05
    zs = _make_zip_structure([figure])

    encoded = _json.dumps(zs, cls=CustomJSONEncoder, ensure_ascii=False)

    assert "caption_verified" not in encoded
    assert (
        '"hallucination_score": 0.05' in encoded
        or '"hallucination_score":0.05' in encoded.replace(" ", "")
    )
    # Panel block should not repeat a hallucination_score field
    panel_section = encoded.split('"panels"')[1].split("]")[0]
    assert "hallucination_score" not in panel_section


def test_verify_score_matches_similarity_ratio_formula():
    """Score written into the figure is exactly 1 - similarity_ratio/100."""
    manuscript = (
        "Time-lapse imaging of mitochondrial dynamics in HEK293 cells expressing "
        "FLAG-AID-Fzo1, with quantification across replicates."
    )
    figure = _make_figure(
        "Figure 1",
        caption=(
            "Time-lapse imaging of mitochondrial dynamics in HEK293 cells "
            "expressing FLAG-AID-Fzo1."
        ),
    )
    zs = _make_zip_structure([figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 0
    expected = caption_hallucination_score(
        "Time-lapse imaging of mitochondrial dynamics in HEK293 cells expressing FLAG-AID-Fzo1.",
        manuscript,
    )
    assert zs.figures[0].hallucination_score == expected


def test_verify_treats_legacy_placeholder_as_extracted_text():
    """Legacy placeholder text is no longer special-cased or rewritten."""
    manuscript = "Some manuscript text without that caption."
    figure = _make_figure(
        "Figure 3",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        caption_title="",
        panels=[],
    )
    figure.hallucination_score = 1.0
    zs = _make_zip_structure([figure])

    replaced = verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert replaced == 1
    assert zs.figures[0].figure_caption == UNVERIFIED_CAPTION_PLACEHOLDER
    assert zs.figures[0].hallucination_score == caption_hallucination_score(
        UNVERIFIED_CAPTION_PLACEHOLDER, manuscript
    )


# ---------------------------------------------------------------------------
# caption_verified flag
# ---------------------------------------------------------------------------


def test_verify_sets_caption_verified_true_when_present():
    manuscript = (
        "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells under "
        "control conditions."
    )
    figure = _make_figure(
        "Figure 1",
        caption="Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells.",
    )
    zs = _make_zip_structure([figure])

    verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert zs.figures[0].caption_verified is True


def test_verify_sets_caption_verified_false_for_hallucinated():
    manuscript = "Manuscript without the suspicious caption text."
    figure = _make_figure(
        "Figure 8",
        caption="Totally invented analysis of XYZ123 across replicates.",
    )
    zs = _make_zip_structure([figure])

    verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert zs.figures[0].caption_verified is False
    assert zs.figures[0].figure_caption == (
        "Totally invented analysis of XYZ123 across replicates."
    )


def test_verify_sets_caption_verified_false_for_empty():
    manuscript = "Manuscript body without anything relevant for this figure."
    figure = _make_figure("Figure 1", caption="", panels=[])
    zs = _make_zip_structure([figure])

    verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert zs.figures[0].caption_verified is False


def test_verify_sets_caption_verified_false_for_existing_placeholder():
    manuscript = "Unrelated manuscript text."
    figure = _make_figure(
        "Figure 3",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        panels=[],
    )
    zs = _make_zip_structure([figure])

    verify_captions_against_manuscript(zs, manuscript, threshold=90.0)

    assert zs.figures[0].caption_verified is False


# ---------------------------------------------------------------------------
# repair_empty_panel_markers
# ---------------------------------------------------------------------------


def test_repair_drops_empty_panels_and_relabels_sequentially():
    """SLX4IP-style: empty A. and G. -> survivors B,C,D,E,F,H -> relabeled A..F."""
    caption_text = (
        "Figure 2. SLX4IP deficiency causes global replication stress.\n"
        "\n"
        "A.\n"
        "\n"
        "B. U2OS cells were labelled with CldU and IdU.\n"
        "C. RPE1-hTert cells were labelled with CldU and IdU.\n"
        "D. U2OS were subjected to a 20 min EdU pulse.\n"
        "E. RPE1-hTert cells were subjected to a 20 min EdU pulse.\n"
        "F. Abundance ratios across U2OS SLX4IP-/- clones.\n"
        "G.\n"
        "H. Abundance ratios across RPE1-hTert SLX4IP -/- clones.\n"
    )
    figure = _make_figure(
        "Figure 2",
        caption=caption_text,
        panels=[
            Panel(panel_label="A", panel_caption=""),
            Panel(panel_label="B", panel_caption="B. U2OS cells were labelled."),
            Panel(panel_label="C", panel_caption="C. RPE1-hTert cells were labelled."),
            Panel(panel_label="D", panel_caption="D. U2OS were subjected to EdU."),
            Panel(
                panel_label="E",
                panel_caption="E. RPE1-hTert cells were subjected to EdU.",
            ),
            Panel(panel_label="F", panel_caption="F. Abundance ratios across U2OS."),
            Panel(panel_label="G", panel_caption=""),
            Panel(
                panel_label="H", panel_caption="H. Abundance ratios across RPE1-hTert."
            ),
        ],
    )
    figure.caption_verified = True
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 2
    survivors = zs.figures[0].panels
    assert [p.panel_label for p in survivors] == ["A", "B", "C", "D", "E", "F"]

    new_caption = zs.figures[0].figure_caption
    # Orphan marker lines for dropped panels are gone
    assert "\nA.\n" not in new_caption
    assert "\nG.\n" not in new_caption
    # Leading markers have been remapped
    assert "A. U2OS cells were labelled" in new_caption
    assert "B. RPE1-hTert cells were labelled" in new_caption
    assert "C. U2OS were subjected" in new_caption
    assert "D. RPE1-hTert cells were subjected" in new_caption
    assert "E. Abundance ratios across U2OS" in new_caption
    assert "F. Abundance ratios across RPE1-hTert" in new_caption
    # The figure-level title line is preserved
    assert new_caption.splitlines()[0] == (
        "Figure 2. SLX4IP deficiency causes global replication stress."
    )


def test_repair_preserves_in_prose_letter_mentions():
    """Safe mode: in-prose mentions like 'see panel F' must NOT be remapped."""
    caption_text = (
        "Title line about a study.\n"
        "A.\n"
        "B. First real description. Refers to panel F shown later.\n"
        "C. Mentions G. inline as a sentence fragment.\n"
        "F. Real last panel.\n"
    )
    figure = _make_figure(
        "Figure 1",
        caption=caption_text,
        panels=[
            Panel(panel_label="A", panel_caption=""),
            Panel(panel_label="B", panel_caption="B. First real description."),
            Panel(panel_label="C", panel_caption="C. Mentions things."),
            Panel(panel_label="F", panel_caption="F. Real last panel."),
        ],
    )
    figure.caption_verified = True
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 1
    assert [p.panel_label for p in zs.figures[0].panels] == ["A", "B", "C"]
    caption = zs.figures[0].figure_caption
    # Leading markers were remapped
    assert "A. First real description" in caption
    assert "B. Mentions G. inline as a sentence fragment." in caption
    assert "C. Real last panel" in caption
    # Inline mention of "panel F" stays intact (safe mode)
    assert "panel F" in caption
    # Inline "G." inside the prose of the second line stays intact (safe mode)
    assert "Mentions G. inline" in caption


def test_repair_no_op_when_no_panels_are_empty():
    caption_text = "Title.\n" "A. First.\n" "B. Second.\n" "C. Third.\n"
    figure = _make_figure(
        "Figure 1",
        caption=caption_text,
        panels=[
            Panel(panel_label="A", panel_caption="A. First."),
            Panel(panel_label="B", panel_caption="B. Second."),
            Panel(panel_label="C", panel_caption="C. Third."),
        ],
    )
    figure.caption_verified = True
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 0
    assert [p.panel_label for p in zs.figures[0].panels] == ["A", "B", "C"]
    assert zs.figures[0].figure_caption == caption_text


def test_repair_preserves_unverified_figures():
    """Unverified figures carry bbox-only Panels with empty captions - leave them alone."""
    figure = _make_figure(
        "Figure 8",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        panels=[
            Panel(panel_label="", panel_caption="", panel_bbox=[0, 0, 10, 10]),
            Panel(panel_label="", panel_caption="", panel_bbox=[10, 0, 20, 10]),
        ],
    )
    figure.caption_verified = False
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 0
    assert len(zs.figures[0].panels) == 2
    assert zs.figures[0].figure_caption == UNVERIFIED_CAPTION_PLACEHOLDER


def test_repair_handles_panels_in_non_alphabetical_input_order():
    """Even if the LLM emits panels out of order, the relabel mapping is alphabetical."""
    caption_text = "A. one.\n" "B.\n" "C. three.\n" "D. four.\n"
    figure = _make_figure(
        "Figure 1",
        caption=caption_text,
        panels=[
            Panel(panel_label="D", panel_caption="D. four."),
            Panel(panel_label="A", panel_caption="A. one."),
            Panel(panel_label="C", panel_caption="C. three."),
            Panel(panel_label="B", panel_caption=""),
        ],
    )
    figure.caption_verified = True
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 1
    # Surviving panels keep their original *list* order, but their labels are
    # remapped via the alphabetical sort: A->A, C->B, D->C.
    survivors_by_label = sorted(zs.figures[0].panels, key=lambda p: p.panel_label)
    assert [p.panel_label for p in survivors_by_label] == ["A", "B", "C"]
    captions = {p.panel_label: p.panel_caption for p in survivors_by_label}
    assert captions["A"] == "A. one."
    assert captions["B"] == "C. three."
    assert captions["C"] == "D. four."

    caption = zs.figures[0].figure_caption
    assert "\nB.\n" not in caption  # orphan dropped
    assert "A. one." in caption
    assert "B. three." in caption
    assert "C. four." in caption


def test_repair_supports_paren_markers():
    """Caption authors sometimes use 'A)' instead of 'A.'."""
    caption_text = (
        "Title.\n"
        "A)\n"
        "B) U2OS cells were labelled.\n"
        "C) RPE1-hTert cells were labelled.\n"
    )
    figure = _make_figure(
        "Figure 1",
        caption=caption_text,
        panels=[
            Panel(panel_label="A", panel_caption=""),
            Panel(panel_label="B", panel_caption="B) U2OS cells were labelled."),
            Panel(panel_label="C", panel_caption="C) RPE1-hTert cells were labelled."),
        ],
    )
    figure.caption_verified = True
    zs = _make_zip_structure([figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 1
    assert [p.panel_label for p in zs.figures[0].panels] == ["A", "B"]
    caption = zs.figures[0].figure_caption
    assert "\nA)\n" not in caption
    assert "A) U2OS cells were labelled" in caption
    assert "B) RPE1-hTert cells were labelled" in caption


def test_repair_handles_mixed_figures():
    """A verified figure is repaired; an unverified figure next to it is not."""
    verified_caption = "Title.\n" "A.\n" "B. Real text.\n"
    verified_figure = _make_figure(
        "Figure 1",
        caption=verified_caption,
        panels=[
            Panel(panel_label="A", panel_caption=""),
            Panel(panel_label="B", panel_caption="B. Real text."),
        ],
    )
    verified_figure.caption_verified = True

    unverified_figure = _make_figure(
        "Figure 8",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        panels=[
            Panel(panel_label="", panel_caption="", panel_bbox=[0, 0, 10, 10]),
        ],
    )
    unverified_figure.caption_verified = False

    zs = _make_zip_structure([verified_figure, unverified_figure])

    removed = repair_empty_panel_markers(zs)

    assert removed == 1
    assert [p.panel_label for p in zs.figures[0].panels] == ["A"]
    assert "A. Real text." in zs.figures[0].figure_caption
    assert "\nA.\n" not in zs.figures[0].figure_caption
    # Unverified figure untouched
    assert len(zs.figures[1].panels) == 1
    assert zs.figures[1].figure_caption == UNVERIFIED_CAPTION_PLACEHOLDER


# ---------------------------------------------------------------------------
# sort_panels_by_label
# ---------------------------------------------------------------------------


def test_sort_panels_orders_alphabetically_in_verified_figure():
    """Panels arrive in arbitrary order from object detection; output must be A,B,C..."""
    figure = _make_figure(
        "Figure 2",
        caption="Real caption.",
        panels=[
            Panel(panel_label="G", panel_caption="g"),
            Panel(panel_label="A", panel_caption="a"),
            Panel(panel_label="D", panel_caption="d"),
            Panel(panel_label="B", panel_caption="b"),
            Panel(panel_label="C", panel_caption="c"),
            Panel(panel_label="E", panel_caption="e"),
            Panel(panel_label="F", panel_caption="f"),
            Panel(panel_label="H", panel_caption="h"),
        ],
    )
    zs = _make_zip_structure([figure])

    changed = sort_panels_by_label(zs)

    assert changed == 1
    assert [p.panel_label for p in zs.figures[0].panels] == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]


def test_sort_panels_is_case_insensitive():
    figure = _make_figure(
        "Figure 1",
        caption="Real caption.",
        panels=[
            Panel(panel_label="b", panel_caption="b"),
            Panel(panel_label="A", panel_caption="a"),
            Panel(panel_label="c", panel_caption="c"),
        ],
    )
    zs = _make_zip_structure([figure])

    sort_panels_by_label(zs)

    assert [p.panel_label for p in zs.figures[0].panels] == ["A", "b", "c"]


def test_sort_panels_pushes_empty_labels_to_end():
    figure = _make_figure(
        "Figure 1",
        caption="Real caption.",
        panels=[
            Panel(panel_label="", panel_caption=""),
            Panel(panel_label="B", panel_caption="b"),
            Panel(panel_label="", panel_caption=""),
            Panel(panel_label="A", panel_caption="a"),
        ],
    )
    zs = _make_zip_structure([figure])

    sort_panels_by_label(zs)

    labels = [p.panel_label for p in zs.figures[0].panels]
    assert labels == ["A", "B", "", ""]


def test_sort_panels_is_stable_for_unverified_bbox_only_figure():
    """Unverified figures have all-empty labels; stable sort preserves detection order."""
    panels = [
        Panel(panel_label="", panel_caption="", panel_bbox=[0, 0, 10, 10]),
        Panel(panel_label="", panel_caption="", panel_bbox=[10, 0, 20, 10]),
        Panel(panel_label="", panel_caption="", panel_bbox=[20, 0, 30, 10]),
    ]
    figure = _make_figure(
        "Figure 8",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        panels=panels,
    )
    figure.caption_verified = False
    zs = _make_zip_structure([figure])

    sort_panels_by_label(zs)

    # Stable: detection order preserved (bbox x-coordinates ascending)
    assert [p.panel_bbox[0] for p in zs.figures[0].panels] == [0, 10, 20]


def test_sort_panels_no_op_when_already_sorted():
    figure = _make_figure(
        "Figure 1",
        caption="Real caption.",
        panels=[
            Panel(panel_label="A", panel_caption="a"),
            Panel(panel_label="B", panel_caption="b"),
            Panel(panel_label="C", panel_caption="c"),
        ],
    )
    zs = _make_zip_structure([figure])

    changed = sort_panels_by_label(zs)

    assert changed == 0
    assert [p.panel_label for p in zs.figures[0].panels] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# dedupe_consecutive_paragraphs
# ---------------------------------------------------------------------------


def test_dedupe_collapses_repetition_loop_to_single_paragraph():
    """The classic EMBOR data_availability bug: 48 copies of the same sentence."""
    sentence = (
        "The datasets and computer code produced in this study are available "
        "in the following databases:"
    )
    text = "\n\n".join([sentence] * 48 + ["The real content follows here."])

    result = dedupe_consecutive_paragraphs(text)

    assert result.count(sentence) == 1
    assert "The real content follows here." in result
    assert result == f"{sentence}\n\n" + "The real content follows here."


def test_dedupe_preserves_non_consecutive_repetition():
    """``A B A`` is intentional and stays untouched; only consecutive runs collapse."""
    text = "Paragraph A.\n\nParagraph B.\n\nParagraph A."

    result = dedupe_consecutive_paragraphs(text)

    assert result == text


def test_dedupe_is_no_op_on_clean_text():
    text = (
        "The datasets are available in OMERO.\n\n"
        "The mass spec data are at PRIDE.\n\n"
        "The code is on GitHub."
    )

    assert dedupe_consecutive_paragraphs(text) == text


def test_dedupe_handles_whitespace_only_variations_as_duplicates():
    """Trailing whitespace and tabs should not defeat dedupe."""
    text = "Hello world.\n\nHello world.  \n\nHello world.\t"

    result = dedupe_consecutive_paragraphs(text)

    assert result == "Hello world."


def test_dedupe_strips_empty_paragraphs_and_excess_blank_lines():
    text = "First.\n\n\n\nFirst.\n\n\n\nSecond."

    result = dedupe_consecutive_paragraphs(text)

    assert result == "First.\n\nSecond."


def test_dedupe_returns_empty_for_empty_input():
    assert dedupe_consecutive_paragraphs("") == ""
    assert dedupe_consecutive_paragraphs(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# conflicting_panels stays empty for unverified figures (regression test)
# ---------------------------------------------------------------------------


def test_embor_figure8_json_regression_finalize_from_output_file():
    """Regression: stale EMBOR output had score 0 + conflicting_panels for Figure 8."""
    import json as _json

    from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
        CustomJSONEncoder,
        Figure,
        ZipStructure,
    )

    path = "data/output/EMBOR-2025-62929V1-T.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except FileNotFoundError:
        return  # optional fixture in CI

    manuscript = data.get("manuscript_text", "")
    fig8 = next(f for f in data["figures"] if f["figure_label"] == "Figure 8")
    figure = Figure(
        figure_label="Figure 8",
        img_files=fig8["img_files"],
        sd_files=fig8.get("sd_files", []),
        figure_caption=fig8["figure_caption"],
        caption_title=fig8.get("caption_title", ""),
        panels=[],
        hallucination_score=fig8.get("hallucination_score", 0),
    )
    if fig8.get("conflicting_panels"):
        figure._conflicting_panels = list(fig8["conflicting_panels"])
    zs = ZipStructure(
        manuscript_id=data.get("manuscript_id", ""),
        xml=data.get("xml", ""),
        docx=data.get("docx", ""),
        pdf=data.get("pdf", ""),
        appendix=data.get("appendix", []),
        figures=[figure],
        manuscript_text=manuscript,
    )

    finalize_figure_output(zs, manuscript, threshold=90.0)

    assert zs.figures[0].hallucination_score > 0.0
    assert getattr(zs.figures[0], "_conflicting_panels", []) == []
    encoded = _json.dumps(zs, cls=CustomJSONEncoder, ensure_ascii=False)
    assert "conflicting_panels" not in encoded


def test_unverified_figure_has_no_conflicting_panels_in_serialized_json():
    """Regression: old runs leaked conflicting_panels into the JSON output for
    figures whose caption could not be verified. The field is pipeline-only
    (_conflicting_panels) and must never appear in serialized output."""
    import json as _json

    from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
        CustomJSONEncoder,
    )

    figure = _make_figure(
        "Figure 8",
        caption=UNVERIFIED_CAPTION_PLACEHOLDER,
        panels=[
            Panel(panel_label="", panel_caption="", panel_bbox=[0, 0, 10, 10]),
        ],
    )
    figure.caption_verified = False
    # Simulate the assignment performed by _handle_unverified_figure
    figure._conflicting_panels = []
    zs = _make_zip_structure([figure])

    encoded = _json.dumps(zs, cls=CustomJSONEncoder, ensure_ascii=False)

    assert "conflicting_panels" not in encoded


# ---------------------------------------------------------------------------
# audit_caption_hallucination_scores
# ---------------------------------------------------------------------------


def test_audit_flags_old_llm_apology_caption_with_int_zero_score():
    """Real bug from EMBOR-2025-62929V1-T.json: stored score=0 for ``"Figure 8
    not present in text"``. The audit must flag this as a discrepancy because
    1 - similarity_ratio/100 of that caption against the manuscript is clearly
    not zero."""
    manuscript = (
        "Title here. We performed experiments on mitochondrial dynamics in HEK293 cells. "
        "Figure 1 shows the western blot analysis. Figure 8 was mentioned in passing. "
        "Methods. Results. Many other paragraphs of real manuscript text follow."
    )
    serialized = {
        "manuscript_text": manuscript,
        "figures": [
            {
                "figure_label": "Figure 8",
                "figure_caption": "Figure 8 not present in text",
                "hallucination_score": 0,  # int, matches the old bug
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    assert len(report) == 1
    row = report[0]
    assert row["figure_label"] == "Figure 8"
    assert row["caption_verified"] is True  # missing field defaults to True
    assert row["expected_score"] is not None
    assert row["expected_score"] > 0.0
    assert row["stored_score"] == 0.0
    assert row["discrepancy"] is True


def test_audit_passes_when_stored_score_matches_recomputed():
    """A correctly-scored verified figure must not raise a discrepancy."""
    manuscript = (
        "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells under "
        "control conditions. Additional methods follow."
    )
    serialized = {
        "manuscript_text": manuscript,
        "figures": [
            {
                "figure_label": "Figure 1",
                "figure_caption": (
                    "Western blot analysis of FLAG-AID-Fzo1 in HEK293 cells."
                ),
                "hallucination_score": 0.0,
                "caption_verified": True,
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    assert len(report) == 1
    row = report[0]
    assert row["similarity_ratio"] >= 90.0
    assert row["caption_verified"] is True
    assert row["expected_score"] is not None
    assert row["expected_score"] <= 0.1
    assert row["discrepancy"] is False


def test_audit_scores_placeholder_like_any_other_caption():
    """Legacy placeholder text is now just caption text for audit purposes."""
    serialized = {
        "manuscript_text": "Unrelated manuscript body.",
        "figures": [
            {
                "figure_label": "Figure 8",
                "figure_caption": UNVERIFIED_CAPTION_PLACEHOLDER,
                "hallucination_score": 0.32,
                "caption_verified": False,
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    row = report[0]
    assert row["caption_verified"] is False
    assert row["expected_score"] is not None


def test_audit_recomputes_even_when_caption_verified_false():
    """caption_verified is internal state; scores are recomputed from text."""
    serialized = {
        "manuscript_text": "Manuscript body.",
        "figures": [
            {
                "figure_label": "Figure 5",
                "figure_caption": "some non-placeholder text",
                "hallucination_score": 0.42,
                "caption_verified": False,
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    row = report[0]
    assert row["caption_verified"] is False
    assert row["expected_score"] is not None


def test_audit_flags_empty_caption_with_zero_stored_score():
    """Empty caption (caption_verified True by default for old data) is a
    discrepancy when stored as 0: expected score is 1.0."""
    serialized = {
        "manuscript_text": "Some manuscript text.",
        "figures": [
            {
                "figure_label": "Figure 3",
                "figure_caption": "",
                "hallucination_score": 0,
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    row = report[0]
    assert row["caption_verified"] is True  # default for old data
    assert row["expected_score"] == 1.0
    assert row["discrepancy"] is True


def test_audit_accepts_explicit_manuscript_override():
    """When manuscript text is missing in JSON, the override argument is used."""
    serialized = {
        # No manuscript_text key at all
        "figures": [
            {
                "figure_label": "Figure 1",
                "figure_caption": "Some caption that is in the manuscript override.",
                "hallucination_score": 0.0,
            }
        ],
    }
    override = (
        "Long manuscript override that contains Some caption that is in the "
        "manuscript override word for word."
    )

    report = audit_caption_hallucination_scores(serialized, manuscript_text=override)

    assert report[0]["similarity_ratio"] >= 90.0
    assert report[0]["discrepancy"] is False


def test_audit_truncates_long_caption_preview():
    """Captions over 80 chars are previewed with an ellipsis (table readability)."""
    long_caption = "a" * 200
    serialized = {
        "manuscript_text": "x",
        "figures": [
            {
                "figure_label": "Figure 1",
                "figure_caption": long_caption,
                "hallucination_score": 1.0,
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    assert report[0]["caption_preview"].endswith("...")
    assert len(report[0]["caption_preview"]) <= 83


def test_audit_handles_non_numeric_stored_score():
    """Defensive: a malformed stored score should not crash the audit."""
    serialized = {
        "manuscript_text": "anything",
        "figures": [
            {
                "figure_label": "Figure 1",
                "figure_caption": "x",
                "hallucination_score": "not a number",
            }
        ],
    }

    report = audit_caption_hallucination_scores(serialized)

    assert report[0]["stored_score"] == 0.0  # default fallback
    assert report[0]["discrepancy"] is True  # 0.0 vs expected non-zero


def test_audit_returns_empty_when_no_figures():
    assert audit_caption_hallucination_scores({"figures": []}) == []
    assert audit_caption_hallucination_scores({}) == []
