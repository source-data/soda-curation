"""Tests for the panel source benchmark's ground-truth helpers (stub packages, scoring)."""

import zipfile
from types import SimpleNamespace

import pytest

from src.soda_curation.pipeline.assign_panel_source.assign_panel_source_base import (
    PanelSourceAssigner,
)
from tests.test_pipeline.benchmark.panel_source_gt import (
    figure_inputs,
    figure_listing,
    figures_from_ground_truth,
    has_sd_ground_truth,
    score_figure,
    write_stub_package,
)

Z1 = "suppl_data/Figure 1.zip"
Z2 = "suppl_data/Figure 2.zip"
EV = "suppl_data/Figure EV1.zip"


@pytest.fixture
def gt():
    return {
        "sd_files": ["suppl_data/Dataset EV1.xlsx"],
        "unassigned_sd_files": [f"{EV}:EV1/a.tif", "suppl_data/Checklist.pdf"],
        "figures": [
            {
                "figure_label": "Figure 1",
                "sd_files": [Z1, f"{Z1}:Figure 1/all.xlsx"],
                "unassigned_sd_files": [f"{Z1}:Figure 1/README.txt"],
                "panels": [
                    {"panel_label": "a", "sd_files": [f"{Z1}:Figure 1/1A/a.csv"]},
                    {
                        "panel_label": "b",
                        "sd_files": [
                            f"{Z1}:Figure 1/1B-C/bc.xlsx",
                            "suppl_data/1B.csv.gz",
                        ],
                    },
                    {"panel_label": "c", "sd_files": [f"{Z1}:Figure 1/1B-C/bc.xlsx"]},
                ],
            },
            {
                "figure_label": "Figure 2",
                "sd_files": [Z2],
                "unassigned_sd_files": [
                    f"{Z2}:__MACOSX/Figure 2/._x.tif",
                    f"{Z2}:Figure 2/.DS_Store",
                ],
                "panels": [
                    {"panel_label": "A", "sd_files": [f"{Z2}:Figure 2/2A/x.tif"]}
                ],
            },
            {
                "figure_label": "Figure 3",
                "sd_files": [],
                "panels": [{"panel_label": "A"}],
            },
        ],
    }


def test_figure_inputs_are_containers_then_direct_files(gt):
    assert figure_inputs(gt["figures"][0]) == [Z1, "suppl_data/1B.csv.gz"]
    assert figure_inputs(gt["figures"][2]) == []


def test_figure_listing_filters_system_files(gt):
    assert figure_listing(gt, gt["figures"][0]) == {
        f"{Z1}:Figure 1/all.xlsx",
        f"{Z1}:Figure 1/README.txt",
        f"{Z1}:Figure 1/1A/a.csv",
        f"{Z1}:Figure 1/1B-C/bc.xlsx",
        "suppl_data/1B.csv.gz",
    }
    assert figure_listing(gt, gt["figures"][1]) == {f"{Z2}:Figure 2/2A/x.tif"}
    assert figure_listing(gt, gt["figures"][2]) == set()


def test_stub_package_matches_assigner_listing(gt, tmp_path):
    write_stub_package(gt, tmp_path)
    with zipfile.ZipFile(tmp_path / Z1) as z:
        assert all(z.read(n) == b"" for n in z.namelist())
    assert (tmp_path / "suppl_data/1B.csv.gz").exists()
    assert (tmp_path / EV).exists()  # unowned containers are written too
    fake = SimpleNamespace(extraction_dir=tmp_path)
    for fig in gt["figures"]:
        listed = PanelSourceAssigner._get_zip_contents(fake, figure_inputs(fig))
        assert set(listed) == figure_listing(gt, fig)
        assert len(listed) == len(set(listed))


def test_figures_from_ground_truth_have_labels_but_no_assignments(gt):
    figures = figures_from_ground_truth(gt)
    assert [p.panel_label for p in figures[0].panels] == ["a", "b", "c"]
    assert all(p.sd_files == [] for f in figures for p in f.panels)
    assert figures[0].sd_files == [Z1, "suppl_data/1B.csv.gz"]
    assert figures[0].unassigned_sd_files == []


def test_has_sd_ground_truth(gt):
    assert has_sd_ground_truth(gt)
    assert not has_sd_ground_truth({"figures": [{"panels": [{"panel_label": "A"}]}]})


def test_score_perfect_case_insensitive(gt):
    fig = gt["figures"][0]
    listing = figure_listing(gt, fig)
    expected = {p["panel_label"]: p["sd_files"] for p in fig["panels"]}
    actual = {k.upper(): v for k, v in expected.items()}
    s = score_figure(expected, actual, listing)
    assert (s["precision"], s["recall"], s["shared_recall"], s["off_panel"]) == (
        1,
        1,
        1,
        1,
    )
    assert s["counts"]["shared_expected"] == 2


def test_score_shared_file_left_unassigned_and_wrong_extra(gt):
    fig = gt["figures"][0]
    listing = figure_listing(gt, fig)
    expected = {p["panel_label"]: p["sd_files"] for p in fig["panels"]}
    actual = {
        "A": [f"{Z1}:Figure 1/1A/a.csv", f"{Z1}:Figure 1/README.txt"],
        "B": ["suppl_data/1B.csv.gz"],
        "C": [],
    }
    s = score_figure(expected, actual, listing)
    # correct: A/a.csv, B/1B.csv.gz; wrong: A/README; missed: B and C bc.xlsx
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 4)
    assert s["shared_recall"] == 0
    # off-panel files: all.xlsx kept, README put on A
    assert s["off_panel"] == pytest.approx(1 / 2)


def test_score_ignores_files_outside_listing_and_undefined_scores():
    s = score_figure({"A": ["x", "other figure"]}, {}, {"x"})
    assert s["recall"] == 0
    assert (
        s["precision"] is None and s["shared_recall"] is None and s["off_panel"] is None
    )
    assert s["counts"]["expected"] == 1
