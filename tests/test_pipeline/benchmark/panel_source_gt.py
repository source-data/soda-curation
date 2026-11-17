"""Panel source assignment from ground truth: stub packages, input structure, scoring.

The assignment step only reads file names: each figure's source-data containers
(zips, listed with ``infolist``) and direct files. So instead of the manuscript
package, the benchmark writes stub zips whose entries are the ground truth's file
names (empty), and builds figures and panels from the ground truth.

Ground-truth source-data fields (see scripts/check_gt_sd.py):
    figures[].panels[].sd_files       files on that panel
    figures[].sd_files                container zip(s), then figure-level files
    figures[].unassigned_sd_files     unassigned files in the figure's containers
    sd_files, unassigned_sd_files     manuscript level
Files inside a zip are written ``<zip>:<entry>``.
"""

import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
)

# Same filter as PanelSourceAssigner._get_zip_contents.
_SYSTEM_FILES = ("Thumbs.db", "desktop.ini", ".DS_Store")


def is_listed(name: str) -> bool:
    """Whether the assignment step would list this zip entry."""
    return not (
        name.endswith("/")
        or "__MACOSX" in name
        or any(s in name for s in _SYSTEM_FILES)
        or name.startswith(".")
    )


def _container(name: str) -> str:
    return name.split(":", 1)[0] if ":" in name else ""


def _figure_files(fig: Dict[str, Any]) -> List[str]:
    files = list(fig.get("sd_files", [])) + list(fig.get("unassigned_sd_files", []))
    for panel in fig.get("panels", []):
        files += panel.get("sd_files", [])
    return files


def all_files(gt: Dict[str, Any]) -> Set[str]:
    """Every source-data path in the ground truth."""
    files = set(gt.get("sd_files", [])) | set(gt.get("unassigned_sd_files", []))
    for fig in gt["figures"]:
        files |= set(_figure_files(fig))
    return files


def figure_inputs(fig: Dict[str, Any]) -> List[str]:
    """The figure's source-data inputs as the XML would give them: containers and direct files."""
    inputs: List[str] = []
    for name in _figure_files(fig):
        source = _container(name) or name
        if source not in inputs:
            inputs.append(source)
    return inputs


def figure_listing(gt: Dict[str, Any], fig: Dict[str, Any]) -> Set[str]:
    """The file list the assignment step sees for this figure."""
    files = all_files(gt)
    listing: Set[str] = set()
    for source in figure_inputs(fig):
        if source.endswith(".zip"):
            prefix = source + ":"
            listing |= {
                f for f in files if f.startswith(prefix) and is_listed(f[len(prefix) :])
            }
        else:
            listing.add(source)
    return listing


def has_sd_ground_truth(gt: Dict[str, Any]) -> bool:
    """Whether any panel of the manuscript has source-data files."""
    return any(p.get("sd_files") for f in gt["figures"] for p in f.get("panels", []))


def write_stub_package(gt: Dict[str, Any], root: Path) -> None:
    """Write stub zips (empty entries) and empty direct files under ``root``."""
    files = all_files(gt)
    entries: Dict[str, Set[str]] = {}
    for name in files:
        if ":" in name:
            entries.setdefault(_container(name), set()).add(name.split(":", 1)[1])
        elif name.endswith(".zip"):
            entries.setdefault(name, set())
    for container, names in entries.items():
        path = root / container
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w") as z:
            for entry in sorted(names):
                z.writestr(entry, b"")
    for name in files:
        if ":" not in name and not name.endswith(".zip"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def figures_from_ground_truth(gt: Dict[str, Any]) -> List[Figure]:
    """Input figures: ground-truth panel labels, no assignments, XML-like sd_files."""
    return [
        Figure(
            figure_label=fig["figure_label"],
            img_files=fig.get("img_files", []),
            sd_files=figure_inputs(fig),
            panels=[
                Panel(
                    panel_label=p["panel_label"],
                    panel_caption=p.get("panel_caption", ""),
                )
                for p in fig.get("panels", [])
            ],
        )
        for fig in gt["figures"]
    ]


def _pairs(panels: Dict[str, Iterable[str]], listing: Set[str]) -> Set[tuple]:
    return {
        (p.upper(), f) for p, files in panels.items() for f in files if f in listing
    }


def score_figure(
    expected: Dict[str, Iterable[str]],
    actual: Dict[str, Iterable[str]],
    listing: Set[str],
) -> Dict[str, Any]:
    """Score one figure's assignments against the ground truth.

    Pairs are (panel label, file), labels compared case-insensitively, restricted to
    files the step could see. Scores are None where undefined (nothing to measure).
        precision     share of assigned pairs that are correct
        recall        share of expected pairs that were assigned
        shared_recall recall on files expected on two or more panels
        off_panel     share of files expected on no panel that stayed off panels
    """
    exp, act = _pairs(expected, listing), _pairs(actual, listing)
    hits = exp & act
    per_file: Dict[str, int] = {}
    for _, f in exp:
        per_file[f] = per_file.get(f, 0) + 1
    shared = {(p, f) for p, f in exp if per_file[f] > 1}
    off = listing - {f for _, f in exp}
    off_kept = off - {f for _, f in act}

    def ratio(n: int, d: int):
        return n / d if d else None

    return {
        "precision": ratio(len(hits), len(act)),
        "recall": ratio(len(hits), len(exp)),
        "shared_recall": ratio(len(shared & act), len(shared)),
        "off_panel": ratio(len(off_kept), len(off)),
        "counts": {
            "expected": len(exp),
            "assigned": len(act),
            "correct": len(hits),
            "shared_expected": len(shared),
            "shared_correct": len(shared & act),
            "off_panel_expected": len(off),
            "off_panel_kept": len(off_kept),
            "listed": len(listing),
        },
    }
