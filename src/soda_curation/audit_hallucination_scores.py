"""
CLI for auditing hallucination scores in an existing pipeline-output JSON.

Re-runs the rapidfuzz-based ``caption_partial_ratio`` against the stored
manuscript text and reports the per-figure score that the current pipeline
*would* produce, alongside the score that the JSON currently contains.

Usage:

    poetry run python -m soda_curation.audit_hallucination_scores \\
        data/output/EMBOR-2025-62929V1-T.json

    # Override manuscript text (e.g. when auditing very old outputs that
    # did not persist manuscript_text):
    poetry run python -m soda_curation.audit_hallucination_scores \\
        data/output/old_run.json --manuscript-text path/to/manuscript.txt

Exit code is ``1`` when at least one figure's stored score disagrees with
the recomputed expected score (useful as a CI guardrail).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from ._main_utils import audit_caption_hallucination_scores


def _format_table(report: List[dict]) -> str:
    """Pretty-print the audit report as a fixed-width table."""
    header = (
        f"{'figure':<14}"
        f"{'partial_ratio':>15}"
        f"{'stored':>10}"
        f"{'expected':>10}"
        f"{'verified':>10}"
        f"{'discrepancy':>13}"
        f"  caption_preview"
    )
    separator = "-" * len(header)
    rows: List[str] = [header, separator]
    for r in report:
        expected_cell = (
            f"{r['expected_score']:>10.4f}"
            if r["expected_score"] is not None
            else f"{'n/a':>10}"
        )
        if r.get("unverifiable"):
            discrepancy_cell = f"{'unverifiable':>13}"
        else:
            discrepancy_cell = f"{('YES' if r['discrepancy'] else 'no'):>13}"
        rows.append(
            f"{r['figure_label']:<14}"
            f"{r['partial_ratio']:>15.2f}"
            f"{r['stored_score']:>10.4f}"
            f"{expected_cell}"
            f"{str(r['caption_verified']):>10}"
            f"{discrepancy_cell}"
            f"  {r['caption_preview']}"
        )
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit hallucination scores in a pipeline-output JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a pipeline-output JSON (e.g. data/output/EMBOR-2025-...json)",
    )
    parser.add_argument(
        "--manuscript-text",
        type=Path,
        default=None,
        help=(
            "Optional path to a manuscript text file to use instead of the "
            "manuscript_text stored inside the JSON."
        ),
    )
    parser.add_argument(
        "--discrepancy-tolerance",
        type=float,
        default=0.05,
        help=(
            "Absolute difference between stored and recomputed score above "
            "which a figure is flagged as a discrepancy (default: 0.05)."
        ),
    )
    args = parser.parse_args(argv)

    with args.json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    manuscript_text: Optional[str] = None
    if args.manuscript_text is not None:
        with args.manuscript_text.open("r", encoding="utf-8") as f:
            manuscript_text = f.read()

    report = audit_caption_hallucination_scores(
        data,
        manuscript_text=manuscript_text,
        discrepancy_tolerance=args.discrepancy_tolerance,
    )

    if not report:
        print("No figures found in", args.json_path, file=sys.stderr)
        return 0

    print(_format_table(report))

    discrepancies = [r for r in report if r["discrepancy"]]
    if discrepancies:
        print(
            f"\nDETECTED {len(discrepancies)} discrepancy(ies). "
            "Run the pipeline again to refresh the JSON.",
            file=sys.stderr,
        )
        return 1
    print("\nAll stored hallucination_score values agree with recomputed scores.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
