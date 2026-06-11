"""Utility functions supporting the main entry point."""

import html
import logging
import os
import re
import shutil
import string
import tempfile
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from src.soda_curation.pipeline.manuscript_structure.manuscript_structure import (
    Figure,
    Panel,
    ZipStructure,
)

logger = logging.getLogger(__name__)


def validate_paths(
    zip_path: str, config_path: str, output_path: Optional[str] = None
) -> None:
    """
    Validate input paths and ensure output directory exists.

    Args:
        zip_path: Path to input ZIP file
        config_path: Path to configuration file
        output_path: Optional path to output JSON file

    Raises:
        ValueError: If required paths are not provided
        FileNotFoundError: If input files don't exist
        zipfile.BadZipFile: If ZIP file is corrupted or invalid
    """
    if not zip_path:
        raise ValueError("ZIP path must be provided")
    if not config_path:
        raise ValueError("config path must be provided")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file {zip_path} does not exist")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    # Validate that the ZIP file is actually a valid ZIP file
    try:
        import zipfile

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Try to read the file list to validate it's a proper ZIP
            file_list = zip_ref.namelist()
            if not file_list:
                raise zipfile.BadZipFile(f"ZIP file {zip_path} is empty")
    except zipfile.BadZipFile as e:
        raise zipfile.BadZipFile(f"Invalid ZIP file {zip_path}: {e}")
    except Exception as e:
        raise zipfile.BadZipFile(f"Cannot read ZIP file {zip_path}: {e}")

    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Ensured output directory exists: {output_dir}")


def setup_extract_dir() -> Path:
    """
    Create and return a temporary extraction directory.

    Returns:
        Path to temporary extraction directory

    Note:
        The caller is responsible for cleaning up this directory using cleanup_extract_dir
    """
    temp_dir = tempfile.mkdtemp(prefix="soda_curation_")
    extract_dir = Path(temp_dir)
    logger.info(f"Created temporary extraction directory: {extract_dir}")
    return extract_dir


def cleanup_extract_dir(extract_dir: Path) -> None:
    """
    Clean up temporary extraction directory.

    Args:
        extract_dir: Path to temporary extraction directory
    """
    if extract_dir and extract_dir.exists():
        try:
            shutil.rmtree(extract_dir)
            logger.info(f"Cleaned up temporary directory: {extract_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {str(e)}")


def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from text while preserving content.

    Args:
        text: HTML text to strip

    Returns:
        str: Text with HTML tags removed
    """
    if not text:
        return ""

    # Use BeautifulSoup to strip HTML tags
    try:
        # Only use BeautifulSoup if the text contains HTML-like content
        if "<" in text and ">" in text:
            soup = BeautifulSoup(text, "html.parser")
            # Get text while preserving spaces between elements
            return soup.get_text(" ", strip=True)
        return text
    except Exception:
        # Fallback to regex if BeautifulSoup fails
        return re.sub(r"<[^>]+>", " ", text)


def normalize(s: str, do_not_remove: str = "", do: Optional[List[str]] = None) -> str:
    if not s:
        return ""

    if do is None:
        do = [
            "ctrl",
            "strip",
            "lower",
            "html_unescape",
            "html_tags",
            "punctuation",
            "unicode",
            "special_chars",
        ]

    # ---- 1. Preserve newline and carriage returns. ----
    def remove_control_characters(txt: str, excluded: List[str] = ["Cc", "Cf"]) -> str:
        cleaned = []
        for ch in txt:
            # Keep newline and carriage returns
            if ch in ("\n", "\r"):
                cleaned.append(ch)
            # Remove all other control/format characters
            elif unicodedata.category(ch) not in excluded:
                cleaned.append(ch)
        return "".join(cleaned)

    # ---- 2. Actual normalize steps ----
    if "ctrl" in do:
        s = remove_control_characters(s)

    if "strip" in do:
        s = s.strip()

    if "lower" in do:
        s = s.lower()

    if "html_unescape" in do:
        s = html.unescape(s)

    if "html_tags" in do:
        s = strip_html_tags(s)

    # First, ensure line breaks are treated as word separators by adding space.
    s = re.sub(r"(\S)(\n|\r)(\S)", r"\1 \3", s)

    # Then replace line breaks with spaces
    s = s.replace("\n", " ").replace("\r", " ")

    if "html_tags" in do:
        # Ensure space after closing tags
        s = re.sub(r"(</[^>]+>)(\S)", r"\1 \2", s)
        # Ensure space before opening tags
        s = re.sub(r"(\S)(<[^/][^>]*>)", r"\1 \2", s)

    if "punctuation" in do:
        punctuation = string.punctuation
        for c in do_not_remove:
            punctuation = punctuation.replace(c, "")
        s = re.sub(f"[{re.escape(punctuation)}]", " ", s)

    if "unicode" in do:
        s = (
            unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )

    # Remove multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def normalize_text(
    text: str,
    strip_html: bool = True,
    keep_chars: str = "",
    config: Optional[dict] = None,
) -> str:
    """
    Enhanced text normalization with configurable options:
      - Removes HTML tags (optional)
      - Lowercases text
      - Normalizes whitespace and line breaks
      - Preserves special character sequences
      - Supports Unicode normalization
      - Configurable punctuation handling

    Args:
        text: Text to normalize
        strip_html: Whether to remove HTML tags
        keep_chars: Characters to preserve when removing punctuation
        config: Additional configuration options

    Returns:
        str: Normalized text
    """
    if not text:
        return ""

    # Set up normalization options based on parameters
    if config is None:
        config = {}

    # Build the list of normalization operations
    operations = ["strip", "lower", "line_breaks", "special_chars"]

    # Add optional operations based on parameters
    if strip_html:
        operations.append("html_tags")
        operations.append("html_unescape")

    # Add advanced operations if requested in config
    if config.get("normalize_unicode", False):
        operations.append("unicode")

    if config.get("remove_punctuation", False):
        operations.append("punctuation")

    if config.get("remove_control_chars", True):
        operations.append("ctrl")

    # Apply the enhanced normalization with our configured options
    return normalize(text, do_not_remove=keep_chars, do=operations)


def exact_match_check(
    extracted_text: str, source_text: str, strip_html: bool = False
) -> bool:
    """
    Check if normalized extracted text exists within normalized source text.

    Both texts are HTML by default (manuscript text and AI extractions are
    verbatim HTML), so HTML markup is part of the comparison unless
    ``strip_html=True`` is passed.

    Args:
        extracted_text: Text to check for hallucination
        source_text: Original source text to compare against
        strip_html: Whether to strip HTML markup before comparing

    Returns:
        bool: True if extract is found in source, False otherwise
    """
    if not extracted_text or not source_text:
        return False

    # Create normalized versions of both texts
    norm_extracted = normalize_text(extracted_text, strip_html=strip_html)
    norm_source = normalize_text(source_text, strip_html=strip_html)

    # Check if the normalized extracted text is in the normalized source
    return norm_extracted in norm_source


def fuzzy_match_score(
    extracted_text: str, source_text: str, strip_html: bool = False
) -> float:
    """
    Calculate fuzzy match similarity score between extracted text and source text.

    HTML markup is included in the comparison by default so that the score
    also verifies the AI preserved the input HTML verbatim; pass
    ``strip_html=True`` to compare visible text only.

    Args:
        extracted_text: Text to check for hallucination
        source_text: Original source text to compare against
        strip_html: Whether to strip HTML markup before comparing

    Returns:
        float: Similarity score between 0-100
    """
    if not extracted_text or not source_text:
        return 0.0

    # Create normalized versions for fuzzy matching
    norm_extracted = normalize_text(extracted_text, strip_html=strip_html)
    norm_source = normalize_text(source_text, strip_html=strip_html)

    # Get the best partial ratio score
    return fuzz.partial_ratio(norm_extracted, norm_source)


UNVERIFIED_CAPTION_PLACEHOLDER = (
    "FIGURE CAPTION NOT PRESENT OR POSSIBLY HALLUCINATED, PLEASE CHECK."
)

_LEADING_FIGURE_LABEL_RE = re.compile(
    r"^\s*(?:extended\s+view\s+)?(?:figure|fig\.?)\s*"
    r"(?P<label>(?:ev\s*)?\d+[a-z]?|\d+[a-z]?\s*ev)\b",
    re.IGNORECASE,
)


def _normalized_figure_label_scope(text: str) -> Optional[tuple[str, bool]]:
    """
    Return (figure_number, is_ev) for a leading figure label, if present.

    Examples:
    - "Figure 8" -> ("8", False)
    - "Figure EV8" -> ("8", True)
    - "Figure 8EV" -> ("8", True)

    Captions without a leading figure label return None; we do not reject those
    here because many extractor outputs contain only the caption body/panels.
    """
    match = _LEADING_FIGURE_LABEL_RE.match(text or "")
    if not match:
        return None

    label = re.sub(r"\s+", "", match.group("label")).upper()
    is_ev = label.startswith("EV") or label.endswith("EV")
    label = label.replace("EV", "")
    return label, is_ev


def _caption_label_mismatches_figure(figure_label: str, caption: str) -> bool:
    """True when a caption is clearly for a different figure namespace/number."""
    target = _normalized_figure_label_scope(figure_label)
    extracted = _normalized_figure_label_scope(caption)
    if target is None or extracted is None:
        return False
    return target != extracted


def caption_similarity_ratio(
    caption: str, manuscript_text: str, strip_html: bool = False
) -> float:
    """
    Normalized rapidfuzz ratio for ``caption`` against its best manuscript span.

    A direct ``fuzz.ratio(caption, full_manuscript)`` is not useful because a
    caption is much shorter than the manuscript and would score near zero even
    when it is present verbatim. Instead, we first locate the best aligned span
    in the manuscript, then score ``caption`` against that span with
    ``fuzz.ratio``. This preserves the "caption inside large text" behavior
    while penalizing hallucinated prefixes/suffixes that ``partial_ratio`` can
    be too forgiving about.

    Both the manuscript and the AI-extracted caption are HTML, so by default
    (``strip_html=False``) HTML markup takes part in the comparison: a high
    score certifies that the AI returned the exact HTML present in the input.
    Pass ``strip_html=True`` to compare visible text only.
    """
    if not caption or not manuscript_text:
        return 0.0
    norm_caption = normalize_text(caption, strip_html=strip_html)
    norm_source = normalize_text(manuscript_text, strip_html=strip_html)
    if not norm_caption or not norm_source:
        return 0.0

    alignment = fuzz.partial_ratio_alignment(norm_caption, norm_source)
    matched_span = norm_source[alignment.dest_start : alignment.dest_end]
    if not matched_span:
        return 0.0
    return float(fuzz.ratio(norm_caption, matched_span))


def caption_partial_ratio(
    caption: str, manuscript_text: str, strip_html: bool = False
) -> float:
    """
    Backwards-compatible name for the caption similarity score.

    Historically this returned ``fuzz.partial_ratio``. It now returns the
    ratio-against-best-span score used for hallucination scoring.
    """
    return caption_similarity_ratio(caption, manuscript_text, strip_html=strip_html)


def caption_hallucination_score(
    caption: str, manuscript_text: str, strip_html: bool = False
) -> float:
    """
    Hallucination score for one caption in [0, 1].

    Defined as ``1 - similarity_ratio/100``: 0 means the caption is present
    verbatim (after normalization), 1 means it is fully absent. This is the
    single source of truth for the per-figure hallucination signal exposed to
    the frontend.
    """
    return max(
        0.0,
        min(
            1.0,
            1.0
            - caption_similarity_ratio(caption, manuscript_text, strip_html=strip_html)
            / 100.0,
        ),
    )


def caption_present_in_manuscript(
    caption: str,
    manuscript_text: str,
    threshold: float = 90.0,
    strip_html: bool = False,
) -> tuple[bool, float]:
    """Convenience: returns (similarity_ratio >= threshold, similarity_ratio)."""
    score = caption_similarity_ratio(caption, manuscript_text, strip_html=strip_html)
    return score >= threshold, score


def verify_captions_against_manuscript(
    zip_structure: ZipStructure,
    manuscript_text: str,
    threshold: float = 90.0,
    strip_html: bool = False,
) -> int:
    """
    Score every figure caption with rapidfuzz and mark suspect captions.

    Provider-independent guardrail. For each figure in ``zip_structure.figures``:

    - **Score (sent to the frontend):** ``figure.hallucination_score`` is
      always set to ``1 - similarity_ratio/100`` computed against the
      LLM-extracted caption (0 = verbatim, 1 = absent). This is the single,
      continuous, deterministic signal the frontend can threshold itself.
      Empty captions yield score ``1.0`` naturally (``similarity_ratio == 0``).
    - **Cascade (pipeline-internal guard):** when the caption is empty *or*
      ``similarity_ratio < threshold``, the figure is marked
      ``caption_verified = False`` and the caption-derived panel list is
      cleared so downstream steps cannot fabricate panel metadata. The caption
      text itself is never modified: empty stays empty, and suspect non-empty
      captions are returned exactly as extracted.

    Returns the number of figures marked unverified this run.
    """
    unverified = 0
    for figure in zip_structure.figures:
        figure._hallucination_source_caption = figure.figure_caption  # type: ignore[attr-defined]

        if _caption_label_mismatches_figure(figure.figure_label, figure.figure_caption):
            logger.warning(
                "Caption label does not match requested figure; clearing caption",
                extra={
                    "figure_label": figure.figure_label,
                    "caption_preview": figure.figure_caption[:120],
                    "reason": "caption_label_mismatch",
                },
            )
            figure.figure_caption = ""
            figure.caption_title = ""
            figure.panels = []
            figure._conflicting_panels = []
            figure.hallucination_score = 1.0
            figure.caption_verified = False
            unverified += 1
            continue

        similarity_ratio = caption_similarity_ratio(
            figure.figure_caption, manuscript_text, strip_html=strip_html
        )
        figure.hallucination_score = caption_hallucination_score(
            figure.figure_caption, manuscript_text, strip_html=strip_html
        )

        if not figure.figure_caption or similarity_ratio < threshold:
            figure._conflicting_panels = []
            logger.warning(
                "Caption not found in manuscript; marking as unverified",
                extra={
                    "figure_label": figure.figure_label,
                    "similarity_ratio": similarity_ratio,
                    "threshold": threshold,
                    "hallucination_score": figure.hallucination_score,
                },
            )
            figure.panels = []
            figure.caption_verified = False
            unverified += 1
        else:
            figure.caption_verified = True

    return unverified


def _hallucination_score_source_caption(figure: Figure) -> str:
    """Caption text used for rapidfuzz scoring."""
    stored = getattr(figure, "_hallucination_source_caption", None)
    if stored and str(stored).strip():
        return str(stored)
    return figure.figure_caption or ""


def finalize_figure_output(
    zip_structure: ZipStructure,
    manuscript_text: str,
    threshold: float = 90.0,
    strip_html: bool = False,
) -> int:
    """
    Mandatory last pass before JSON: rapidfuzz scores + unverified cleanup.

    Even when an earlier pipeline step was skipped (e.g. ``verify_captions`` is
    non-critical and failed), this guarantees:

    - Every figure's ``hallucination_score`` is ``1 - similarity_ratio/100``.
    - Figures with zero or one panel are normalized to exactly one panel object.
    - Figures below ``threshold`` keep their extracted caption text, but their
      caption-derived panels and private conflict metadata are cleared.
    """
    if not (manuscript_text or "").strip():
        logger.error(
            "Cannot finalize figure output: manuscript_text is empty",
            extra={"figure_count": len(zip_structure.figures)},
        )
        return 0

    unverified = 0
    for figure in zip_structure.figures:
        source_caption = figure.figure_caption or ""
        if source_caption:
            figure._hallucination_source_caption = source_caption  # type: ignore[attr-defined]

        if _caption_label_mismatches_figure(figure.figure_label, source_caption):
            logger.warning(
                "Caption label does not match requested figure during finalization",
                extra={
                    "figure_label": figure.figure_label,
                    "caption_preview": source_caption[:120],
                    "reason": "caption_label_mismatch",
                },
            )
            figure.figure_caption = ""
            figure.caption_title = ""
            figure.panels = []
            figure._conflicting_panels = []
            figure.hallucination_score = 1.0
            figure.caption_verified = False
            unverified += 1
        else:
            similarity_ratio = caption_similarity_ratio(
                source_caption, manuscript_text, strip_html=strip_html
            )
            figure.hallucination_score = caption_hallucination_score(
                source_caption, manuscript_text, strip_html=strip_html
            )

            if not source_caption.strip() or similarity_ratio < threshold:
                figure.caption_verified = False
                figure._conflicting_panels = []
                figure.panels = []
                unverified += 1
            else:
                figure.caption_verified = True

        ensure_single_panel_figure(figure)

    return unverified


def ensure_single_panel_figure(figure: Figure) -> bool:
    """
    Normalize figures with zero or one panel to exactly one panel object.

    A figure with no panel substructure (0 panels) and one with a single panel
    are treated the same: the output always contains one ``Panel`` with label
    ``A`` (when unlabeled) and the figure caption as ``panel_caption`` when the
    panel caption is empty. Figures with two or more panels are unchanged.
    """
    if len(figure.panels) > 1:
        return False

    figure_caption = figure.figure_caption or ""

    if len(figure.panels) == 1:
        panel = figure.panels[0]
        if not (panel.panel_label or "").strip():
            panel.panel_label = "A"
        if not (panel.panel_caption or "").strip():
            panel.panel_caption = figure_caption
        return False

    figure.panels = [
        Panel(
            panel_label="A",
            panel_caption=figure_caption,
        )
    ]
    return True


# Match a panel-marker letter at the start of a line, optionally followed by
# end-of-line/whitespace, with either ``.`` or ``)`` as the separator. Safe mode:
# only leading markers (start of line) are eligible for rewriting; in-prose
# mentions like ``"see panel F"`` or ``"shown in G)"`` are intentionally left
# untouched.
_PANEL_MARKER_LEADING = re.compile(r"^([A-Z])([.)])(?=\s|$)", re.MULTILINE)

# Match an orphan line that is just a single marker letter, e.g. ``"A."`` or
# ``"G)"`` possibly surrounded by whitespace. Used to strip the empty marker
# lines after their phantom panel has been dropped.
_PANEL_MARKER_ORPHAN_LINE = re.compile(r"^\s*([A-Z])[.)]\s*$")


def _strip_orphan_marker_lines(text: str, dropped_labels: set[str]) -> str:
    """Remove lines whose entire content is a marker for a dropped panel."""
    if not text or not dropped_labels:
        return text
    upper_dropped = {label.upper() for label in dropped_labels if label}
    kept_lines: list[str] = []
    for line in text.splitlines():
        m = _PANEL_MARKER_ORPHAN_LINE.match(line)
        if m and m.group(1).upper() in upper_dropped:
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _rewrite_leading_panel_markers(text: str, relabel: Dict[str, str]) -> str:
    """
    Atomically remap leading panel-marker letters using ``relabel`` (uppercase).

    Only matches at the start of a line (re.MULTILINE), so in-prose mentions
    are not touched. All substitutions are computed against the original text
    in a single ``re.sub`` pass, so chained mappings like ``B->A, C->B`` cannot
    cascade.
    """
    if not text or not relabel:
        return text
    upper_map = {k.upper(): v for k, v in relabel.items()}

    def _replace(match: "re.Match[str]") -> str:
        letter = match.group(1).upper()
        suffix = match.group(2)
        new_letter = upper_map.get(letter)
        if new_letter is None:
            return match.group(0)
        return f"{new_letter}{suffix}"

    return _PANEL_MARKER_LEADING.sub(_replace, text)


def repair_empty_panel_markers(zip_structure: ZipStructure) -> int:
    """
    Repair phantom panel markers emitted by LLM caption extractors.

    LLM caption extractors faithfully echo whatever the author wrote, which
    occasionally includes stray panel-letter markers like ``"A."`` or
    ``"G."`` followed by no descriptive text. These do not correspond to real
    panels in the figure but they offset every subsequent panel label
    downstream — e.g. a figure that actually has 6 panels ``A..F`` ends up
    with 8 panel objects labelled ``A, B, C, D, E, F, G, H`` where ``A`` and
    ``G`` are empty.

    For every figure with ``caption_verified == True`` this function:

    1. Drops panels whose ``panel_caption`` is empty/whitespace.
    2. Relabels the survivors sequentially ``A, B, C, ...`` in alphabetical
       order of their previous labels (stable wrt. the original sort order).
    3. Rewrites the *leading* panel markers inside ``figure.figure_caption``
       (start-of-line only, safe mode) using the same mapping, and strips
       orphan marker lines that belonged to the dropped panels. In-prose
       mentions ("see panel G", "shown in F)") are left untouched on purpose:
       relabeling them is heuristic and risky.

    Unverified figures are skipped — their empty-caption ``Panel`` objects
    legitimately expose bounding boxes from object detection and must not be
    relabeled.

    Returns the total number of panels dropped across all figures.
    """
    total_dropped = 0
    for figure in zip_structure.figures:
        if not getattr(figure, "caption_verified", True):
            continue
        if not figure.panels:
            continue

        original_count = len(figure.panels)
        survivors = [p for p in figure.panels if (p.panel_caption or "").strip()]
        dropped_panels = [
            p for p in figure.panels if not (p.panel_caption or "").strip()
        ]
        dropped_count = len(dropped_panels)

        if dropped_count == 0:
            continue

        dropped_labels = {p.panel_label for p in dropped_panels if p.panel_label}

        # Sort survivors by their original label so the first alphabetically
        # becomes ``A``, the next ``B``, etc. This is what the figures actually
        # look like in practice (panels are alphabetical left-to-right /
        # top-to-bottom), and matches the user's expectation that a dropped
        # ``A`` means the original ``B`` is the real ``A``.
        survivors_sorted = sorted(
            survivors, key=lambda p: (p.panel_label or "").upper()
        )
        relabel: Dict[str, str] = {}
        for idx, panel in enumerate(survivors_sorted):
            new_label = chr(ord("A") + idx)
            old_label = (panel.panel_label or "").upper()
            if old_label and old_label != new_label:
                relabel[old_label] = new_label

        for panel in survivors:
            old = (panel.panel_label or "").upper()
            if old in relabel:
                panel.panel_label = relabel[old]

        new_caption = figure.figure_caption or ""
        if new_caption:
            new_caption = _strip_orphan_marker_lines(new_caption, dropped_labels)
            new_caption = _rewrite_leading_panel_markers(new_caption, relabel)

        logger.warning(
            "Repaired phantom panel markers in figure",
            extra={
                "figure_label": figure.figure_label,
                "dropped_panel_count": dropped_count,
                "original_panel_count": original_count,
                "remaining_panel_count": len(survivors),
                "dropped_labels": sorted(dropped_labels),
                "relabel_map": relabel,
            },
        )

        figure.panels = survivors
        figure.figure_caption = new_caption
        total_dropped += dropped_count

    return total_dropped


def sort_panels_by_label(zip_structure: ZipStructure) -> int:
    """
    Sort each figure's ``panels`` list alphabetically by ``panel_label``.

    - Empty labels are pushed to the end of the list so bbox-only panels in
      unverified figures don't push real labels around.
    - Sort is case-insensitive and stable, so panels with the same uppercase
      label keep their relative order from the previous pipeline steps.

    Returns the number of figures whose panel order actually changed.
    """
    changed = 0
    for figure in zip_structure.figures:
        if not figure.panels:
            continue
        before = [id(p) for p in figure.panels]
        figure.panels.sort(
            key=lambda p: (
                not (p.panel_label or "").strip(),  # empties last
                (p.panel_label or "").upper(),
            )
        )
        after = [id(p) for p in figure.panels]
        if before != after:
            changed += 1
    return changed


# ---------------------------------------------------------------------------
# Defensive text cleanup for LLM outputs
# ---------------------------------------------------------------------------


def audit_caption_hallucination_scores(
    zip_structure_dict: Dict,
    manuscript_text: Optional[str] = None,
    discrepancy_tolerance: float = 0.05,
    strip_html: bool = False,
) -> List[Dict]:
    """
    Recompute each figure's hallucination score from a serialized pipeline output.

    Provides a provider-independent cross-check for the unified
    ``1 - similarity_ratio/100`` scoring on any pipeline JSON. Useful both for
    auditing historical runs that predate this scoring and for spot-checking
    new runs.

    Semantics match :func:`verify_captions_against_manuscript`:

    - The stored ``hallucination_score`` should equal
      ``1 - similarity_ratio/100`` of the current ``figure_caption`` against
      the manuscript text. This is true even when the caption is marked
      unverified: the caption text is preserved in JSON, so the score remains
      fully recomputable.

    Returns one dict per figure with ``figure_label``, ``caption_preview``,
    ``similarity_ratio``, ``stored_score``, ``expected_score``,
    ``caption_verified``, and ``discrepancy``.
    """
    if manuscript_text is None:
        manuscript_text = str(zip_structure_dict.get("manuscript_text", "") or "")

    report: List[Dict] = []
    for fig in zip_structure_dict.get("figures", []):
        caption = str(fig.get("figure_caption", "") or "")
        stored_raw = fig.get("hallucination_score", 0.0)
        try:
            stored = float(stored_raw)
        except (TypeError, ValueError):
            stored = 0.0

        caption_verified = bool(fig.get("caption_verified", True))
        similarity = caption_similarity_ratio(
            caption, manuscript_text, strip_html=strip_html
        )
        expected = max(0.0, min(1.0, 1.0 - similarity / 100.0))
        discrepancy = abs(stored - expected) > discrepancy_tolerance

        report.append(
            {
                "figure_label": fig.get("figure_label", ""),
                "caption_preview": (
                    (caption[:80] + "...") if len(caption) > 80 else caption
                ),
                "similarity_ratio": round(similarity, 2),
                # Backwards-compatible key for the current CLI/tests.
                "partial_ratio": round(similarity, 2),
                "stored_score": round(stored, 4),
                "expected_score": round(expected, 4),
                "caption_verified": caption_verified,
                "discrepancy": discrepancy,
            }
        )
    return report


def dedupe_consecutive_paragraphs(text: str) -> str:
    """
    Collapse runs of identical paragraphs back to a single occurrence.

    Models occasionally fall into a repetition loop when generating long-form
    text and emit the same sentence (or paragraph) dozens of times in a row
    before recovering. The classic case observed in the wild is a data
    availability section that starts with ``"The datasets and computer code
    produced in this study are available in the following databases:"``
    repeated 40+ times before the real URLs appear.

    This helper splits ``text`` on blank-line boundaries (``\\n\\n+``),
    compares each paragraph against the previous one after whitespace
    normalization, and drops consecutive duplicates. Non-consecutive
    repetition (e.g. ``A B A``) is intentionally preserved.

    Empty/blank paragraphs are also dropped so we never emit triple newlines.
    """
    if not text:
        return text
    parts = re.split(r"\n{2,}", text)
    deduped: list[str] = []
    last_norm: Optional[str] = None
    for part in parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if not normalized:
            continue
        if normalized == last_norm:
            continue
        deduped.append(part.strip())
        last_norm = normalized
    return "\n\n".join(deduped)


def calculate_hallucination_score(
    extracted_text: str, source_text: str, strip_html: bool = False
) -> float:
    """
    Calculate a 0-1 hallucination possibility score.

    Uses the same formula as :func:`caption_hallucination_score`:
    ``1 - similarity_ratio/100`` against the best matching manuscript span.
    """
    return caption_hallucination_score(
        extracted_text, source_text, strip_html=strip_html
    )
