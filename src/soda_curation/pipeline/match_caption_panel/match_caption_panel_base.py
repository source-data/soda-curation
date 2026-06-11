import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from pydantic import BaseModel

from ...pipeline.prompt_handler import PromptHandler
from ..ai_observability import summarize_text
from ..manuscript_structure.manuscript_structure import Panel, ZipStructure
from .object_detection import convert_to_pil_image, create_object_detection

logger = logging.getLogger(__name__)


class PanelObject(BaseModel):
    """Model for a list of panels."""

    panel_label: str
    panel_caption: str


class MatchPanelCaption(ABC):
    def __init__(
        self, config: Dict[str, Any], prompt_handler: PromptHandler, extract_dir: Path
    ):
        """Initialize with configuration."""
        self.config = config
        self.prompt_handler = prompt_handler
        self.extract_dir = Path(extract_dir)
        self.figure_images: Dict = {}
        self._validate_config()
        self.object_detector = create_object_detection(config)

    @staticmethod
    def _align_panel_object_with_catalog(
        panel_obj: PanelObject, original_panels: Dict[str, Panel]
    ) -> PanelObject:
        """Map vision output to caption-derived labels; caption always comes from catalog."""
        label = (panel_obj.panel_label or "").strip()
        if not label:
            return PanelObject(panel_label="", panel_caption="")
        for key, orig in original_panels.items():
            if key.upper() == label.upper():
                return PanelObject(panel_label=key, panel_caption=orig.panel_caption)
        logger.warning(
            "Panel label %r from vision matcher is not in caption-derived panels %s; "
            "treating crop as unlabeled for assignment",
            label,
            list(original_panels.keys()),
        )
        return PanelObject(panel_label="", panel_caption="")

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    def process_figures(self, zip_structure: ZipStructure) -> ZipStructure:
        """Process all figures in the manuscript."""
        self.zip_structure = zip_structure
        self.figure_images = {}
        for figure in zip_structure.figures:
            try:
                logger.info(
                    "Processing figure for panel-caption matching",
                    extra={
                        "operation": "main.match_caption_panel",
                        "figure_label": figure.figure_label,
                        "figure_caption_summary": summarize_text(figure.figure_caption),
                        "caption_verified": getattr(figure, "caption_verified", True),
                    },
                )

                if not getattr(figure, "caption_verified", True):
                    self._handle_unverified_figure(figure)
                    continue

                # Store original panels in a dictionary for quick lookup by label
                original_panels = {panel.panel_label: panel for panel in figure.panels}

                # Convert figure file to PIL Image
                full_path = self.extract_dir / figure.img_files[0]
                if not full_path.exists():
                    raise FileNotFoundError(f"File not found: {full_path}")
                image, _ = convert_to_pil_image(str(full_path))
                self.figure_images[figure.figure_label] = image

                # Debug: Check what we got from convert_to_pil_image
                logger.debug(
                    f"convert_to_pil_image returned: image type={type(image)}, image={image}"
                )

                # Additional validation before passing to detect_panels
                if not hasattr(image, "mode") or not hasattr(image, "size"):
                    logger.error(
                        f"Invalid image object for {figure.figure_label}: "
                        f"type={type(image)}, has_mode={hasattr(image, 'mode')}, "
                        f"has_size={hasattr(image, 'size')}, value={image}"
                    )
                    raise TypeError(
                        f"convert_to_pil_image returned invalid object: {type(image)}"
                    )

                # Get only bounding boxes from detection
                detected_regions = self.object_detector.detect_panels(image)
                logger.info(
                    "Detected panel candidate regions",
                    extra={
                        "operation": "main.match_caption_panel",
                        "figure_label": figure.figure_label,
                        "detected_region_count": len(detected_regions),
                    },
                )

                if not detected_regions:
                    logger.warning(
                        f"No panels detected in figure {figure.figure_label}"
                    )
                    continue

                # Process each detected region and collect AI results
                panel_matches = []
                for idx, detection in enumerate(detected_regions):
                    if detection["confidence"] < 0.25:
                        logger.warning(
                            f"Low confidence detection ({detection['confidence']:.2f}) in figure {figure.figure_label}"
                        )
                        continue

                    encoded_image = self._extract_panel_image(image, detection["bbox"])

                    if encoded_image:
                        logger.info(
                            "Sending panel crop to AI matcher",
                            extra={
                                "operation": "main.match_caption_panel",
                                "figure_label": figure.figure_label,
                                "detection_index": idx,
                                "bbox_confidence": detection["confidence"],
                                "caption_summary": summarize_text(
                                    figure.figure_caption
                                ),
                                "encoded_image_chars": len(encoded_image),
                            },
                        )
                        panel_object = self._match_panel_caption(
                            encoded_image,
                            figure.figure_caption,
                            figure.panels,
                        )
                        panel_object = (
                            PanelObject(**json.loads(panel_object))
                            if isinstance(panel_object, str)
                            else panel_object
                        )
                        panel_object = self._align_panel_object_with_catalog(
                            panel_object, original_panels
                        )

                        # Store the detection index with the panel match
                        panel_matches.append(
                            {
                                "panel_object": panel_object,
                                "detection": detection,
                                "detection_idx": idx,
                            }
                        )

                # Resolve any duplicate panel label assignments and add unmatched detections
                processed_panels = self._resolve_panel_conflicts(
                    figure, panel_matches, original_panels
                )

                # Update figure panels
                figure.panels = processed_panels

            except FileNotFoundError as e:
                logger.warning(
                    "Figure image missing; continuing with next figure",
                    extra={
                        "operation": "main.match_caption_panel",
                        "figure_label": figure.figure_label,
                        "severity": "recoverable",
                        "reason": "figure_image_missing",
                        "error": str(e),
                    },
                )
                continue
            except Exception as e:
                logger.warning(
                    "Recoverable failure while processing figure; continuing",
                    extra={
                        "operation": "main.match_caption_panel",
                        "figure_label": figure.figure_label,
                        "severity": "recoverable",
                        "reason": "figure_processing_error",
                        "error": str(e),
                    },
                )
                continue

        return zip_structure

    def get_figure_images_and_captions(self) -> List[Tuple[str, str, str]]:
        """Return base64-encoded figure images and their captions.

        Returns:
            List of (figure_label, base64_encoded_image, figure_caption) tuples.
            Call ``process_figures`` first to populate the image cache.
        """
        result = []
        if not hasattr(self, "zip_structure") or not self.zip_structure:
            logger.warning("No zip structure available. Run process_figures first.")
            return result

        for figure in self.zip_structure.figures:
            try:
                if figure.figure_label in self.figure_images:
                    image = self.figure_images[figure.figure_label]
                    buffered = io.BytesIO()
                    image.save(buffered, format="JPEG", quality=80)
                    encoded_image = base64.b64encode(buffered.getvalue()).decode(
                        "utf-8"
                    )
                    result.append(
                        (figure.figure_label, encoded_image, figure.figure_caption)
                    )
                else:
                    logger.warning(
                        f"Figure image not found in cache: {figure.figure_label}"
                    )
            except Exception as e:
                logger.error(f"Error encoding figure {figure.figure_label}: {str(e)}")

        return result

    def _handle_unverified_figure(self, figure: Any) -> None:
        """
        Honest detection-only path for figures whose caption could not be verified.

        Runs object detection so we still surface the spatial truth (bounding
        boxes + confidences) but does not call the vision LLM and does not run
        conflict resolution. Each detected region becomes a Panel with
        ``panel_label=""`` and ``panel_caption=""``. This avoids inventing
        labels we cannot back up with caption text.
        """
        figure.panels = []
        figure._conflicting_panels = []
        if not figure.img_files:
            logger.warning(
                "Unverified figure has no image files; nothing to detect",
                extra={
                    "operation": "main.match_caption_panel",
                    "figure_label": figure.figure_label,
                    "severity": "recoverable",
                    "reason": "no_image_for_unverified_figure",
                },
            )
            return
        full_path = self.extract_dir / figure.img_files[0]
        if not full_path.exists():
            logger.warning(
                "Unverified figure image missing; skipping detection",
                extra={
                    "operation": "main.match_caption_panel",
                    "figure_label": figure.figure_label,
                    "severity": "recoverable",
                    "reason": "figure_image_missing",
                },
            )
            return
        try:
            image, _ = convert_to_pil_image(str(full_path))
            detections = self.object_detector.detect_panels(image)
        except Exception as exc:
            logger.warning(
                "Object detection failed for unverified figure; skipping",
                extra={
                    "operation": "main.match_caption_panel",
                    "figure_label": figure.figure_label,
                    "severity": "recoverable",
                    "reason": "detection_error",
                    "error": str(exc),
                },
            )
            return

        kept = [d for d in detections if d.get("confidence", 0.0) >= 0.25]
        figure.panels = [
            Panel(
                panel_label="",
                panel_caption="",
                panel_bbox=list(detection["bbox"]),
                confidence=float(detection["confidence"]),
                sd_files=[],
                ai_response=None,
            )
            for detection in kept
        ]
        logger.info(
            "Emitted bbox-only panels for unverified figure",
            extra={
                "operation": "main.match_caption_panel",
                "figure_label": figure.figure_label,
                "detection_count": len(detections),
                "kept_detection_count": len(kept),
            },
        )

    def _extract_panel_image(
        self, pil_image: Image.Image, bbox: List[float]
    ) -> Optional[str]:
        """
        Extract a panel image from a figure based on bounding box coordinates.

        This method crops the PIL Image according to the bounding box,
        and returns the panel image as a base64 encoded string.

        Args:
            pil_image (Image.Image): The PIL Image object of the entire figure.
            bbox (List[float]): Bounding box coordinates [x1, y1, x2, y2] in relative format.

        Returns:
            Optional[str]: Base64 encoded string of the panel image, or None if extraction fails.
        """
        try:
            width, height = pil_image.size
            left, top, right, bottom = [
                int(coord * width if i % 2 == 0 else coord * height)
                for i, coord in enumerate(bbox)
            ]
            panel = pil_image.crop((left, top, right, bottom))
            if panel.mode != "RGB":
                panel = panel.convert("RGB")

            buffered = io.BytesIO()
            panel.save(buffered, format="JPEG", quality=80)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error extracting panel image: {str(e)}")
            return None

    def _find_best_position_match(
        self, matches: List[Dict], original_bbox: List[float]
    ) -> Optional[Dict]:
        """
        Find the panel match that is closest in position to the original panel.

        Args:
            matches: List of panel matches
            original_bbox: Bounding box coordinates of the original panel

        Returns:
            Best match or None if no matches
        """
        if not matches or not original_bbox:
            return None

        # Calculate center points
        def get_center(bbox):
            return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        original_center = get_center(original_bbox)

        # Find match with closest center point
        best_match = None
        min_distance = float("inf")

        for match in matches:
            bbox = match["detection"]["bbox"]
            center = get_center(bbox)

            # Calculate squared distance (avoid square root for efficiency)
            distance = (center[0] - original_center[0]) ** 2 + (
                center[1] - original_center[1]
            ) ** 2

            if distance < min_distance:
                min_distance = distance
                best_match = match

        return best_match

    @abstractmethod
    def _match_panel_caption(
        self,
        encoded_image: str,
        figure_caption: str,
        allowed_panels: Optional[List[Panel]] = None,
    ) -> PanelObject:
        """Pick which caption-derived panel a crop belongs to (label only in practice)."""
        pass

    @staticmethod
    def _get_next_available_label(used_labels: Set[str]) -> str:
        """Return the next unused alphabetical panel label (A, B, …, Z, AA, AB, …)."""
        label_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        upper_used = {lbl.upper() for lbl in used_labels}
        for char in label_sequence:
            if char not in upper_used:
                return char
        for char1 in label_sequence:
            for char2 in label_sequence:
                double_char = char1 + char2
                if double_char not in upper_used:
                    return double_char
        return "unknown"

    def _add_unmatched_detections(
        self,
        panel_matches: List[Dict],
        used_detection_indices: Set[int],
        matched_labels: Set[str],
        figure_label: str,
    ) -> List[Panel]:
        """Create panels for detections that were not claimed by any caption label."""
        new_panels: List[Panel] = []
        unmatched = [
            m for m in panel_matches if m["detection_idx"] not in used_detection_indices
        ]
        for match in unmatched:
            panel_object = match["panel_object"]
            detection = match["detection"]
            panel_label = panel_object.panel_label.strip()
            if not panel_label or panel_label.upper() in {
                lbl.upper() for lbl in matched_labels
            }:
                panel_label = self._self._get_next_available_label(matched_labels)
            logger.info(
                f"Adding new panel {panel_label} from unmatched detection in figure {figure_label}"
            )
            new_panels.append(
                Panel(
                    panel_label=panel_label,
                    panel_caption=panel_object.panel_caption,
                    panel_bbox=detection["bbox"],
                    confidence=detection["confidence"],
                    sd_files=[],
                    ai_response=None,
                )
            )
            matched_labels.add(panel_label.upper())
        return new_panels

    def _resolve_panel_conflicts(
        self, figure: Any, panel_matches: List[Dict], original_panels: Dict[str, Panel]
    ) -> List[Panel]:
        """
        Resolve conflicts when multiple detected panels are assigned the same label,
        and ensure all detected panels are preserved with sequential labeling.

        Handles case-insensitive panel label matching throughout.

        Args:
            figure: The figure containing panels
            panel_matches: List of dicts with panel_object, detection, and detection_idx
            original_panels: Dict mapping panel labels to original Panel objects

        Returns:
            List of resolved Panel objects without duplicates
        """

        # Create case-insensitive lookup for original panels
        original_panels_ci = {}
        for key, panel in original_panels.items():
            original_panels_ci[key.upper()] = (
                key,
                panel,
            )  # Store original case with panel

        # Group by panel label (case-insensitive)
        label_to_matches = {}
        for match in panel_matches:
            panel_label = match["panel_object"].panel_label
            # Handle empty labels specially by assigning them a unique temporary key
            if not panel_label.strip():
                # Give each empty label a unique identifier
                panel_label = f"__empty_{match['detection_idx']}"
            else:
                # Use uppercase for matching but preserve original case
                match["original_case_label"] = panel_label
                panel_label = panel_label.upper()

            if panel_label not in label_to_matches:
                label_to_matches[panel_label] = []
            label_to_matches[panel_label].append(match)

        # Track panels with conflict resolution
        processed_panels = []
        conflicts_found = False
        used_detection_indices = set()  # Track which detections have been used

        # Pipeline-only conflict log (never serialized; see Figure dataclass).
        figure._conflicting_panels = []

        # Track used labels (case-insensitive)
        matched_labels = set()

        # Process each unique panel label
        for panel_label_upper, matches in label_to_matches.items():
            # Check if this is a temporarily assigned empty label
            is_empty_label = panel_label_upper.startswith("__empty_")

            # Get the original label (or use empty string for empty labels)
            if is_empty_label:
                original_label = ""
                # Use uppercase for the label key
                panel_label_key = ""
            else:
                # Look for a case-insensitive match in original panels
                panel_label_key = panel_label_upper
                # If we have an original panel with this label (ignoring case),
                # use its original casing instead of the detected casing
                if panel_label_key in original_panels_ci:
                    original_label = original_panels_ci[panel_label_key][
                        0
                    ]  # Use original case
                else:
                    # No original panel matching this label, use the original case from detection
                    original_label = matches[0].get(
                        "original_case_label", panel_label_upper
                    )

            # Look up the original panel if it exists
            original_panel = None
            if panel_label_key in original_panels_ci:
                _, original_panel = original_panels_ci[panel_label_key]

            if not original_panel and not is_empty_label and panel_label_key:
                # This is potentially a panel detected in the image but not in original text
                logger.info(
                    f"New panel label '{original_label}' identified via image detection for figure {figure.figure_label}"
                )
                original_panel = None  # Keep None, but continue processing

            if len(matches) == 1:
                # No conflict for this panel label
                match = matches[0]
                detection = match["detection"]
                used_detection_indices.add(match["detection_idx"])

                # For empty labels, assign the next available letter
                if is_empty_label:
                    original_label = self._get_next_available_label(matched_labels)
                    logger.info(
                        f"Assigning sequential label '{original_label}' to unlabeled panel in figure {figure.figure_label}"
                    )

                if original_panel:
                    # Create panel with original caption and new bbox
                    panel = Panel(
                        panel_label=original_label,  # Use the original or newly assigned label
                        # Use the original caption, not the one from match_caption_panel
                        panel_caption=original_panel.panel_caption,
                        panel_bbox=detection["bbox"],
                        confidence=detection["confidence"],
                        # Preserve all other original data
                        sd_files=(
                            original_panel.sd_files
                            if hasattr(original_panel, "sd_files")
                            else []
                        ),
                        ai_response=(
                            original_panel.ai_response
                            if hasattr(original_panel, "ai_response")
                            else None
                        ),
                    )
                else:
                    # Create a new panel from detection without original data
                    panel = Panel(
                        panel_label=original_label,  # Use the original or newly assigned label
                        panel_caption=match["panel_object"].panel_caption,
                        panel_bbox=detection["bbox"],
                        confidence=detection["confidence"],
                        sd_files=[],
                        ai_response=None,
                    )
                processed_panels.append(panel)
                # Track the used label (case-insensitive)
                matched_labels.add(original_label.upper())

            else:
                # Conflict: multiple detections assigned the same label
                conflicts_found = True
                logger.warning(
                    f"Found {len(matches)} panels with label {original_label} in figure {figure.figure_label}"
                )

                # First attempt: resolve by finding closest match to original position
                best_match = None
                if (
                    original_panel
                    and hasattr(original_panel, "panel_bbox")
                    and original_panel.panel_bbox
                ):
                    best_match = self._find_best_position_match(
                        matches, original_panel.panel_bbox
                    )
                    if best_match:
                        logger.info(
                            f"Resolved conflict for panel {original_label} using position similarity"
                        )

                # If no position match was found or original position wasn't available,
                # fall back to using the detection with highest confidence
                if not best_match:
                    best_match = max(
                        matches, key=lambda m: m["detection"]["confidence"]
                    )
                    logger.info(
                        f"Resolved conflict for panel {original_label} using confidence score"
                    )

                # Create panel with the best match
                detection = best_match["detection"]
                used_detection_indices.add(best_match["detection_idx"])

                # For empty labels, assign the next available letter
                if is_empty_label:
                    original_label = self._get_next_available_label(matched_labels)
                    logger.info(
                        f"Assigning sequential label '{original_label}' to conflicting unlabeled panel in figure {figure.figure_label}"
                    )

                if original_panel:
                    panel = Panel(
                        panel_label=original_label,  # Use the original or newly assigned label
                        # Use the original caption, not the one from match_caption_panel
                        panel_caption=original_panel.panel_caption,
                        panel_bbox=detection["bbox"],
                        confidence=detection["confidence"],
                        # Preserve all other original data
                        sd_files=(
                            original_panel.sd_files
                            if hasattr(original_panel, "sd_files")
                            else []
                        ),
                        ai_response=(
                            original_panel.ai_response
                            if hasattr(original_panel, "ai_response")
                            else None
                        ),
                    )
                else:
                    # Create a new panel from detection without original data
                    panel = Panel(
                        panel_label=original_label,  # Use the original or newly assigned label
                        panel_caption=best_match["panel_object"].panel_caption,
                        panel_bbox=detection["bbox"],
                        confidence=detection["confidence"],
                        sd_files=[],
                        ai_response=None,
                    )
                processed_panels.append(panel)
                # Track the used label (case-insensitive)
                matched_labels.add(original_label.upper())

                # Track the conflicting matches that were not used
                for conflict_match in matches:
                    if conflict_match != best_match:  # Skip the one we're using
                        used_detection_indices.add(conflict_match["detection_idx"])

                        # For conflicts with empty labels, assign sequential labels right away
                        if is_empty_label:
                            conflict_label = self._get_next_available_label(
                                matched_labels
                            )
                            matched_labels.add(conflict_label.upper())

                            # Create a new panel for this detection with sequential label
                            conflict_panel = Panel(
                                panel_label=conflict_label,
                                panel_caption=conflict_match[
                                    "panel_object"
                                ].panel_caption,
                                panel_bbox=conflict_match["detection"]["bbox"],
                                confidence=conflict_match["detection"]["confidence"],
                                sd_files=[],
                                ai_response=None,
                            )
                            processed_panels.append(conflict_panel)
                            logger.info(
                                f"Created new panel with label '{conflict_label}' from conflicting detection in figure {figure.figure_label}"
                            )
                        else:
                            # Standard conflict handling for non-empty labels
                            figure._conflicting_panels.append(
                                {
                                    "panel_label": original_label,
                                    "detection_idx": conflict_match["detection_idx"],
                                    "confidence": conflict_match["detection"][
                                        "confidence"
                                    ],
                                    "bbox": conflict_match["detection"]["bbox"],
                                }
                            )

        if conflicts_found:
            logger.info(f"Resolved panel conflicts in figure {figure.figure_label}")

        # Check for original panels that didn't get matched to any detection
        for label, panel in original_panels.items():
            if label.upper() not in [lab.upper() for lab in matched_labels]:
                logger.warning(
                    f"Original panel {label} not matched to any detection in figure {figure.figure_label}"
                )
                # Add the original panel without a bbox
                processed_panels.append(panel)
                matched_labels.add(label.upper())  # Update matched labels

        processed_panels.extend(
            self._add_unmatched_detections(
                panel_matches,
                used_detection_indices,
                matched_labels,
                figure.figure_label,
            )
        )
        return processed_panels
