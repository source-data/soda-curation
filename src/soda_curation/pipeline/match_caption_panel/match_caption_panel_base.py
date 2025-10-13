import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PIL import Image
from pydantic import BaseModel

from ...pipeline.prompt_handler import PromptHandler
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
        self.extract_dir = Path(
            extract_dir
        )  # This is now the manuscript-specific directory
        self._validate_config()
        # Initialize object detector using the create_object_detection helper
        self.object_detector = create_object_detection(config)

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    def process_figures(self, zip_structure: ZipStructure) -> ZipStructure:
        """Process all figures in the manuscript."""
        self.zip_structure = zip_structure
        for figure in zip_structure.figures:
            try:
                # Store original panels in a dictionary for quick lookup by label
                original_panels = {panel.panel_label: panel for panel in figure.panels}

                # Convert figure file to PIL Image
                full_path = self.extract_dir / figure.img_files[0]
                if not full_path.exists():
                    raise FileNotFoundError(f"File not found: {full_path}")
                image, _ = convert_to_pil_image(str(full_path))

                # Debug: Check what we got from convert_to_pil_image
                logger.debug(
                    f"convert_to_pil_image returned: image type={type(image)}, image={image}"
                )

                # Get only bounding boxes from detection
                detected_regions = self.object_detector.detect_panels(image)

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
                        panel_object = self._match_panel_caption(
                            encoded_image, figure.figure_caption
                        )
                        panel_object = (
                            PanelObject(**json.loads(panel_object))
                            if isinstance(panel_object, str)
                            else panel_object
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

            except Exception as e:
                logger.error(f"Error processing figure {figure.figure_label}: {str(e)}")
                continue

        return zip_structure

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

            buffered = io.BytesIO()
            panel.save(buffered, format="PNG")
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
        self, panel_image: Image.Image, figure_caption: str
    ) -> Dict[str, str]:
        """Match a panel image with its caption using AI."""
        pass

    def _resolve_panel_conflicts(
        self, figure: Any, panel_matches: List[Dict], original_panels: Dict[str, Panel]
    ) -> List[Panel]:
        """
        Resolve conflicts when multiple detected panels are assigned the same label,
        and ensure all detected panels are preserved with sequential labeling.

        This implementation handles case-insensitive matching of panel labels,
        so that 'A' and 'a' are treated as the same label.

        Args:
            figure: The figure containing panels
            panel_matches: List of dictionaries with panel_object, detection, and detection_idx
            original_panels: Dictionary mapping panel labels to original Panel objects

        Returns:
            List of resolved Panel objects without duplicates
        """

        # Helper function for sequential labeling - define this at the top
        def get_next_available_label(used_labels: Set[str]) -> str:
            """Get the next available panel label in alphabetical sequence."""
            # Standard panel label sequence
            label_sequence = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for char in label_sequence:
                if char.upper() not in [label.upper() for label in used_labels]:
                    return char
            # If we use up all letters, start with AA, AB, etc.
            for char1 in label_sequence:
                for char2 in label_sequence:
                    double_char = char1 + char2
                    if double_char.upper() not in [
                        label.upper() for label in used_labels
                    ]:
                        return double_char
            return "unknown"  # Fallback

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

        # Initialize tracking for conflicting panels
        if not hasattr(figure, "conflicting_panels"):
            figure.conflicting_panels = []

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
                    original_label = get_next_available_label(matched_labels)
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
                        sd_files=original_panel.sd_files
                        if hasattr(original_panel, "sd_files")
                        else [],
                        ai_response=original_panel.ai_response
                        if hasattr(original_panel, "ai_response")
                        else None,
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
                    original_label = get_next_available_label(matched_labels)
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
                        sd_files=original_panel.sd_files
                        if hasattr(original_panel, "sd_files")
                        else [],
                        ai_response=original_panel.ai_response
                        if hasattr(original_panel, "ai_response")
                        else None,
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
                            conflict_label = get_next_available_label(matched_labels)
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
                            figure.conflicting_panels.append(
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

        # Add unmatched detections as new panels with sequential labels
        unmatched_detections = [
            match
            for match in panel_matches
            if match["detection_idx"] not in used_detection_indices
        ]

        # Process unmatched detections
        for match in unmatched_detections:
            panel_object = match["panel_object"]
            detection = match["detection"]

            # Check if the AI assigned a valid label that isn't already used
            panel_label = panel_object.panel_label.strip()
            if panel_label and panel_label.upper() not in [
                lab.upper() for lab in matched_labels
            ]:
                # Use the AI-assigned label (like "D")
                pass
            else:
                # Find the next available label in sequence
                panel_label = get_next_available_label(matched_labels)

            logger.info(
                f"Adding new panel {panel_label} from unmatched detection in figure {figure.figure_label}"
            )

            panel = Panel(
                panel_label=panel_label,
                panel_caption=panel_object.panel_caption,
                panel_bbox=detection["bbox"],
                confidence=detection["confidence"],
                sd_files=[],
                ai_response=None,
            )
            processed_panels.append(panel)
            matched_labels.add(
                panel_label.upper()
            )  # Update matched labels (case-insensitive)

        return processed_panels
