import base64
import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

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
                # Store original panels to preserve ALL data
                original_panels = {panel.panel_label: panel for panel in figure.panels}

                # Convert figure file to PIL Image
                full_path = self.extract_dir / figure.img_files[0]
                if not full_path.exists():
                    raise FileNotFoundError(f"File not found: {full_path}")
                image, _ = convert_to_pil_image(str(full_path))

                # Get only bounding boxes from detection
                detected_regions = self.object_detector.detect_panels(image)

                if not detected_regions:
                    logger.warning(
                        f"No panels detected in figure {figure.figure_label}"
                    )
                    continue

                # Process each detected region
                processed_panels = []
                for detection in detected_regions:
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

                        # Get original panel if it exists
                        original_panel = original_panels.get(panel_object.panel_label)

                        # Create new panel preserving ALL original data
                        panel = Panel(
                            panel_label=panel_object.panel_label,
                            panel_caption=panel_object.panel_caption,
                            panel_bbox=detection["bbox"],
                            confidence=detection["confidence"],
                            # Preserve original data exactly as is
                            sd_files=original_panel.sd_files if original_panel else [],
                            ai_response=original_panel.ai_response
                            if original_panel
                            else None,
                        )
                        processed_panels.append(panel)

                # Preserve any panels that weren't detected but existed in original
                unmatched_labels = set(original_panels.keys()) - {
                    p.panel_label for p in processed_panels
                }
                for label in unmatched_labels:
                    processed_panels.append(original_panels[label])

                # Update figure panels while preserving all other figure attributes
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

    @abstractmethod
    def _match_panel_caption(
        self, panel_image: Image.Image, figure_caption: str
    ) -> Dict[str, str]:
        """Match a panel image with its caption using AI."""
        pass
