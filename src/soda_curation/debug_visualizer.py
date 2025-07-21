#!/usr/bin/env python3
"""
Debug visualizer for SODA Curation QC pipeline.

This module helps debug the QC pipeline by extracting and saving individual figure images
from the figure data JSON files, allowing you to see exactly what the AI is analyzing.
"""

import argparse
import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .data_storage import load_figure_data

logger = logging.getLogger(__name__)


class DebugVisualizer:
    """Debug visualizer for extracting and saving figure images."""

    def __init__(self, output_dir: str = "data/debug_images"):
        """Initialize the debug visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_images_from_json(
        self, json_file: str, prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract images from figure data JSON and save as PNG files.

        Args:
            json_file: Path to the figure data JSON file
            prefix: Optional prefix for output filenames

        Returns:
            Dictionary mapping figure labels to saved file paths
        """
        logger.info(f"Loading figure data from: {json_file}")

        # Load the figure data
        figure_data = load_figure_data(json_file)

        if not figure_data:
            logger.error(f"No figure data loaded from {json_file}")
            return {}

        logger.info(f"Found {len(figure_data)} figures to extract")

        # Create a subdirectory for this dataset
        dataset_name = prefix if prefix is not None else Path(json_file).stem
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        saved_files = {}

        for i, (figure_label, image_data, figure_caption) in enumerate(figure_data):
            try:
                # Clean up figure label for filename
                safe_filename = self._sanitize_filename(figure_label)

                # Save the image
                image_path = self._save_base64_image(
                    image_data, dataset_dir / f"{safe_filename}.png"
                )

                # Save the caption as a text file
                caption_path = dataset_dir / f"{safe_filename}_caption.txt"
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(f"Figure: {figure_label}\n\n")
                    f.write(figure_caption)

                saved_files[figure_label] = str(image_path)
                logger.info(f"Saved {figure_label} to {image_path}")

            except Exception as e:
                logger.error(f"Error processing {figure_label}: {str(e)}")
                continue

        # Create a summary file
        self._create_summary_file(dataset_dir, figure_data, saved_files)

        logger.info(
            f"Successfully extracted {len(saved_files)} images to {dataset_dir}"
        )
        return saved_files

    def _save_base64_image(self, base64_data: str, output_path: Path) -> Path:
        """Save a base64 encoded image to file."""
        try:
            # Decode the base64 data
            image_bytes = base64.b64decode(base64_data)

            # Open with PIL to ensure it's valid and can be resaved
            with Image.open(BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary (in case of RGBA or other modes)
                if img.mode not in ("RGB", "L"):  # L is grayscale
                    img = img.convert("RGB")

                # Save as PNG
                img.save(output_path, "PNG")

                # Log image info
                logger.debug(
                    f"Saved image: {output_path} (Size: {img.size}, Mode: {img.mode})"
                )

            return output_path

        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            raise

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize a string to be safe for use as a filename."""
        # Replace spaces and special characters
        safe = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "._-")
        return safe.lower()

    def _create_summary_file(
        self,
        output_dir: Path,
        figure_data: List[Tuple[str, str, str]],
        saved_files: Dict[str, str],
    ) -> None:
        """Create a summary HTML file for easy viewing of all figures."""
        summary_path = output_dir / "summary.html"

        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Figure Debug Summary</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .figure { border: 1px solid #ddd; margin: 20px 0; padding: 20px; }",
            "        .figure-image { max-width: 800px; width: 100%; }",
            "        .figure-label { font-size: 18px; font-weight: bold; color: #333; }",
            "        .figure-caption { margin-top: 10px; font-size: 14px; color: #666; }",
            "        .error { color: red; }",
            "    </style>",
            "</head>",
            "<body>",
            f"    <h1>Figure Debug Summary - {output_dir.name}</h1>",
            f"    <p>Total figures: {len(figure_data)}</p>",
            f"    <p>Successfully extracted: {len(saved_files)}</p>",
        ]

        for figure_label, image_data, figure_caption in figure_data:
            html_content.append('<div class="figure">')
            html_content.append(f'    <div class="figure-label">{figure_label}</div>')

            if figure_label in saved_files:
                safe_filename = self._sanitize_filename(figure_label)
                html_content.append(
                    f'    <img src="{safe_filename}.png" class="figure-image" alt="{figure_label}">'
                )
            else:
                html_content.append(
                    '    <div class="error">Failed to extract image</div>'
                )

            # Truncate very long captions for HTML display
            display_caption = figure_caption
            if len(display_caption) > 500:
                display_caption = display_caption[:500] + "..."

            html_content.append(
                f'    <div class="figure-caption">{display_caption}</div>'
            )
            html_content.append("</div>")

        html_content.extend(["</body>", "</html>"])

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))

        logger.info(f"Created summary file: {summary_path}")

    def analyze_image_properties(self, json_file: str) -> Dict[str, Any]:
        """Analyze properties of images in the figure data."""
        logger.info(f"Analyzing image properties from: {json_file}")

        figure_data = load_figure_data(json_file)

        if not figure_data:
            return {}

        analysis = {
            "total_figures": len(figure_data),
            "image_sizes": [],
            "data_sizes": [],
            "modes": [],
            "errors": [],
        }

        for figure_label, image_data, figure_caption in figure_data:
            try:
                # Decode and analyze the image
                image_bytes = base64.b64decode(image_data)

                with Image.open(BytesIO(image_bytes)) as img:
                    analysis["image_sizes"].append(img.size)
                    analysis["data_sizes"].append(len(image_data))
                    analysis["modes"].append(img.mode)

            except Exception as e:
                analysis["errors"].append(f"{figure_label}: {str(e)}")

        return analysis


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Debug visualizer for SODA Curation QC pipeline"
    )
    parser.add_argument("figure_data", help="Path to figure data JSON file")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/debug_images",
        help="Output directory for extracted images (default: data/debug_images)",
    )
    parser.add_argument(
        "--prefix", "-p", help="Prefix for output filenames (default: uses filename)"
    )
    parser.add_argument(
        "--analyze",
        "-a",
        action="store_true",
        help="Analyze image properties without extracting",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    visualizer = DebugVisualizer(args.output_dir)

    if args.analyze:
        # Just analyze properties
        analysis = visualizer.analyze_image_properties(args.figure_data)
        print("\nImage Analysis Results:")
        print(f"Total figures: {analysis['total_figures']}")
        print(f"Image sizes: {set(analysis['image_sizes'])}")
        print(f"Image modes: {set(analysis['modes'])}")
        print(
            f"Data size range: {min(analysis['data_sizes']) if analysis['data_sizes'] else 0} - {max(analysis['data_sizes']) if analysis['data_sizes'] else 0} characters"
        )
        if analysis["errors"]:
            print(f"Errors: {len(analysis['errors'])}")
            for error in analysis["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
    else:
        # Extract images
        saved_files = visualizer.extract_images_from_json(args.figure_data, args.prefix)
        print(
            f"\nSuccessfully extracted {len(saved_files)} figures to {args.output_dir}"
        )

        # Print file paths
        for figure_label, file_path in saved_files.items():
            print(f"  {figure_label}: {file_path}")


if __name__ == "__main__":
    main()
