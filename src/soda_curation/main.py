import argparse
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from .config import load_config
from .data_availability.data_availability_openai import DataAvailabilityExtractorGPT
from .logging_config import setup_logging
from .pipeline.assign_panel_source.assign_panel_source import PanelSourceAssigner
from .pipeline.extract_captions.extract_captions_claude import (
    FigureCaptionExtractorClaude,
)
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.manuscript_structure.exceptions import (
    NoManuscriptFileError,
    NoXMLFileFoundError,
)
from .pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    Figure,
    Panel,
    ZipStructure,
    full_path,
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from .pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from .pipeline.object_detection.object_detection import (
    convert_to_pil_image,
    create_object_detection,
)

CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
    }
}


class FigureProcessor:
    """Class to handle figure processing including panel detection, caption matching and source assignment."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.object_detector = create_object_detection(config)
        self.panel_caption_matcher = MatchPanelCaptionOpenAI(config)
        self.panel_source_assigner = PanelSourceAssigner(config)

    def process_figures(self, zip_structure: ZipStructure) -> ZipStructure:
        """Process figures and their panels."""
        logging.info("Processing figures and panels")

        for figure in zip_structure.figures:
            if figure._full_img_files:
                try:
                    # First detect panels using object detection
                    pil_image, _ = convert_to_pil_image(figure._full_img_files[0])
                    detected_panels = self.object_detector.detect_panels(pil_image)

                    # Create initial panel objects with detection results
                    panels = [
                        Panel(
                            panel_label="",  # Will be filled by panel_caption_matcher
                            panel_caption="",  # Will be filled by panel_caption_matcher
                            panel_bbox=panel["panel_bbox"],
                            confidence=panel["confidence"],
                            ai_response="",
                            sd_files=[],
                        )
                        for panel in detected_panels
                    ]
                    figure.panels = panels

                    # If figure has valid caption, process panel captions and sources
                    if (
                        figure.figure_caption
                        and figure.figure_caption != "Figure caption not found."
                    ):
                        logging.info(
                            f"Processing panels for figure {figure.figure_label}"
                        )
                        # Match panel captions
                        figure = self.panel_caption_matcher.match_captions(figure)
                        # Assign panel sources - now process individual figure
                        figure = self.panel_source_assigner.assign_panel_source(figure)
                    else:
                        logging.warning(
                            f"Skipping panel caption matching for {figure.figure_label} - No valid caption"
                        )

                except Exception as e:
                    logging.error(
                        f"Error processing figure {figure.figure_label}: {str(e)}"
                    )

        return zip_structure

    def _process_single_figure(self, figure: Figure) -> Figure:
        """Process a single figure."""
        if figure._full_img_files:
            pil_image, _ = convert_to_pil_image(figure._full_img_files[0])
            detected_panels = self.object_detector.detect_panels(pil_image)

            # Create panels with basic information from detection
            figure.panels = [
                Panel(
                    panel_label="",
                    panel_caption="",
                    panel_bbox=panel["panel_bbox"],
                    confidence=panel["confidence"],
                    ai_response="",
                    sd_files=[],
                )
                for panel in detected_panels
            ]

            # Only proceed with caption matching and source assignment if figure has valid caption
            if (
                figure.figure_caption
                and figure.figure_caption != "Figure caption not found."
            ):
                figure = self.panel_caption_matcher.match_captions(figure)
                figure = self.panel_source_assigner.assign_panel_source(figure)

        return figure

    def _create_empty_panels(self, figure: Figure) -> List[Panel]:
        """Create empty panel objects for figures without captions."""
        if not figure._full_img_files:
            return []

        pil_image, _ = convert_to_pil_image(figure._full_img_files[0])
        detected_panels = self.object_detector.detect_panels(pil_image)

        return [
            Panel(
                panel_label="",
                panel_caption="",
                panel_bbox=panel["panel_bbox"],
                confidence=0.0,
                ai_response="",
                sd_files=[],
            )
            for panel in detected_panels
        ]


def get_file_tree(directory: str) -> Dict[str, Any]:
    """
    Recursively generate a file tree from a directory.

    Args:
        directory (str): The directory to generate the file tree from.

    Returns:
        Dict[str, Any]: A nested dictionary representing the file tree.
    """
    file_tree = {}
    for root, dirs, files in os.walk(directory):
        current = file_tree
        rel_path = os.path.relpath(root, directory)
        if rel_path != ".":
            path_parts = rel_path.split(os.sep)
            for part in path_parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            current[relative_path] = None
    logging.debug(f"Generated file tree: {json.dumps(file_tree, indent=2)}")
    return file_tree


def extract_zip_contents(zip_path: str, extract_dir: Path) -> None:
    """
    Extract the contents of a ZIP file to a directory.

    Args:
        zip_path (str): Path to the ZIP file.
        extract_dir (Path): Directory where the ZIP contents will be extracted.

    Raises:
        zipfile.BadZipFile: If the ZIP file is invalid.
        Exception: If an error occurs during extraction.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        logging.error(f"Invalid ZIP file: {zip_path}")
        raise
    except Exception as e:
        logging.error(f"Error extracting ZIP file: {str(e)}")
        raise


def get_manuscript_structure(zip_path: str, extract_dir: str) -> ZipStructure:
    """
    Extract the manuscript structure from a ZIP file.

    Args:
        zip_path (str): Path to the ZIP file containing the manuscript data.
        extract_dir (str): Directory where ZIP contents are extracted.

    Returns:
        ZipStructure: A structured representation of the manuscript.
    """
    xml_extractor = XMLStructureExtractor(zip_path, extract_dir)
    structure = xml_extractor.extract_structure()
    logging.info(
        f"Manuscript structure: id={structure.manuscript_id}, xml={structure.xml}, docx={structure.docx}, pdf={structure.pdf}"
    )
    logging.info(f"Number of figures: {len(structure.figures)}")
    return structure


def update_file_paths(zip_structure: ZipStructure, extract_dir: str) -> ZipStructure:
    """
    Update the file paths in the ZipStructure object with full paths.

    Args:
        zip_structure (ZipStructure): The ZipStructure object to update.
        extract_dir (str): The directory where the ZIP contents are extracted.

    Returns:
        ZipStructure: The updated ZipStructure object.
    """
    if zip_structure.docx:
        zip_structure._full_docx = full_path(extract_dir, zip_structure.docx)
        logging.info(f"Full DOCX path: {zip_structure._full_docx}")
    if zip_structure.pdf:
        zip_structure._full_pdf = full_path(extract_dir, zip_structure.pdf)
        logging.info(f"Full PDF path: {zip_structure._full_pdf}")

    zip_structure._full_appendix = []
    for app in zip_structure.appendix:
        if isinstance(app, str):
            zip_structure._full_appendix.append(full_path(extract_dir, app))
        elif isinstance(app, dict):
            zip_structure._full_appendix.append(app)
        else:
            logging.warning(f"Unexpected appendix item type: {type(app)}")

    for figure in zip_structure.figures:
        figure._full_img_files = [
            full_path(extract_dir, img) for img in figure.img_files
        ]
        figure._full_sd_files = [full_path(extract_dir, sd) for sd in figure.sd_files]

    return zip_structure


def extract_figure_captions(
    zip_structure: ZipStructure, config: Dict[str, Any], expected_figure_count: int
) -> ZipStructure:
    """
    Extract figure captions from the manuscript using the configured AI model.

    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.
        config (Dict[str, Any]): The configuration settings for the AI model.
        expected_figure_count (int): The expected number of figures in the manuscript.

    Returns:
        ZipStructure: The updated ZipStructure object with extracted figure captions.
    """
    ai_provider = config.get("ai", "openai")

    expected_figure_labels = []
    for figure in zip_structure.figures:
        expected_figure_labels.append(figure.figure_label)

    if ai_provider == "openai":
        from .pipeline.extract_captions.extract_captions_openai import (
            FigureCaptionExtractorGpt,
        )

        caption_extractor = FigureCaptionExtractorGpt(config["openai"])
    elif ai_provider == "anthropic":
        from .pipeline.extract_captions.extract_captions_claude import (
            FigureCaptionExtractorClaude,
        )

        caption_extractor = FigureCaptionExtractorClaude(config["anthropic"])
    else:
        raise ValueError(f"Unsupported AI provider: {ai_provider}")

    result = caption_extractor.extract_captions(
        zip_structure._full_docx,
        zip_structure,
        expected_figure_count,
        expected_figure_labels=", ".join(expected_figure_labels),
    )
    return result


def update_sd_files(zip_structure: ZipStructure) -> ZipStructure:
    """
    Update the source data file paths in the ZipStructure object.

    Args:
        zip_structure (ZipStructure): The ZipStructure object to update.

    Returns:
        ZipStructure: The updated ZipStructure object.
    """
    for figure in zip_structure.figures:
        if figure._full_sd_files:
            zip_file = os.path.basename(figure._full_sd_files[0])
            figure.sd_files = [f"{zip_structure.manuscript_id}/suppl_data/{zip_file}"]
        else:
            figure.sd_files = []
    return zip_structure


def is_excel_file(filename):
    """
    Check if a file is an Excel file based on the extension.

    Args:
        filename (str): The name of the file.

    Returns:
        bool: True if the file is an Excel file, False otherwise.
    """
    return filename.lower().endswith(".xlsx")


def extract_panels(zip_structure: ZipStructure, config: Dict[str, Any]) -> ZipStructure:
    """
    Extract panels from the figures using an object detection model.

    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.
        config (Dict[str, Any]): The configuration settings for the object detection model.

    Returns:
        ZipStructure: The updated ZipStructure object with extracted panels.
    """
    object_detector = create_object_detection(config)
    for figure in zip_structure.figures:
        if figure._full_img_files:
            original_img_path = figure._full_img_files[0]
            if os.path.exists(original_img_path):
                try:
                    pil_image, _ = convert_to_pil_image(original_img_path)
                    panels = object_detector.detect_panels(pil_image)
                    figure.panels = [
                        Panel(
                            panel_label=p["panel_label"],
                            panel_caption=p["panel_caption"],
                            panel_bbox=p["panel_bbox"],
                            confidence=p["confidence"],
                            ai_response=None,
                        )
                        for p in panels
                    ]
                    logging.info(
                        f"Detected {len(panels)} panels in {figure.img_files[0]}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error during panel detection for {figure.img_files[0]}: {str(e)}"
                    )
            else:
                logging.warning(f"Image file not found: {figure.img_files[0]}")
    return zip_structure


def match_panel_caption(
    zip_structure: ZipStructure, config: Dict[str, Any]
) -> ZipStructure:
    """
    Match panel captions to figure captions using an AI model.

    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.
        config (Dict[str, Any]): The configuration settings for the AI model.

    Returns:
        ZipStructure: The updated ZipStructure object with matched panel captions.
    """
    panel_caption_matcher = MatchPanelCaptionOpenAI(config)
    result = panel_caption_matcher.match_captions(zip_structure)
    return result


def assign_panel_source(
    zip_structure: ZipStructure, config: Dict[str, Any], extract_dir: str
) -> ZipStructure:
    """
    Assign source data files to panels based on the panel captions.

    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.
        config (Dict[str, Any]): The configuration settings for the source assignment model.
        extract_dir (str): The directory where the ZIP contents are extracted.

    Returns:
        ZipStructure: The updated ZipStructure object with assigned panel source data files.
    """
    file_tree = get_file_tree(extract_dir)
    logging.info(f"File tree structure: {json.dumps(file_tree, indent=2)}")
    panel_source_assigner = PanelSourceAssigner(config)
    zip_structure = panel_source_assigner.assign_panel_source(zip_structure)

    # Ensure non_associated_sd_files is initialized
    if not hasattr(zip_structure, "non_associated_sd_files"):
        zip_structure.non_associated_sd_files = []

    for figure in zip_structure.figures:
        assigned_files = set()
        if figure._full_sd_files:
            zip_filename = os.path.basename(figure._full_sd_files[0])
            figure_number = figure.figure_label.split()[-1]

            # Update panel.sd_files paths and collect assigned files
            for panel in figure.panels:
                # Update panel.sd_files paths
                panel.sd_files = [
                    f"{zip_filename}:Figure {figure_number}/{os.path.basename(file)}"
                    for file in panel.sd_files
                    if isinstance(file, str)
                    and not ("__MACOSX" in file or file.endswith(".DS_Store"))
                ]
                assigned_files.update(panel.sd_files)
        else:
            for panel in figure.panels:
                panel.sd_files = []

        # Collect unassigned files from the figure's source data ZIP files
        if figure._full_sd_files:
            for sd_file in figure._full_sd_files:
                zip_filename = os.path.basename(sd_file)
                figure_number = figure.figure_label.split()[-1]
                with zipfile.ZipFile(sd_file, "r") as zip_ref:
                    all_sd_files_in_zip = set(zip_ref.namelist())

                # Remove unwanted files
                all_sd_files_in_zip = set(
                    file
                    for file in all_sd_files_in_zip
                    if not ("__MACOSX" in file or file.endswith(".DS_Store"))
                )

                # Adjust paths to match the format in assigned_files
                all_sd_files = set(
                    f"{zip_filename}:Figure {figure_number}/{os.path.basename(file)}"
                    for file in all_sd_files_in_zip
                )

                # Determine unassigned files
                unassigned_files = all_sd_files - assigned_files

                # Add unassigned files to non_associated_sd_files
                zip_structure.non_associated_sd_files.extend(unassigned_files)

    return zip_structure


def process_figures(self, zip_structure: ZipStructure) -> ZipStructure:
    """Process figures and their panels."""
    logging.info("Processing figures and panels")

    for figure in zip_structure.figures:
        # First extract panels using object detection
        if figure._full_img_files:
            pil_image, _ = convert_to_pil_image(figure._full_img_files[0])
            detected_panels = self.object_detector.detect_panels(pil_image)

            # If figure has no valid caption, create basic panel objects
            if (
                not figure.figure_caption
                or figure.figure_caption == "Figure caption not found."
            ):
                figure.panels = [
                    Panel(
                        panel_label="",
                        panel_caption="",
                        panel_bbox=panel["panel_bbox"],
                        confidence=0.0,
                        ai_response="",
                        sd_files=[],
                    )
                    for panel in detected_panels
                ]
                continue

            # Only proceed with panel caption matching if figure has a valid caption
            if figure.figure_caption:
                figure = self.panel_caption_matcher.match_captions(figure)
                figure = self.panel_source_assigner.assign_panel_source(figure)

    return zip_structure


def process_ev_materials(zip_structure: ZipStructure) -> ZipStructure:
    """
    Process EV materials in the manuscript and assign them to the appropriate figures.

    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.

    Returns:
        ZipStructure: The updated ZipStructure object with EV materials assigned to figures.
    """
    ev_materials = []
    for figure in zip_structure.figures:
        if figure._full_sd_files:
            for file in figure._full_sd_files:
                if re.search(
                    r"(Figure|Table|Dataset)\s*EV",
                    os.path.basename(file),
                    re.IGNORECASE,
                ):
                    ev_materials.append(file)

    for material in ev_materials:
        material_name = os.path.basename(material)
        match = re.search(
            r"(Figure|Table|Dataset)\s*EV(\d+)", material_name, re.IGNORECASE
        )
        if match:
            ev_type = match.group(1).capitalize()
            ev_number = match.group(2)
            ev_figure = next(
                (
                    fig
                    for fig in zip_structure.figures
                    if fig.figure_label == f"{ev_type} EV{ev_number}"
                ),
                None,
            )
            if ev_figure:
                ev_figure.sd_files.append(material_name)
                logging.info(f"Assigned {material_name} to {ev_figure.figure_label}")
            else:
                zip_structure.non_associated_sd_files.append(material_name)
                logging.info(
                    f"Added {material_name} to non_associated_sd_files (no matching EV figure found)"
                )
        else:
            zip_structure.non_associated_sd_files.append(material_name)
            logging.info(
                f"Added {material_name} to non_associated_sd_files (not recognized as EV material)"
            )

    return zip_structure


def process_zip_structure(zip_structure):
    """
    Process the ZipStructure object before serializing to JSON.

    Args:
        zip_structure (ZipStructure): The ZipStructure object to process.

    Returns:
        ZipStructure: The processed ZipStructure object.
    """
    # Update appendix paths - strip to just the relative path
    processed_appendix = []
    for item in zip_structure.appendix:
        if isinstance(item, dict):
            path = item["object_id"]
            # Get everything after the manuscript_id/
            _, *parts = path.split("/")
            processed_appendix.append("/".join(parts))
        elif isinstance(item, str):
            _, *parts = item.split("/")
            processed_appendix.append("/".join(parts))
    zip_structure.appendix = processed_appendix

    figure_sd_files = set()

    for figure in zip_structure.figures:
        # Update image files paths - strip to just the relative path
        figure.img_files = []
        for img_file in figure._full_img_files:
            # Split path and get everything after manuscript_id
            parts = img_file.split("/")
            try:
                idx = parts.index(zip_structure.manuscript_id)
                figure.img_files.append("/".join(parts[idx + 1 :]))
            except ValueError:
                # If manuscript_id not in path, take everything after input/
                try:
                    idx = parts.index("input")
                    figure.img_files.append("/".join(parts[idx + 2 :]))
                except ValueError:
                    # If all else fails, use the last two parts of the path
                    figure.img_files.append("/".join(parts[-2:]))

        # Update source data file paths
        if figure._full_sd_files:
            figure.sd_files = []
            for file in figure._full_sd_files:
                if file.endswith(".zip"):
                    parts = file.split("/")
                    try:
                        idx = parts.index(zip_structure.manuscript_id)
                        figure.sd_files.append("/".join(parts[idx + 1 :]))
                    except ValueError:
                        try:
                            idx = parts.index("input")
                            figure.sd_files.append("/".join(parts[idx + 2 :]))
                        except ValueError:
                            figure.sd_files.append("/".join(parts[-2:]))
            figure_sd_files.update(figure.sd_files)

        # Process panels
        for panel in figure.panels:
            if figure.sd_files:
                # Update panel source data paths
                new_sd_files = []
                for file in panel.sd_files:
                    if isinstance(file, str) and not (
                        "__MACOSX" in file or file.endswith(".DS_Store")
                    ):
                        if ":" in file:
                            # Split into zip path and internal path
                            zip_part, internal_path = file.split(":", 1)

                            # Clean up zip part - already handled by previous code
                            zip_parts = zip_part.split("/")
                            try:
                                idx = zip_parts.index(zip_structure.manuscript_id)
                                clean_zip_path = "/".join(zip_parts[idx + 1 :])
                            except ValueError:
                                try:
                                    idx = zip_parts.index("input")
                                    clean_zip_path = "/".join(zip_parts[idx + 2 :])
                                except ValueError:
                                    clean_zip_path = "/".join(zip_parts[-2:])

                            # Simply preserve the original internal path
                            new_sd_files.append(f"{clean_zip_path}:{internal_path}")

                panel.sd_files = new_sd_files

    # Remove any files with __MACOSX or .DS_Store
    if hasattr(zip_structure, "non_associated_sd_files"):
        zip_structure.non_associated_sd_files = [
            file
            for file in zip_structure.non_associated_sd_files
            if "__MACOSX" not in file
            and not file.endswith(".DS_Store")
            and ":" in file  # Only keep files within ZIP archives
        ]

        # Clean up paths in non_associated_sd_files
        cleaned_non_associated = []
        for file in zip_structure.non_associated_sd_files:
            if ":" in file:
                zip_part, internal_path = file.split(":", 1)
                zip_parts = zip_part.split("/")
                try:
                    idx = zip_parts.index(zip_structure.manuscript_id)
                    clean_zip_path = "/".join(zip_parts[idx + 1 :])
                except ValueError:
                    try:
                        idx = zip_parts.index("input")
                        clean_zip_path = "/".join(zip_parts[idx + 2 :])
                    except ValueError:
                        clean_zip_path = "/".join(zip_parts[-2:])
                cleaned_non_associated.append(f"{clean_zip_path}:{internal_path}")
        zip_structure.non_associated_sd_files = cleaned_non_associated

    # Final cleanup - remove private fields
    for attr in list(vars(zip_structure)):
        if attr.startswith("_"):
            delattr(zip_structure, attr)

    return zip_structure


def main(zip_path: str, config_path: str, output_path: str = None) -> str:
    setup_logging(CONFIG)
    logging.info("Application has started.")

    class DocumentContentCache:
        """Cache for document content to avoid multiple extractions."""

        def __init__(self):
            self._content = {}

        def get_content(self, doc_path: str, extractor) -> str:
            """Get document content, extracting only if not already cached."""
            if doc_path not in self._content:
                logging.info(f"Extracting content from {doc_path}")
                self._content[doc_path] = extractor._extract_docx_content(doc_path)
            return self._content[doc_path]

    # Initialize content cache
    doc_cache = DocumentContentCache()
    # Now proceed with the rest of your application
    logging.info("Application has started.")

    output_json = ""
    if not zip_path or not config_path:
        raise ValueError("ZIP path and config path must be provided")

    config = load_config(config_path)
    setup_logging(config)

    # Update the AI provider validation
    if config["ai"] not in ["openai", "anthropic"]:
        raise ValueError(f"Invalid AI provider: {config['ai']}")

    zip_file = Path(zip_path)
    extract_dir = zip_file.parent / zip_file.stem

    try:
        extract_zip_contents(zip_path, extract_dir)

        # Add extract_dir to config
        config["extract_dir"] = str(extract_dir)

        try:
            zip_structure = get_manuscript_structure(zip_path, str(extract_dir))
            # Add AI configuration to zip_structure
            zip_structure.ai_provider = config["ai"]
            zip_structure.ai_config = {
                "provider": config["ai"],
                "model": config[config["ai"]]["model"],
                "temperature": config[config["ai"]]["temperature"],
                "top_p": config[config["ai"]]["top_p"],
            }

            # Add provider-specific parameters
            if config["ai"] == "openai":
                zip_structure.ai_config.update(
                    {"max_tokens": config["openai"].get("max_tokens", 0)}
                )
            elif config["ai"] == "anthropic":
                zip_structure.ai_config.update(
                    {
                        "max_tokens_to_sample": config["anthropic"].get(
                            "max_tokens_to_sample", 0
                        ),
                        "top_k": config["anthropic"].get("top_k", 0),
                    }
                )

        except NoXMLFileFoundError as e:
            logging.error(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})
        except NoManuscriptFileError as e:
            logging.error(f"Error: {str(e)}")
            return json.dumps({"error": str(e)})

        zip_structure = update_file_paths(zip_structure, str(extract_dir))

        expected_figure_count = len(
            [
                fig
                for fig in zip_structure.figures
                if not re.search(r"EV", fig.figure_label, re.IGNORECASE)
            ]
        )
        expected_figure_labels = [
            fig.figure_label
            for fig in zip_structure.figures
            if not re.search(r"EV", fig.figure_label, re.IGNORECASE)
        ]
        logging.info(f"Expected figure count: {expected_figure_count}")

        # Initialize extractors based on AI provider
        logging.info("Initializing extractors")
        if config["ai"] == "openai":
            caption_extractor = FigureCaptionExtractorGpt(config["openai"])
        elif config["ai"] == "anthropic":
            caption_extractor = FigureCaptionExtractorClaude(config["anthropic"])
        else:
            raise ValueError(f"Unsupported AI provider: {config['ai']}")

        data_extractor = DataAvailabilityExtractorGPT(config)
        # Extract document content once and cache it
        logging.info("Extracting document content")
        doc_content = doc_cache.get_content(zip_structure._full_docx, caption_extractor)

        # Extract captions using cached content
        logging.info("Starting caption extraction process")
        zip_structure = caption_extractor.extract_captions(
            doc_content,
            zip_structure,
            expected_figure_count,
            expected_figure_labels=expected_figure_labels,
        )

        # Extract data availability using same cached content
        logging.info("Starting data availability extraction")
        data_records = data_extractor.extract_data_availability(doc_content)
        zip_structure.data_availability = data_records
        logging.info("Completed data availability extraction")

        # Process panels - create object detector and other components once
        object_detector = create_object_detection(config)
        panel_caption_matcher = MatchPanelCaptionOpenAI(config)
        panel_source_assigner = PanelSourceAssigner(config)

        # Process each figure
        for figure in zip_structure.figures:
            if figure._full_img_files:
                # try:
                # First detect panels using object detection
                pil_image, _ = convert_to_pil_image(figure._full_img_files[0])
                detected_panels = object_detector.detect_panels(pil_image)

                # Create initial panel objects with detection results
                panels = [
                    Panel(
                        panel_label="",  # Will be filled by panel_caption_matcher if caption exists
                        panel_caption="",  # Will be filled by panel_caption_matcher if caption exists
                        panel_bbox=panel["panel_bbox"],
                        confidence=panel["confidence"],
                        ai_response="",
                        sd_files=[],
                    )
                    for panel in detected_panels
                ]
                figure.panels = panels

                # If figure has valid caption, process panel captions and sources
                if (
                    figure.figure_caption
                    and figure.figure_caption != "Figure caption not found."
                ):
                    logging.info(f"Processing panels for figure {figure.figure_label}")
                    # Match panel captions
                    figure = panel_caption_matcher.match_captions(figure)
                    # Assign panel sources
                    figure = panel_source_assigner.assign_panel_source(figure)

                else:
                    logging.warning(
                        f"Skipping panel caption matching for {figure.figure_label} - No valid caption {figure.figure_caption}"
                    )

            # except Exception as e:
            #     logging.error(f"Error processing figure {figure.figure_label}: {str(e)}")

        # Process EV materials
        zip_structure = process_ev_materials(zip_structure)

        # Final processing before output
        zip_structure = process_zip_structure(zip_structure)

        output_json = json.dumps(
            zip_structure, cls=CustomJSONEncoder, ensure_ascii=False, indent=2
        )

        if output_path:
            output_path = (
                output_path
                if output_path.startswith("/app/")
                else f"/app/output/{os.path.basename(output_path)}"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_json)
                logging.info(f"Output written to {output_path}")
            except Exception as e:
                logging.error(f"An error occurred while writing to the file: {e}")

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
        output_json = json.dumps({"error": str(e)})

    finally:
        try:
            shutil.rmtree(extract_dir)
            logging.info(f"Cleaned up extracted files in {extract_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up extracted files: {str(e)}")

    return output_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a ZIP file using soda-curation"
    )
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument(
        "--config", required=True, help="Path to the configuration file"
    )
    parser.add_argument("--output", help="Path to the output JSON file")
    args = parser.parse_args()

    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)
