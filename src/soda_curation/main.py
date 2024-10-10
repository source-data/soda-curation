import argparse
import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from .logging_config import setup_logging
from .pipeline.assign_panel_source.assign_panel_source import PanelSourceAssigner
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
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

logger = logging.getLogger(__name__)

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
        if rel_path != '.':
            path_parts = rel_path.split(os.sep)
            for part in path_parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            current[relative_path] = None
    logger.debug(f"Generated file tree: {json.dumps(file_tree, indent=2)}")
    return file_tree

def check_duplicate_panels(figure: Figure) -> str:
    """
    Check if a figure contains duplicate panel labels.

    Args:
        figure (Figure): The figure object to check for duplicate panels.

    Returns:
        str: "true" if duplicate panels are found, "false" otherwise.
    """
    panel_labels = [panel.panel_label for panel in figure.panels]
    return "true" if len(panel_labels) != len(set(panel_labels)) else "false"


def get_manuscript_structure(zip_path: str, extract_dir: str) -> ZipStructure:
    """
    Extract the manuscript structure from a ZIP file.

    Args:
        zip_path (str): Path to the ZIP file containing the manuscript data.
        extract_dir (str): Directory where ZIP contents are extracted.

    Returns:
        ZipStructure: A structured representation of the manuscript.
    """
    try:
        logger.info("Processing ZIP structure")
        xml_extractor = XMLStructureExtractor(zip_path, extract_dir)
        structure = xml_extractor.extract_structure()

        logger.info(f"Initial structure: manuscript_id={structure.manuscript_id}, xml={structure.xml}, docx={structure.docx}, pdf={structure.pdf}")
        logger.info(f"Number of figures: {len(structure.figures)}")

        # Update file path
        if structure.docx:
            structure._full_docx = full_path(extract_dir, structure.docx)
            logger.info(f"Full DOCX path: {structure._full_docx}")
        if structure.pdf:
            structure._full_pdf = full_path(extract_dir, structure.pdf)
            logger.info(f"Full PDF path: {structure._full_pdf}")
        
        # Process appendix items
        structure._full_appendix = []
        for app in structure.appendix:
            if isinstance(app, str):
                structure._full_appendix.append(full_path(extract_dir, app))
            elif isinstance(app, dict):
                structure._full_appendix.append(app)  # Keep dictionaries as they are
            else:
                logger.warning(f"Unexpected appendix item type: {type(app)}")

        for i, figure in enumerate(structure.figures):
            logger.info(f"Processing figure {i+1}: {figure.figure_label}")
            figure._full_img_files = [full_path(extract_dir, img) for img in figure.img_files]
            figure._full_sd_files = [full_path(extract_dir, sd) for sd in figure.sd_files]

        logger.info("ZIP structure processed successfully")
        return structure
    except Exception as e:
        logger.exception(f"Failed to process ZIP structure: {str(e)}")
        return ZipStructure(errors=[f"Manuscript structure extraction error: {str(e)}"])



def extract_figure_captions(
    zip_structure: ZipStructure, config: Dict[str, Any], expected_figure_count: int
) -> ZipStructure:
    """
    Extract figure captions from the manuscript.

    Args:
        zip_structure (ZipStructure): The current structure of the manuscript.
        config (Dict[str, Any]): Configuration dictionary for the extraction process.
        expected_figure_count (int): The expected number of figures in the manuscript.

    Returns:
        ZipStructure: Updated structure with extracted figure captions.
    """
    try:
        if config["ai"] == "openai":
            caption_extractor = FigureCaptionExtractorGpt(config["openai"])
        else:
            raise ValueError(f"Unknown AI provider: {config['ai']}")

        if zip_structure._full_docx and os.path.exists(zip_structure._full_docx):
            logger.info(f"Extracting captions from DOCX: {zip_structure._full_docx}")
            result = caption_extractor.extract_captions(zip_structure._full_docx, zip_structure, expected_figure_count)
        elif zip_structure._full_pdf and os.path.exists(zip_structure._full_pdf):
            logger.info(f"Extracting captions from PDF: {zip_structure._full_pdf}")
            result = caption_extractor.extract_captions(zip_structure._full_pdf, zip_structure, expected_figure_count)
        else:
            logger.warning("No DOCX or PDF file found for caption extraction")
            raise FileNotFoundError("No DOCX or PDF file found for caption extraction")

        return result
    except Exception as e:
        logger.error(f"Failed to extract figure captions: {str(e)}")
        zip_structure.errors.append(f"Figure caption extraction error: {str(e)}")
        return zip_structure

def extract_panels(zip_structure: ZipStructure, config: Dict[str, Any]) -> ZipStructure:
    """
    Detect and extract panels from figure images.

    Args:
        zip_structure (ZipStructure): The current structure of the manuscript.
        config (Dict[str, Any]): Configuration dictionary for the panel extraction process.

    Returns:
        ZipStructure: Updated structure with extracted panels.
    """
    try:
        logger.info("Performing object detection on figure images")
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
                                ai_response=None,  # or p.get("ai_response") if it exists
                            )
                            for p in panels
                        ]
                        logger.info(
                            f"Detected {len(panels)} panels in {figure.img_files[0]}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error during panel detection for {figure.img_files[0]}: {str(e)}"
                        )
                else:
                    logger.warning(f"Image file not found: {figure.img_files[0]}")
        return zip_structure
    except Exception as e:
        logger.error(f"Failed to extract panels: {str(e)}")
        zip_structure.errors.append(f"Panel extraction error: {str(e)}")
        return zip_structure


def match_panel_caption(
    zip_structure: ZipStructure, config: Dict[str, Any]
) -> ZipStructure:
    """
    Match extracted panel captions with their corresponding panels.

    Args:
        zip_structure (ZipStructure): The current structure of the manuscript with extracted panels.
        config (Dict[str, Any]): Configuration dictionary for the caption matching process.

    Returns:
        ZipStructure: Updated structure with matched panel captions and duplicate panel flags.
    """
    try:
        logger.info("Matching panel captions")
        if config["ai"] == "openai":
            panel_caption_matcher = MatchPanelCaptionOpenAI(config)
        else:
            raise ValueError(f"Unknown AI provider: {config['ai']}")

        for figure in zip_structure.figures:
            if figure._full_img_files:
                original_img_path = figure._full_img_files[0]
                if os.path.exists(original_img_path):
                    pil_image, _ = convert_to_pil_image(original_img_path)
                    figure._pil_image = (
                        pil_image  # Store the PIL image in the figure object
                    )

        result = panel_caption_matcher.match_captions(zip_structure)

        for figure in result.figures:
            figure.duplicated_panels = check_duplicate_panels(figure)
            logger.info(
                f"Figure {figure.figure_label} flag: {figure.duplicated_panels}"
            )
            if hasattr(figure, "_pil_image"):
                del figure._pil_image  # Remove the temporary PIL image

        return result
    except Exception as e:
        logger.error(f"Failed to match panel captions: {str(e)}")
        zip_structure.errors.append(f"Panel caption matching error: {str(e)}")
        return zip_structure


def main(zip_path: str, config_path: str, output_path: str = None) -> str:
    if not zip_path or not config_path:
        raise ValueError("ZIP path and config path must be provided")

    config = load_config(config_path)
    setup_logging(config)

    if config["ai"] not in ["openai"]:
        raise ValueError(f"Invalid AI provider: {config['ai']}")

    zip_file = Path(zip_path)
    extract_dir = zip_file.parent / zip_file.stem
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        return json.dumps(ZipStructure(errors=[f"Invalid ZIP file: {zip_path}"]), cls=CustomJSONEncoder)
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {str(e)}")
        return json.dumps(ZipStructure(errors=[f"Error extracting ZIP file: {str(e)}"]), cls=CustomJSONEncoder)

    zip_structure = get_manuscript_structure(zip_path, str(extract_dir))
    logger.info(f"Manuscript structure: id={zip_structure.manuscript_id}, xml={zip_structure.xml}, docx={zip_structure.docx}, pdf={zip_structure.pdf}")
    logger.info(f"Number of figures: {len(zip_structure.figures)}")

    # Calculate expected figure count (excluding EV figures)
    expected_figure_count = len([fig for fig in zip_structure.figures if not re.search(r'EV', fig.figure_label, re.IGNORECASE)])
    logger.info(f"Expected figure count: {expected_figure_count}")

    if zip_structure.docx:
        logger.info("Attempting DOCX caption extraction")
        zip_structure = extract_figure_captions(zip_structure, config, expected_figure_count)
    elif zip_structure.pdf:
        logger.info("No DOCX found. Attempting PDF caption extraction")
        zip_structure = extract_figure_captions(zip_structure, config, expected_figure_count)
    else:
        logger.error("Neither DOCX nor PDF file found. Cannot extract captions.")
        zip_structure.errors.append("No DOCX or PDF file found for caption extraction")

    logger.info(f"After extraction: docx={zip_structure.docx}, pdf={zip_structure.pdf}")
    logger.info(f"Final captions: {[figure.figure_caption for figure in zip_structure.figures]}")

    # Update config['extract_dir'] to the full path including the manuscript ID
    config["extract_dir"] = str(extract_dir)

    zip_structure = extract_panels(zip_structure, config)
    zip_structure = match_panel_caption(zip_structure, config)

    # New step: Assign panel source data
    file_tree = get_file_tree(extract_dir)
    logger.info(f"File tree structure: {json.dumps(file_tree, indent=2)}")
    logger.info("Starting panel source assignment process")
    panel_source_assigner = PanelSourceAssigner(config)
    zip_structure = panel_source_assigner.assign_panel_source(zip_structure)
    logger.info("Panel source assignment process completed")

    # Move EV-related source data files to appendix and ensure correct paths for sd_files
    ev_sd_files = []
    for figure in zip_structure.figures:
        non_ev_sd_files = []
        for sd_file in figure.sd_files:
            basename = os.path.basename(sd_file)
            if re.search(r'(Figure|Table|Dataset)\s*EV', basename, re.IGNORECASE):
                if sd_file not in zip_structure.appendix:
                    ev_sd_files.append(sd_file)
            else:
                # Ensure the path is relative to the ZIP file structure
                non_ev_sd_files.append(os.path.join('suppl_data', basename))
        figure.sd_files = non_ev_sd_files

        # Update _full_sd_files to exclude EV files
        figure._full_sd_files = [
            f for f in figure._full_sd_files 
            if not re.search(r'(Figure|Table|Dataset)\s*EV', os.path.basename(f), re.IGNORECASE)
        ]

    # Add EV files to appendix if not already present
    for ev_file in ev_sd_files:
        if ev_file not in zip_structure.appendix:
            zip_structure.appendix.append(ev_file)

    # Remove duplicates from appendix while preserving order
    seen = set()
    deduped_appendix = []
    for item in zip_structure.appendix:
        if isinstance(item, dict):
            # For dictionary items, use a tuple of its items for hashing
            item_hash = tuple(sorted(item.items()))
        else:
            item_hash = item
        if item_hash not in seen:
            seen.add(item_hash)
            deduped_appendix.append(item)
    zip_structure.appendix = deduped_appendix

    # Remove duplicates from non_associated_sd_files and exclude files already in appendix
    zip_structure.non_associated_sd_files = list(set(zip_structure.non_associated_sd_files))
    zip_structure.non_associated_sd_files = [
        file for file in zip_structure.non_associated_sd_files 
        if file not in zip_structure.appendix
    ]

    # Ensure all paths in appendix are relative to the ZIP file structure
    for i, item in enumerate(zip_structure.appendix):
        if isinstance(item, str):
            zip_structure.appendix[i] = os.path.join('suppl_data', os.path.basename(item))
        elif isinstance(item, dict) and 'object_id' in item:
            item['object_id'] = os.path.join('suppl_data', os.path.basename(item['object_id']))

    # Simplify appendix structure and add EV figures
    simplified_appendix = []
    ev_figures = []
    for item in zip_structure.appendix:
        if isinstance(item, dict):
            simplified_appendix.append(item['object_id'])
        else:
            simplified_appendix.append(item)

    # Move EV figures to appendix
    for figure in zip_structure.figures[:]:
        if figure.figure_label.startswith("Figure EV"):
            ev_figures.extend(figure.img_files)
            zip_structure.figures.remove(figure)

    simplified_appendix.extend(ev_figures)

    # Remove duplicates while preserving order
    zip_structure.appendix = list(dict.fromkeys(simplified_appendix))

    # Remove items from non_associated_sd_files if they're in appendix
    zip_structure.non_associated_sd_files = [
        file for file in zip_structure.non_associated_sd_files 
        if file not in zip_structure.appendix
    ]

    # Ensure all paths in appendix are relative to the ZIP file structure
    zip_structure.appendix = [
        os.path.join('suppl_data', os.path.basename(file)) 
        for file in zip_structure.appendix
    ]

    # Remove duplicate entries from non_associated_sd_files
    zip_structure.non_associated_sd_files = list(dict.fromkeys(zip_structure.non_associated_sd_files))

    output_json = json.dumps(zip_structure, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

    if output_path:
        output_path = output_path if output_path.startswith("/app/") else f"/app/output/{os.path.basename(output_path)}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_json)
            logger.info(f"Output written to {output_path}")
        except Exception as e:
            logger.error(f"An error occurred while writing to the file: {e}")

    # Clean up extracted files
    try:
        shutil.rmtree(extract_dir)
        logger.info(f"Cleaned up extracted files in {extract_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up extracted files: {str(e)}")

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
