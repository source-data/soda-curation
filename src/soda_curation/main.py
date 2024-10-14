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
        logger.info(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {str(e)}")
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
    logger.info(f"Manuscript structure: id={structure.manuscript_id}, xml={structure.xml}, docx={structure.docx}, pdf={structure.pdf}")
    logger.info(f"Number of figures: {len(structure.figures)}")
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
        logger.info(f"Full DOCX path: {zip_structure._full_docx}")
    if zip_structure.pdf:
        zip_structure._full_pdf = full_path(extract_dir, zip_structure.pdf)
        logger.info(f"Full PDF path: {zip_structure._full_pdf}")
    
    zip_structure._full_appendix = []
    for app in zip_structure.appendix:
        if isinstance(app, str):
            zip_structure._full_appendix.append(full_path(extract_dir, app))
        elif isinstance(app, dict):
            zip_structure._full_appendix.append(app)
        else:
            logger.warning(f"Unexpected appendix item type: {type(app)}")

    for figure in zip_structure.figures:
        figure._full_img_files = [full_path(extract_dir, img) for img in figure.img_files]
        figure._full_sd_files = [full_path(extract_dir, sd) for sd in figure.sd_files]

    return zip_structure

def extract_figure_captions(zip_structure: ZipStructure, config: Dict[str, Any], expected_figure_count: int) -> ZipStructure:
    """
    Extract figure captions from the manuscript using an AI model.
    
    Args:
        zip_structure (ZipStructure): The structured representation of the manuscript.
        config (Dict[str, Any]): The configuration settings for the AI model.
        expected_figure_count (int): The expected number of figures in the manuscript.
        
    Returns:
        ZipStructure: The updated ZipStructure object with extracted figure captions.
    """
    caption_extractor = FigureCaptionExtractorGpt(config["openai"])
    result = caption_extractor.extract_captions(zip_structure._full_docx, zip_structure, expected_figure_count)
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
                    logger.info(f"Detected {len(panels)} panels in {figure.img_files[0]}")
                except Exception as e:
                    logger.error(f"Error during panel detection for {figure.img_files[0]}: {str(e)}")
            else:
                logger.warning(f"Image file not found: {figure.img_files[0]}")
    return zip_structure

def match_panel_caption(zip_structure: ZipStructure, config: Dict[str, Any]) -> ZipStructure:
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

def assign_panel_source(zip_structure: ZipStructure, config: Dict[str, Any], extract_dir: str) -> ZipStructure:
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
    logger.info(f"File tree structure: {json.dumps(file_tree, indent=2)}")
    panel_source_assigner = PanelSourceAssigner(config)
    zip_structure = panel_source_assigner.assign_panel_source(zip_structure)

    # Ensure non_associated_sd_files is initialized
    if not hasattr(zip_structure, 'non_associated_sd_files'):
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
                    if isinstance(file, str) and not ("__MACOSX" in file or file.endswith('.DS_Store'))
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
                with zipfile.ZipFile(sd_file, 'r') as zip_ref:
                    all_sd_files_in_zip = set(zip_ref.namelist())

                # Remove unwanted files
                all_sd_files_in_zip = set(
                    file for file in all_sd_files_in_zip if not ("__MACOSX" in file or file.endswith('.DS_Store'))
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
                if re.search(r'(Figure|Table|Dataset)\s*EV', os.path.basename(file), re.IGNORECASE):
                    ev_materials.append(file)

    for material in ev_materials:
        material_name = os.path.basename(material)
        match = re.search(r'(Figure|Table|Dataset)\s*EV(\d+)', material_name, re.IGNORECASE)
        if match:
            ev_type = match.group(1).capitalize()
            ev_number = match.group(2)
            ev_figure = next((fig for fig in zip_structure.figures if fig.figure_label == f"{ev_type} EV{ev_number}"), None)
            if ev_figure:
                ev_figure.sd_files.append(material_name)
                logger.info(f"Assigned {material_name} to {ev_figure.figure_label}")
            else:
                zip_structure.non_associated_sd_files.append(material_name)
                logger.info(f"Added {material_name} to non_associated_sd_files (no matching EV figure found)")
        else:
            zip_structure.non_associated_sd_files.append(material_name)
            logger.info(f"Added {material_name} to non_associated_sd_files (not recognized as EV material)")

    return zip_structure

def process_zip_structure(zip_structure):
    """
    Process the ZipStructure object before serializing to JSON.
    
    Args:
        zip_structure (ZipStructure): The ZipStructure object to process.
        
    Returns:
        ZipStructure: The processed ZipStructure object.
    """
    manuscript_id = zip_structure.manuscript_id

    # Update appendix paths
    processed_appendix = []
    for item in zip_structure.appendix:
        if isinstance(item, dict):
            processed_appendix.append(f"{manuscript_id}/{item['object_id']}")
        elif isinstance(item, str):
            processed_appendix.append(f"{manuscript_id}/{item}")
    zip_structure.appendix = processed_appendix

    non_associated_sd_files = set()

    for figure in zip_structure.figures:
        # Update sd_files to only include the ZIP file
        if figure._full_sd_files:
            figure.sd_files = [os.path.relpath(figure._full_sd_files[0], '/app/input/')]
        else:
            figure.sd_files = []

        # Update img_files to include the manuscript ID in the path
        figure.img_files = [os.path.relpath(img_file, '/app/input/') for img_file in figure._full_img_files]

        # Process panels
        for panel in figure.panels:
            if figure._full_sd_files:
                zip_filename = os.path.basename(figure._full_sd_files[0])
                panel.sd_files = [
                    f"{zip_filename}:Figure {figure.figure_label.split()[-1]}/{file.split('/', 1)[-1]}"
                    for file in panel.sd_files
                    if isinstance(file, str) and not ("__MACOSX" in file or file.endswith('.DS_Store'))
                ]
            else:
                non_associated_sd_files.update(file for file in panel.sd_files if isinstance(file, str))
                panel.sd_files = []

    # Process non_associated_sd_files
    for file in zip_structure.non_associated_sd_files:
        if isinstance(file, str):
            if file.endswith(('.xlsx', '.pdf')):
                non_associated_sd_files.add(f"{manuscript_id}/suppl_data/{os.path.basename(file)}")
            elif ':' in file:
                # It's already in a ZIP file format, just clean it up
                zip_name, internal_path = file.split(':', 1)
                non_associated_sd_files.add(f"{os.path.basename(zip_name)}:{internal_path.split('/', 1)[-1]}")
            else:
                # Assume it's from a ZIP file
                figure_number = next(
                    (fig.figure_label.split()[-1] for fig in zip_structure.figures 
                     for panel in fig.panels
                     if any(file in sd_file for sd_file in panel.sd_files if isinstance(sd_file, str))),
                    ''
                )
                if figure_number:
                    non_associated_sd_files.add(f"Figure {figure_number}.zip:Figure {figure_number}/{file}")
                else:
                    non_associated_sd_files.add(file)

    # Remove duplicates, files with __MACOSX or .DS_Store, and files that are properly associated with panels
    all_panel_sd_files = set()
    for figure in zip_structure.figures:
        for panel in figure.panels:
            all_panel_sd_files.update(sd_file for sd_file in panel.sd_files if isinstance(sd_file, str))

    zip_structure.non_associated_sd_files = list(
        non_associated_sd_files - all_panel_sd_files
    )

    # Remove any files with __MACOSX or .DS_Store
    zip_structure.non_associated_sd_files = [
        file for file in zip_structure.non_associated_sd_files
        if "__MACOSX" not in file and not file.endswith('.DS_Store')
    ]

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

    extract_zip_contents(zip_path, extract_dir)

    zip_structure = get_manuscript_structure(zip_path, str(extract_dir))
    zip_structure = update_file_paths(zip_structure, str(extract_dir))

    expected_figure_count = len([fig for fig in zip_structure.figures if not re.search(r'EV', fig.figure_label, re.IGNORECASE)])
    logger.info(f"Expected figure count: {expected_figure_count}")

    logger.info("Starting caption extraction process")
    zip_structure = extract_figure_captions(zip_structure, config, expected_figure_count)
    logger.info("Caption extraction process completed")

    logger.info("Updating source data files")
    zip_structure = update_sd_files(zip_structure)
    logger.info("Source data files updated")

    missing_captions = [fig.figure_label for fig in zip_structure.figures if fig.figure_caption == "Figure caption not found."]
    if missing_captions:
        logger.warning(f"Captions still missing for figures: {', '.join(missing_captions)}")
    else:
        logger.info("All figure captions successfully extracted")

    config["extract_dir"] = str(extract_dir)

    zip_structure = extract_panels(zip_structure, config)
    zip_structure = match_panel_caption(zip_structure, config)

    logger.info("Starting panel source assignment process")
    zip_structure = assign_panel_source(zip_structure, config, str(extract_dir))
    logger.info("Panel source assignment process completed")

    zip_structure = process_ev_materials(zip_structure)
    zip_structure = process_zip_structure(zip_structure)

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

    try:
        shutil.rmtree(extract_dir)
        logger.info(f"Cleaned up extracted files in {extract_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up extracted files: {str(e)}")

    return output_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a ZIP file using soda-curation")
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", help="Path to the output JSON file")
    args = parser.parse_args()

    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)