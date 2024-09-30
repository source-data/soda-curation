import argparse
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from .logging_config import setup_logging
from .pipeline.extract_captions.extract_captions_anthropic import (
    FigureCaptionExtractorClaude,
)
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.manuscript_structure.manuscript_structure import (
    CustomJSONEncoder,
    Figure,
    Panel,
    ZipStructure,
    full_path,
)
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor
from .pipeline.match_caption_panel.match_caption_panel_anthropic import (
    MatchPanelCaptionClaude,
)
from .pipeline.match_caption_panel.match_caption_panel_openai import (
    MatchPanelCaptionOpenAI,
)
from .pipeline.object_detection.object_detection import (
    convert_to_pil_image,
    create_object_detection,
)

logger = logging.getLogger(__name__)

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
        xml_extractor = XMLStructureExtractor(str(zip_path), extract_dir)
        structure = xml_extractor.extract_structure()
        
        # Update file paths
        structure._full_docx = full_path(extract_dir, structure.docx) if structure.docx else None
        structure._full_pdf = full_path(extract_dir, structure.pdf) if structure.pdf else None
        structure._full_appendix = [full_path(extract_dir, app) for app in structure.appendix]
        
        for figure in structure.figures:
            figure._full_img_files = [full_path(extract_dir, img) for img in figure.img_files]
            figure._full_sd_files = [full_path(extract_dir, sd) for sd in figure.sd_files]
        
        logger.info("ZIP structure processed successfully")
        return structure
    except Exception as e:
        logger.error(f"Failed to process ZIP structure: {str(e)}")
        return ZipStructure(errors=[f"Manuscript structure extraction error: {str(e)}"])

def extract_figure_captions(zip_structure: ZipStructure, config: Dict[str, Any]) -> ZipStructure:
    """
    Extract figure captions from the manuscript.

    Args:
        zip_structure (ZipStructure): The current structure of the manuscript.
        config (Dict[str, Any]): Configuration dictionary for the extraction process.

    Returns:
        ZipStructure: Updated structure with extracted figure captions.
    """
    try:
        if config['ai'] == 'openai':
            caption_extractor = FigureCaptionExtractorGpt(config['openai'])
        elif config['ai'] == 'anthropic':
            caption_extractor = FigureCaptionExtractorClaude(config['anthropic'])
        else:
            raise ValueError(f"Unknown AI provider: {config['ai']}")

        if zip_structure._full_docx and os.path.exists(zip_structure._full_docx):
            logger.info(f"Extracting captions from DOCX: {zip_structure._full_docx}")
            result = caption_extractor.extract_captions(zip_structure._full_docx, zip_structure)
        elif zip_structure._full_pdf and os.path.exists(zip_structure._full_pdf):
            logger.info(f"Extracting captions from PDF: {zip_structure._full_pdf}")
            result = caption_extractor.extract_captions(zip_structure._full_pdf, zip_structure)
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
                                ai_response=None  # or p.get("ai_response") if it exists
                            )
                            for p in panels
                        ]
                        logger.info(f"Detected {len(panels)} panels in {figure.img_files[0]}")
                    except Exception as e:
                        logger.error(f"Error during panel detection for {figure.img_files[0]}: {str(e)}")
                else:
                    logger.warning(f"Image file not found: {figure.img_files[0]}")
        return zip_structure
    except Exception as e:
        logger.error(f"Failed to extract panels: {str(e)}")
        zip_structure.errors.append(f"Panel extraction error: {str(e)}")
        return zip_structure

def match_panel_caption(zip_structure: ZipStructure, config: Dict[str, Any]) -> ZipStructure:
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
        if config['ai'] == 'openai':
            panel_caption_matcher = MatchPanelCaptionOpenAI(config)
        elif config['ai'] == 'anthropic':
            panel_caption_matcher = MatchPanelCaptionClaude(config)
        else:
            raise ValueError(f"Unknown AI provider: {config['ai']}")

        for figure in zip_structure.figures:
            if figure._full_img_files:
                original_img_path = figure._full_img_files[0]
                if os.path.exists(original_img_path):
                    pil_image, _ = convert_to_pil_image(original_img_path)
                    figure._pil_image = pil_image  # Store the PIL image in the figure object

        result = panel_caption_matcher.match_captions(zip_structure)
        
        for figure in result.figures:
            figure.duplicated_panels = check_duplicate_panels(figure)
            logger.info(f"Figure {figure.figure_label} flag: {figure.duplicated_panels}")
            if hasattr(figure, '_pil_image'):
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

    if config['ai'] not in ['openai', 'anthropic']:
        raise ValueError(f"Invalid AI provider: {config['ai']}")

    zip_file = Path(zip_path)
    extract_dir = zip_file.parent / zip_file.stem
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"Extracted ZIP contents to: {extract_dir}")

    # Extract ZIP contents
    zip_file = Path(zip_path)
    extract_dir = zip_file.parent / zip_file.stem
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {zip_path}")
        return json.dumps(ZipStructure(errors=[f"Invalid ZIP file: {zip_path}"]), cls=CustomJSONEncoder)
    except Exception as e:
        logger.error(f"Error extracting ZIP file: {str(e)}")
        return json.dumps(ZipStructure(errors=[f"Error extracting ZIP file: {str(e)}"]), cls=CustomJSONEncoder)

    zip_structure = get_manuscript_structure(zip_path, str(extract_dir))
    
    # Update config['extract_dir'] to the full path including the manuscript ID
    config['extract_dir'] = str(extract_dir)
    
    # Try DOCX extraction first
    logger.info("Attempting DOCX caption extraction")
    zip_structure = extract_figure_captions(zip_structure, config)
    
    # If no captions were extracted from DOCX, try PDF
    if all(figure.figure_caption == "Figure caption not found." for figure in zip_structure.figures):
        logger.info("No captions found in DOCX, attempting extraction from PDF")
        zip_structure.docx = None  # Set docx to None when falling back to PDF
        zip_structure._full_docx = None
        zip_structure = extract_figure_captions(zip_structure, config)
    
    logger.info(f"After PDF extraction: docx={zip_structure.docx}, pdf={zip_structure.pdf}")
    logger.info(f"Final captions: {[figure.figure_caption for figure in zip_structure.figures]}")
    
    zip_structure = extract_panels(zip_structure, config)
    zip_structure = match_panel_caption(zip_structure, config)

    output_json = json.dumps(zip_structure, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

    if output_path:
        output_path = (
            output_path
            if output_path.startswith("/app/")
            else f"/app/output/{os.path.basename(output_path)}"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser(description="Process a ZIP file using soda-curation")
    parser.add_argument("--zip", required=True, help="Path to the input ZIP file")
    parser.add_argument("--config", required=True, help="Path to the configuration file")
    parser.add_argument("--output", help="Path to the output JSON file")
    args = parser.parse_args()

    output_json = main(args.zip, args.config, args.output)
    if not args.output:
        print(output_json)