import argparse
import json
import zipfile
import sys
import os
import logging
from pathlib import Path
from .config import load_config
from .logging_config import setup_logging
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.extract_captions.extract_captions_anthropic import FigureCaptionExtractorClaude
from .pipeline.object_detection.object_detection import create_object_detection
from .pipeline.match_caption_panel.match_caption_panel_openai import MatchPanelCaptionOpenAI
from .pipeline.match_caption_panel.match_caption_panel_anthropic import MatchPanelCaptionClaude
from .pipeline.manuscript_structure.manuscript_structure import ZipStructure, Figure, CustomJSONEncoder
from .pipeline.manuscript_structure.manuscript_xml_parser import XMLStructureExtractor

def check_duplicate_panels(figure: Figure) -> str:
    panel_labels = [panel.get('panel_label', '') for panel in figure.panels]
    return 'true' if len(panel_labels) != len(set(panel_labels)) else 'false'

def main():
    parser = argparse.ArgumentParser(description="Process a ZIP file using soda-curation")
    parser.add_argument("--zip", help="Path to the input ZIP file")
    parser.add_argument("--config", help="Path to the configuration file")
    parser.add_argument('-o', '--output', type=str, required=False, help='Path to the output file.')
    
    args = parser.parse_args()

    if not args.zip or not args.config:
        print("Usage: python -m soda_curation.main --zip <path_to_zip_file> --config <path_to_config_file>")
        sys.exit(1)

    config = load_config(args.config)
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    
    # Check OpenAI configuration
    if 'openai' not in config or 'api_key' not in config['openai']:
        logger.error("OpenAI configuration is missing or incomplete in the config file")
        sys.exit(1)
    
    # Set up debug directory
    if config.get('debug', {}).get('enabled', False):
        debug_dir = Path(config['debug']['debug_dir'])
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug directory created at: {debug_dir}")
        config['debug']['debug_dir'] = str(debug_dir)
    else:
        config['debug'] = {'enabled': False, 'debug_dir': None}
    
    zip_path = Path(args.zip)
    if not zip_path.is_file():
        logger.error(f"ZIP file not found: {args.zip}")
        sys.exit(1)

    extract_dir = zip_path.parent / zip_path.stem
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted ZIP contents to: {extract_dir}")
        logger.debug(f"Contents of extracted directory: {os.listdir(extract_dir)}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {args.zip}")
        sys.exit(1)

    config['extract_dir'] = str(extract_dir)

    # Initialize components
    if config['ai'] == 'openai':
        caption_extractor = FigureCaptionExtractorGpt(config['openai'])
        panel_caption_matcher = MatchPanelCaptionOpenAI(config)
    elif config['ai'] == 'anthropic':
        caption_extractor = FigureCaptionExtractorClaude(config['anthropic'])
        panel_caption_matcher = MatchPanelCaptionClaude(config)
    else:
        logger.error(f"Unknown AI provider: {config['ai']}")
        sys.exit(1)

    object_detector = create_object_detection(config)

    # Process ZIP structure
    logger.info("Processing ZIP structure")
    try:
        xml_extractor = XMLStructureExtractor(str(zip_path), str(extract_dir))
        result = xml_extractor.extract_structure()
        logger.info("ZIP structure processed successfully")
        logger.debug(f"Extracted structure: {json.dumps(result, cls=CustomJSONEncoder, indent=2)}")
    except Exception as e:
        logger.error(f"Failed to process ZIP structure: {str(e)}")
        sys.exit(1)

    # Update file paths
    result.docx = os.path.join(extract_dir, result.docx) if result.docx else None
    result.pdf = os.path.join(extract_dir, result.pdf) if result.pdf else None
    for figure in result.figures:
        figure.img_files = [os.path.join(extract_dir, img) for img in figure.img_files]
        figure.sd_files = [os.path.join(extract_dir, sd) for sd in figure.sd_files]

    # Log file paths for debugging
    logger.debug(f"DOCX path: {result.docx}")
    logger.debug(f"PDF path: {result.pdf}")
    for figure in result.figures:
        logger.debug(f"Figure {figure.figure_label} image files: {figure.img_files}")

    # Extract captions
    if result.docx and os.path.exists(result.docx):
        logger.info(f"Extracting captions from DOCX: {result.docx}")
        result = caption_extractor.extract_captions(result.docx, result)
        if all(figure.figure_caption == "Figure caption not found." for figure in result.figures):
            logger.info("No captions found in DOCX, falling back to PDF")
            if result.pdf and os.path.exists(result.pdf):
                logger.info(f"Extracting captions from PDF: {result.pdf}")
                result = caption_extractor.extract_captions(result.pdf, result)
    elif result.pdf and os.path.exists(result.pdf):
        logger.info(f"Extracting captions from PDF: {result.pdf}")
        result = caption_extractor.extract_captions(result.pdf, result)
    else:
        logger.warning("No DOCX or PDF file found for caption extraction")

    # Perform object detection
    logger.info("Performing object detection on figure images")
    for figure in result.figures:
        if figure.img_files:
            img_path = figure.img_files[0]
            if os.path.exists(img_path):
                try:
                    panels = object_detector.detect_panels(img_path)
                    figure.panels = panels
                    logger.info(f"Detected {len(panels)} panels in {img_path}")
                except Exception as e:
                    logger.error(f"Error during panel detection for {img_path}: {str(e)}")
            else:
                logger.warning(f"Image file not found: {img_path}")
                logger.debug(f"Current working directory: {os.getcwd()}")
                logger.debug(f"Contents of {os.path.dirname(img_path)}: {os.listdir(os.path.dirname(img_path))}")

    # Match panel captions
    logger.info("Matching panel captions")
    result = panel_caption_matcher.match_captions(result)

    # Check for duplicate panels
    for figure in result.figures:
        figure.duplicated_panels = check_duplicate_panels(figure)
        logger.info(f"Figure {figure.figure_label} flag: {figure.duplicated_panels}")

    # Output results
    output_json = json.dumps(result, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)
    logger.info(output_json)

    if args.output:
        output_path = args.output if args.output.startswith('/app/') else f'/app/output/{os.path.basename(args.output)}'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                json.dump(result, outfile, cls=CustomJSONEncoder, ensure_ascii=False, indent=4)
            logger.info(f"JSON data has been written to {output_path}")
        except Exception as e:
            logger.error(f"An error occurred while writing to the file: {e}")

    # Clean up extracted files
    logger.info("Cleaning up extracted files")
    for root, dirs, files in os.walk(extract_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(extract_dir)

if __name__ == "__main__":
    main()
