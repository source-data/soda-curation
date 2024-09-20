import argparse
import json
import zipfile
import sys
import os
import logging
from pathlib import Path
from .config import load_config
from .logging_config import setup_logging
from .pipeline.zip_structure.zip_structure_openai import StructureZipFileGPT
from .pipeline.zip_structure.zip_structure_anthropic import StructureZipFileClaude
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.extract_captions.extract_captions_anthropic import FigureCaptionExtractorClaude
from .pipeline.object_detection.object_detection import create_object_detection
from .pipeline.match_caption_panel.match_caption_panel_openai import MatchPanelCaptionOpenAI
from .pipeline.zip_structure.zip_structure_base import CustomJSONEncoder

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
    if 'openai' not in config:
        logger.error("OpenAI configuration is missing from the config file")
        sys.exit(1)
    
    if 'api_key' not in config['openai']:
        logger.error("API key is missing from the OpenAI configuration")
        sys.exit(1)
    
    # Set up debug directory
    if config.get('debug', {}).get('enabled', False):
        debug_dir = Path(config['debug']['debug_dir'])
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug directory created at: {debug_dir}")
        config['debug']['debug_dir'] = str(debug_dir)
    else:
        config['debug'] = {'enabled': False, 'debug_dir': None}
    
    # Log configuration (be careful not to log sensitive information like API keys)
    logger.debug(f"Configuration loaded: {json.dumps({k: v for k, v in config.items() if k != 'openai'}, indent=2)}")
    logger.debug(f"OpenAI config keys: {', '.join(config['openai'].keys())}")

    zip_path = Path(args.zip)
    
    if not zip_path.is_file():
        logger.error(f"ZIP file not found: {args.zip}")
        sys.exit(1)

    extract_dir = zip_path.parent / zip_path.stem
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(extract_dir)
        logger.info(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {args.zip}")
        sys.exit(1)

    # Add extract_dir to config for use in other components
    config['extract_dir'] = str(extract_dir)

    if config['ai'] == 'openai':
        zip_processor = StructureZipFileGPT(config['openai'])
        caption_extractor = FigureCaptionExtractorGpt(config['openai'])
        # Pass the entire config including extract_dir
        panel_caption_matcher = MatchPanelCaptionOpenAI(config)
    elif config['ai'] == 'anthropic':
        zip_processor = StructureZipFileClaude(config['anthropic'])
        caption_extractor = FigureCaptionExtractorClaude(config['anthropic'])
        # TODO: Implement Anthropic version of panel caption matcher
        panel_caption_matcher = None
    else:
        logger.error(f"Unknown AI provider: {config['ai']}")
        sys.exit(1)

    # Initialize object detection
    object_detector = create_object_detection(config)

    logger.info("Processing ZIP structure")
    result = zip_processor.process_zip_structure(file_list)
    if result:
        logger.info("ZIP structure processed successfully")
        
        # Try to extract captions from DOCX first
        docx_file = result.docx
        if docx_file:
            docx_path = os.path.join(extract_dir, docx_file)
            if os.path.exists(docx_path):
                logger.info(f"Extracting captions from DOCX: {docx_path}")
                result = caption_extractor.extract_captions(docx_path, result)
            else:
                logger.warning(f"DOCX file not found at expected path: {docx_path}")
        else:
            logger.warning("No DOCX file found in the ZIP structure")

        # If no captions were found in DOCX or DOCX doesn't exist, try PDF
        if all(figure.figure_caption == "Figure caption not found." for figure in result.figures):
            logger.info("No captions found in DOCX. Attempting to extract from PDF.")
            pdf_file = result.pdf
            if pdf_file:
                pdf_path = os.path.join(extract_dir, pdf_file)
                if os.path.exists(pdf_path):
                    logger.info(f"Extracting captions from PDF: {pdf_path}")
                    result = caption_extractor.extract_captions(pdf_path, result)
                else:
                    logger.warning(f"PDF file not found at expected path: {pdf_path}")
            else:
                logger.warning("No PDF file found in the ZIP structure")
        
        logger.info("Captions extraction process completed")

        # Perform object detection on figure images
        logger.info("Performing object detection on figure images")
        for figure in result.figures:
            if figure.img_files:
                img_path = os.path.join(extract_dir, figure.img_files[0])
                if os.path.exists(img_path):
                    try:
                        panels = object_detector.detect_panels(img_path)
                        figure.panels = panels
                        logger.info(f"Detected {len(panels)} panels in {img_path}")
                    except Exception as e:
                        logger.error(f"Error during panel detection for {img_path}: {str(e)}")
                        figure.panels = []
                else:
                    logger.warning(f"Image file not found: {img_path}")

        # Match panel captions
        if panel_caption_matcher:
            logger.info("Matching panel captions")
            result = panel_caption_matcher.match_captions(result)
        else:
            logger.warning("Panel caption matching not available for the selected AI provider")

        logger.info(json.dumps(result, cls=CustomJSONEncoder, ensure_ascii=False, indent=2).encode('utf-8').decode())
        
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created directory: {output_dir}")
            # Check if the output file exists
            if os.path.exists(args.output):
                logger.warning(f"File {args.output} already exists and will be overwritten.")
            
            # Write JSON data to the output file with pretty formatting
            try:
                with open(args.output, 'w') as outfile:
                    json.dumps(result, cls=CustomJSONEncoder, ensure_ascii=False, indent=4).encode('utf-8').decode()
                logger.info(f"JSON data has been written to {args.output}")
            except Exception as e:
                logger.error(f"An error occurred while writing to the file: {e}")
    else:
        logger.error("Failed to process ZIP structure")
        sys.exit(1)



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
