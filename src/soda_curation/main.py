import argparse
import json
import zipfile
import sys
import os
from pathlib import Path
from .config import load_config
from .pipeline.zip_structure.zip_structure_openai import StructureZipFileGPT
from .pipeline.zip_structure.zip_structure_anthropic import StructureZipFileClaude
from .pipeline.extract_captions.extract_captions_openai import FigureCaptionExtractorGpt
from .pipeline.extract_captions.extract_captions_anthropic import FigureCaptionExtractorClaude
from .pipeline.zip_structure.zip_structure_base import CustomJSONEncoder

def main():
    parser = argparse.ArgumentParser(description="Process a ZIP file using soda-curation")
    parser.add_argument("--zip", help="Path to the input ZIP file")
    parser.add_argument("--config", help="Path to the configuration file")
    args = parser.parse_args()

    if not args.zip or not args.config:
        print("Usage: python -m soda_curation.main --zip <path_to_zip_file> --config <path_to_config_file>")
        sys.exit(1)

    config = load_config(args.config)
    zip_path = Path(args.zip)
    
    if not zip_path.is_file():
        print(f"Error: ZIP file not found: {args.zip}")
        sys.exit(1)

    extract_dir = zip_path.parent / zip_path.stem
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(extract_dir)
        print(f"Extracted ZIP contents to: {extract_dir}")
    except zipfile.BadZipFile:
        print(f"Error: Invalid ZIP file: {args.zip}")
        sys.exit(1)

    if config['ai'] == 'openai':
        zip_processor = StructureZipFileGPT(config['openai'])
        caption_extractor = FigureCaptionExtractorGpt(config['openai'])
    elif config['ai'] == 'anthropic':
        zip_processor = StructureZipFileClaude(config['anthropic'])
        caption_extractor = FigureCaptionExtractorClaude(config['anthropic'])
    else:
        print(f"Error: Unknown AI provider: {config['ai']}")
        sys.exit(1)

    print("Debug: Processing ZIP structure")
    result = zip_processor.process_zip_structure(file_list)
    if result:
        print("Debug: ZIP structure processed successfully")
        
        # Get the DOCX file path from the ZIP structure result
        docx_file = result.docx
        if not docx_file:
            print("Error: No DOCX file found in the ZIP structure")
            sys.exit(1)
        
        docx_path = os.path.join(extract_dir, docx_file)
        if not os.path.exists(docx_path):
            print(f"Error: DOCX file not found at expected path: {docx_path}")
            sys.exit(1)
        
        print(f"Debug: Using DOCX file: {docx_path}")
        print("Debug: Extracting captions")
        result = caption_extractor.extract_captions(docx_path, result)
        print("Debug: Captions extraction process completed")
        print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
    else:
        print("Error: Failed to process ZIP structure")
        sys.exit(1)

    # Clean up extracted files
    print(f"Debug: Cleaning up extracted files")
    for root, dirs, files in os.walk(extract_dir, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(extract_dir)

if __name__ == "__main__":
    main()
