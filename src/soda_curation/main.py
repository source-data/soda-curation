import argparse
import json
import zipfile
import sys
from pathlib import Path
from .config import load_config
from .pipeline.zip_structure.zip_structure_openai import StructureZipFileGPT
from .pipeline.zip_structure.zip_structure_anthropic import StructureZipFileClaude
from .pipeline.zip_structure.zip_structure_base import CustomJSONEncoder

def main():
    """
    Main function to process a ZIP file using soda-curation.

    This function parses command-line arguments, loads the configuration,
    processes the ZIP file, and outputs the results as JSON.

    Command-line arguments:
    --zip: Path to the input ZIP file
    --config: Path to the configuration file

    The function will exit with an error message if required arguments are missing,
    if the ZIP file is not found or invalid, or if processing fails.
    """
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

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
    except zipfile.BadZipFile:
        print(f"Error: Invalid ZIP file: {args.zip}")
        sys.exit(1)

    if config['ai'] == 'openai':
        processor = StructureZipFileGPT(config['openai'])
    elif config['ai'] == 'anthropic':
        processor = StructureZipFileClaude(config['anthropic'])
    else:
        print(f"Error: Unknown AI provider: {config['ai']}")
        sys.exit(1)

    result = processor.process_zip_structure(file_list)
    if result:
        print(json.dumps(result, indent=2, cls=CustomJSONEncoder))
    else:
        print("Error: Failed to process ZIP structure")
        sys.exit(1)

if __name__ == "__main__":
    main()