#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 <path_to_zip_file> [path_to_output_file]"
    echo "  <path_to_zip_file>: Path to the input ZIP file (required)"
    echo "  [path_to_output_file]: Path to the output file (optional)"
}

# Check if at least one argument is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    print_usage
    exit 1
fi

ZIP_FILE="$1"
ZIP_FILENAME=$(basename "$ZIP_FILE")

if [ ! -f "$ZIP_FILE" ]; then
    echo "Error: File $ZIP_FILE does not exist."
    exit 1
fi

# Get absolute paths
ABSOLUTE_ZIP_PATH=$(realpath "$ZIP_FILE")
ABSOLUTE_CONFIG_PATH=$(realpath "config.yaml")

# Initialize OUTPUT_ARGS
OUTPUT_ARGS=""

# Check if output file is provided
if [ "$#" -eq 2 ]; then
    OUTPUT_FILE="$2"
    OUTPUT_DIRNAME=$(dirname "$OUTPUT_FILE")
    OUTPUT_FILENAME=$(basename "$OUTPUT_FILE")
    ABSOLUTE_OUTPUT_DIR=$(realpath "$OUTPUT_DIRNAME")
    OUTPUT_ARGS="--output /app/output/$OUTPUT_FILENAME"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIRNAME"
    
    # Add output directory volume mount
    OUTPUT_VOLUME="-v $ABSOLUTE_OUTPUT_DIR:/app/output"
else
    OUTPUT_VOLUME=""
    OUTPUT_ARGS=""
fi

# Use a CPU-only base image
docker run -it \
    -v "$ABSOLUTE_CONFIG_PATH:/app/config.yaml" \
    -v "$ABSOLUTE_ZIP_PATH:/app/input/$ZIP_FILENAME" \
    $OUTPUT_VOLUME \
    soda-curation-cpu \
    --zip "/app/input/$ZIP_FILENAME" \
    --config "/app/config.yaml" \
    $OUTPUT_ARGS

