#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage: $0 <path_to_zip_file> [path_to_output_file]"
    echo "  <path_to_zip_file>: Path to the input ZIP file (required)"
    echo "  [path_to_output_file]: Path to the output file (optional)"
}

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

ABSOLUTE_ZIP_PATH=$(realpath "$ZIP_FILE")
ABSOLUTE_CONFIG_PATH=$(realpath "config.yaml")

# Create logs directory if it doesn't exist
mkdir -p logs

OUTPUT_ARGS=""
if [ "$#" -eq 2 ]; then
    OUTPUT_FILE="$2"
    OUTPUT_DIRNAME=$(dirname "$OUTPUT_FILE")
    OUTPUT_FILENAME=$(basename "$OUTPUT_FILE")
    ABSOLUTE_OUTPUT_DIR=$(realpath "$OUTPUT_DIRNAME")
    OUTPUT_ARGS="--output /app/output/$OUTPUT_FILENAME"
    mkdir -p "$OUTPUT_DIRNAME"
    OUTPUT_VOLUME="-v $ABSOLUTE_OUTPUT_DIR:/app/output"
else
    OUTPUT_VOLUME=""
    OUTPUT_ARGS=""
fi

docker run -it \
  -v "$ABSOLUTE_CONFIG_PATH:/app/config.yaml" \
  -v "$ABSOLUTE_ZIP_PATH:/app/input/$ZIP_FILENAME" \
  -v "$(pwd)/logs:/app/logs" \
  $OUTPUT_VOLUME \
  soda-curation-cpu \
  --zip "/app/input/$ZIP_FILENAME" \
  --config "/app/config.yaml" \
  $OUTPUT_ARGS