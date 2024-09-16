#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_zip_file>"
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

# Use a CPU-only base image
docker run -it \
    -v "$ABSOLUTE_CONFIG_PATH:/app/config.yaml" \
    -v "$ABSOLUTE_ZIP_PATH:/app/input/$ZIP_FILENAME" \
    soda-curation-cpu \
    --zip "/app/input/$ZIP_FILENAME" \
    --config "/app/config.yaml"
