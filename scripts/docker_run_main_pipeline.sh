#!/usr/bin/env bash
# Build the CPU dev image and run the main curation pipeline on a ZIP under data/archives/.
#
# Usage:
#   ./scripts/docker_run_main_pipeline.sh [archive.zip]
# Default archive: EMBOJ-2025-121381.zip
#
# Requires: Docker, data/archives/<zip>, .env.dev with API keys

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-soda-curation-cpu}"
ZIP_NAME="${1:-EMBOJ-2025-121381.zip}"
STEM="${ZIP_NAME%.zip}"

if [[ ! -f "$ROOT/data/archives/$ZIP_NAME" ]]; then
  echo "Missing archive: $ROOT/data/archives/$ZIP_NAME" >&2
  exit 1
fi

echo "Building $IMAGE (no GIT_ACCESS_TOKEN; public deps only) ..."
docker build -t "$IMAGE" -f Dockerfile.cpu --target development \
  --build-arg DEPLOYMENT_ENV=dev .

echo "Running main pipeline on $ZIP_NAME -> data/output/${STEM}_main.json ..."
docker run --rm \
  --env-file "$ROOT/.env.dev" \
  -v "$ROOT/data:/app/data" \
  "$IMAGE" \
  poetry run python -m src.soda_curation.main \
    --zip "/app/data/archives/$ZIP_NAME" \
    --config /app/config.yaml \
    --output "/app/data/output/${STEM}_main.json"

echo "Done. Output: $ROOT/data/output/${STEM}_main.json"
