#!/usr/bin/env bash
# run.sh - Run the graphrag-pipeline in Docker with memory limits
#
# Usage:
#   ./run.sh sources/flashrnn.txt --dry-run
#   ./run.sh sources/flashrnn.txt --ner-only
#   ./run.sh sources/flashrnn.txt              # full pipeline (NER + LLM + FalkorDB)
#
# Prereqs:
#   - docker build -t graphrag-pipeline .
#   - FalkorDB running on host port 6379 (or use docker-compose.yml)
#   - ANTHROPIC_API_KEY set in env (for LLM extraction)

set -euo pipefail

IMAGE_NAME="graphrag-pipeline"
SOURCE_FILE="${1:?Usage: ./run.sh <source-file> [--dry-run] [--ner-only]}"
shift

# Build if image doesn't exist
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Building $IMAGE_NAME..."
    docker build -t "$IMAGE_NAME" .
fi

exec docker run --rm \
    --memory=8g \
    --cpus=4 \
    --network=host \
    -e "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}" \
    -e "ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so" \
    -v "$(cd "$(dirname "$0")/sources" && pwd)":/app/sources:ro \
    "$IMAGE_NAME" \
    --source "$SOURCE_FILE" \
    "$@"
