#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOCKERFILE="$PROJECT_DIR/docker/Dockerfile.sandbox"

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "Sandbox Dockerfile is missing: $DOCKERFILE" >&2
    exit 1
fi
if ! docker image inspect quantmind-oss:latest >/dev/null 2>&1; then
    echo "Base image quantmind-oss:latest must be built or loaded first" >&2
    exit 1
fi

docker build \
    --pull=false \
    --build-arg BASE_IMAGE=quantmind-oss:latest \
    --file "$DOCKERFILE" \
    --tag quantmind-sandbox:latest \
    "$PROJECT_DIR"

label="$(docker image inspect quantmind-sandbox:latest \
    --format '{{ index .Config.Labels "quantmind.user_code_sandbox" }}')"
if [[ "$label" != true ]]; then
    echo "Built sandbox image failed its required identity label check" >&2
    exit 1
fi

echo "Built quantmind-sandbox:latest from quantmind-oss:latest"
