#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="rggl-cli"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for --build flag
if [[ "${1:-}" == "--build" ]]; then
  echo "🔨 Building image..."
  docker build -t "$IMAGE_NAME" "$DIR"
  shift
fi

# Pass through OPENAI_API_KEY if it's set in host env
ENV_ARGS=()
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  ENV_ARGS+=("-e" "OPENAI_API_KEY=$OPENAI_API_KEY")
fi

# Run the CLI
exec docker run --rm -it "${ENV_ARGS[@]}" "$IMAGE_NAME" "$@"
