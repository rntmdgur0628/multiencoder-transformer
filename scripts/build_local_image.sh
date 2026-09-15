#!/usr/bin/env bash
# Build only on the server HOST. It does not start a container or touch data.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
IMAGE_NAME="${IMAGE_NAME:-multiencoder-transformer:local}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"

if ! command -v docker >/dev/null; then
  echo "HOST ERROR: docker is not available on this server host." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "HOST ERROR: Docker daemon is unavailable or this account lacks access." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/Dockerfile" || ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
  echo "HOST ERROR: PROJECT_DIR is not the multiencoder-transformer checkout: $PROJECT_DIR" >&2
  exit 1
fi

echo "Building image on SERVER HOST"
echo "  project: $PROJECT_DIR"
echo "  image:   $IMAGE_NAME"
echo "  base:    $BASE_IMAGE"
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --tag "$IMAGE_NAME" \
  "$PROJECT_DIR"
docker image inspect "$IMAGE_NAME" --format 'Built image: {{.RepoTags}}'
