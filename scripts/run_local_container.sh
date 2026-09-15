#!/usr/bin/env bash
# Start an interactive GPU container from the SERVER HOST with explicit mounts.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
IMAGE_NAME="${IMAGE_NAME:-multiencoder-transformer:local}"
: "${MOISES_DIR:?Set MOISES_DIR to the SERVER-HOST MoisesDB root first.}"
: "${CACHE_DIR:?Set CACHE_DIR to a writable SERVER-HOST cache path first.}"
: "${RUN_DIR:?Set RUN_DIR to a writable SERVER-HOST run path first.}"
GPU_ID="${GPU_ID:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-multiencoder-shell-$(date +%Y%m%d-%H%M%S)}"

if ! command -v docker >/dev/null; then
  echo "HOST ERROR: docker is not available on this server host." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "HOST ERROR: Docker daemon is unavailable or this account lacks access." >&2
  exit 1
fi
for path in "$PROJECT_DIR" "$MOISES_DIR"; do
  if [[ ! -d "$path" ]]; then
    echo "HOST ERROR: required host directory does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
  echo "HOST ERROR: PROJECT_DIR is not the project checkout: $PROJECT_DIR" >&2
  exit 1
fi
if ! find "$MOISES_DIR" -name data.json -print -quit | grep -q .; then
  echo "HOST ERROR: MOISES_DIR has no data.json: $MOISES_DIR" >&2
  exit 1
fi
mkdir -p "$CACHE_DIR" "$RUN_DIR"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
  echo "HOST ERROR: image is missing: $IMAGE_NAME. Run scripts/build_local_image.sh first." >&2
  exit 1
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "HOST ERROR: container name already exists: $CONTAINER_NAME" >&2
  exit 1
fi

echo "Starting container from SERVER HOST"
echo "  host project: $PROJECT_DIR  -> /workspace/project"
echo "  host Moises:  $MOISES_DIR  -> /workspace/data/moises (read-only)"
echo "  host cache:   $CACHE_DIR   -> /workspace/cache"
echo "  host runs:    $RUN_DIR     -> /workspace/runs"
echo "  GPU:          $GPU_ID"

docker run --rm -it \
  --name "$CONTAINER_NAME" \
  --gpus "device=$GPU_ID" \
  -v "$PROJECT_DIR:/workspace/project" \
  -v "$MOISES_DIR:/workspace/data/moises:ro" \
  -v "$CACHE_DIR:/workspace/cache" \
  -v "$RUN_DIR:/workspace/runs" \
  -e PYTHONPATH=/workspace/project/src \
  -e MOISES_ROOT=/workspace/data/moises \
  -e CACHE_ROOT=/workspace/cache \
  -e RUN_ROOT=/workspace/runs \
  -w /workspace/project \
  "$IMAGE_NAME" \
  bash -lc '
    set -e
    echo "CONTAINER CHECK: mounts"
    test -f /workspace/project/pyproject.toml
    test -d /workspace/data/moises
    test -d /workspace/cache
    test -d /workspace/runs
    find /workspace/data/moises -name data.json -print -quit
    echo "CONTAINER CHECK: Python/CUDA"
    python - <<"PY"
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CONTAINER ERROR: CUDA is not visible")
print("gpu:", torch.cuda.get_device_name(0))
PY
    echo "Container ready. Run: python -m msr.cli make-moises-splits ..."
    exec bash
  '
