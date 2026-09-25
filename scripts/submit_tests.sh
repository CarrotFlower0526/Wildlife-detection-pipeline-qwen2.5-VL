#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export WILDLIFE_REPO_ROOT="$repo_root"
export PYTHON_BIN="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
export MODEL_PATH="${MODEL_PATH:-$repo_root/checkpoints/Qwen2.5-VL-3B-Instruct-AWQ}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python not found: %s. Set PYTHON_BIN to the project environment interpreter.\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    printf 'Model not found: %s. Install the checkpoint or set MODEL_PATH.\n' "$MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$repo_root/logs"
sbatch --chdir="$repo_root" \
    --output="$repo_root/logs/wildlife-%A_%a.out" \
    --error="$repo_root/logs/wildlife-%A_%a.err" \
    "$@" "$repo_root/scripts/evaluate_tests.sbatch"