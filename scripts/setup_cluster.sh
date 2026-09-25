#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_base="${PYTHON_BASE:-python3.12}"
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    "$python_base" -m venv "$repo_root/.venv"
fi
python_bin="$repo_root/.venv/bin/python"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12 for this CUDA/AWQ environment"'
"$python_bin" -m pip install --upgrade pip
"$python_bin" -m pip install 'torch==2.6.0' 'torchvision==0.21.0' \
    --index-url https://download.pytorch.org/whl/cu124
"$python_bin" -m pip install -e . 'torch==2.6.0' 'torchvision==0.21.0' 'autoawq==0.2.9'
"$python_bin" -m pip check
"$python_bin" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", local_dir="checkpoints/Qwen2.5-VL-3B-Instruct-AWQ")'