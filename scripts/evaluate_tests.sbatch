#!/usr/bin/env bash
#SBATCH --job-name=wildlife-test
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-1%2

set -euo pipefail

: "${SLURM_JOB_ID:?Submit this job with scripts/submit_tests.sh}"
: "${WILDLIFE_REPO_ROOT:?Submit this job with scripts/submit_tests.sh}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the project environment interpreter}"
: "${MODEL_PATH:?Set MODEL_PATH to the local checkpoint directory}"

splits=(cis_test trans_test)
task_id="${SLURM_ARRAY_TASK_ID:?A Slurm array task ID is required}"
if [[ "$task_id" != 0 && "$task_id" != 1 ]]; then
    printf 'Unsupported array task ID: %s\n' "$task_id" >&2
    exit 1
fi
split="${splits[$task_id]}"
cd "$WILDLIFE_REPO_ROOT"

export PYTHONPATH="$WILDLIFE_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

output_dir="${OUTPUT_ROOT:-$WILDLIFE_REPO_ROOT/outputs}/$split"
mkdir -p "$output_dir"
exec 9>"$output_dir/.run.lock"
if ! flock -n 9; then
    printf 'Another job is writing to %s\n' "$output_dir" >&2
    exit 1
fi
printf 'Job: %s | Split: %s | Node: %s | Started: %s\n' "$SLURM_JOB_ID" "$split" "$(hostname)" "$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this allocation"; print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'

cli=("$PYTHON_BIN" -m wildlife_detection.cli)
"${cli[@]}" --dataset-root "$WILDLIFE_REPO_ROOT/datasets" --split "$split" --dry-run \
    | tee "$output_dir/validation.json"

inference_args=(
    --dataset-root "$WILDLIFE_REPO_ROOT/datasets"
    --split "$split"
    --model-path "$MODEL_PATH"
    --output-dir "$output_dir"
    --batch-size "${BATCH_SIZE:-1}"
    --max-new-tokens "${MAX_NEW_TOKENS:-512}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    inference_args+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "${MAX_PIXELS:-}" ]]; then
    inference_args+=(--max-pixels "$MAX_PIXELS")
fi

"${cli[@]}" "${inference_args[@]}"
"$PYTHON_BIN" - "$output_dir/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
if metrics["evaluated_images"] != metrics["samples"] or metrics["failed_images"]:
    raise SystemExit(
        f'Incomplete inference: {metrics["evaluated_images"]}/{metrics["samples"]} images, '
        f'{metrics["failed_images"]} failures. Inspect predictions.jsonl before retrying.'
    )
PY
printf 'Completed %s inference and evaluation: %s\n' "$split" "$(date -Is)"