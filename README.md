# Qwen2.5-VL Wildlife Object Detection Evaluation

This repository evaluates `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` as a prompted object detector on the existing Caltech Camera Traps COCO-style splits. It preserves raw model responses, converts valid boxes to COCO detections, and reports standard COCO AP/AR metrics.

## Setup

Use a CUDA-compatible environment and install the matching PyTorch build first, then:

```bash
pip install -e .
```

Place the downloaded model at:

```text
checkpoints/Qwen2.5-VL-3B-Instruct-AWQ/
```

The model is loaded with `local_files_only=True`; the pipeline never downloads it implicitly.

## Validate the dataset

This works before the model has been downloaded:

```bash
wildlife-detect --split cis_val --max-samples 20 --dry-run
```

## Run inference and evaluation

Start with a small smoke test:

```bash
wildlife-detect \
  --split cis_val \
  --max-samples 20 \
  --output-dir outputs/cis_val_smoke
```

Run the complete validation split:

```bash
wildlife-detect --split cis_val --output-dir outputs/cis_val
```

Available splits are `train`, `cis_val`, `cis_test`, `trans_val`, and `trans_test`. Use `--batch-size N` if GPU memory permits. `--max-pixels` can reduce vision-token memory (for example `1003520`, corresponding to `1280 * 28 * 28`).

Inference appends one record per image to `predictions.jsonl` and resumes it by default. A record contains parsed detections, the raw response, parse warnings, exceptions, and timing. The final files are:

- `predictions.jsonl`: resumable, auditable per-image output
- `coco_predictions.json`: standard COCO result format
- `metrics.json`: run summary and COCO AP/AR values

To regenerate metrics without loading the model:

```bash
wildlife-detect \
  --split cis_val \
  --output-dir outputs/cis_val \
  --evaluate-only outputs/cis_val/predictions.jsonl
```

Custom COCO paths can be passed with `--annotations` and `--image-dir`.

## Structured JSON generation

New inference uses LM Format Enforcer to filter allowed tokens during decoding,
not just instructions in the prompt. The JSON schema requires a top-level array
of objects with exactly these fields:

```json
[{"label":"bobcat","bbox":[10,20,100,200],"score":0.9}]
```

`label` must be one of the dataset's detection categories, `bbox` must contain
exactly four numbers, and `score` must be numeric. `[]` is valid for an empty
image. Markdown fences, extra fields, and early EOS before a complete value are
blocked. Tokenizer data is cached; parser state is fresh for each generation call.

Strict JSON/schema validation also runs before parsing every new response.
`--max-new-tokens` is still a hard limit: a response cut off before closing its
JSON is retained in `raw_output` and recorded as a failure, not repaired or
silently accepted. Increase the limit when inspecting a failure confirms truncation.
The schema enforces structure, not detection correctness, box ordering/bounds,
or confidence calibration; existing box/score postprocessing still applies.

New records are tagged `output_format: "json_schema_v1"`. Use a fresh output
directory for a constrained run; resuming older unconstrained records is rejected
to avoid silently mixing protocols. Existing results can still be evaluated
without changing them. Example:

```bash
wildlife-detect --split cis_val --max-samples 20 --output-dir outputs/cis_val_structured
```

## Coordinates and legacy results

New inference prompts and parses boxes in actual supplied-image pixels. JSONL
records declare `coordinate_space: "image"` and store `image_width` and
`image_height`; `width` and `height` remain the annotation dimensions. COCO
export scales each box to annotation pixels exactly once. Ground-truth
annotation IDs are mapped to positive integers for COCO evaluation without
modifying the dataset files.

Older JSONL files lack coordinate metadata and require an explicit interpretation:

```bash
wildlife-detect --split cis_test \
  --evaluate-only outputs/cis_test/predictions.jsonl \
  --legacy-coordinate-space image \
  --output-dir outputs/cis_test_rescored
```

`image` assumes the saved detection boxes use the actual image file dimensions;
`annotation` leaves those boxes unchanged. The choice only affects unmarked
records and is recorded in the metrics. Original JSONL records, raw responses,
and failures are preserved; use a separate output directory to retain old metrics.
Legacy rescoring does not repair malformed responses or rerun inference.
Do not mix old and new inference runs: use a fresh output directory when an old
JSONL lacks coordinate metadata. New marked runs still resume normally.
Images labeled with a category but lacking boxes remain in evaluation, matching
the existing protocol; legacy scores remain conditional on the coordinate assumption.

## Cluster test jobs

On `yf201@login.cs.duke.edu`, check GPU availability using
`~/scripts/check_gpu_usage.sh` and submit both test splits with
`bash scripts/submit_tests.sh` after preparing the cluster checkout:

```bash
PYTHON_BASE=/home/users/yf201/miniconda3/envs/pilot/bin/python bash scripts/setup_cluster.sh
```

Setup installs PyTorch 2.6.0 with CUDA 12.4, matching torchvision and AutoAWQ,
then downloads the checkpoint. Jobs load the checkpoint offline. Each array task
requests one A6000, four CPUs, 32 GB RAM, and four days in `compsci-gpu`.
Each array task validates its dataset, runs inference, and writes COCO metrics to
`outputs/cis_test/` or `outputs/trans_test/`. See `logs/wildlife-*.{out,err}` for
job output. Validation checks file presence, readable image headers, and positive
annotation dimensions, and reports resized images. It does not fully decode images.
Jobs fail if inference has missing or failed records.
The jobs resume predictions on resubmission, including skipping failed records
as described below.

For a short GPU check with separate outputs, run:

```bash
OUTPUT_ROOT="$PWD/outputs/smoke" MAX_SAMPLES=3 bash scripts/submit_tests.sh --time=00:30:00 --wait
```

Monitor with `squeue -u "$USER"`. Array task `0` is `cis_test` and task `1` is
`trans_test`. To resume only one split, pass `--array=0` or `--array=1` to
`scripts/submit_tests.sh`. `PYTHON_BIN`, `MODEL_PATH`, `OUTPUT_ROOT`, `BATCH_SIZE`,
and `MAX_PIXELS` can be overridden through environment variables.

## Final result analysis

After `cis_test`, `trans_test`, and `train_5000` have stopped, run:

```bash
.venv/bin/python scripts/analyze_results.py
```

This CPU-only report checks exact image coverage and duplicates, rescales unmarked
test predictions under the explicit supplied-image-coordinate assumption, scores
the marked training predictions, and summarizes response failures, missed objects,
class counts, and corrective actions. It preserves the original results and writes
`outputs/final_analysis/report.md`, `summary.json`, and per-run corrected COCO
predictions and metrics. Incomplete runs are flagged instead of scored as complete.
The first-5,000 training subset and the changed prompt are noted as comparison limits.

For Slurm automation, use an `afterany` dependency on the inference jobs, not
`afterok`: the inference scripts exit nonzero when individual responses failed,
even when all images have been processed and metrics exist. Analysis reports
those failures rather than silently retrying or removing them.

## Notes

- The dataset category `empty` is treated as a negative image, not an object class.
- The prompt requires supplied-image pixel `xyxy` boxes. The parser also accepts `bbox_2d`, fenced JSON, and normalized `[0, 1]` coordinates; COCO export converts to annotation pixels.
- Qwen's text confidence is not calibrated. It is retained for COCO ranking, with a fallback score of `1.0` when absent.
- Failed generations are recorded rather than silently discarded. Resume skips every existing record, including failures; remove a failed JSONL line or use a fresh output directory to retry it.
