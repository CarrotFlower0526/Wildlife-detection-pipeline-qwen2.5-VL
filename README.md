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

## Notes

- The dataset category `empty` is treated as a negative image, not an object class.
- The prompt requires original-image pixel `xyxy` boxes. The parser also accepts `bbox_2d`, fenced JSON, and normalized `[0, 1]` coordinates.
- Qwen's text confidence is not calibrated. It is retained for COCO ranking, with a fallback score of `1.0` when absent.
- Failed generations are recorded rather than silently discarded. Resume skips every existing record, including failures; remove a failed JSONL line or use a fresh output directory to retry it.
