# Wildlife detection training and evaluation results

Completed on October 19, 2026. Both Qwen2.5-VL models completed three epochs of
LoRA fine-tuning, followed by full validation. A supervised YOLO11-L detector
achieved the highest validation mAP among the six declared candidates. The
selected configuration, **YOLO11-L at 1280 pixels with test-time augmentation
(TTA)**, achieved **35.09% mAP and 52.03% AP50 on the independent cross-camera
test split**.

Selection used full `trans_val` mAP, with `cis_val` mAP as the tie-breaker. The
selected configuration was saved before test inference. Test results were not
used to choose models, tune settings, or change thresholds.

## Downloads

- [Best model and inference code](artifacts/best-model.zip?raw=true) — 47.4 MB;
  includes `best.pt`, the inference recipe, category mapping, pinned core
  dependencies, and instructions for Linux with an NVIDIA GPU.
- [Complete test predictions](artifacts/test-predictions.zip?raw=true) — 16.7 MB;
  includes both test JSONL files and the image coverage report.
- [Full metrics and configuration](report.json),
  [validation comparison CSV](validation-comparison.csv), and
  [test per-class AP CSV](test-per-class.csv).
- [Image coverage](coverage.json),
  [detector training history](detector-training.csv), and
  [file checksums](SHA256SUMS.txt).

`report.json` contains the full-precision experimental results. Tables below and
the comparison CSVs are summaries of those results. The model package is the
same verified package delivered after the run.

## Full validation comparison

All metrics below are percentages. mAP denotes COCO AP averaged over IoU
thresholds 0.50–0.95. `cis` refers to the same-camera split and `trans` to the
cross-camera split. Empty-image false-positive rate (FPR) is the fraction of
genuine empty images with at least one detection above the stated score.

| Candidate | trans mAP | trans AP50 | cis mAP | trans empty FPR at 0.5 | cis empty FPR at 0.5 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL 3B frozen | 0.33 | 0.53 | 0.33 | 0.00 | 0.00 |
| Qwen2.5-VL 3B LoRA | 18.64 | 28.49 | 42.05 | 19.67 | 74.47 |
| Qwen2.5-VL 7B frozen | 12.43 | 20.69 | 14.31 | 1.64 | 4.82 |
| Qwen2.5-VL 7B LoRA | 19.05 | 30.66 | 42.82 | 73.77 | 89.75 |
| YOLO11-L 1024 | 38.63 | 52.99 | 56.71 | 0.00 | 3.12 |
| **YOLO11-L 1280 + TTA** | **39.29** | **54.12** | **59.29** | **1.64** | **3.92** |

Fine-tuning improved Qwen detection mAP, but also increased empty-image false
positives. The selected detector improved cross-camera validation mAP by
**26.85 percentage points over frozen 7B** and **20.24 points over 7B LoRA**.
YOLO11-L 1024 had slightly lower mAP and fewer false positives; the declared
selection criterion was mAP, which selected 1280 + TTA.

Qwen scores are an uncalibrated geometric mean of response-token probabilities,
shared by all boxes in an image. YOLO retains its native detection scores.
Identical numerical thresholds therefore do not establish equivalent operating
points across methods. The cross-camera validation split contains only 61
genuine empty images, so its FPR estimates have a small denominator.

## Independent test results for the selected model

| Split | Images with predictions | Images in primary evaluation | mAP | AP50 | AR100 | Empty FPR at 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Same-camera `cis_test` | 15,827 | 13,876 | 46.51 | 68.97 | 56.66 | 4.32 (51/1,180) |
| Cross-camera `trans_test` | 23,275 | 19,811 | 35.09 | 52.03 | 48.46 | 4.22 (75/1,778) |

AR100 is COCO average recall across IoU thresholds with at most 100 detections per
image. It is not recall at a single confidence threshold. The predefined FPR
thresholds are reported without selecting a new threshold using test results:

| Split | Empty FPR at 0.05 | At 0.25 | At 0.5 | At 0.75 |
|---|---:|---:|---:|---:|
| `cis_test` | 35.85 | 12.63 | 4.32 | 0.59 |
| `trans_test` | 56.30 | 16.70 | 4.22 | 0.45 |

The prediction files cover all **39,102 test images**. Their IDs were checked
against the actual dataset records, with no duplicates, missing or unexpected
images. Box coordinates, category IDs, and score ranges passed structural checks.
Only the selected model was evaluated on the held-out tests; there is no
before-and-after test comparison for the other candidates.

Per-class AP is included in the [CSV](test-per-class.csv) and
[full report](report.json). Cross-camera AP remains low for fox (0.00%), rodent
(3.54%), badger (4.85%), and bird (9.40%). A missing class metric means there is
no evaluable ground truth for that class in that split, not zero AP. The official
15-category scheme includes `car`; the reported aggregate is not an
animals-only metric.

## Data and evaluation protocol

The experiment uses `cct20-extended-negatives-v1`. The original training data
contains 12,099 boxed images and 1,454 label-only positives. It adds 3,000
explicitly empty images from CCT cameras disjoint from all five benchmark
splits, with one frame per sequence. This is an extended-data result, not a
strict original-training-data-only benchmark.

YOLO trains on 15,099 boxed/empty images. Qwen additionally uses the 1,454
label-only positives for an auxiliary image-level classification instruction;
they are not trained as empty detection targets. Annotation boxes are converted
to decoded-image pixels once.

Primary box metrics exclude known label-only positives while retaining genuine
empty images. This excludes 1,951 images from `cis_test` and 3,464 from
`trans_test`, although inference covers every image. The JSON report also
contains secondary `legacy_all_images` scores. Comparisons must use the same
evaluation protocol; the older AWQ pipeline outputs elsewhere in this repository
are not the frozen BF16 baselines in this campaign.

## Training and inference configuration

| Model | Training settings | Completed training |
|---|---|---|
| Qwen2.5-VL 3B | LoRA rank 16, learning rate 1e-4 | 3 epochs; first epoch selected by validation screening |
| Qwen2.5-VL 7B | LoRA rank 32, learning rate 5e-5 | 3 epochs; final epoch selected by validation screening |
| YOLO11-L | 1024 pixels, AdamW, initial learning rate 0.001, patience 15 | 42 epochs with early stopping; best native validation epoch 27 |

Qwen uses frozen vision weights, assistant-only language loss, BF16, batch size
one, and gradient accumulation of 16. Epoch selection uses reproducible
512-image screening subsets per validation split, followed by full validation
of the selected adapter. YOLO starts from the pinned COCO-pretrained checkpoint
and uses batch size eight. Its native training validation selects the checkpoint;
the six final candidates are compared using the shared COCO evaluator.

The selected inference configuration uses `imgsz=1280`, `augment=True`,
`conf=0.001`, NMS `iou=0.6`, and `max_det=100`. The low inference threshold retains
predictions for AP evaluation; FPR at 0.5 is computed from these saved predictions.
The model package requires no Qwen base model. Extract it beneath your home
directory and follow its `README.txt`, or run the existing repository entry point
after installing the locked environment:

```bash
mkdir -p "$HOME/.cache/wildlife-inference/tmp"
export TMPDIR="$HOME/.cache/wildlife-inference/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
python -m scripts.campaign.predict \
  --bundle /path/under/home/to/best-model/selected_model \
  --images /path/under/home/to/images \
  --output "$HOME/wildlife-predictions.jsonl"
```

Use a new output filename. The entry point uses CUDA device 0. The package pins
core inference dependencies; the source commit's `uv.lock` records the full
training environment. See the [campaign protocol](../../scripts/campaign/README.md)
for additional configuration details.

## Reproducibility and limitations

- Scientific source commit:
  [`c5e74a5a3b57b90cf7df922d222c96fa3fca9274`](https://github.com/CarrotFlower0526/Wildlife-detection-pipeline-qwen2.5-VL/commit/c5e74a5a3b57b90cf7df922d222c96fa3fca9274).
- Workflow: `workflow-20261017T230843Z-5299c860`. Preparation and the GPU campaign
  each completed in their original attempt. Publishing this summary does not
  rerun training or evaluation.
- Hardware: one NVIDIA H200 NVL. Python 3.12.11, PyTorch 2.6.0+cu124,
  Ultralytics 8.3.203, Transformers 4.51.3, and PEFT 0.15.2.
- Best weight SHA256:
  `0f8323d337216a121d7a49fc05f8fb1562dbeb804aa7591bf01f77fccb563526`.
  Downloaded result files matched the published SHA256 values. Both archives
  passed CRC checks, and the packaged command's `--help` entry point ran.
- This is the best of the six declared candidates using one seed (20261017).
  Repeat-run variance and performance outside CCT20 have not been measured.
  Annotation completeness within the remaining boxed images is an evaluation
  assumption. Rare-class weaknesses and the expanded negative training data
  remain relevant when interpreting the results.
