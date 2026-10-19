# Wildlife improvement campaign

This experiment compares frozen and LoRA-tuned Qwen2.5-VL 3B/7B models with a
supervised YOLO11-L detector. It selects on validation, then evaluates one selected
configuration on the two held-out test splits. It does not promise a globally
optimal model or a predetermined accuracy improvement.

The run is complete. See the [final results and model downloads](../../reports/wildlife-improvement-2026-10-19/README.md)
for validation comparisons, the selected model's independent test performance,
and artifact checksums.

## Data and evaluation

- Original upstream: `79f7e06508cd234cf7b05933f3e3de438849a694`.
- Official CCT20 splits are preserved. Training has 12,099 boxed images and 1,454
  label-only positives, and no empty images. Label-only positives are never given
  an empty detection target. Qwen uses those images for an auxiliary image-level
  classification instruction; the detector trains only on boxed/empty images.
- The extended-data protocol adds 3,000 explicitly empty images, one per sequence,
  from CCT cameras absent from **all five** benchmark splits. This must not be
  represented as a strict original-training-data-only benchmark.
- Boxes are converted once from annotation coordinates into decoded-image pixels.
- Primary detection metrics exclude known label-only positives and retain genuine
  empty images. Secondary test metrics retain all images for comparison with the
  legacy protocol. Completeness of annotations within boxed images is assumed.
- Per-class AP, average recall, missed positive images, empty-image false-positive
  rates and malformed responses accompany aggregate AP.
- Qwen generates label/box JSON without invented confidence supervision. Its
  response-token likelihood is an explicitly uncalibrated ranking proxy shared
  across boxes within an image. Detector scores retain native model semantics;
  identical numerical thresholds are not assumed equivalent between methods.

## Candidate selection

3B uses rank 16 LoRA with learning rate 1e-4; 7B uses rank 32 with learning rate
5e-5. Both train three epochs at up to 1,003,520 image pixels, batch size one,
gradient accumulation 16, BF16 and gradient checkpointing. Vision weights remain
frozen. Assistant responses alone contribute to language loss. Each epoch is
screened on reproducible class/location-stratified validation subsets; the best
screened checkpoint is evaluated on the full validation splits.

YOLO11-L starts from the official COCO-pretrained checkpoint and trains at 1024
pixels for up to 80 epochs with patience 15 on trans_val. Its 1024-pixel inference
and 1280-pixel test-time augmentation are separate validation candidates.

The highest full trans_val mAP50:95 wins, with full cis_val mAP as tie-breaker.
Frozen Qwen baselines remain eligible, so a failed fine-tuning experiment cannot
be presented as an improvement. Selection is saved before test inference. Test
results must not be used to tune subsequent candidates and still called held-out
confirmation; a changed experiment would need a new independent confirmation set.

These are proposed initial configurations, not scientifically mandated limits.
Additional candidate search requires evidence from the validation comparisons.
The current campaign uses one seed and reports that limitation.

## Fleet execution

The project is `wildlife-detection`; the experiment is `wildlife-improvement`.
The immutable model revisions are:

- `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3`
- `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5`

Generate the lock with uv 0.11.21, commit all sources, then submit from the clean
root checkout with `codex-fleet submit wildlife-detection wildlife-improvement`.
The environment targets Python 3.12.11 and Linux CUDA 12.4. Legacy AWQ support is
an optional dependency; this campaign uses original BF16 model weights.

Preparation is a CPU stage; the continuous comparison/training stage uses one
exclusive GPU because the initial activation-memory requirement is not measured.
The GPU stage preserves completed candidate results, inference records, optimizer
checkpoints and selected adapters. Input identities use Fleet's role identities,
not machine-specific paths. A code defect requires a corrected source revision;
reuse only stages whose inputs and scientific meaning remain unchanged.

The result includes `report.json`, `report.md`, both test prediction logs and
`selected_model/`. A detector bundle contains its weights. A Qwen bundle contains
the adapter when applicable and pins the external base revision in `recipe.json`.
Run a delivered bundle with:

```bash
python -m scripts.campaign.predict --bundle SELECTED_MODEL --images IMAGE_DIRECTORY --output predictions.jsonl
```

For a Qwen winner, also pass `--base-model` pointing to the pinned local base.
Generated files and all ad-hoc caches stay under Fleet's home-backed temporary
directory or the declared outputs.
