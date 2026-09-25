# Wildlife Evaluation Analysis

Generated: 2026-09-25T06:32:21.754924+00:00

Original predictions are preserved. Scores retain failed images as empty detections.

| Run | Coverage | Failed Responses | mAP 50:95 | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|---:|
| cis_test | 15827/15827 | 132 | 9.95% | 19.22% | 9.45% | 18.85% |
| trans_test | 23275/23275 | 219 | 6.32% | 13.26% | 5.37% | 12.30% |
| train_5000 | 5000/5000 | 36 | 11.80% | 23.91% | 9.79% | 19.23% |

## Interpretation

The two test runs used the legacy prompt and assumed coordinate conversion. The first 5,000 training images used the corrected prompt and explicit coordinates. Differences therefore combine split composition and inference-protocol changes; do not attribute them solely to generalization. No model training was performed.

The training subset is the first 5,000 images in annotation-file order, not a random or stratified sample. Do not treat it as a representative full-training score.

Text-generated confidence is not calibrated. Review precision/recall and failure examples before choosing a threshold.

## cis_test

- Predicted boxes: 9426
- Box-annotated positive images with no predicted box: 5619/12696
- Positive images without boxes: 1951
- Legacy coordinate interpretation applied to 15827 records.
- Error types: {"model response does not contain a JSON array or object": 132}

| Class | GT Boxes | Predicted Boxes |
|---|---:|---:|
| bobcat | 772 | 1147 |
| opossum | 4010 | 2059 |
| coyote | 1182 | 684 |
| raccoon | 981 | 459 |
| bird | 752 | 397 |
| dog | 725 | 379 |
| cat | 1528 | 360 |
| squirrel | 510 | 1244 |
| rabbit | 1559 | 521 |
| skunk | 162 | 121 |
| rodent | 146 | 843 |
| badger | 4 | 816 |
| deer | 140 | 68 |
| car | 791 | 114 |
| fox | 2 | 214 |

### Corrective Actions

- Legacy boxes were explicitly interpreted as supplied-image pixels and scaled to annotation dimensions; annotation IDs were normalized to integers. Keep the original outputs and verify overlays before citing these assumption-dependent scores.
- Inspect the listed response errors. Retry only failed images in a separate run using the corrected prompt, then merge by image ID with coordinate metadata and rescore. Ordinary resume skips failures. For malformed JSON, test schema-constrained generation on validation data; increase the token limit only for genuinely truncated responses.
- Inspect skipped-box warnings and raw responses. Enforce a four-number xyxy schema and valid category names; add regression cases for observed formats rather than silently fabricating missing coordinates.
- Positive images without boxes are retained for compatibility with existing scoring. Confirm the dataset's official box-evaluation protocol: do not assume such images are negatives. Keep genuine empty images when reporting a box-annotated subset.
- Review missed positive images across lighting, size, and occlusion. Benchmark a dedicated detector for candidate boxes, optionally using the VLM to classify crops. Tune prompts and thresholds on validation splits, not these test results.
- Audit possible class confusion/overprediction for rodent, badger, fox. Box-count ratios alone do not prove hallucination. Validate crop classifications and consider training a classifier on the training labels.

## trans_test

- Predicted boxes: 11626
- Box-annotated positive images with no predicted box: 9806/18033
- Positive images without boxes: 3464
- Legacy coordinate interpretation applied to 23275 records.
- Error types: {"model response does not contain a JSON array or object": 219}

| Class | GT Boxes | Predicted Boxes |
|---|---:|---:|
| bobcat | 1979 | 1430 |
| opossum | 4663 | 2251 |
| coyote | 1902 | 715 |
| raccoon | 4803 | 1386 |
| bird | 180 | 268 |
| dog | 961 | 524 |
| cat | 1366 | 406 |
| squirrel | 782 | 1212 |
| rabbit | 722 | 252 |
| skunk | 273 | 151 |
| rodent | 26 | 884 |
| badger | 15 | 1733 |
| deer | 0 | 8 |
| car | 1113 | 85 |
| fox | 1 | 321 |

### Corrective Actions

- Legacy boxes were explicitly interpreted as supplied-image pixels and scaled to annotation dimensions; annotation IDs were normalized to integers. Keep the original outputs and verify overlays before citing these assumption-dependent scores.
- Inspect the listed response errors. Retry only failed images in a separate run using the corrected prompt, then merge by image ID with coordinate metadata and rescore. Ordinary resume skips failures. For malformed JSON, test schema-constrained generation on validation data; increase the token limit only for genuinely truncated responses.
- Inspect skipped-box warnings and raw responses. Enforce a four-number xyxy schema and valid category names; add regression cases for observed formats rather than silently fabricating missing coordinates.
- Positive images without boxes are retained for compatibility with existing scoring. Confirm the dataset's official box-evaluation protocol: do not assume such images are negatives. Keep genuine empty images when reporting a box-annotated subset.
- Review missed positive images across lighting, size, and occlusion. Benchmark a dedicated detector for candidate boxes, optionally using the VLM to classify crops. Tune prompts and thresholds on validation splits, not these test results.
- Audit possible class confusion/overprediction for rodent, badger, fox. Box-count ratios alone do not prove hallucination. Validate crop classifications and consider training a classifier on the training labels.

## train_5000

- Predicted boxes: 3462
- Box-annotated positive images with no predicted box: 1829/4439
- Positive images without boxes: 561
- Legacy coordinate interpretation applied to 0 records.
- Error types: {"model response does not contain a JSON array or object": 36}

| Class | GT Boxes | Predicted Boxes |
|---|---:|---:|
| bobcat | 322 | 358 |
| opossum | 1259 | 610 |
| coyote | 231 | 112 |
| raccoon | 430 | 225 |
| bird | 160 | 69 |
| dog | 170 | 77 |
| cat | 587 | 176 |
| squirrel | 345 | 623 |
| rabbit | 828 | 276 |
| skunk | 64 | 36 |
| rodent | 151 | 559 |
| badger | 5 | 263 |
| deer | 6 | 5 |
| car | 0 | 34 |
| fox | 0 | 39 |

### Corrective Actions

- Inspect the listed response errors. Retry only failed images in a separate run using the corrected prompt, then merge by image ID with coordinate metadata and rescore. Ordinary resume skips failures. For malformed JSON, test schema-constrained generation on validation data; increase the token limit only for genuinely truncated responses.
- Inspect skipped-box warnings and raw responses. Enforce a four-number xyxy schema and valid category names; add regression cases for observed formats rather than silently fabricating missing coordinates.
- Positive images without boxes are retained for compatibility with existing scoring. Confirm the dataset's official box-evaluation protocol: do not assume such images are negatives. Keep genuine empty images when reporting a box-annotated subset.
- Review missed positive images across lighting, size, and occlusion. Benchmark a dedicated detector for candidate boxes, optionally using the VLM to classify crops. Tune prompts and thresholds on validation splits, not these test results.
- Audit possible class confusion/overprediction for rodent, badger, car, fox. Box-count ratios alone do not prove hallucination. Validate crop classifications and consider training a classifier on the training labels.
