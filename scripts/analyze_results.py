import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from wildlife_detection.cli import _coco_predictions, _read_jsonl
from wildlife_detection.data import CocoWildlifeDataset, default_annotation_path, default_image_dir
from wildlife_detection.metrics import evaluate_coco


RUNS = (("cis_test", "cis_test", None, "image"),
        ("trans_test", "trans_test", None, "image"),
        ("train_5000", "train", 5000, None))


def analyze_run(repo_root, output_root, name, split, limit, legacy_space):
    source = repo_root / "outputs" / name / "predictions.jsonl"
    destination = output_root / name
    if destination.resolve() == source.parent.resolve():
        raise ValueError("Analysis outputs must not overwrite the original run directory")
    dataset = CocoWildlifeDataset(
        default_annotation_path(repo_root / "datasets", split),
        default_image_dir(repo_root / "datasets"),
    )
    samples = list(dataset.iter_samples(limit))
    expected = {str(sample.image_id) for sample in samples}
    records = _read_jsonl(source)
    identifiers = [str(record["image_id"]) for record in records]
    observed = set(identifiers)
    result = {
        "name": name, "split": split, "expected_images": len(samples),
        "records": len(records), "unique_images": len(observed),
        "missing_images": len(expected - observed), "extra_images": len(observed - expected),
        "duplicate_records": len(records) - len(observed),
        "predictions_file": str(source), "status": "incomplete", "fixes": [],
    }
    if result["missing_images"] or result["extra_images"] or result["duplicate_records"]:
        result["fixes"].append(
            "Do not interpret partial scores as full results. Inspect Slurm logs and restore "
            "exactly one record per selected image; resume missing images without overwriting saved work."
        )
        return result
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    errors = Counter(record["error"] for record in records if record.get("error"))
    warnings = Counter(warning for record in records for warning in record.get("warnings", []))
    predictions = _coco_predictions(records, samples, legacy_space)
    metrics, _ = evaluate_coco(dataset.coco, predictions, [sample.image_id for sample in samples])
    category_names = {category["id"]: category["name"] for category in dataset.detection_categories}
    ground_truth = Counter()
    boxed_images = set()
    positive_images = set()
    for annotation in dataset.coco["annotations"]:
        image_id = str(annotation["image_id"])
        if image_id not in expected or annotation["category_id"] not in category_names:
            continue
        positive_images.add(image_id)
        if annotation.get("bbox") and annotation["bbox"][2] > 0 and annotation["bbox"][3] > 0:
            boxed_images.add(image_id)
            ground_truth[annotation["category_id"]] += 1
    predicted_classes = Counter(prediction["category_id"] for prediction in predictions)
    missed_images = sum(str(record["image_id"]) in boxed_images and not record.get("detections")
                        for record in records)
    legacy_records = sum("coordinate_space" not in record for record in records)
    result.update({
        "status": "analyzed", "predictions_sha256": digest,
        "detections": len(predictions), "failed_images": sum(errors.values()),
        "error_counts": dict(errors), "warning_counts": dict(warnings),
        "legacy_records": legacy_records, "legacy_coordinate_space": legacy_space,
        "coco_coordinate_space": "annotation", "metrics": metrics,
        "boxed_positive_images": len(boxed_images), "missed_boxed_positive_images": missed_images,
        "label_only_positive_images": len(positive_images - boxed_images),
        "class_counts": {label: {"ground_truth_boxes": ground_truth[category_id],
                                  "predicted_boxes": predicted_classes[category_id]}
                         for category_id, label in category_names.items()},
        "generation_seconds": sum(record.get("seconds", 0) for record in records),
    })
    if legacy_records:
        result["fixes"].append(
            "Legacy boxes were explicitly interpreted as supplied-image pixels and scaled to "
            "annotation dimensions; annotation IDs were normalized to integers. Keep the original "
            "outputs and verify overlays before citing these assumption-dependent scores."
        )
    if errors:
        result["fixes"].append(
            "Inspect the listed response errors. Retry only failed images in a separate run using "
            "the corrected prompt, then merge by image ID with coordinate metadata and rescore. "
            "Ordinary resume skips failures. For malformed JSON, test schema-constrained generation "
            "on validation data; increase the token limit only for genuinely truncated responses."
        )
    if warnings:
        result["fixes"].append(
            "Inspect skipped-box warnings and raw responses. Enforce a four-number xyxy schema "
            "and valid category names; add regression cases for observed formats rather than "
            "silently fabricating missing coordinates."
        )
    if positive_images - boxed_images:
        result["fixes"].append(
            "Positive images without boxes are retained for compatibility with existing scoring. "
            "Confirm the dataset's official box-evaluation protocol: do not assume such images "
            "are negatives. Keep genuine empty images when reporting a box-annotated subset."
        )
    if missed_images:
        result["fixes"].append(
            "Review missed positive images across lighting, size, and occlusion. Benchmark a "
            "dedicated detector for candidate boxes, optionally using the VLM to classify crops. "
            "Tune prompts and thresholds on validation splits, not these test results."
        )
    overpredicted = [label for label, counts in result["class_counts"].items()
                    if counts["predicted_boxes"] >= 20
                    and counts["predicted_boxes"] > 3 * max(counts["ground_truth_boxes"], 1)]
    if overpredicted:
        result["fixes"].append(
            "Audit possible class confusion/overprediction for " + ", ".join(overpredicted) +
            ". Box-count ratios alone do not prove hallucination. Validate crop classifications "
            "and consider training a classifier on the training labels."
        )
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Predictions changed during analysis: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "coco_predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
    return result


def percentage(value):
    return f"{100 * value:.2f}%" if value >= 0 else "N/A"


def render_report(results):
    lines = ["# Wildlife Evaluation Analysis", "",
             f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
             "Original predictions are preserved. Scores retain failed images as empty detections.", "",
             "| Run | Coverage | Failed Responses | mAP 50:95 | AP50 | AP75 | AR100 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        name = result["name"]
        if result["status"] != "analyzed":
            lines.append(f"| {name} | {result['status']} | - | - | - | - | - |")
            continue
        metrics = result["metrics"]
        scores = " | ".join(percentage(metrics[key]) for key in
                            ("AP@[.50:.95]", "AP@.50", "AP@.75", "AR_100"))
        lines.append(f"| {name} | {result['unique_images']}/{result['expected_images']} | "
                     f"{result['failed_images']} | {scores} |")
    lines += ["", "## Interpretation", "",
              "The two test runs used the legacy prompt and assumed coordinate conversion. "
              "The first 5,000 training images used the corrected prompt and explicit coordinates. "
              "Differences therefore combine split composition and inference-protocol changes; "
              "do not attribute them solely to generalization. No model training was performed.", "",
              "The training subset is the first 5,000 images in annotation-file order, not a "
              "random or stratified sample. Do not treat it as a representative full-training score.", "",
              "Text-generated confidence is not calibrated. Review precision/recall and failure "
              "examples before choosing a threshold."]
    for result in results:
        lines += ["", f"## {result['name']}", ""]
        if result.get("analysis_error"):
            lines.append(f"Analysis error: {result['analysis_error']}")
        if result["status"] == "analyzed":
            missed = result["missed_boxed_positive_images"]
            positives = result["boxed_positive_images"]
            lines += [f"- Predicted boxes: {result['detections']}",
                      f"- Box-annotated positive images with no predicted box: {missed}/{positives}",
                      f"- Positive images without boxes: {result['label_only_positive_images']}",
                      f"- Legacy coordinate interpretation applied to {result['legacy_records']} records.",
                      f"- Error types: {json.dumps(result['error_counts'])}", "",
                      "| Class | GT Boxes | Predicted Boxes |", "|---|---:|---:|"]
            for label, counts in result["class_counts"].items():
                lines.append(f"| {label} | {counts['ground_truth_boxes']} | {counts['predicted_boxes']} |")
        lines += ["", "### Corrective Actions", ""]
        lines += [f"- {fix}" for fix in result["fixes"]] or ["- No automated integrity issues detected."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Analyze completed wildlife runs without rerunning inference")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output = args.output_root or args.repo_root / "outputs" / "final_analysis"
    results = []
    for name, split, limit, legacy_space in RUNS:
        try:
            result = analyze_run(args.repo_root, output, name, split, limit, legacy_space)
        except Exception as error:
            result = {"name": name, "status": "analysis_failed", "analysis_error": str(error),
                      "fixes": ["Resolve the reported input/analysis error and rerun this analysis; original outputs are untouched."]}
        results.append(result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    report = render_report(results)
    (output / "report.md").write_text(report)
    print(report)
    if any(result["status"] != "analyzed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()