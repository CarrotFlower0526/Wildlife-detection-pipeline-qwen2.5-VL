from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

from .data import CocoWildlifeDataset, default_annotation_path, default_image_dir
from .metrics import evaluate_coco
from .parsing import parse_detections


def _batches(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL on wildlife object detection")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--split", default="cis_val", choices=["train", "cis_val", "cis_test", "trans_val", "trans_test"]
    )
    parser.add_argument("--annotations", type=Path, help="override the split annotation JSON")
    parser.add_argument("--image-dir", type=Path, help="override the image directory")
    parser.add_argument(
        "--model-path", type=Path, default=Path("checkpoints/Qwen2.5-VL-3B-Instruct-AWQ")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cis_val"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int)
    parser.add_argument("--max-pixels", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate data without loading the model")
    parser.add_argument(
        "--evaluate-only", type=Path, metavar="PREDICTIONS_JSONL", help="only score an existing JSONL"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    annotation_file = args.annotations or default_annotation_path(args.dataset_root, args.split)
    image_dir = args.image_dir or default_image_dir(args.dataset_root)
    dataset = CocoWildlifeDataset(annotation_file, image_dir)
    samples = list(dataset.iter_samples(args.max_samples))
    summary = {
        "annotation_file": str(annotation_file),
        "image_dir": str(image_dir),
        "split": args.split,
        "samples": len(samples),
        "categories": dataset.category_names,
    }
    if args.dry_run:
        summary.update(dataset.validate(args.max_samples, check_images=True))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if summary["missing_images"] or summary["invalid_sizes"]:
            raise SystemExit(1)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.evaluate_only or (args.output_dir / "predictions.jsonl")
    existing = _read_jsonl(jsonl_path) if (args.evaluate_only or not args.no_resume) else []
    completed = {str(record["image_id"]) for record in existing}

    if not args.evaluate_only:
        from tqdm import tqdm

        from .model import QwenDetector

        detector = QwenDetector(
            args.model_path, args.max_new_tokens, args.min_pixels, args.max_pixels
        )
        pending = [sample for sample in samples if str(sample.image_id) not in completed]
        mode = "a" if existing and not args.no_resume else "w"
        with jsonl_path.open(mode, encoding="utf-8") as handle:
            progress = tqdm(total=len(pending), desc="detect", unit="image")
            for batch in _batches(pending, args.batch_size):
                started = time.perf_counter()
                try:
                    outputs = detector.predict(batch, dataset.category_names)
                    errors = [None] * len(batch)
                except Exception as exc:  # Persist batch failures so long jobs remain inspectable.
                    outputs = [""] * len(batch)
                    errors = [f"{type(exc).__name__}: {exc}"] * len(batch)
                elapsed = (time.perf_counter() - started) / len(batch)
                for sample, raw_output, error in zip(batch, outputs, errors):
                    warnings: list[str] = []
                    detections = []
                    if error is None:
                        try:
                            detections, warnings = parse_detections(
                                raw_output, dataset.category_name_to_id, sample.width, sample.height
                            )
                        except ValueError as exc:
                            error = str(exc)
                    record = {
                        "image_id": sample.image_id,
                        "file_name": sample.file_name,
                        "width": sample.width,
                        "height": sample.height,
                        "detections": [
                            {
                                "label": d.label,
                                "category_id": d.category_id,
                                "bbox_xyxy": list(d.bbox_xyxy),
                                "bbox": d.bbox_xywh,
                                "score": d.score,
                            }
                            for d in detections
                        ],
                        "raw_output": raw_output,
                        "warnings": warnings,
                        "error": error,
                        "seconds": elapsed,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    progress.update(1)
            progress.close()
        existing = _read_jsonl(jsonl_path)

    selected_ids = {str(sample.image_id) for sample in samples}
    records = [record for record in existing if str(record["image_id"]) in selected_ids]
    coco_predictions = [
        {
            "image_id": record["image_id"],
            "category_id": detection["category_id"],
            "bbox": detection["bbox"],
            "score": detection["score"],
        }
        for record in records
        for detection in record.get("detections", [])
    ]
    image_ids = [record["image_id"] for record in records]
    metrics, report = evaluate_coco(dataset.coco, coco_predictions, image_ids)
    run_summary = {
        **summary,
        "evaluated_images": len(image_ids),
        "detections": len(coco_predictions),
        "failed_images": sum(bool(record.get("error")) for record in records),
        "metrics": metrics,
    }
    (args.output_dir / "coco_predictions.json").write_text(
        json.dumps(coco_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report)
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
