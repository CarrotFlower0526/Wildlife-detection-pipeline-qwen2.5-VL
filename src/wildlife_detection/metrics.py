from __future__ import annotations

import contextlib
import io
from typing import Any


METRIC_NAMES = [
    "AP@[.50:.95]", "AP@.50", "AP@.75", "AP_small", "AP_medium", "AP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
]


def evaluate_coco(
    source_coco: dict[str, Any], predictions: list[dict[str, Any]], image_ids: list[str | int]
) -> tuple[dict[str, float], str]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError("pycocotools is required for metrics; run: pip install -e .") from exc

    ids = set(image_ids)
    categories = [c for c in source_coco["categories"] if c["name"].lower() != "empty"]
    category_ids = {c["id"] for c in categories}
    annotations = []
    for annotation in source_coco["annotations"]:
        if annotation["image_id"] not in ids or annotation["category_id"] not in category_ids:
            continue
        bbox = annotation.get("bbox")
        if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        item = dict(annotation)
        item["area"] = float(item.get("area", bbox[2] * bbox[3]))
        item["iscrowd"] = int(item.get("iscrowd", 0))
        annotations.append(item)
    ground_truth = {
        "info": source_coco.get("info", {}),
        "licenses": source_coco.get("licenses", []),
        "images": [im for im in source_coco["images"] if im["id"] in ids],
        "annotations": annotations,
        "categories": categories,
    }
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        gt = COCO()
        gt.dataset = ground_truth
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
            evaluator = COCOeval(gt, dt, "bbox")
            evaluator.params.imgIds = image_ids
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            values = [float(x) for x in evaluator.stats]
        else:
            capture.write("No detections were produced; all metrics are 0.\n")
            values = [0.0] * 12
    return dict(zip(METRIC_NAMES, values)), capture.getvalue()
