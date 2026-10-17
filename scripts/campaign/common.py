from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path


def configure_paths():
    # Fleet's TMPDIR is backed by the login user's home, including in bubblewrap.
    root = Path(os.environ.get("TMPDIR", Path.home() / ".cache/wildlife/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(root)
    for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
        directory = root / name.lower()
        directory.mkdir(exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return root


def read_json(path):
    return json.loads(Path(path).read_text())


def read_prediction_log(path):
    """Recover only an incomplete final write; never hide corruption in complete lines."""
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    complete_end = payload.rfind(b"\n") + 1
    records = [json.loads(line) for line in payload[:complete_end].splitlines() if line.strip()]
    if complete_end != len(payload):
        with path.open("r+b") as stream:
            stream.truncate(complete_end)
    return records


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def grouped_annotations(coco):
    grouped = defaultdict(list)
    for annotation in coco["annotations"]:
        grouped[str(annotation["image_id"])].append(annotation)
    return grouped


def clean_sample(image, annotations, categories, actual_size):
    """Distinguish genuine negative, boxed positive, and unsupervised positive."""
    width, height = actual_size
    sx, sy = width / image["width"], height / image["height"]
    boxes, labels = [], set()
    for annotation in annotations:
        label = categories[annotation["category_id"]]
        labels.add(label)
        box = annotation.get("bbox")
        if label == "empty" or not box or len(box) != 4:
            continue
        if not all(math.isfinite(float(v)) for v in box) or box[2] <= 0 or box[3] <= 0:
            continue
        x, y, w, h = box
        x1, y1 = max(0.0, x * sx), max(0.0, y * sy)
        x2, y2 = min(width, (x + w) * sx), min(height, (y + h) * sy)
        if x2 > x1 and y2 > y1:
            boxes.append({"label": label, "category_id": annotation["category_id"],
                          "bbox": [x1, y1, x2 - x1, y2 - y1]})
    if boxes:
        status = "boxed"
    elif labels == {"empty"}:
        status = "empty"
    else:
        status = "label_only"
    return {"id": str(image["id"]), "file_name": f"images/{image['id']}.jpg",
            "width": width, "height": height, "location": str(image.get("location", "")),
            "seq_id": str(image.get("seq_id", image["id"])),
            "status": status, "boxes": boxes, "labels": sorted(labels)}


def screening_sample(rows, limit=512, seed=20261017):
    """Round-robin class/location strata. Full validation remains the final selector."""
    groups = defaultdict(list)
    for row in rows:
        key = (tuple(row["labels"]), row["location"])
        groups[key].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    rng.shuffle(keys)
    result = []
    while len(result) < min(limit, len(rows)):
        for key in keys:
            if groups[key] and len(result) < limit:
                result.append(groups[key].pop())
    return result


def negative_candidates(metadata, benchmark_splits, limit=3000):
    """Additional empty images from cameras entirely outside the CCT20 benchmark."""
    blocked_images = {str(i["id"]) for s in benchmark_splits for i in s["images"]}
    blocked_sequences = {str(i.get("seq_id", i["id"])) for s in benchmark_splits for i in s["images"]}
    blocked_locations = {str(i["location"]) for s in benchmark_splits for i in s["images"]}
    empty_ids = {c["id"] for c in metadata["categories"] if c["name"] == "empty"}
    annotations = grouped_annotations(metadata)
    groups = defaultdict(list)
    for image in metadata["images"]:
        cats = {a["category_id"] for a in annotations[str(image["id"]) ]}
        if (cats and cats <= empty_ids and str(image["id"]) not in blocked_images
                and str(image.get("seq_id", image["id"])) not in blocked_sequences
                and str(image["location"]) not in blocked_locations):
            groups[str(image["location"])].append(image)
    rng = random.Random(20261017)
    for group in groups.values():
        rng.shuffle(group)
    selected, sequences = [], set()
    while any(groups.values()) and len(selected) < limit:
        for location in sorted(groups):
            if not groups[location] or len(selected) >= limit:
                continue
            image = groups[location].pop()
            seq = str(image.get("seq_id", image["id"]))
            if seq not in sequences:
                selected.append(image)
                sequences.add(seq)
    if len(selected) != limit:
        raise ValueError(f"Only {len(selected)} eligible independent negative sequences")
    return selected


def input_identity():
    path = os.environ.get("CODEX_FLEET_VERIFIED_INPUTS_PATH")
    return {r["role"]: r["identity"] for r in read_json(path)["roles"]} if path else {}


def experiment_identity(config):
    return {"inputs": input_identity(), "config": config}


def coco_ground_truth(rows, categories, include_label_only=False):
    selected = [r for r in rows if include_label_only or r["status"] != "label_only"]
    id_map = {r["id"]: index + 1 for index, r in enumerate(selected)}
    annotations = []
    for row in selected:
        for box in row["boxes"]:
            annotations.append({"id": len(annotations) + 1, "image_id": id_map[row["id"]],
                                "category_id": box["category_id"], "bbox": box["bbox"],
                                "area": box["bbox"][2] * box["bbox"][3], "iscrowd": 0})
    return {"info": {}, "images": [{"id": id_map[r["id"]], "width": r["width"],
             "height": r["height"]} for r in selected], "annotations": annotations,
             "categories": categories}, id_map


def evaluate(rows, predictions, categories, include_label_only=False):
    import contextlib
    import io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    expected = {r["id"] for r in rows}
    if set(predictions) != expected:
        raise ValueError("Predictions must have exactly one entry for every selected image")
    ground_truth, id_map = coco_ground_truth(rows, categories, include_label_only)
    flattened = [{**p, "image_id": id_map[image_id]} for image_id, boxes in predictions.items()
                 if image_id in id_map for p in boxes]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = ground_truth; gt.createIndex()
        if flattened:
            dt = gt.loadRes(flattened)
        else:
            dt = COCO(); dt.dataset = {**ground_truth, "annotations": []}; dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        valid = precision[precision >= 0]
        name = next(c["name"] for c in categories if c["id"] == category_id)
        per_class[name] = float(np.mean(valid)) if valid.size else None
    boxed = [r for r in rows if r["status"] == "boxed"]
    empty = [r for r in rows if r["status"] == "empty"]
    return {"map": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1]),
            "ap75": float(evaluator.stats[2]), "ar100": float(evaluator.stats[8]),
            "per_class_ap": per_class, "evaluated_images": len(id_map),
            "label_only_excluded": 0 if include_label_only else sum(r["status"] == "label_only" for r in rows),
            "boxed_images": len(boxed),
            "boxed_images_without_predictions": sum(not predictions[r["id"]] for r in boxed),
            "empty_images": len(empty),
            "empty_fpr_by_threshold": {str(t): (sum(any(p["score"] >= t for p in predictions[r["id"]])
                for r in empty) / len(empty) if empty else None) for t in (0.05, 0.25, 0.5, 0.75)}}


def selection_key(metrics):
    # Primary objective: generalization to unseen camera locations. Tie-break with cis mAP.
    return (metrics["trans_val"]["map"], metrics["cis_val"]["map"])
