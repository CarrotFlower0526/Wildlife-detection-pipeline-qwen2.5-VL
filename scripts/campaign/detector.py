from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

from .common import atomic_json, read_prediction_log


def prepare_yolo(data, scratch, categories, train_rows, validation_rows):
    mapping = {c["id"]: index for index, c in enumerate(categories)}
    root = Path(scratch) / "yolo_dataset"
    for split, rows in (("train", train_rows), ("val", validation_rows)):
        images = root / "images" / split; images.mkdir(parents=True, exist_ok=True)
        labels = root / "labels" / split; labels.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row["status"] == "label_only":
                continue
            image_path = images / f"{row['id']}.jpg"
            if not image_path.exists():
                image_path.symlink_to((Path(data) / row["file_name"]).resolve())
            lines = []
            for box in row["boxes"]:
                x, y, width, height = box["bbox"]
                lines.append(f"{mapping[box['category_id']]} {(x+width/2)/row['width']:.8f} "
                             f"{(y+height/2)/row['height']:.8f} {width/row['width']:.8f} {height/row['height']:.8f}")
            (labels / f"{row['id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml = root / "data.yaml"
    # JSON is valid YAML and safely encodes paths and class names.
    yaml.write_text(json.dumps({"path": str(root), "train": "images/train", "val": "images/val",
                                "names": {i: c["name"] for i, c in enumerate(categories)}}))
    return yaml


def train(data, scratch, categories, rows, validation, directory, epochs=80,
          device=0, image_size=1024, batch_size=8, workers=4, amp=True):
    import torch
    from ultralytics import YOLO

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    config = prepare_yolo(data, scratch, categories, rows, validation)
    durable = directory / "resume.pt"
    if durable.exists():
        checkpoint = torch.load(durable, map_location="cpu", weights_only=False)
        # Ultralytics stores absolute dataset/output paths. Rebind to this attempt,
        # preserving optimizer, EMA, scheduler epoch, and scientific settings.
        checkpoint["train_args"].update(data=str(config), project=str(directory), name="fit", save_dir=str(directory / "fit"))
        rebound = directory / "resume-rebound.pt"
        torch.save(checkpoint, rebound)
        model = YOLO(str(rebound))
    else:
        model = YOLO(str(Path(data) / "pretrained/yolo11l.pt"))

    def preserve_checkpoint(trainer):
        temporary = directory / "resume-writing.pt"
        shutil.copyfile(trainer.last, temporary)
        os.replace(temporary, durable)
        atomic_json(directory / "checkpoint_state.json", {"epoch": trainer.epoch + 1})

    model.add_callback("on_model_save", preserve_checkpoint)
    model.train(data=str(config), project=str(directory), name="fit", exist_ok=True,
                resume=bool(durable.exists()), epochs=epochs, patience=15, imgsz=image_size, batch=batch_size,
                device=device, workers=workers, amp=amp, optimizer="AdamW", lr0=0.001, lrf=0.01,
                seed=20261017, deterministic=True, close_mosaic=10,
                cache=False, plots=False, save=True, verbose=False)
    best = Path(model.trainer.best)
    shutil.copyfile(best, directory / "best.pt")
    atomic_json(directory / "training_complete.json", {"epochs_requested": epochs,
                "early_stopping_patience": 15, "selection_split": "trans_val", "imgsz": image_size})
    del model
    gc.collect(); torch.cuda.empty_cache()
    return directory / "best.pt"


def infer(weights, data, categories, rows, cache, *, augment=False, image_size=1024):
    import torch
    from ultralytics import YOLO

    cache = Path(cache); cache.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for record in read_prediction_log(cache):
        if record["id"] in records:
            raise ValueError("Duplicate prediction record")
        records[record["id"]] = record["detections"]
    pending = [r for r in rows if r["id"] not in records]
    if pending:
        model = YOLO(str(weights))
        with cache.open("a") as stream:
            for start in range(0, len(pending), 8):
                batch = pending[start:start+8]
                results = model.predict(source=[str(Path(data) / r["file_name"]) for r in batch],
                        stream=True, device=0, imgsz=image_size, augment=augment, batch=8,
                        conf=0.001, iou=0.6, max_det=100, verbose=False, save=False)
                observed = 0
                for row, result in zip(batch, results):
                    observed += 1
                    boxes = []
                    for xyxy, score, label in zip(result.boxes.xyxy.cpu().tolist(),
                            result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                        x1, y1, x2, y2 = xyxy
                        boxes.append({"category_id": categories[int(label)]["id"],
                                      "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
                    records[row["id"]] = boxes
                    stream.write(json.dumps({"id": row["id"], "detections": boxes}) + "\n"); stream.flush()
                if observed != len(batch):
                    raise ValueError("Detector did not return one result per image")
        del model; gc.collect(); torch.cuda.empty_cache()
    return {r["id"]: records[r["id"]] for r in rows}, {"failed_responses": 0, "augment": augment, "image_size": image_size}
