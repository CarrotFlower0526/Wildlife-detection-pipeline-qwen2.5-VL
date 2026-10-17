from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from .common import (atomic_json, clean_sample, configure_paths, grouped_annotations,
                     negative_candidates, read_json)

BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
SPLITS = ("train", "cis_val", "trans_val", "cis_test", "trans_test")


def download(url, path, expected_md5=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return {"url": url, "bytes": path.stat().st_size, "reused": True}
    part = path.with_suffix(path.suffix + ".part")
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block); md5.update(block); sha.update(block)
    if expected_md5 and md5.hexdigest() != expected_md5:
        part.unlink()
        raise ValueError(f"Published checksum mismatch: {url}")
    os.replace(part, path)
    return {"url": url, "bytes": path.stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def image_from_benchmark(archive, output):
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith((".jpg", ".jpeg")):
                destination = output / Path(member.name).name
                if destination.exists():
                    continue
                # Read file bytes only: archive paths, links, ownership are never extracted.
                with tar.extractfile(member) as source, destination.open("wb") as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-negatives", type=int, default=3000)
    args = parser.parse_args()
    scratch = configure_paths()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    images = out / "images"; images.mkdir(exist_ok=True)
    sources = {}
    annotation_tar = scratch / "eccv_18_annotations.tar.gz"
    sources["annotations"] = download(BASE + annotation_tar.name, annotation_tar,
                                       "66a1f481b44aa1edadf75c9cfbd27aba")
    with tarfile.open(annotation_tar) as tar:
        datasets = {split: json.load(tar.extractfile(f"eccv_18_annotation_files/{split}_annotations.json"))
                    for split in SPLITS}
    image_tar = scratch / "eccv_18_all_images_sm.tar.gz"
    sources["benchmark_images"] = download(BASE + image_tar.name, image_tar,
                                            "8143c17aa2a12872b66f284ff211531f")
    image_from_benchmark(image_tar, images)
    image_tar.unlink()  # The decoded images are the durable input for all downstream runs.

    categories = [c for c in datasets["train"]["categories"] if c["name"] != "empty"]
    category_map = {c["id"]: c["name"] for c in datasets["train"]["categories"]}
    manifests = {}
    for split, coco in datasets.items():
        by_image = grouped_annotations(coco)
        rows = []
        for entry in coco["images"]:
            with Image.open(images / (str(entry["id"]) + ".jpg")) as image:
                image.load()
                size = image.size
            rows.append(clean_sample(entry, by_image[str(entry["id"])], category_map, size))
        manifests[split] = rows
        atomic_json(out / f"{split}.json", rows)

    # CCT20's official training split has no negatives. Use only explicitly empty
    # sequences at locations absent from ALL five benchmark splits, and disclose
    # this as an extended-data experiment rather than the original paper protocol.
    extra = []
    if args.extra_negatives:
        full_metadata = scratch / "caltech_camera_traps.json.zip"
        sources["extra_negative_metadata"] = download(BASE + "labels/" + full_metadata.name, full_metadata)
        with zipfile.ZipFile(full_metadata) as archive:
            full = json.loads(archive.read(next(n for n in archive.namelist() if n.endswith(".json"))))
        candidates = negative_candidates(full, list(datasets.values()), args.extra_negatives)
        image_base = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/"

        def prepare_negative(entry):
            destination = images / f"{entry['id']}.jpg"
            if not destination.exists():
                with urllib.request.urlopen(image_base + entry["file_name"], timeout=120) as response:
                    raw = response.read()
                with Image.open(io.BytesIO(raw)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    image.save(destination, quality=95)
            with Image.open(destination) as image:
                image.load(); size = image.size
            row = clean_sample(entry, [{"category_id": 30}], category_map, size)
            row["source"] = "external_cct_nonbenchmark_camera_empty"
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            extra = list(pool.map(prepare_negative, candidates))
        sources["extra_negative_images"] = {"base_url": image_base, "count": len(extra)}
    training = [r for r in manifests["train"] if r["status"] != "label_only"] + extra
    atomic_json(out / "train_detection.json", training)
    atomic_json(out / "extra_negatives.json", extra)
    atomic_json(out / "categories.json", categories)
    model_path = out / "pretrained" / "yolo11l.pt"
    sources["detector"] = download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt", model_path)
    report = {"protocol": "cct20-extended-negatives-v1", "sources": sources,
              "training_images": len(training), "extra_negative_images": len(extra),
              "negative_source": "CCT cameras disjoint from all CCT20 benchmark cameras; one frame per sequence",
              "excluded_label_only_training_images": sum(r["status"] == "label_only" for r in manifests["train"]),
              "splits": {name: dict(Counter(r["status"] for r in rows)) for name, rows in manifests.items()},
              "categories": categories,
              "evaluation": "Primary detection scores exclude known label-only positives and retain true empty images. Legacy all-image scores are secondary. Box completeness within boxed images remains an annotation assumption."}
    atomic_json(out / "dataset_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
