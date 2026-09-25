from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from PIL import Image


@dataclass(frozen=True)
class Sample:
    image_id: str | int
    file_name: str
    image_path: Path
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]

    @property
    def image_size(self) -> tuple[int, int]:
        with Image.open(self.image_path) as image:
            return image.size


class CocoWildlifeDataset:
    """Small read-only adapter for the repository's COCO-style JSON files."""

    def __init__(self, annotation_file: str | Path, image_dir: str | Path):
        self.annotation_file = Path(annotation_file)
        self.image_dir = Path(image_dir)
        with self.annotation_file.open("r", encoding="utf-8") as handle:
            self.coco: dict[str, Any] = json.load(handle)

        required = {"images", "annotations", "categories"}
        missing = required.difference(self.coco)
        if missing:
            raise ValueError(f"COCO JSON is missing keys: {sorted(missing)}")

        grouped: dict[str | int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco["annotations"]:
            grouped[annotation["image_id"]].append(annotation)
        self._annotations = grouped
        self.images: list[dict[str, Any]] = self.coco["images"]
        self.categories: list[dict[str, Any]] = self.coco["categories"]

    @property
    def detection_categories(self) -> list[dict[str, Any]]:
        # "empty" is an image-level negative label, not a box category.
        return [c for c in self.categories if c["name"].strip().lower() != "empty"]

    @property
    def category_names(self) -> list[str]:
        return [str(c["name"]) for c in self.detection_categories]

    @property
    def category_name_to_id(self) -> dict[str, int]:
        return {str(c["name"]).strip().lower(): int(c["id"]) for c in self.detection_categories}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Sample:
        image = self.images[index]
        image_id = image["id"]
        return Sample(
            image_id=image_id,
            file_name=image["file_name"],
            image_path=self.image_dir / image["file_name"],
            width=int(image["width"]),
            height=int(image["height"]),
            annotations=tuple(self._annotations.get(image_id, [])),
        )

    def iter_samples(self, limit: int | None = None) -> Iterator[Sample]:
        stop = len(self) if limit is None else min(limit, len(self))
        for index in range(stop):
            yield self[index]

    def validate(self, limit: int | None = None, check_images: bool = True) -> dict[str, int]:
        checked = missing = invalid = 0
        resized = unreadable = 0
        for sample in self.iter_samples(limit):
            checked += 1
            if sample.width <= 0 or sample.height <= 0:
                invalid += 1
            if check_images:
                if not sample.image_path.is_file():
                    missing += 1
                    continue
                try:
                    resized += sample.image_size != (sample.width, sample.height)
                except OSError:
                    unreadable += 1
        return {
            "images_checked": checked, "missing_images": missing, "invalid_sizes": invalid,
            "resized_images": resized, "unreadable_images": unreadable,
        }


def default_annotation_path(dataset_root: str | Path, split: str) -> Path:
    return Path(dataset_root) / "eccv_18_annotation_files" / f"{split}_annotations.json"


def default_image_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "eccv_18_all_images_sm"
