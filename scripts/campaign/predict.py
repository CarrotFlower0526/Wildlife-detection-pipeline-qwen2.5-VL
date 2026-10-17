"""Run a delivered selected_model bundle on a directory of new images."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .common import configure_paths, read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-model", type=Path)
    args = p.parse_args()
    configure_paths()
    recipe = read_json(args.bundle / "recipe.json")
    winner = recipe["winner"]
    rows = []
    for path in sorted(args.images.rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        with Image.open(path) as image:
            width, height = image.size
        relative = str(path.relative_to(args.images))
        rows.append({"id": relative, "file_name": relative, "width": width, "height": height})
    if not rows:
        raise ValueError("No supported images found")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path; inputs may have changed")
    if winner["kind"] == "detector":
        from .detector import infer
        infer(args.bundle / "best.pt", args.images, recipe["categories"], rows, args.output,
              augment=winner["augment"], image_size=winner["image_size"])
    else:
        from .qwen import infer, load_model
        if not args.base_model:
            p.error("--base-model must point to the exact base revision in recipe.json")
        adapter = args.bundle / "adapter" if winner["adapter"] else None
        model, processor = load_model(args.base_model, adapter)
        infer(model, processor, rows, args.images, recipe["categories"], args.output)


if __name__ == "__main__":
    main()
