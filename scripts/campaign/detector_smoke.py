"""Exercise the production detector training wrapper on tiny synthetic CPU data."""
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from .common import configure_paths, read_json
from .detector import train


def main():
    scratch = configure_paths()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    root = Path(tempfile.mkdtemp(prefix="detector-smoke-", dir=scratch))
    (root / "images").mkdir()
    (root / "pretrained").mkdir()
    YOLO("yolo11n.yaml").save(root / "pretrained/yolo11l.pt")
    rows = []
    for i in range(6):
        image = Image.new("RGB", (64, 64), (30 + i*10, 30, 30))
        ImageDraw.Draw(image).rectangle((10, 12, 40, 45), fill=(180, 100, 80))
        image.save(root / f"images/{i}.jpg")
        rows.append({"id": str(i), "file_name": f"images/{i}.jpg", "width": 64, "height": 64,
                     "status": "boxed", "boxes": [{"category_id": 6, "label": "bobcat", "bbox": [10, 12, 30, 33]}]})
    result = root / "checkpoint"
    best = train(root, root / "scratch", [{"id": 6, "name": "bobcat"}], rows[:4], rows[4:],
                 result, epochs=1, device="cpu", image_size=64, batch_size=2, workers=0, amp=False)
    assert best.is_file() and (result / "training_complete.json").is_file()
    checkpoint = torch.load(result / "resume.pt", map_location="cpu", weights_only=False)
    assert checkpoint["optimizer"] is not None and checkpoint["epoch"] == 0
    assert read_json(result / "checkpoint_state.json")["epoch"] == 1
    print(json.dumps({"status": "passed", "architecture": "YOLO11n fixture", "device": "cpu",
                      "checks": ["production YOLO data conversion", "real training epoch", "native validation", "best weights", "durable optimizer checkpoint"],
                      "uncovered": "Full YOLO11-L weights, CUDA memory, interrupted cross-host resume and convergence"}))


if __name__ == "__main__":
    main()
