from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

from .common import (atomic_json, configure_paths, evaluate, experiment_identity,
                     read_json, screening_sample, selection_key)

CONFIG = {"protocol": "cct20-extended-negatives-v1", "seed": 20261017,
          "screen_images_per_split": 512,
          "qwen3b": {"rank": 16, "learning_rate": 0.0001, "epochs": 3},
          "qwen7b": {"rank": 32, "learning_rate": 0.00005, "epochs": 3},
          "detector": {"model": "yolo11l", "epochs": 80, "imgsz": 1024, "patience": 15},
          "selection": "Highest full trans_val mAP50:95; ties broken by full cis_val mAP50:95",
          "test_policy": "Freeze one candidate before any test inference; no test-driven retuning"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model3b", type=Path, required=True)
    parser.add_argument("--model7b", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    scratch = configure_paths()
    import torch
    from . import detector, qwen

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This experiment requires a CUDA device supporting BF16")
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    args.result.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(CONFIG)
    state_path = args.checkpoint / "state.json"
    state = read_json(state_path) if state_path.exists() else {
        "identity": identity, "completed": [],
        "progress": {"completed": 0, "total": 12, "unit": "experiment stages", "elapsed_seconds": 0.0}}
    if state["identity"] != identity:
        raise RuntimeError("Checkpoint belongs to different inputs or scientific settings")
    atomic_json(state_path, state)
    stages = args.checkpoint / "stages"

    def stage(name, action):
        path = stages / f"{name}.json"
        if path.exists():
            receipt = read_json(path)
        else:
            print(f"START {name}", flush=True)
            start = time.monotonic()
            result = action()
            receipt = {"result": result, "elapsed_seconds": time.monotonic() - start}
            atomic_json(path, receipt)
        if name not in state["completed"]:
            state["completed"].append(name)
            state["progress"]["completed"] = len(state["completed"])
            state["progress"]["elapsed_seconds"] += receipt["elapsed_seconds"]
            atomic_json(state_path, state)
            print(f"COMPLETE {name}", flush=True)
        return receipt["result"]

    categories = read_json(args.data / "categories.json")
    validation = {split: read_json(args.data / f"{split}.json") for split in ("cis_val", "trans_val")}
    screen = {split: screening_sample([r for r in rows if r["status"] != "label_only"],
                                     CONFIG["screen_images_per_split"]) for split, rows in validation.items()}
    train_detection = read_json(args.data / "train_detection.json")
    train_qwen = train_detection + [r for r in read_json(args.data / "train.json") if r["status"] == "label_only"]
    atomic_json(args.checkpoint / "screen_ids.json", {s: [r["id"] for r in rows] for s, rows in screen.items()})
    models = {"qwen3b": args.model3b, "qwen7b": args.model7b}
    candidates = {}

    def score_qwen(name, kind, base, adapter=None):
        model, processor = qwen.load_model(base, adapter)
        result = {"kind": kind, "base": name, "adapter": str(adapter.relative_to(args.checkpoint)) if adapter else None,
                  "metrics": {}}
        for split, rows in validation.items():
            predictions, diagnostics = qwen.infer(model, processor, rows, args.data, categories,
                        args.checkpoint / "predictions" / f"{name}-{kind}-{split}.jsonl")
            result["metrics"][split] = {**evaluate(rows, predictions, categories), **diagnostics}
        del model, processor
        gc.collect(); torch.cuda.empty_cache()
        return result

    for name, base in models.items():
        candidates[name + "-baseline"] = stage(name + "-baseline", lambda n=name, b=base: score_qwen(n, "baseline", b))
        training_dir = args.checkpoint / name

        def fit_qwen(name=name, base=base, training_dir=training_dir):
            complete = training_dir / "training_complete.json"
            if not complete.exists():
                qwen.train(base, train_qwen, args.data, categories, training_dir, CONFIG[name], screen)
            best = read_json(training_dir / "best_screen.json")
            return {"adapter": str((training_dir / best["adapter"]).relative_to(args.checkpoint)), "training": read_json(complete)}

        fitted = stage(name + "-training", fit_qwen)
        adapter = args.checkpoint / fitted["adapter"]
        candidates[name + "-lora"] = stage(name + "-lora", lambda n=name, b=base, a=adapter: score_qwen(n, "lora", b, a))

    detector_dir = args.checkpoint / "detector"

    def fit_detector():
        if not (detector_dir / "training_complete.json").exists():
            detector.train(args.data, scratch, categories, train_detection, validation["trans_val"],
                           detector_dir, epochs=CONFIG["detector"]["epochs"])
        return {"weights": "detector/best.pt", "training": read_json(detector_dir / "training_complete.json")}

    fitted_detector = stage("detector-training", fit_detector)
    weights = args.checkpoint / fitted_detector["weights"]

    def score_detector(augment):
        result = {"kind": "detector", "weights": fitted_detector["weights"], "augment": augment,
                  "image_size": 1280 if augment else 1024, "metrics": {}}
        for split, rows in validation.items():
            predictions, diagnostics = detector.infer(weights, args.data, categories, rows,
                  args.checkpoint / "predictions" / f"detector-{augment}-{split}.jsonl",
                  augment=augment, image_size=result["image_size"])
            result["metrics"][split] = {**evaluate(rows, predictions, categories), **diagnostics}
        return result

    candidates["detector-1024"] = stage("detector-validation", lambda: score_detector(False))
    candidates["detector-1280-tta"] = stage("detector-tta-validation", lambda: score_detector(True))

    # This file is committed before test labels or images are read by this process.
    selection = stage("selection", lambda: {"name": max(candidates, key=lambda n: selection_key(candidates[n]["metrics"])),
                                            "rule": CONFIG["selection"], "candidates": candidates})
    winner = candidates[selection["name"]]
    tests = {}
    for split in ("cis_test", "trans_test"):
        def test_selected(split=split):
            rows = read_json(args.data / f"{split}.json")
            cache = args.checkpoint / "predictions" / f"selected-{split}.jsonl"
            if winner["kind"] == "detector":
                predictions, diagnostic = detector.infer(args.checkpoint / winner["weights"], args.data, categories,
                    rows, cache, augment=winner["augment"], image_size=winner["image_size"])
            else:
                adapter = args.checkpoint / winner["adapter"] if winner["adapter"] else None
                model, processor = qwen.load_model(models[winner["base"]], adapter)
                predictions, diagnostic = qwen.infer(model, processor, rows, args.data, categories, cache)
                del model, processor; gc.collect(); torch.cuda.empty_cache()
            return {"primary": evaluate(rows, predictions, categories),
                    "legacy_all_images": evaluate(rows, predictions, categories, include_label_only=True),
                    "diagnostics": diagnostic}
        tests[split] = stage("selected-" + split, test_selected)

    selected_model = args.result / "selected_model"; selected_model.mkdir(exist_ok=True)
    if winner["kind"] == "detector":
        shutil.copyfile(args.checkpoint / winner["weights"], selected_model / "best.pt")
    elif winner["adapter"]:
        shutil.copytree(args.checkpoint / winner["adapter"], selected_model / "adapter", dirs_exist_ok=True)
    model_references = {"qwen3b": {"repo": "Qwen/Qwen2.5-VL-3B-Instruct", "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3"},
                        "qwen7b": {"repo": "Qwen/Qwen2.5-VL-7B-Instruct", "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5"}}
    atomic_json(selected_model / "recipe.json", {"winner": winner, "name": selection["name"],
                "categories": categories, "base_models": model_references,
                "prediction_command": "python -m scripts.campaign.predict --bundle SELECTED_MODEL --images IMAGE_DIR --output predictions.jsonl [--base-model QWEN_BASE_DIR]"})
    report = {"configuration": CONFIG, "input_identity": identity["inputs"],
              "dataset": read_json(args.data / "dataset_report.json"), "selection": selection,
              "test": tests, "gpu": torch.cuda.get_device_name(0),
              "software": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "ultralytics")},
              "limitations": ["Best among the declared candidates; not a proof of global optimality.",
                  "Additional negatives come from nonbenchmark CCT cameras; this is not a strict original-data-only CCT20 result.",
                  "Known label-only positives are excluded from primary box AP; annotations on remaining images are assumed complete.",
                  "Qwen response likelihood is an uncalibrated ranking proxy, not an object probability.",
                  "Single training seed; repeat-run variance has not been measured."]}
    atomic_json(args.result / "report.json", report)
    lines = ["# Wildlife detection experiment", "", f"Selected: **{selection['name']}**", "",
             CONFIG["selection"], "", "| Candidate | cis_val mAP | trans_val mAP |", "|---|---:|---:|"]
    for name, value in candidates.items():
        lines.append(f"| {name} | {value['metrics']['cis_val']['map']:.4f} | {value['metrics']['trans_val']['map']:.4f} |")
    lines += ["", "## Held-out test results", "", "| Split | mAP50:95 | AP50 | AR100 |", "|---|---:|---:|---:|"]
    for split, value in tests.items():
        m = value["primary"]; lines.append(f"| {split} | {m['map']:.4f} | {m['ap50']:.4f} | {m['ar100']:.4f} |")
    lines += ["", "## Interpretation", ""] + ["- " + value for value in report["limitations"]]
    (args.result / "report.md").write_text("\n".join(lines) + "\n")
    for split in tests:
        shutil.copyfile(args.checkpoint / "predictions" / f"selected-{split}.jsonl", args.result / f"{split}-predictions.jsonl")
    print(json.dumps({"selected": selection["name"], "test": tests}), flush=True)


if __name__ == "__main__":
    main()
