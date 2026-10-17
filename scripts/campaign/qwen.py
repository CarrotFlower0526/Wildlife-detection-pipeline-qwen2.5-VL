from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from .common import atomic_json, read_json, read_prediction_log, selection_key


def question(row, categories, classification=False):
    names = ", ".join(c["name"] for c in categories)
    if classification:
        return f"List the visible target categories in this camera-trap image. Allowed categories: {names}. Return only a JSON array of category strings."
    return (f"Detect every visible target in this camera-trap image. Allowed labels: {names}. "
            f"The supplied image is {row['width']} pixels wide and {row['height']} pixels high. "
            'Return only a JSON array of objects with keys "label" and "bbox". '
            'bbox is [x1,y1,x2,y2] in supplied-image pixels. Include small, partly occluded, '
            'and night-time animals. Return [] only if no allowed target is visible.')


def target(row):
    if row["status"] == "label_only":
        return json.dumps([x for x in row["labels"] if x != "empty"], separators=(",", ":"))
    result = []
    for box in sorted(row["boxes"], key=lambda b: (b["bbox"][0], b["bbox"][1], b["label"])):
        x, y, w, h = box["bbox"]
        result.append({"label": box["label"], "bbox": [round(x, 2), round(y, 2), round(x+w, 2), round(y+h, 2)]})
    return json.dumps(result, separators=(",", ":"))


def messages(prompt, answer=None):
    value = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        value.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return value


class Collator:
    def __init__(self, processor, dataset, categories):
        self.processor, self.dataset, self.categories = processor, Path(dataset), categories

    def __call__(self, rows):
        from PIL import Image
        images, texts, prefixes = [], [], []
        for row in rows:
            with Image.open(self.dataset / row["file_name"]) as image:
                images.append(image.convert("RGB"))
            prompt = question(row, self.categories, row["status"] == "label_only")
            texts.append(self.processor.apply_chat_template(messages(prompt, target(row)), tokenize=False))
            prefixes.append(self.processor.apply_chat_template(messages(prompt), tokenize=False, add_generation_prompt=True))
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for i, (prefix, image) in enumerate(zip(prefixes, images)):
            prefix_ids = self.processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"][0]
            # Prefix identity checks mask the *actual* expanded image-token prompt.
            if not batch["input_ids"][i, :len(prefix_ids)].equal(prefix_ids):
                raise ValueError("Assistant prefix is not an exact token prefix; refusing incorrect supervision")
            labels[i, :len(prefix_ids)] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any():
            raise ValueError("No assistant target tokens")
        batch["labels"] = labels
        return batch


def load_model(model_path, adapter=None, train=False, max_pixels=1003520):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                              min_pixels=256*28*28, max_pixels=max_pixels)
    processor.tokenizer.padding_side = "right" if train else "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
    model.to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    return model, processor


def prediction_objects(raw, row, categories, confidence):
    """No score is learned from fabricated confidence labels; use sequence likelihood."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Expected JSON array")
    names = {c["name"]: c["id"] for c in categories}
    detections = []
    for obj in payload:
        if not isinstance(obj, dict) or obj.get("label") not in names:
            raise ValueError("Invalid detection label")
        box = obj.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            raise ValueError("Invalid detection box")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Inverted/empty detection box")
        x1, x2 = max(0, x1), min(row["width"], x2)
        y1, y2 = max(0, y1), min(row["height"], y2)
        if x2 > x1 and y2 > y1:
            detections.append({"category_id": names[obj["label"]], "bbox": [x1, y1, x2-x1, y2-y1],
                               "score": confidence})
    return detections


def infer(model, processor, rows, dataset, categories, cache):
    import torch
    from PIL import Image
    cache = Path(cache); cache.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    for record in read_prediction_log(cache):
        if record["id"] in records:
            raise ValueError("Duplicate cached prediction")
        records[record["id"]] = record
    model.eval()
    processor.tokenizer.padding_side = "left"
    with cache.open("a") as stream:
        for row in rows:
            if row["id"] in records:
                continue
            with Image.open(Path(dataset) / row["file_name"]) as source:
                image = source.convert("RGB")
            text = processor.apply_chat_template(messages(question(row, categories)), tokenize=False, add_generation_prompt=True)
            batch = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = model.generate(**batch, do_sample=False, max_new_tokens=768, use_cache=True,
                                        return_dict_in_generate=True, output_scores=True)
            generated = result.sequences[:, batch["input_ids"].shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            # Mean token probability is an explicit, uncalibrated ranking proxy.
            # It is not a detector posterior; all boxes in the response share it.
            log_probs = [torch.log_softmax(logits[0].float(), -1)[generated[0, i]].item()
                         for i, logits in enumerate(result.scores)]
            confidence = math.exp(sum(log_probs) / len(log_probs)) if log_probs else 0.0
            error = None
            try:
                boxes = prediction_objects(raw, row, categories, confidence)
            except (ValueError, TypeError) as exc:
                boxes, error = [], str(exc)
            record = {"id": row["id"], "detections": boxes, "raw_output": raw, "error": error}
            # A torn final line can be discarded on restart, without losing preceding records.
            stream.write(json.dumps(record) + "\n"); stream.flush()
            records[row["id"]] = record
            del result, batch
    expected = {r["id"] for r in rows}
    return {k: v["detections"] for k, v in records.items() if k in expected}, {
        "failed_responses": sum(bool(records[k]["error"]) for k in expected),
        "score_semantics": "Uncalibrated geometric mean of response-token probabilities, shared across each image's boxes"}


def train(model_path, rows, dataset, categories, directory, config, screen_rows):
    import shutil
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from .common import evaluate

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    model, processor = load_model(model_path, train=True)
    modules = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and "visual" not in name
               and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}]
    if not modules:
        raise ValueError("No language projection layers found")
    model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=2*config["rank"],
                          lora_dropout=0.05, target_modules=modules, bias="none", task_type="CAUSAL_LM"))
    model.enable_input_require_grads()
    model.config.use_cache = False

    class SaveAndEvaluate(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            atomic_json(checkpoint / "complete.json", {"step": state.global_step})
        def on_epoch_end(self, args, state, control, **kwargs):
            model.config.use_cache = True
            metrics = {}
            for split, validation in screen_rows.items():
                cache = directory / f"screen-epoch-{state.epoch:.2f}-{split}.jsonl"
                # An interrupted epoch may be trained again from an earlier step.
                # Never mix its former predictions with those from the new weights.
                if cache.exists():
                    cache.unlink()
                preds, diagnostic = infer(model, processor, validation, dataset, categories,
                                          cache)
                metrics[split] = {**evaluate(validation, preds, categories), **diagnostic}
            best_path = directory / "best_screen.json"
            prior = read_json(best_path) if best_path.exists() else None
            if prior is None or selection_key(metrics) > selection_key(prior["metrics"]):
                adapter_name = f"epoch-{state.epoch:.2f}-adapter"
                staging = directory / (adapter_name + "-writing")
                if staging.exists(): shutil.rmtree(staging)
                model.save_pretrained(staging); processor.save_pretrained(staging)
                final = directory / adapter_name
                if final.exists(): shutil.rmtree(final)
                staging.rename(final)
                atomic_json(best_path, {"epoch": state.epoch, "metrics": metrics, "adapter": adapter_name})
            model.train(); processor.tokenizer.padding_side = "right"
            model.config.use_cache = False

    checkpoints = sorted(directory.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    complete = [p for p in checkpoints if (p / "complete.json").exists()]
    for path in checkpoints:
        if path not in complete:
            shutil.rmtree(path)
    arguments = TrainingArguments(output_dir=str(directory), num_train_epochs=config["epochs"],
        per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps", save_steps=100, save_total_limit=2, logging_steps=10,
        report_to="none", remove_unused_columns=False, dataloader_num_workers=0,
        seed=20261017, data_seed=20261017, max_grad_norm=1.0)
    trainer = Trainer(model=model, args=arguments, train_dataset=rows,
                      data_collator=Collator(processor, dataset, categories), callbacks=[SaveAndEvaluate()])
    trainer.train(resume_from_checkpoint=str(complete[-1]) if complete else None)
    atomic_json(directory / "training_complete.json", {"config": config,
                "best": read_json(directory / "best_screen.json"), "target_modules": modules})
    del trainer, model, processor
    gc.collect(); torch.cuda.empty_cache()
    return directory / read_json(directory / "best_screen.json")["adapter"]
