"""Validate real Qwen processor/collator/LoRA gradients with a tiny CPU model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from .common import configure_paths
from .qwen import Collator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processor", required=True)
    args = p.parse_args()
    scratch = configure_paths()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True,
                                             min_pixels=56*56, max_pixels=56*56)
    processor.tokenizer.padding_side = "right"
    config = Qwen2_5_VLConfig.from_pretrained(args.processor, local_files_only=True,
              vocab_size=len(processor.tokenizer), hidden_size=64,
              intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
              num_key_value_heads=2, max_position_embeddings=4096,
              rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]},
              vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 4, "out_hidden_size": 64, "patch_size": 14,
                "temporal_patch_size": 2, "spatial_merge_size": 2, "window_size": 112,
                "fullatt_block_indexes": [0]})
    model = Qwen2_5_VLForConditionalGeneration(config)
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8,
               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    (scratch / "smoke-images").mkdir(exist_ok=True)
    Image.new("RGB", (100, 80), "gray").save(scratch / "smoke-images/one.jpg")
    rows = [{"id": "one", "file_name": "smoke-images/one.jpg", "width": 100, "height": 80,
             "status": "boxed", "labels": ["bobcat"],
             "boxes": [{"label": "bobcat", "category_id": 6, "bbox": [10, 20, 30, 40]}]},
            {"id": "two", "file_name": "smoke-images/one.jpg", "width": 100, "height": 80,
             "status": "empty", "labels": ["empty"], "boxes": []}]
    batch = Collator(processor, scratch, [{"id": 6, "name": "bobcat"}])(rows)
    assert (batch["labels"] == -100).any() and (batch["labels"] != -100).any()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    for name, parameter in model.named_parameters():
        if "visual" in name:
            assert not parameter.requires_grad
    output = model(**batch)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    assert gradients and any(g.abs().sum().item() > 0 for g in gradients)
    model.zero_grad()
    model.enable_input_require_grads()
    model.config.use_cache = False
    trainer = Trainer(model=model, train_dataset=rows,
        data_collator=Collator(processor, scratch, [{"id": 6, "name": "bobcat"}]),
        args=TrainingArguments(output_dir=str(scratch / "smoke-trainer"), use_cpu=True,
             max_steps=1, per_device_train_batch_size=2, learning_rate=1e-4,
             gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
             remove_unused_columns=False, save_strategy="no", report_to="none", disable_tqdm=True))
    trained = trainer.train()
    assert trainer.state.global_step == 1 and math.isfinite(trained.training_loss)
    print(json.dumps({"status": "passed", "device": "cpu", "architecture": "tiny Qwen2.5-VL",
                      "batch_size": len(rows), "loss": output.loss.item(),
                      "supervised_tokens": int((batch["labels"] != -100).sum()),
                      "checks": ["real image-token expansion", "assistant-only loss", "padding masking",
                                 "vision frozen", "finite forward loss", "nonzero LoRA gradients", "Trainer optimizer step"],
                      "uncovered": "Full pretrained weights, CUDA memory and long-run convergence"}))


if __name__ == "__main__":
    main()
