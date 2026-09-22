from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image


class QwenDetector:
    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 512,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"model directory not found: {model_path}. Place Qwen2.5-VL-3B-Instruct-AWQ there first."
            )
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("model dependencies are missing; run: pip install -e .") from exc

        try:
            # AutoAWQ's Triton GEMM kernel produces wrong dtypes on very new GPU
            # architectures (e.g. Blackwell/sm_120); force the pure-PyTorch fallback.
            from awq.modules.linear import gemm as _awq_gemm

            _awq_gemm.TRITON_AVAILABLE = False
        except ImportError:
            pass

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, **processor_kwargs
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).eval()
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def prompt(categories: Sequence[str], width: int, height: int) -> str:
        labels = ", ".join(categories)
        return (
            "Detect every visible target object in this wildlife camera image. "
            f"Allowed labels only: [{labels}]. The original image is {width} pixels wide and "
            f"{height} pixels high. Return ONLY a JSON array. Each item must be "
            '{"label":"one allowed label","bbox":[x1,y1,x2,y2],"score":0.0}. '
            "bbox must use original-image pixel coordinates, with top-left and bottom-right corners. "
            "Use [] when no allowed object is visible. Do not add Markdown or explanations."
        )

    def predict(self, samples: Sequence, categories: Sequence[str]) -> list[str]:
        messages = []
        images = []
        for sample in samples:
            messages.append(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt(categories, sample.width, sample.height)},
                    ],
                }]
            )
            with Image.open(sample.image_path) as image:
                images.append(image.convert("RGB"))

        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
