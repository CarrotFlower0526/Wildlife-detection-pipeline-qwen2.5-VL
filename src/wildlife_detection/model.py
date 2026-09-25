from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from .parsing import detection_schema


class QwenDetector:
    def __init__(
        self,
        model_path: str | Path,
        max_new_tokens: int = 512,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"model directory not found: {model_path}. Place Qwen2.5-VL-3B-Instruct-AWQ there first."
            )
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("model dependencies are missing; run: pip install -e .") from exc

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
            try:
                from awq.modules.linear import gemm as _awq_gemm

                _awq_gemm.TRITON_AVAILABLE = False
            except ImportError:
                pass

        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        quantization_config = getattr(config, "quantization_config", None)
        if quantization_config and quantization_config.get("quant_method") == "awq":
            excluded_modules = list(quantization_config.get("modules_to_not_convert") or [])
            if "lm_head" not in excluded_modules:
                excluded_modules.append("lm_head")
            quantization_config["modules_to_not_convert"] = excluded_modules

        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, **processor_kwargs
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).eval()
        self.torch = torch
        self.max_new_tokens = max_new_tokens

    def _prefix_allowed_tokens(self, categories: Sequence[str]):
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )

        if getattr(self, "_tokenizer_data", None) is None:
            self._tokenizer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        parser = JsonSchemaParser(
            detection_schema(categories),
            config=CharacterLevelParserConfig(max_json_array_length=self.max_new_tokens),
        )
        return build_transformers_prefix_allowed_tokens_fn(self._tokenizer_data, parser)

    @staticmethod
    def prompt(categories: Sequence[str], width: int, height: int) -> str:
        labels = ", ".join(categories)
        return (
            "Detect every visible target object in this wildlife camera image. "
            f"Allowed labels only: [{labels}]. The supplied image is {width} pixels wide and "
            f"{height} pixels high. Return ONLY a JSON array. Each item must be "
            '{"label":"one allowed label","bbox":[x1,y1,x2,y2],"score":0.0}. '
            "bbox must use supplied-image pixel coordinates, with top-left and bottom-right corners. "
            "Use [] when no allowed object is visible. Do not add Markdown or explanations."
        )

    def predict(self, samples: Sequence, categories: Sequence[str]) -> list[str]:
        messages = []
        images = []
        for sample in samples:
            with Image.open(sample.image_path) as image:
                images.append(image.convert("RGB"))
                width, height = image.size
            messages.append(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt(categories, width, height)},
                    ],
                }]
            )
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        inputs = self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        prefix_allowed_tokens_fn = self._prefix_allowed_tokens(categories)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                forced_eos_token_id=None,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
