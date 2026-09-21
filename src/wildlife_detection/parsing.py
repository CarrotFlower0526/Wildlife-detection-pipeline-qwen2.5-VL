from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float

    @property
    def bbox_xywh(self) -> list[float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return [x1, y1, x2 - x1, y2 - y1]


def _candidate_payloads(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    starts = [i for i, char in enumerate(text) if char in "[{"]
    for start in starts:
        opener = text[start]
        closer = "]" if opener == "[" else "}"
        end = text.rfind(closer)
        if end > start:
            candidates.append(text[start : end + 1])
    return candidates


def _decode_payload(text: str) -> Any:
    for candidate in _candidate_payloads(text):
        for decoder in (json.loads, ast.literal_eval):
            try:
                return decoder(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    raise ValueError("model response does not contain a JSON array or object")


def _objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("detections", "objects", "results", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if any(key in payload for key in ("bbox", "bbox_2d", "box")):
            return [payload]
    return []


def _canonical_label(value: Any, category_map: dict[str, int]) -> str | None:
    label = str(value or "").strip().lower().replace("_", " ")
    aliases = {name.lower(): name.lower() for name in category_map}
    aliases.update({f"{name}s": name for name in category_map if not name.endswith("s")})
    aliases.update({"possum": "opossum", "automobile": "car", "vehicle": "car"})
    return aliases.get(label)


def parse_detections(
    text: str,
    category_map: dict[str, int],
    width: int,
    height: int,
) -> tuple[list[Detection], list[str]]:
    """Parse common Qwen localization outputs; malformed items are reported and skipped."""
    payload = _decode_payload(text)
    detections: list[Detection] = []
    warnings: list[str] = []
    for index, obj in enumerate(_objects(payload)):
        label = _canonical_label(
            obj.get("label", obj.get("class", obj.get("category", obj.get("name")))), category_map
        )
        if label is None:
            warnings.append(f"item {index}: unknown label")
            continue
        box = obj.get("bbox", obj.get("bbox_2d", obj.get("box")))
        if isinstance(box, dict):
            box = [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            warnings.append(f"item {index}: invalid bbox")
            continue
        try:
            coords = [float(v) for v in box]
        except (TypeError, ValueError):
            warnings.append(f"item {index}: non-numeric bbox")
            continue
        if not all(math.isfinite(v) for v in coords):
            warnings.append(f"item {index}: non-finite bbox")
            continue
        # Accept explicitly normalized responses in addition to requested pixel coordinates.
        if all(0.0 <= v <= 1.0 for v in coords):
            coords = [coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height]
        x1, x2 = sorted((max(0.0, min(width, coords[0])), max(0.0, min(width, coords[2]))))
        y1, y2 = sorted((max(0.0, min(height, coords[1])), max(0.0, min(height, coords[3]))))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            warnings.append(f"item {index}: zero-area bbox")
            continue
        try:
            score = float(obj.get("score", obj.get("confidence", 1.0)))
        except (TypeError, ValueError):
            score = 1.0
        score = max(0.0, min(1.0, score)) if math.isfinite(score) else 1.0
        detections.append(Detection(label, category_map[label], (x1, y1, x2, y2), score))
    return detections, warnings
