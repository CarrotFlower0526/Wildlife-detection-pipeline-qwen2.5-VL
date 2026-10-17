import json

import pytest

from scripts.campaign.common import (clean_sample, coco_ground_truth, evaluate, negative_candidates,
                                     read_prediction_log, screening_sample, selection_key)
from scripts.campaign.qwen import prediction_objects, question, target


CATEGORIES = [{"id": 6, "name": "bobcat"}]


def sample(status="boxed", identity="one"):
    return {"id": identity, "file_name": f"images/{identity}.jpg", "width": 100, "height": 100,
            "location": "1", "seq_id": identity, "status": status,
            "labels": ["empty" if status == "empty" else "bobcat"],
            "boxes": [{"label": "bobcat", "category_id": 6, "bbox": [10, 20, 30, 40]}] if status == "boxed" else []}


def test_coordinate_conversion_and_label_only_not_negative():
    image = {"id": "a", "width": 200, "height": 200}
    cats = {6: "bobcat", 30: "empty"}
    scaled = clean_sample(image, [{"category_id": 6, "bbox": [20, 40, 60, 80]}], cats, (100, 100))
    assert scaled["boxes"][0]["bbox"] == [10, 20, 30, 40]
    assert clean_sample(image, [{"category_id": 6}], cats, (100, 100))["status"] == "label_only"
    assert clean_sample(image, [], cats, (100, 100))["status"] == "label_only"
    assert clean_sample(image, [{"category_id": 30}], cats, (100, 100))["status"] == "empty"
    assert clean_sample(image, [{"category_id": 6, "bbox": [0, 0, -1, 2]}], cats, (100, 100))["status"] == "label_only"


def test_ground_truth_excludes_unboxed_positives_and_retains_empty():
    rows = [sample(), sample("empty", "negative"), sample("label_only", "unboxed")]
    gt, mapping = coco_ground_truth(rows, CATEGORIES)
    assert set(mapping) == {"one", "negative"}
    assert len(gt["annotations"]) == 1
    assert set(coco_ground_truth(rows, CATEGORIES, True)[1]) == {"one", "negative", "unboxed"}


def test_perfect_predictions_and_empty_response_recall():
    rows = [sample(), sample("empty", "negative"), sample("label_only", "unboxed")]
    box = {"category_id": 6, "bbox": [10, 20, 30, 40], "score": 0.9}
    scores = evaluate(rows, {"one": [box], "negative": [], "unboxed": [box]}, CATEGORIES)
    assert scores["map"] == pytest.approx(1.0)
    assert scores["label_only_excluded"] == 1
    assert scores["empty_fpr_by_threshold"]["0.5"] == 0
    empty = evaluate(rows, {r["id"]: [] for r in rows}, CATEGORIES)
    assert empty["ar100"] == 0
    assert empty["boxed_images_without_predictions"] == 1
    with pytest.raises(ValueError, match="exactly one"):
        evaluate(rows, {"one": []}, CATEGORIES)


def test_external_negatives_exclude_all_benchmark_locations_and_sequences():
    benchmark = [{"images": [{"id": "used", "location": 1, "seq_id": "used-seq"}]}]
    images = [{"id": "same-camera", "location": "1", "seq_id": "new"},
              {"id": "same-sequence", "location": "2", "seq_id": "used-seq"},
              {"id": "good-a", "location": "2", "seq_id": "ok"},
              {"id": "good-b", "location": "2", "seq_id": "ok"},
              {"id": "positive", "location": "3", "seq_id": "positive"}]
    annotations = [{"image_id": i["id"], "category_id": 6 if i["id"] == "positive" else 30} for i in images]
    metadata = {"categories": CATEGORIES + [{"id": 30, "name": "empty"}], "images": images, "annotations": annotations}
    selected = negative_candidates(metadata, benchmark, 1)
    assert len(selected) == 1 and selected[0]["id"].startswith("good-")
    with pytest.raises(ValueError, match="Only 1"):
        negative_candidates(metadata, benchmark, 2)


def test_training_targets_and_inference_contract():
    assert json.loads(target(sample())) == [{"label": "bobcat", "bbox": [10, 20, 40, 60]}]
    assert target(sample("empty")) == "[]"
    assert json.loads(target(sample("label_only"))) == ["bobcat"]
    assert "category strings" in question(sample(), CATEGORIES, True)
    prediction = prediction_objects(target(sample()), sample(), CATEGORIES, 0.73)[0]
    assert prediction["bbox"] == [10, 20, 30, 40] and prediction["score"] == 0.73
    with pytest.raises(ValueError):
        prediction_objects('[{"label":"bobcat","bbox":[30,0,10,1]}]', sample(), CATEGORIES, 0.7)


def test_resume_removes_only_torn_tail(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":')
    assert read_prediction_log(path) == [{"id": "one"}]
    assert path.read_bytes() == b'{"id":"one"}\n'
    path.write_bytes(b'not json\n')
    with pytest.raises(json.JSONDecodeError):
        read_prediction_log(path)


def test_screening_is_stable_unique_and_selection_prefers_new_cameras():
    rows = [sample(identity=str(i)) for i in range(100)]
    first = screening_sample(rows, 17)
    assert first == screening_sample(rows, 17)
    assert len({r["id"] for r in first}) == 17
    assert selection_key({"cis_val": {"map": .8}, "trans_val": {"map": .2}}) < selection_key({"cis_val": {"map": .3}, "trans_val": {"map": .4}})
