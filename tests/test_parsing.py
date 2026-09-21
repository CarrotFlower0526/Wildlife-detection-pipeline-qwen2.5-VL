import unittest

from wildlife_detection.parsing import parse_detections


CATEGORIES = {"bobcat": 6, "opossum": 1, "car": 33}


class ParseDetectionsTest(unittest.TestCase):
    def test_json_fence_and_qwen_bbox_key(self):
        text = '```json\n[{"label":"bobcat","bbox_2d":[10,20,100,200],"confidence":.8}]\n```'
        detections, warnings = parse_detections(text, CATEGORIES, 2048, 1494)
        self.assertEqual(warnings, [])
        self.assertEqual(detections[0].bbox_xywh, [10.0, 20.0, 90.0, 180.0])
        self.assertEqual(detections[0].score, 0.8)

    def test_normalized_and_unknown(self):
        text = '[{"label":"cars","bbox":[.1,.2,.9,.8]}, {"label":"bear","bbox":[1,2,3,4]}]'
        detections, warnings = parse_detections(text, CATEGORIES, 100, 50)
        self.assertEqual(detections[0].bbox_xyxy, (10.0, 10.0, 90.0, 40.0))
        self.assertEqual(len(warnings), 1)

    def test_empty_array(self):
        detections, warnings = parse_detections("[]", CATEGORIES, 100, 100)
        self.assertEqual(detections, [])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
