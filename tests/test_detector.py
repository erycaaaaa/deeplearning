from threading import RLock
from types import SimpleNamespace
import urllib.error

import numpy as np
import pytest
from PIL import Image

from services.classifier import Prediction
from services.detector import BananaDetector, DetectorError, ensure_weights, find_banana_class


class Tensor:
    def __init__(self, values):
        self.values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class RecordingClassifier:
    def __init__(self):
        self.sizes = []

    def classify(self, image):
        self.sizes.append(image.size)
        return Prediction(0, (0.7, 0.1, 0.1, 0.1))


def detector_for(boxes):
    instance = BananaDetector.__new__(BananaDetector)
    instance.banana_class = 9  # Deliberately not COCO's fixed numeric index.
    instance._lock = RLock()
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=boxes)]

    instance.model = SimpleNamespace(predict=predict)
    return instance, calls


def test_metadata_class_lookup():
    assert find_banana_class({9: "banana", 3: "apple"}) == 9
    assert find_banana_class(["apple", "banana"]) == 1
    with pytest.raises(DetectorError, match="banana"):
        find_banana_class({1: "apple"})


@pytest.mark.parametrize("boxes", [None, SimpleNamespace(xyxy=Tensor([]), conf=Tensor([]), cls=Tensor([]))])
def test_no_detection_never_classifies_full_image(boxes):
    detector, _ = detector_for(boxes)
    classifier = RecordingClassifier()
    result = detector.detect_and_classify(Image.new("RGB", (100, 120)), classifier)
    assert classifier.sizes == []
    assert result.detections == []
    assert sum(result.counts.values()) == 0


def test_pipeline_filters_classes_clips_boxes_and_counts_actual_crops():
    boxes = SimpleNamespace(
        xyxy=Tensor([[-10, -10, 50, 60], [30, 40, 120, 130], [1, 2, 3, 4], [70, 80, 60, 90]]),
        conf=Tensor([0.8, 0.9, 0.8, 0.9]), cls=Tensor([9, 9, 2, 9]),
    )
    detector, calls = detector_for(boxes)
    classifier = RecordingClassifier()
    result = detector.detect_and_classify(Image.new("RGB", (100, 120)), classifier, 0.25)
    assert len(result.detections) == len(classifier.sizes) == sum(result.counts.values()) == 2
    assert [x.object_id for x in result.detections] == [1, 2]
    assert result.detections[0].box == (0, 0, 50, 60)
    assert result.detections[1].box == (30, 40, 100, 120)
    assert classifier.sizes == [(52, 62), (73, 83)]
    assert calls[0]["classes"] == [9]
    assert calls[0]["iou"] == 0.45
    assert calls[0]["save"] is False
    assert calls[0]["save_crop"] is False


def test_threshold_validation():
    detector, calls = detector_for(None)
    for threshold in (0, 1, float("nan")):
        with pytest.raises(DetectorError):
            detector.detect_and_classify(Image.new("RGB", (100, 100)), RecordingClassifier(), threshold)
    assert calls == []


def test_failed_weight_download_has_actionable_error_and_no_partial_file(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("test offline")

    monkeypatch.setattr("services.detector.urllib.request.urlopen", offline)
    target = tmp_path / "models" / "yolov8n.pt"
    with pytest.raises(DetectorError, match="Klasifikasi Pisang tetap dapat digunakan"):
        ensure_weights(target)
    assert not target.exists()
    assert not target.with_suffix(".pt.part").exists()
