from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from services.classifier import DEFAULT_MODEL_PATH, load_classifier
from services.detector import DEFAULT_WEIGHTS_PATH, load_detector
from utils.image_utils import validate_image

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_original_checkpoint_single_sample():
    if not DEFAULT_MODEL_PATH.is_file():
        pytest.skip("Original checkpoint not available")
    path = ROOT / "samples" / "single_banana.jpeg"
    if not path.is_file():
        pytest.skip("Single sample not available")
    classifier = load_classifier()
    assert classifier.model.input_shape == (None, 224, 224, 3)
    assert classifier.model.output_shape == (None, 4)
    prediction = classifier.classify(validate_image(path.read_bytes(), path.name))
    assert np.isfinite(prediction.probabilities).all()
    assert sum(prediction.probabilities) == pytest.approx(1, abs=1e-4)
    assert all(0 <= p <= 1 for p in prediction.probabilities)


@pytest.mark.integration
def test_original_pipeline_multiple_and_blank():
    if not DEFAULT_MODEL_PATH.is_file() or not DEFAULT_WEIGHTS_PATH.is_file():
        pytest.skip("Download real model weights before integration testing")
    path = next((p for p in (ROOT / "samples").glob("multiple_bananas.*")), None)
    if path is None:
        pytest.skip("Multi sample not available")
    image = validate_image(path.read_bytes(), path.name)
    detector, classifier = load_detector(), load_classifier()
    result = detector.detect_and_classify(image, classifier)
    assert len(result.detections) == sum(result.counts.values())
    assert result.annotated_image.size == image.size
    for item in result.detections:
        x1, y1, x2, y2 = item.box
        assert 0 <= x1 < x2 <= image.width
        assert 0 <= y1 < y2 <= image.height
        assert sum(item.prediction.probabilities) == pytest.approx(1, abs=1e-4)
    blank = detector.detect_and_classify(Image.new("RGB", (640, 480), "white"), classifier)
    assert blank.detections == []
