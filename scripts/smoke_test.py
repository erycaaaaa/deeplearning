"""Exercise both real models. Only sample images are written to test-results/."""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.classifier import load_classifier
from services.detector import load_detector
from utils.image_utils import validate_image


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "test-results"
    output.mkdir(exist_ok=True)
    classifier = load_classifier()
    detector = load_detector()
    report = {"input_shape": classifier.model.input_shape, "output_shape": classifier.model.output_shape}
    for stem in ("single_banana", "multiple_bananas"):
        path = next((p for p in (root / "samples").glob(f"{stem}.*")), None)
        if path is None:
            report[stem] = "SKIPPED: sample unavailable"
            continue
        image = validate_image(path.read_bytes(), path.name)
        if stem == "single_banana":
            prediction = classifier.classify(image)
            report[stem] = {"label": prediction.label, "probabilities": prediction.probabilities}
        else:
            result = detector.detect_and_classify(image, classifier)
            result.annotated_image.save(output / "multiple_bananas_annotated.png")
            report[stem] = {"boxes": len(result.detections), "counts": result.counts,
                            "objects": [{"id": d.object_id, "xyxy": d.box,
                                         "crop_xyxy": d.crop_box,
                                         "yolo_confidence": d.detection_confidence,
                                         "label": d.prediction.label,
                                         "probabilities": d.prediction.probabilities}
                                        for d in result.detections]}
    empty = detector.detect_and_classify(Image.new("RGB", (640, 480), "white"), classifier)
    report["blank_image_boxes"] = len(empty.detections)
    assert not empty.detections, "Blank image unexpectedly produced detections"
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
