"""YOLOv8 COCO banana detection followed by the original Keras classifier."""
from __future__ import annotations

import math
import os
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from threading import RLock

import numpy as np
import streamlit as st
from PIL import Image

from services.classifier import BananaClassifier, CLASS_NAMES, Prediction
from utils.image_utils import annotate_detections, clamp_box, padded_box, to_png_bytes

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS_PATH = ROOT / "models" / "yolov8n.pt"
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "true")


class DetectorError(RuntimeError):
    pass


def ensure_weights(path: Path = DEFAULT_WEIGHTS_PATH) -> Path:
    """Download only public model weights; uploaded images never leave memory."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.part")
    try:
        request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "BananaRipeness/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response, temporary.open("wb") as target:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 30 * 1024 * 1024:
                    raise DetectorError("Unduhan bobot YOLO melebihi ukuran yang diharapkan.")
                target.write(chunk)
        if total < 1_000_000:
            raise DetectorError("Unduhan bobot YOLO tidak lengkap.")
        temporary.replace(path)
    except (OSError, urllib.error.URLError, DetectorError) as exc:
        temporary.unlink(missing_ok=True)
        raise DetectorError(
            f"Bobot YOLO belum tersedia dan unduhan gagal: {exc}. "
            f"Unduh yolov8n.pt melalui tautan di README, lalu simpan di {path}. "
            "Klasifikasi Pisang tetap dapat digunakan."
        ) from exc
    return path


def find_banana_class(names: dict[int, str] | list[str]) -> int:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for index, name in items:
        if str(name).strip().lower() == "banana":
            return int(index)
    raise DetectorError("Metadata model YOLO tidak memiliki kelas 'banana'. Gunakan yolov8n.pt COCO.")


@dataclass(frozen=True)
class Detection:
    object_id: int
    box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    detection_confidence: float
    prediction: Prediction


@dataclass
class DetectionResult:
    detections: list[Detection]
    annotated_image: Image.Image
    threshold: float

    @cached_property
    def png_bytes(self) -> bytes:
        return to_png_bytes(self.annotated_image)

    @property
    def counts(self) -> dict[str, int]:
        counter = Counter(item.prediction.label for item in self.detections)
        return {label: counter[label] for label in CLASS_NAMES}


class BananaDetector:
    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH):
        try:
            Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
            from ultralytics import YOLO, settings

            settings.update({"sync": False, "hub": False})
            weights = ensure_weights(weights_path)
            self.model = YOLO(str(weights), task="detect")
            self.banana_class = find_banana_class(self.model.names)
            self._lock = RLock()
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"YOLO gagal dimuat: {exc}. Periksa requirements.txt dan models/yolov8n.pt.") from exc

    def detect_and_classify(self, image: Image.Image, classifier: BananaClassifier,
                            confidence: float = 0.25) -> DetectionResult:
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            raise DetectorError("Confidence threshold harus berada pada rentang 0.05–0.95.")
        rgb = image.convert("RGB")
        try:
            with self._lock:
                results = self.model.predict(
                    source=rgb, conf=confidence, iou=0.45,
                    classes=[self.banana_class], agnostic_nms=False,
                    imgsz=640, max_det=100, device="cpu", verbose=False,
                    save=False, save_txt=False, save_crop=False,
                )
        except Exception as exc:
            raise DetectorError(f"Deteksi gagal: {exc}. Coba foto beresolusi lebih kecil.") from exc
        detections = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            coordinates = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for coordinates_row, score, class_id in zip(coordinates, scores, classes):
                if int(class_id) != self.banana_class or not np.isfinite(score):
                    continue
                if not confidence <= float(score) <= 1.0:
                    continue
                box = clamp_box(coordinates_row, rgb.size)
                if box is None:
                    continue
                crop_bounds = padded_box(box, rgb.size, fraction=0.03)
                prediction = classifier.classify(rgb.crop(crop_bounds))
                detections.append(Detection(len(detections) + 1, box, crop_bounds,
                                            float(score), prediction))
        # An empty list intentionally never triggers whole-image classification.
        return DetectionResult(detections, annotate_detections(rgb, detections), confidence)


@st.cache_resource(show_spinner=False)
def load_detector(weights_path: str = str(DEFAULT_WEIGHTS_PATH)) -> BananaDetector:
    return BananaDetector(Path(weights_path))
