"""MobileNetV2 inference with the exact training class order and preprocessing."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
_plot_cache = ROOT / ".cache" / "matplotlib"
_plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_plot_cache))
DEFAULT_MODEL_PATH = ROOT / "models" / "mobilenet_model.keras"
# This order is part of the checkpoint contract. Never sort it.
CLASS_NAMES = ("matang", "mentah", "setengah-matang", "terlalu-matang")
CLASS_COLORS = ("#E99A25", "#27834D", "#F2C94C", "#9A5033")
CLASS_DESCRIPTIONS = (
    "Kulit tampak matang. Pisang biasanya siap dikonsumsi atau dipasarkan.",
    "Kulit tampak masih mentah. Buah biasanya memerlukan waktu pematangan tambahan.",
    "Kulit menunjukkan tahap peralihan menuju matang.",
    "Kulit tampak sangat matang, dengan perubahan warna atau bercak yang lebih banyak.",
)
CLASS_ADVICE = (
    "Prioritaskan penjualan dan konsumsi dalam waktu dekat. Hindari tekanan saat penanganan.",
    "Simpan di tempat sejuk dan berventilasi pada suhu ruang. Periksa perubahan warna berkala.",
    "Siapkan untuk penjualan berikutnya dan pantau pematangan setiap hari.",
    "Periksa langsung sebelum digunakan. Jika masih layak, segera olah; pisahkan buah berjamur atau rusak.",
)


class ClassifierError(RuntimeError):
    """A checkpoint or its actual output is incompatible with this application."""


@dataclass(frozen=True)
class Prediction:
    class_index: int
    probabilities: tuple[float, float, float, float]

    @property
    def label(self) -> str:
        return CLASS_NAMES[self.class_index]

    @property
    def confidence(self) -> float:
        return self.probabilities[self.class_index]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """EXIF -> RGB -> TF bilinear resize -> float32 /255 -> (1,224,224,3).

    tf.image.resize (antialias=False) intentionally matches read_image in the
    supplied notebook. Pillow's bilinear implementation has different sampling.
    The checkpoint itself maps [0,1] to [-1,1]; do not preprocess_input here.
    """
    import tensorflow as tf

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32)
    resized = tf.image.resize(pixels, (224, 224), method="bilinear", antialias=False)
    normalized = tf.clip_by_value(tf.cast(resized, tf.float32) / 255.0, 0.0, 1.0)
    return np.expand_dims(normalized.numpy(), axis=0).astype(np.float32, copy=False)


def prediction_from_output(output: Any) -> Prediction:
    probabilities = np.asarray(output, dtype=np.float32)
    if probabilities.shape != (1, 4):
        raise ClassifierError(f"Output model harus (1, 4), diterima {probabilities.shape}.")
    row = probabilities[0]
    if (not np.all(np.isfinite(row)) or np.any(row < 0) or np.any(row > 1)
            or not np.isclose(row.sum(), 1.0, atol=1e-4)):
        raise ClassifierError("Model tidak menghasilkan empat probabilitas softmax yang valid.")
    return Prediction(int(np.argmax(row)), tuple(float(p) for p in row))


class BananaClassifier:
    def __init__(self, model_path: str | Path):
        import tensorflow as tf

        self.path = Path(model_path).resolve()
        if not self.path.is_file():
            raise ClassifierError(
                f"Model tidak ditemukan. Letakkan checkpoint asli di: {self.path}"
            )
        try:
            self.model = tf.keras.models.load_model(self.path, compile=False)
        except Exception as exc:
            raise ClassifierError(
                f"Gagal memuat {self.path.name}: {exc}. Gunakan Python 3.11/3.12, "
                "TensorFlow 2.21.0 dan Keras 3.13.2 sesuai requirements.txt."
            ) from exc
        if (self.model.input_shape != (None, 224, 224, 3)
                or self.model.output_shape != (None, 4)):
            raise ClassifierError(
                f"Shape checkpoint tidak sesuai: input={self.model.input_shape}, "
                f"output={self.model.output_shape}. Diperlukan satu input RGB dan empat kelas."
            )
        scalers = [layer for layer in self.model.layers
                   if isinstance(layer, tf.keras.layers.Rescaling)]
        if not any(np.isclose(layer.scale, 2.0) and np.isclose(layer.offset, -1.0)
                   for layer in scalers):
            raise ClassifierError("Checkpoint harus memiliki Rescaling(2.0, offset=-1.0).")
        if getattr(self.model.layers[-1].activation, "__name__", "") != "softmax":
            raise ClassifierError("Layer output checkpoint harus menggunakan softmax.")
        # st.cache_resource shares resources between sessions. Serialize model calls.
        self._lock = RLock()

    def classify(self, image: Image.Image) -> Prediction:
        batch = preprocess_image(image)
        with self._lock:
            output = self.model(batch, training=False).numpy()
        return prediction_from_output(output)


@st.cache_resource(show_spinner=False)
def load_classifier(model_path: str = str(DEFAULT_MODEL_PATH)) -> BananaClassifier:
    return BananaClassifier(model_path)
