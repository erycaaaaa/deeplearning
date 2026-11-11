import numpy as np
import pytest
from PIL import Image, ImageOps

from services.classifier import CLASS_NAMES, ClassifierError, prediction_from_output, preprocess_image


def test_class_order_is_checkpoint_contract():
    assert CLASS_NAMES == ("matang", "mentah", "setengah-matang", "terlalu-matang")
    result = prediction_from_output([[0.03, 0.1, 0.85, 0.02]])
    assert result.label == "setengah-matang"
    assert result.confidence == pytest.approx(0.85)


@pytest.mark.parametrize("output", [
    [[0.1, 0.2, 0.3]], [[0.1, 0.2, 0.3, 0.1]],
    [[float("nan"), 0, 0, 1]], [[-0.1, 0.1, 0, 1]],
    [[0, 0, 0, 2]], [0, 0, 0, 1],
])
def test_invalid_probabilities_raise_instead_of_fabrication(output):
    with pytest.raises(ClassifierError):
        prediction_from_output(output)


def test_preprocessing_exactly_matches_notebook_read_image():
    import tensorflow as tf

    pixels = np.random.default_rng(42).integers(0, 256, (113, 317, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    actual = preprocess_image(image)
    expected = tf.clip_by_value(tf.cast(tf.image.resize(pixels, (224, 224)), tf.float32) / 255.0, 0, 1)
    np.testing.assert_allclose(actual[0], expected.numpy(), atol=1e-7)
    assert actual.shape == (1, 224, 224, 3)
    assert actual.dtype == np.float32
    assert 0 <= actual.min() <= actual.max() <= 1
    assert np.all(preprocess_image(Image.new("RGB", (45, 60), "white")) == 1)
    assert np.all(preprocess_image(Image.new("RGB", (45, 60), "black")) == 0)


def test_preprocessing_applies_exif_before_resize():
    image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (45, 70, 3), dtype=np.uint8))
    image.getexif()[274] = 6
    np.testing.assert_array_equal(preprocess_image(image), preprocess_image(ImageOps.exif_transpose(image)))
