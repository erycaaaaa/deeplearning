from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from services.classifier import Prediction
from utils.image_utils import (
    ImageValidationError, MAX_BYTES, annotate_detections, clamp_box,
    padded_box, to_png_bytes, validate_image,
)


def encode(image, format="PNG", **kwargs):
    buffer = BytesIO()
    image.save(buffer, format=format, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize("data,filename,message", [
    (b"", "x.jpg", "kosong"),
    (b"not an image", "x.jpg", "rusak"),
    (b"x" * (MAX_BYTES + 1), "x.jpg", "10 MB"),
    (b"any", "x.webp", "Format"),
])
def test_reject_bad_uploads(data, filename, message):
    with pytest.raises(ImageValidationError, match=message):
        validate_image(data, filename)


def test_verify_content_and_extreme_dimensions():
    with pytest.raises(ImageValidationError, match="JPEG atau PNG"):
        validate_image(encode(Image.new("RGB", (64, 64)), "GIF"), "fake.jpg")
    for size in [(31, 64), (12001, 32), (5000, 5000)]:
        with pytest.raises(ImageValidationError):
            validate_image(encode(Image.new("RGB", size)), "photo.png")


def test_truncated_photo_rejected():
    data = encode(Image.new("RGB", (100, 100)), "JPEG")
    with pytest.raises(ImageValidationError):
        validate_image(data[:len(data) // 2], "broken.jpg")


def test_exif_orientation_and_rgb():
    exif = Image.Exif()
    exif[274] = 6  # 90-degree clockwise rotation
    decoded = validate_image(encode(Image.new("RGB", (80, 40), "red"), "JPEG", exif=exif), "phone.jpeg")
    assert decoded.size == (40, 80)
    assert decoded.mode == "RGB"
    assert decoded.getexif().get(274) is None


def test_transparency_white_background_and_grayscale():
    result = validate_image(encode(Image.new("RGBA", (32, 32), (255, 0, 0, 0))), "x.png")
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert validate_image(encode(Image.new("L", (32, 32))), "x.png").mode == "RGB"


def test_xyxy_bounds_and_padding():
    assert clamp_box((-4.4, -8, 110, 150), (100, 120)) == (0, 0, 100, 120)
    assert clamp_box((10.8, 20.3, 30.2, 40.2), (100, 120)) == (10, 20, 31, 41)
    for box in [(20, 20, 10, 10), (1, 1, 1, 2), (float("nan"), 0, 2, 3), (101, 0, 200, 10)]:
        assert clamp_box(box, (100, 120)) is None
    assert padded_box((0, 0, 100, 120), (100, 120)) == (0, 0, 100, 120)
    assert padded_box((20, 20, 80, 80), (100, 100)) == (18, 18, 82, 82)


def test_annotation_uses_original_coordinates_and_preserves_source():
    source = Image.new("RGB", (400, 300), "white")
    prediction = Prediction(1, (0.1, 0.7, 0.1, 0.1))
    item = SimpleNamespace(object_id=1, box=(80, 90, 320, 250), prediction=prediction)
    annotated = annotate_detections(source, [item])
    assert annotated.size == source.size
    assert source.getpixel((80, 200)) == (255, 255, 255)
    assert annotated.getpixel((80, 200)) == (39, 131, 77)
    assert annotated.getpixel((79, 200)) == (255, 255, 255)
    assert Image.open(BytesIO(to_png_bytes(annotated))).format == "PNG"
    assert annotate_detections(source, []).tobytes() == source.tobytes()
