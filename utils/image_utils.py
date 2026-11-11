"""Safe in-memory decoding, bounded xyxy coordinates, and PNG annotation."""
from __future__ import annotations

import math
import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from PIL.Image import open as pillow_open

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 24_000_000
MAX_SIDE = 12_000
MIN_SIDE = 32
ALLOWED_FORMATS = {"JPEG", "PNG"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ImageValidationError(ValueError):
    pass


def validate_image(data: bytes, filename: str = "photo.jpg") -> Image.Image:
    """Verify content, not just extension; never write the uploaded data to disk."""
    if not data:
        raise ImageValidationError("File kosong. Pilih foto JPG, JPEG, atau PNG yang valid.")
    if len(data) > MAX_BYTES:
        raise ImageValidationError("Ukuran foto melebihi 10 MB. Kompres foto lalu coba lagi.")
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ImageValidationError("Format tidak didukung. Gunakan JPG, JPEG, atau PNG.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # Keep Pillow's original decoder. Ultralytics patches Image.open
            # globally and may try to import a HEIF plugin for corrupt JPEGs.
            with pillow_open(BytesIO(data)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise ImageValidationError("Isi file harus benar-benar berupa JPEG atau PNG.")
                width, height = probe.size
                if min(width, height) < MIN_SIDE:
                    raise ImageValidationError("Foto terlalu kecil. Gunakan minimal 32 × 32 piksel.")
                if max(width, height) > MAX_SIDE or width * height > MAX_PIXELS:
                    raise ImageValidationError(
                        "Resolusi terlalu besar. Gunakan maksimal 24 megapiksel "
                        "dan 12.000 piksel pada setiap sisi."
                    )
                if getattr(probe, "n_frames", 1) != 1:
                    raise ImageValidationError("Gunakan foto statis, bukan PNG beranimasi.")
                probe.verify()
            with pillow_open(BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                # Transparent photos receive a neutral white background.
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    rgb = Image.new("RGB", rgba.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                    return rgb
                return oriented.convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Resolusi foto terlalu besar untuk diproses dengan aman.") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageValidationError("Foto rusak atau tidak dapat dibaca. Pilih file lain.") from exc


def clamp_box(xyxy: Iterable[float], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    coords = tuple(float(value) for value in xyxy)
    if len(coords) != 4 or not all(math.isfinite(value) for value in coords):
        return None
    width, height = size
    x1 = max(0, min(width, math.floor(coords[0])))
    y1 = max(0, min(height, math.floor(coords[1])))
    x2 = max(0, min(width, math.ceil(coords[2])))
    y2 = max(0, min(height, math.ceil(coords[3])))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def padded_box(box: tuple[int, int, int, int], size: tuple[int, int],
               fraction: float = 0.03) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x, pad_y = math.ceil((x2 - x1) * fraction), math.ceil((y2 - y1) * fraction)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(size[0], x2 + pad_x), min(size[1], y2 + pad_y)


def annotation_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("DejaVuSans-Bold.ttf", "C:/Windows/Fonts/seguisb.ttf",
                      "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate_detections(image: Image.Image, detections: Iterable[Any]) -> Image.Image:
    """Draw in the original EXIF-corrected image coordinate system.

    Labels scale with canvas width to remain readable in a mobile preview.
    Numbers link overlapping labels to the separate, accessible result cards.
    """
    from services.classifier import CLASS_COLORS

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    font_size = max(14, round(width * 0.034))
    line_width = max(2, round(width * 0.006))
    margin = max(2, round(width * 0.008))
    valid = [(item, clamp_box(item.box, canvas.size)) for item in detections]
    valid = [(item, box) for item, box in valid if box is not None]
    # Draw every box first so a subsequent box cannot cross an earlier label.
    for item, box in valid:
        x1, y1, x2, y2 = box
        color = CLASS_COLORS[item.prediction.class_index]
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=line_width)
    occupied = []
    labels = []
    for item, box in valid:
        x1, y1, x2, y2 = box
        color = CLASS_COLORS[item.prediction.class_index]
        text = f"{item.object_id} {item.prediction.label.capitalize()}"
        size = font_size
        font = annotation_font(size)
        bounds = draw.textbbox((0, 0), text, font=font)
        while bounds[2] - bounds[0] + margin * 2 > width and size > 6:
            size -= 1
            font = annotation_font(size)
            bounds = draw.textbbox((0, 0), text, font=font)
        label_width = min(width, bounds[2] - bounds[0] + margin * 2)
        label_height = min(height, bounds[3] - bounds[1] + margin * 2)
        left = max(0, min(x1, width - label_width))
        top = max(0, min(y1 - label_height if y1 >= label_height else y1, height - label_height))
        # Prefer the space immediately above the object, then adjacent rows.
        # Dense bunches otherwise place long class labels on top of each other.
        candidates = [(left, top)]
        stride = label_height + margin
        for offset in range(stride, height + stride, stride):
            for candidate_y in (top - offset, top + offset):
                if 0 <= candidate_y <= height - label_height:
                    candidates.append((left, candidate_y))
        for candidate_left, candidate_top in candidates:
            candidate = (candidate_left, candidate_top,
                         candidate_left + label_width, candidate_top + label_height)
            if all(candidate[2] + margin <= old[0] or old[2] + margin <= candidate[0]
                   or candidate[3] + margin <= old[1] or old[3] + margin <= candidate[1]
                   for old in occupied):
                left, top = candidate_left, candidate_top
                break
        occupied.append((left, top, left + label_width, top + label_height))
        if top + label_height < y1 - margin or top > y1 + margin:
            anchor_x = max(left, min(round((x1 + x2) / 2), left + label_width - 1))
            anchor_y = top + label_height if top < y1 else top
            draw.line([(round((x1 + x2) / 2), y1), (anchor_x, anchor_y)], fill=color,
                      width=max(1, line_width // 2))
        foreground = "white" if item.prediction.class_index in (1, 3) else "#18271E"
        labels.append(((left, top, left + label_width - 1, top + label_height - 1), color,
                       (left + margin - bounds[0], top + margin - bounds[1]), text, foreground, font))
    # Draw text last so leader lines from neighboring objects cannot obscure it.
    for rectangle, color, origin, text, foreground, font in labels:
        draw.rectangle(rectangle, fill=color)
        draw.text(origin, text, fill=foreground, font=font)
    return canvas


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
