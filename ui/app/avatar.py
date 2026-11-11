"""Luna's round avatar: a square cut from her profile picture, where the owner chose.

The crop is the square's centre (``x``, ``y``, as fractions of the picture's width and
height) and ``zoom``: 1 is the largest square the picture holds, 2 a square half as
wide. It is kept beside the profile picture and forgotten when the picture changes.
Until one is chosen, a tall picture is cut from its top, where a portrait's head is,
so the head is never clipped.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

MAX_ZOOM = 6.0
# The default for a tall picture: a square about 70 % of its width, its top edge 2 %
# below the picture's top, centred across: her head and shoulders.
DEFAULT_PORTRAIT_ZOOM = 1.4
DEFAULT_TOP_MARGIN = 0.02
SIZE = 256


def clamp(crop: dict[str, float], width: int, height: int) -> dict[str, float]:
    """The crop moved and zoomed just enough for its square to lie inside the picture."""
    zoom = min(max(float(crop.get("zoom", 1.0)), 1.0), MAX_ZOOM)
    side = min(width, height) / zoom
    half_x, half_y = side / 2 / width, side / 2 / height
    x = min(max(float(crop.get("x", 0.5)), half_x), 1 - half_x)
    y = min(max(float(crop.get("y", 0.5)), half_y), 1 - half_y)
    return {"x": round(x, 5), "y": round(y, 5), "zoom": round(zoom, 4)}


def default_crop(width: int, height: int) -> dict[str, float]:
    if height <= width:
        return clamp({"x": 0.5, "y": 0.5, "zoom": 1.0}, width, height)
    side = width / DEFAULT_PORTRAIT_ZOOM
    return clamp({"x": 0.5, "y": DEFAULT_TOP_MARGIN + side / 2 / height, "zoom": DEFAULT_PORTRAIT_ZOOM}, width, height)


def box(crop: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    crop = clamp(crop, width, height)
    side = min(round(min(width, height) / crop["zoom"]), width, height)
    # Rounded once, so the box is exactly square and inside the picture.
    left = min(max(round(crop["x"] * width - side / 2), 0), width - side)
    top = min(max(round(crop["y"] * height - side / 2), 0), height - side)
    return left, top, left + side, top + side


class Avatar:
    def __init__(self, profile: Path) -> None:
        self.profile = profile
        self.crop_path = profile.with_name("avatar.json")

    def size(self) -> tuple[int, int]:
        with Image.open(self.profile) as image:
            return image.size

    def crop(self) -> dict[str, Any]:
        """The crop in use, whether it was chosen, and the picture's size for the editor."""
        width, height = self.size()
        chosen = None
        try:
            chosen = json.loads(self.crop_path.read_text())
        except (OSError, ValueError):
            pass
        crop = clamp(chosen, width, height) if isinstance(chosen, dict) else default_crop(width, height)
        return {**crop, "chosen": isinstance(chosen, dict), "width": width, "height": height}

    def save(self, crop: dict[str, Any]) -> dict[str, Any]:
        width, height = self.size()
        values = {key: float(crop[key]) for key in ("x", "y", "zoom")}
        self.crop_path.write_text(json.dumps(clamp(values, width, height)))
        return self.crop()

    def reset(self) -> None:
        self.crop_path.unlink(missing_ok=True)

    def render(self) -> bytes:
        """The avatar as a square PNG, SIZE pixels a side."""
        crop = self.crop()
        with Image.open(self.profile) as image:
            square = image.convert("RGB").crop(box(crop, *image.size)).resize((SIZE, SIZE), Image.LANCZOS)
        buffer = io.BytesIO()
        square.save(buffer, "PNG")
        return buffer.getvalue()
