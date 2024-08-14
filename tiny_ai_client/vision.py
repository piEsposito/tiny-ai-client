from __future__ import annotations
import base64
from io import BytesIO

from PIL import Image as PIL_Image


def encode_pil_image(image: PIL_Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
