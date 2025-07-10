
import numpy as np
from PIL import Image
import io

def read_image_from_bytes(image_bytes, size=(128, 128)):
    """Read and preprocess image from raw bytes."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)
