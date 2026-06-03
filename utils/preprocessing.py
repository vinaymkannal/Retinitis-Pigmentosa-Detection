"""
utils/preprocessing.py
=======================
Image preprocessing for RP Detection model.
"""

import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Model input size
IMG_SIZE = (224, 224)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a retinal image for model inference.

    Steps:
      1. Read image via OpenCV (BGR -> RGB)
      2. Resize to 224×224
      3. Apply ResNet50 preprocessing (mean subtraction, no /255)
      4. Add batch dimension

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        # Fallback: try PIL
        pil_img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(pil_img)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    # Convert to float and apply ResNet50 preprocessing
    img_float = img_resized.astype(np.float32)
    img_preprocessed = preprocess_input(img_float)

    # Add batch dimension
    return np.expand_dims(img_preprocessed, axis=0)


def load_image_display(image_path: str) -> np.ndarray:
    """
    Load image for display purposes (RGB uint8, resized to 224×224).
    Used by Grad-CAM overlay generation.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        pil_img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(pil_img)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return cv2.resize(img_rgb, IMG_SIZE)


def apply_circular_mask(image: np.ndarray, margin: float = 0.02) -> np.ndarray:
    """
    Apply a circular mask to remove black borders around retinal images.
    Retinal fundus images are typically circular on a black background.

    Args:
        image: RGB image array (H, W, 3)
        margin: fractional margin to inset the circle (default 2%)

    Returns:
        Masked image array
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * (1 - margin))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def validate_image(image_path: str) -> dict:
    """
    Validate that the uploaded file is a readable image.
    Returns dict with 'valid' bool and optional 'error' string.
    """
    try:
        img = Image.open(image_path)
        img.verify()
        return {'valid': True, 'format': img.format, 'size': img.size}
    except Exception as e:
        return {'valid': False, 'error': str(e)}
