"""
utils/gradcam.py
================
Grad-CAM explainability for RP Detection.

Fixed issues:
  - Correct layer navigation: model -> resnet50 -> conv5_block3_out
  - No .numpy() on existing numpy arrays
  - Circular retinal masking applied to heatmap overlay
  - Robust error handling
"""

import numpy as np
import cv2
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)

# Target layer inside ResNet50
RESNET_LAYER_NAME = 'conv5_block3_out'
RESNET_SUBMODEL_NAME = 'resnet50'


def _get_gradcam_model(model):
    """
    Build a Grad-CAM model that outputs:
      - conv5_block3_out activations (inside resnet50 sub-model)
      - final predictions

    Handles two architectures:
      A) model contains a sub-model named 'resnet50'
      B) model itself has the conv layer at top level
    """
    # Try nested resnet50 sub-model first
    try:
        base_model = model.get_layer(RESNET_SUBMODEL_NAME)
        last_conv = base_model.get_layer(RESNET_LAYER_NAME)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )
        logger.info(f"Grad-CAM: using nested layer {RESNET_SUBMODEL_NAME}/{RESNET_LAYER_NAME}")
        return grad_model
    except Exception as e1:
        logger.warning(f"Nested layer approach failed: {e1}. Trying top-level.")

    # Fallback: try direct access
    try:
        last_conv = model.get_layer(RESNET_LAYER_NAME)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv.output, model.output]
        )
        logger.info(f"Grad-CAM: using top-level layer {RESNET_LAYER_NAME}")
        return grad_model
    except Exception as e2:
        logger.warning(f"Direct layer access failed: {e2}. Using last conv layer.")

    # Last resort: find the last Conv2D layer automatically
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
        # Also check sub-models
        if hasattr(layer, 'layers'):
            for sub_layer in layer.layers:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    last_conv_layer = sub_layer

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model for Grad-CAM.")

    logger.info(f"Grad-CAM: using auto-detected conv layer: {last_conv_layer.name}")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    return grad_model


def compute_gradcam_heatmap(model, img_array: np.ndarray, class_idx: int = 0) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given input image.

    Args:
        model: Keras model
        img_array: Preprocessed image array (1, 224, 224, 3)
        class_idx: Class index for gradient computation (0 for binary)

    Returns:
        heatmap: np.ndarray (H, W) float32 in [0, 1]
    """
    grad_model = _get_gradcam_model(model)

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # For binary classification, use the single output neuron
        predictions = tf.convert_to_tensor(predictions)
        loss = predictions[:, 0]

    # Gradients of loss w.r.t. conv feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv outputs by pooled gradients
    conv_outputs = conv_outputs[0]          # (H, W, C)
    pooled_grads = pooled_grads             # (C,)

    # Multiply each channel by its gradient weight
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)           # (H, W)

    # ReLU + normalize to [0, 1]
    heatmap = tf.nn.relu(heatmap)

    # Convert to numpy safely (avoid double .numpy() call)
    if hasattr(heatmap, 'numpy'):
        heatmap = heatmap.numpy()
    heatmap = np.array(heatmap, dtype=np.float32)

    # Normalize
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, apply_mask: bool = True) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: RGB image array (H, W, 3) uint8
        heatmap: Heatmap (H_small, W_small) float32 in [0,1]
        alpha: Blend factor for overlay
        apply_mask: Whether to apply circular retinal mask

    Returns:
        overlay: RGB image with heatmap overlay (H, W, 3) uint8
    """
    from utils.preprocessing import apply_circular_mask

    h, w = original_img.shape[:2]

    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply circular mask to heatmap (remove black border noise)
    if apply_mask:
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        radius = int(min(cx, cy) * 0.97)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        heatmap_resized = heatmap_resized * (mask / 255.0)

    # Convert heatmap to colour map (COLORMAP_JET)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored_rgb, alpha, 0)
    return overlay


def generate_gradcam(model, image_path: str,
                     gradcam_save_path: str,
                     overlay_save_path: str) -> bool:
    """
    Full Grad-CAM pipeline: generate heatmap + overlay, save to disk.

    Args:
        model: Keras model
        image_path: Path to input retinal image
        gradcam_save_path: Path to save coloured heatmap image
        overlay_save_path: Path to save overlay image

    Returns:
        True on success, False on failure
    """
    try:
        from utils.preprocessing import preprocess_image, load_image_display

        # Preprocess for model
        img_array = preprocess_image(image_path)

        # Load original image for display
        original_img = load_image_display(image_path)

        # Compute heatmap
        heatmap = compute_gradcam_heatmap(model, img_array)

        # Build coloured heatmap image
        heatmap_display = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_display)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Save heatmap
        os.makedirs(os.path.dirname(gradcam_save_path), exist_ok=True)
        cv2.imwrite(gradcam_save_path, heatmap_colored)

        # Build and save overlay
        overlay = overlay_heatmap(original_img, heatmap)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(overlay_save_path, overlay_bgr)

        logger.info(f"Grad-CAM saved: {gradcam_save_path}")
        logger.info(f"Overlay saved: {overlay_save_path}")
        return True

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {str(e)}", exc_info=True)
        return False
