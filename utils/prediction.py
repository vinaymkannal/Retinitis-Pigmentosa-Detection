"""
utils/prediction.py
===================
Model loading and inference for RP Detection.
"""

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Class labels — index 0 = Normal, index 1 = RP  (sigmoid > 0.5 → RP)
CLASS_LABELS = {0: 'Normal', 1: 'Retinitis Pigmentosa'}
THRESHOLD = 0.81


def load_model_safe(model_path: str):
    """
    Load the Keras model from disk.
    Provides helpful error messages if model not found.
    """
    import tensorflow as tf

    if not os.path.exists(model_path):
        logger.warning(
            f"Model file not found at {model_path}. "
            "Please place 'rp_detection_final.keras' in the models/ directory."
        )
        # Return a dummy model for development/testing
        logger.info("Creating placeholder model for development mode.")
        return _create_dummy_model()

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model summary layers: {[l.name for l in model.layers]}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Falling back to placeholder model.")
        return _create_dummy_model()


def _create_dummy_model():
    """
    Create a placeholder ResNet50-based model for development/testing
    when the trained weights are not available.
    """
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer_1')
    base = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D(name='gap')(base.output)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dense(256, activation='relu', name='dense_1')(x)
    x = layers.Dropout(0.4, name='dropout_1')(x)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.Dropout(0.3, name='dropout_2')(x)
    output = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=inputs, outputs=output)
    logger.warning("Using DUMMY MODEL — predictions are random. Load your trained model!")
    return model


def predict_image(model, image_path: str) -> dict:
    """
    Run inference on a single retinal image.

    Args:
        model: Loaded Keras model
        image_path: Path to image file

    Returns:
        dict with keys: label, confidence, raw_score, class_idx
    """
    from utils.preprocessing import preprocess_image

    # Preprocess
    img_array = preprocess_image(image_path)

    # Inference
    raw_score = float(model.predict(img_array, verbose=0)[0][0])

    # Classify
    class_idx = 1 if raw_score >= THRESHOLD else 0
    label = CLASS_LABELS[class_idx]

    # Confidence: distance from decision boundary mapped to 50-100%
    if class_idx == 1:
        confidence = round(raw_score * 100, 1)
    else:
        confidence = round((1 - raw_score) * 100, 1)

    return {
        'label': label,
        'confidence': confidence,
        'raw_score': round(raw_score, 4),
        'class_idx': class_idx
    }


def predict_batch(model, image_paths: list) -> list:
    """
    Run inference on a list of image paths.

    Returns:
        List of prediction dicts (same structure as predict_image)
    """
    results = []
    for path in image_paths:
        try:
            results.append(predict_image(model, path))
        except Exception as e:
            logger.error(f"Batch prediction failed for {path}: {e}")
            results.append({'label': 'Error', 'confidence': 0, 'raw_score': -1, 'class_idx': -1})
    return results
