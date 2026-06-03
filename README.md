# RetinaScan AI – Retinitis Pigmentosa Detection System

## Overview

RetinaScan AI is an AI-powered web application for automated detection of Retinitis Pigmentosa (RP) from retinal fundus images. The system uses a deep learning model based on ResNet50 and provides explainable predictions through Grad-CAM visualizations.

The application enables users to upload retinal images, perform disease classification, and visualize the regions that influenced the model's decision.

---

## Features

* Automated Retinitis Pigmentosa detection
* Deep learning-based classification using ResNet50
* Explainable AI using Grad-CAM heatmaps
* Overlay visualization of attention regions
* Interactive Flask web dashboard
* Confidence score and prediction analysis
* Responsive futuristic medical AI interface

---

## Technology Stack

### Backend

* Python
* Flask
* TensorFlow / Keras

### Deep Learning

* ResNet50
* Transfer Learning
* Grad-CAM Explainability

### Frontend

* HTML5
* CSS3
* JavaScript

### Data Processing

* NumPy
* OpenCV
* Matplotlib

---

## Project Structure

```text
Retinitis-Pigmentosa-Detection/
│
├── app.py
├── templates/
│   ├── index.html
│   └── result.html
│
├── utils/
│   ├── preprocessing.py
│   ├── prediction.py
│   ├── gradcam.py
│   └── visualization.py
│
├── training/
│   └── training.ipynb
│
├── static/
│   ├── uploads/
│   ├── gradcam/
│   └── results/
│
└── models/
```

---

## Workflow

1. Upload retinal fundus image.
2. Image preprocessing and normalization.
3. Classification using trained ResNet50 model.
4. Prediction confidence calculation.
5. Grad-CAM generation.
6. Overlay visualization creation.
7. Display results on dashboard.

---

## Explainable AI

The system uses Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight important retinal regions responsible for the model's prediction. This improves model transparency and assists clinical interpretation.

---

## Results

### Classification Output

* Normal Retina
* Retinitis Pigmentosa

### Visual Outputs

* Original Retinal Image
* Grad-CAM Heatmap
* Diagnostic Overlay Visualization

---

## Future Enhancements

* Multi-disease retinal classification
* Cloud deployment
* PDF diagnostic report generation
* Doctor recommendation module
* Real-time API integration

---

## Author

**Vinay M Kannal**

---

## Disclaimer

This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or clinical decision-making.
