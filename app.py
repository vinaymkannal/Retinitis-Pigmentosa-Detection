"""
RP Detection Web Application - Flask Backend
=============================================
Retinitis Pigmentosa AI Detection System
Author: AI Healthcare Project
"""

import os
import uuid
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'rp-detection-secret-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/gradcam'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['MODEL_PATH'] = 'models/rp_detection_final.keras'
app.config['HISTORY_FILE'] = 'static/results/prediction_history.json'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['GRADCAM_FOLDER'], app.config['RESULTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Lazy-load model to avoid startup errors if model file missing
_model = None

def get_model():
    global _model
    if _model is None:
        from utils.prediction import load_model_safe
        _model = load_model_safe(app.config['MODEL_PATH'])
    return _model


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_history():
    """Load prediction history from JSON file."""
    if os.path.exists(app.config['HISTORY_FILE']):
        try:
            with open(app.config['HISTORY_FILE'], 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(entry):
    """Append a prediction entry to history."""
    history = load_history()
    history.insert(0, entry)
    history = history[:100]  # Keep last 100 predictions
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    """Landing page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard with prediction history."""
    history = load_history()
    stats = {
        'total': len(history),
        'rp_count': sum(1 for h in history if h.get('label') == 'Retinitis Pigmentosa'),
        'normal_count': sum(1 for h in history if h.get('label') == 'Normal'),
        'avg_confidence': round(
            np.mean([h.get('confidence', 0) for h in history]) if history else 0, 1
        )
    }
    return render_template('dashboard.html', history=history, stats=stats)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and run prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF.'}), 400

    try:
        # Save uploaded file with unique name
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        filename = f"retina_{unique_id}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filepath}")

        # Load model
        model = get_model()

        # Run prediction
        from utils.prediction import predict_image
        from utils.gradcam import generate_gradcam
        from utils.preprocessing import preprocess_image

        result = predict_image(model, filepath)
        label = result['label']
        confidence = result['confidence']
        raw_score = result['raw_score']

        # Generate Grad-CAM
        gradcam_filename = f"gradcam_{unique_id}.jpg"
        overlay_filename = f"overlay_{unique_id}.jpg"
        gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
        overlay_path = os.path.join(app.config['GRADCAM_FOLDER'], overlay_filename)

        gradcam_ok = generate_gradcam(
            model=model,
            image_path=filepath,
            gradcam_save_path=gradcam_path,
            overlay_save_path=overlay_path
        )

        # Build response
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = {
            'id': unique_id,
            'timestamp': timestamp,
            'filename': filename,
            'label': label,
            'confidence': confidence,
            'raw_score': raw_score,
            'image_url': f'/static/uploads/{filename}',
            'gradcam_url': f'/static/gradcam/{gradcam_filename}' if gradcam_ok else None,
            'overlay_url': f'/static/gradcam/{overlay_filename}' if gradcam_ok else None,
            'interpretation': get_interpretation(label, confidence),
            'risk_level': get_risk_level(label, confidence)
        }

        save_history(entry)
        logger.info(f"Prediction complete: {label} ({confidence}%)")

        return render_template('result.html', result=entry)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/result/<prediction_id>')
def result_page(prediction_id):
    """Render result page for a given prediction."""
    history = load_history()
    entry = next((h for h in history if h.get('id') == prediction_id), None)
    if not entry:
        return render_template('index.html', error='Result not found.')
    return render_template('result.html', result=entry)


@app.route('/download/report/<prediction_id>')
def download_report(prediction_id):
    """Generate and download a PDF report for a prediction."""
    history = load_history()
    entry = next((h for h in history if h.get('id') == prediction_id), None)
    if not entry:
        return jsonify({'error': 'Result not found'}), 404

    try:
        from utils.visualization import generate_pdf_report
        pdf_path = generate_pdf_report(entry, app.config['RESULTS_FOLDER'])
        return send_file(pdf_path, as_attachment=True,
                         download_name=f'RP_Report_{prediction_id}.pdf')
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500


@app.route('/api/history')
def api_history():
    """Return prediction history as JSON."""
    history = load_history()
    return jsonify(history)


@app.route('/api/stats')
def api_stats():
    """Return summary statistics."""
    history = load_history()
    rp = [h for h in history if h.get('label') == 'Retinitis Pigmentosa']
    normal = [h for h in history if h.get('label') == 'Normal']
    return jsonify({
        'total': len(history),
        'rp_count': len(rp),
        'normal_count': len(normal),
        'avg_confidence': round(np.mean([h.get('confidence', 0) for h in history]), 1) if history else 0
    })


@app.route('/api/model-info')
def model_info():
    """Return model status info."""
    model_exists = os.path.exists(app.config['MODEL_PATH'])
    return jsonify({
        'model_path': app.config['MODEL_PATH'],
        'model_loaded': _model is not None,
        'model_file_exists': model_exists,
        'backend': 'TensorFlow/Keras ResNet50'
    })


# ─────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_interpretation(label, confidence):
    """Return clinical interpretation text."""
    if label == 'Retinitis Pigmentosa':
        if confidence >= 90:
            return ("High-confidence detection of Retinitis Pigmentosa patterns. "
                    "The AI model identified strong indicators of RP including peripheral retinal degeneration. "
                    "Immediate consultation with a retinal specialist is strongly recommended.")
        elif confidence >= 75:
            return ("Moderate-to-high likelihood of Retinitis Pigmentosa detected. "
                    "Several characteristic RP markers are present. "
                    "Please schedule an ophthalmology evaluation for comprehensive assessment.")
        else:
            return ("Possible indicators of Retinitis Pigmentosa detected with moderate confidence. "
                    "Early-stage RP features may be present. "
                    "A follow-up examination with an eye specialist is recommended.")
    else:
        if confidence >= 90:
            return ("No significant indicators of Retinitis Pigmentosa detected. "
                    "The retinal image appears within normal parameters. "
                    "Continue with routine annual eye examinations.")
        else:
            return ("The retina appears largely normal with no strong RP indicators. "
                    "Some ambiguity exists — routine follow-up is advised. "
                    "Consult an ophthalmologist if visual symptoms persist.")


def get_risk_level(label, confidence):
    """Return risk level string."""
    if label == 'Retinitis Pigmentosa':
        if confidence >= 85:
            return 'HIGH'
        elif confidence >= 65:
            return 'MODERATE'
        else:
            return 'LOW-MODERATE'
    else:
        if confidence >= 85:
            return 'LOW'
        else:
            return 'LOW-MODERATE'


# ─────────────────────────────────────────
#  ERROR HANDLERS
# ─────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error='Page not found.'), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    logger.info(f"Starting RP Detection Server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
