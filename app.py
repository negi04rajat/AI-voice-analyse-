from flask import Flask, render_template, request, jsonify
import whisper
import os
import re
from datetime import datetime
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac'}

# Load Whisper model
try:
    model = whisper.load_model("small")
    logging.info("Whisper model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Whisper model: {str(e)}")
    model = None  # Fallback to avoid crashes

# Set up logging
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    """Check if the uploaded file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    """Sanitize filename to prevent path traversal attacks."""
    filename = re.sub(r'[^\w.-]', '_', filename)  # Replace invalid characters
    return filename

@app.route('/')
def home():
    """Serve the landing page (index.html)."""
    return render_template('index.html')
@app.route('/prepare')
def upload_form():
    """Navigate to record.html for speech-to-text."""
    return render_template('record.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handles file upload and transcription."""
    if not model:
        return jsonify({"error": "Whisper model not loaded"}), 500

    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No selected file!"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type! Allowed: MP3, WAV, M4A, FLAC"}), 400

        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        sanitized_filename = sanitize_filename(file.filename)
        filename = f"{timestamp}_{sanitized_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Transcribe audio
        try:
            result = model.transcribe(filepath)
        except Exception as e:
            logging.error(f"Whisper transcription error: {str(e)}")
            return jsonify({"error": "Failed to transcribe audio"}), 500

        return jsonify({
            "message": "Transcription successful!",
            "filename": filename,
            "transcription": result.get("text", "")
        }), 200

    except Exception as e:
        logging.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
@app.route('/career-test')
def career_test():
    return render_template('career-test.html')

@app.route('/personality')
def personality_test():
    return render_template('personality.html')

@app.route('/record', methods=['POST'])
def record_audio():
    """Handles live audio recording and transcription."""
    if not model:
        return jsonify({"error": "Whisper model not loaded"}), 500

    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio data received!"}), 400

        audio_data = request.files['audio']
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"live_recording_{timestamp}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_data.save(filepath)

        # Transcribe the recording
        try:
            result = model.transcribe(filepath)
        except Exception as e:
            logging.error(f"Whisper transcription error: {str(e)}")
            return jsonify({"error": "Failed to transcribe live recording"}), 500

        return jsonify({
            "message": "Live recording transcription successful!",
            "filename": filename,
            "transcription": result.get("text", "")
        }), 200

    except Exception as e:
        logging.error(f"Error in record_audio: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
