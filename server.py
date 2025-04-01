from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)

# Load Whisper Model
model = whisper.load_model("base")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)

    # Transcribe Audio using Whisper
    result = model.transcribe(file_path)
    return jsonify({"transcription": result["text"]})

if __name__ == "__main__":
    app.run(debug=True)
