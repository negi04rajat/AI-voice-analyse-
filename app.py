from flask import Flask, render_template, request, jsonify, make_response
import os
import re
from datetime import datetime
import logging
from pydub import AudioSegment
import textstat
from transformers import pipeline
import whisperx
import torch
import google.generativeai as genai  # Gemini import

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Configure uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'webm'}

# Load WhisperX tiny model for faster performance
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("tiny", device, compute_type="float32")

# Load Emotion Classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Configure Gemini API
genai.configure(api_key="")  # <-- Put your API key here
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')

# Pacing thresholds
SLOW_THRESHOLD = 100
FAST_THRESHOLD = 160

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    return re.sub(r'[^\w.-]', '_', filename)

def convert_to_wav(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    audio = None
    try:
        if ext == '.mp3':
            audio = AudioSegment.from_mp3(filepath)
        elif ext == '.webm':
            audio = AudioSegment.from_file(filepath, format='webm')
        elif ext == '.m4a':
            audio = AudioSegment.from_file(filepath, format='m4a')
        elif ext == '.flac':
            audio = AudioSegment.from_file(filepath, format='flac')
        elif ext == '.wav':
            return filepath

        if audio:
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format='wav')
            os.remove(filepath)
            logging.info(f"Converted {ext} to WAV: {wav_path}")
            return wav_path

    except Exception as e:
        logging.error(f"Audio conversion failed: {str(e)}")
        raise

    return filepath

def analyze_emotion(transcript):
    if not transcript.strip():
        return {"neutral": 1.0}
    result = emotion_classifier(transcript[:512])
    if not result:
        return {"neutral": 1.0}
    emotions = result[0]
    return {e["label"].lower(): round(e["score"], 2) for e in emotions}

def analyze_pacing(transcript, duration_minutes=1):
    if not transcript.strip() or duration_minutes <= 0:
        return {"error": "Invalid input"}
    words = transcript.split()
    wpm = len(words) / duration_minutes
    pacing_category = "slow" if wpm < SLOW_THRESHOLD else "fast" if wpm > FAST_THRESHOLD else "normal"
    return {
        "words_per_minute": round(wpm, 5),
        "transcript_wpm": round(len(words) / duration_minutes, 5),
        "pacing_category": pacing_category,
        "readability": round(textstat.flesch_reading_ease(transcript), 5),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(transcript), 5),
        "gunning_fog_index": round(textstat.gunning_fog(transcript), 5)
    }

def analyze_grammar_confidence(transcript):
    if not transcript.strip():
        return {"grammar_feedback": "No transcript found.", "confidence": 0.0}
    
    prompt = f"""You are a professional English language evaluator and interview coach. Analyze the following interview transcript based on the following five categories. Be clear, concise, and constructive in your feedback.
Format your response with proper headers and bullet points for readability.
Analyze:
1. Grammar Issues – Identify major grammar problems (e.g., tense usage, redundancy, fragments, etc.).
2. Answer Clarity – Are the responses understandable and meaningful? (Yes/No) Explain briefly.
3. Suggestions for Improvement – How can the candidate improve grammar, clarity, and expression?
4. Confidence Score – Give a confidence score between 0% to 100% for the transcript's overall quality.
5. Areas for Improvement – Mention specific skills or habits the candidate should work on.

    Text: {transcript}
    """

    try:
        response = gemini_model.generate_content([prompt])
        output = response.candidates[0].content.parts[0].text

        return {
            "grammar_feedback": output,
            "confidence": 0.85  # Static for now, or you can parse from output
        }
    except Exception as e:
        return {"grammar_feedback": f"Error: {str(e)}", "confidence": 0.0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prepare')
def upload_form():
    return render_template('record.html')

@app.route('/career-test')
def career_test():
    return render_template('career-test.html')

@app.route('/personality')
def personality_test():
    return render_template('personality.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400

        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No selected file!"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type!"}), 400

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        sanitized_filename = sanitize_filename(file.filename)
        filename = f"{timestamp}_{sanitized_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        filepath = convert_to_wav(filepath)

        if os.path.getsize(filepath) == 0:
            raise Exception("Uploaded file is empty")

        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=16)

        transcript = ""
        speaker_segments = []
        for seg in result["segments"]:
            speaker = "Speaker"
            text = seg.get("text", "")
            transcript += f"{speaker}: {text}\n"
            speaker_segments.append({"speaker": speaker, "text": text})

        emotion_result = analyze_emotion(transcript)
        pacing_result = analyze_pacing(transcript, duration_minutes=1)
        grammar_confidence_result = analyze_grammar_confidence(transcript)

        summary = "This interview analysis reflects the overall tone and pacing of the speaker."

        emotion_keys = list(emotion_result.keys())
        emotion_values = list(emotion_result.values())

        return render_template("result.html",
                               transcription=transcript,
                               speaker_segments=speaker_segments,
                               emotion_keys=emotion_keys,
                               emotion_values=emotion_values,
                               pacing_results=pacing_result,
                               summary=summary,
                               transcript_wpm=pacing_result['transcript_wpm'],
                               grammar_feedback=grammar_confidence_result["grammar_feedback"],
                               confidence_score=grammar_confidence_result["confidence"])

    except Exception as e:
        logging.error(f"Error in transcribe_audio: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/download', methods=['POST'])
def download_text():
    transcription = request.form.get('text', '')
    if not transcription:
        return "No transcription to download", 400

    response = make_response(transcription)
    response.headers['Content-Disposition'] = 'attachment; filename=transcription.txt'
    response.headers['Content-Type'] = 'text/plain'
    return response

if __name__ == '__main__':
    app.run(debug=True)
