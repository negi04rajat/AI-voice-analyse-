import textstat
from transformers import pipeline

# Initialize the emotion classifier pipeline (DistilRoBERTa-based)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Constants for pacing analysis
SLOW_THRESHOLD = 100  # WPM below this is considered "slow"
FAST_THRESHOLD = 160  # WPM above this is considered "fast"

def analyze_emotion(transcript):
    """
    Analyzes the emotional tone of a given transcript.
    Returns a dictionary with emotion labels and corresponding scores.

    Args:
        transcript (str): The text to analyze.

    Returns:
        dict: A dictionary with emotion labels (e.g., happy, sad) and their scores.
    """
    if not transcript.strip():
        return {"neutral": 1.0}

    # Limit the length of transcript input for processing efficiency (max 512 tokens for DistilRoBERTa)
    result = emotion_classifier(transcript[:512])
    
    # Check if result is empty (handle edge case)
    if not result:
        return {"neutral": 1.0}

    # Extract emotion labels and their scores, rounded to 2 decimal places
    emotions = result[0]
    return {e["label"].lower(): round(e["score"], 2) for e in emotions}

def analyze_pacing(transcript, duration_minutes=1):
    """
    Analyzes the pacing of a given transcript based on words per minute (WPM).
    Categorizes pacing as slow, normal, or fast, and calculates readability scores.

    Args:
        transcript (str): The text to analyze.
        duration_minutes (float): Duration of the speech in minutes.

    Returns:
        dict: A dictionary with WPM, pacing category, and readability scores.
    """
    if not transcript.strip() or duration_minutes <= 0:
        return {"error": "Invalid input. Ensure transcript is non-empty and duration is positive."}

    # Split the transcript into words and calculate words per minute (WPM)
    words = transcript.split()
    wpm = len(words) / duration_minutes

    # Categorize pacing based on WPM
    if wpm < SLOW_THRESHOLD:
        pacing_category = "slow"
    elif wpm > FAST_THRESHOLD:
        pacing_category = "fast"
    else:
        pacing_category = "normal"

    # Compute readability scores
    readability_score = textstat.flesch_reading_ease(transcript)
    grade_level = textstat.flesch_kincaid_grade(transcript)
    gunning_fog_index = textstat.gunning_fog(transcript)

    return {
        "words_per_minute": round(wpm, 2),
        "pacing_category": pacing_category,
        "readability": round(readability_score, 2),
        "flesch_kincaid_grade": round(grade_level, 2),
        "gunning_fog_index": round(gunning_fog_index, 2)
    }

# Example usage
if __name__ == "__main__":
    sample_transcript = "I am feeling a little anxious, but also hopeful about the future."
    print(analyze_emotion(sample_transcript))

    pacing_result = analyze_pacing(sample_transcript, duration_minutes=2)
    print(pacing_result)
